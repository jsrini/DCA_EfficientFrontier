"""Unified income-tax-aware DCA backtest + block-bootstrap Monte Carlo engine.

One core (tax accrual, bootstrap, metrics, strategy runners) parameterized by a WINDOW
config: a dataset, an asset-ROLE -> column-name map, a salary path, and a transition
schedule. Strategies are defined ONCE in asset roles (STOCK/GOLD/BOND/...), so any strategy
whose roles a window supports runs in that window automatically -- the stocks/gold/bonds
mixes run in both the 1995-2026 and 1980-2026 windows from the same code.

Consolidates the tax accrual, bootstrap, metrics, and strategy runners that previously lived in
separate per-window scripts into one window-parameterized engine.

Usage:  python3 dca_core.py 1995     |     python3 dca_core.py 1980
"""
import os, sys, numpy as np, pandas as pd
from scipy.optimize import brentq
from div_rebin import build_div_cadence, respace_received
RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---------------------------------------------------------------------------
# Window configs: dataset + asset-role->column map + decade transition dates.
# ---------------------------------------------------------------------------
WINDOWS = {
    "1995": dict(
        adj="dca_adj_div.csv", yld="dca_yield_div.csv",
        cols={"STOCK": "SPY", "TECH": "QQQ", "INTL": "VGTSX", "BOND": "VBMFX",
              "LONGT": "VUSTX", "CASH": "VFISX", "GOLD": "GOLD", "REIT": "VGSIX"},
        transitions=["2005-01-15", "2015-01-15"]),
    "1980": dict(
        adj="dca_adj_div_80full.csv", yld="dca_yield_div_80full.csv",
        cols={"STOCK": "STOCK", "TECH": "TECH", "INTL": "INTL", "GOLD": "GOLD", "BOND": "BOND",
              "LONGT": "LONGT", "CASH": "CASH", "REIT": "REIT"},
        transitions=["1990-01-15", "2000-01-15"]),
}

# Tax rate on income, by asset ROLE: bonds/Treasuries 20%, equities/REIT 25%, gold none.
ROLE_RATE = {"STOCK": .25, "TECH": .25, "INTL": .25, "REIT": .25,
             "BOND": .20, "LONGT": .20, "CASH": .20, "GOLD": 0.0}
CG = 0.25  # capital-gains rate on realized (hard-reset / momentum) sales
GOLD_FEE = 0.004  # annual management fee on the gold leg, modeled as a daily NAV drag (configurable)
TXN_FEE = 0.001   # transaction fee charged on every buy and every sell (configurable)

# Strategies, defined once in asset roles (weights sum to 1).
STRATEGIES = {
    "30/30/40":      {"STOCK": .30, "GOLD": .30, "BOND": .40},
    "40/25/35":      {"STOCK": .40, "GOLD": .25, "BOND": .35},
    "50/25/25":      {"STOCK": .50, "GOLD": .25, "BOND": .25},
    "40/30/30":      {"STOCK": .40, "GOLD": .30, "BOND": .30},
    "60/40":         {"STOCK": .60, "BOND": .40},
    "PermPort":      {"STOCK": .25, "LONGT": .25, "GOLD": .25, "CASH": .25},
    "Conservative":  {"STOCK": .15, "INTL": .15, "GOLD": .25, "BOND": .20, "REIT": .15, "CASH": .10},
    "Aggressive":    {"TECH": .30, "INTL": .30, "GOLD": .40},
    "US/Intl 50/50": {"STOCK": .50, "INTL": .50},
    "All Weather":   {"STOCK": .30, "LONGT": .40, "BOND": .15, "GOLD": .15},
    "Three-fund":    {"STOCK": .40, "INTL": .20, "BOND": .40},
    "70/30":         {"STOCK": .70, "BOND": .30},
    "80/20":         {"STOCK": .80, "BOND": .20},
}
# Age-based glides: three phases (start, +10yr, +20yr), each naming a strategy above.
GLIDES = {
    "glA": ["Aggressive", "30/30/40", "PermPort"],
    "glB": ["Aggressive", "60/40", "Conservative"],
    "glC": ["US/Intl 50/50", "60/40", "Conservative"],
    "glD": ["50/25/25", "40/30/30", "30/30/40"],
}

def salary(d):
    """75th-percentile income path; extended back before 1995 on the SSA wage index."""
    y = d.year
    if y >= 2000: return 50000.0 * 1.030513 ** (y - 2000)
    if y >= 1995: return 42000.0 * 1.03537 ** (y - 1995)
    return 42000.0 / 1.0464 ** (1995 - y)


class Engine:
    def __init__(self, window, salary_fn=salary, shelter=0.0, rebin=True, gold_fee=GOLD_FEE, txn_fee=TXN_FEE):
        # shelter = fraction of every tax rate that is waived (the account is tax-sheltered).
        #   0.0 = fully taxable account (report premise); 1.0 = fully tax-free account (Roth/401k).
        # rebin = apply the dividend-cadence normalization in the bootstrap (True, the method);
        #   False reproduces the original engines' raw resampled dividend dates (for impact testing).
        # gold_fee = annual management fee on the gold leg, modeled as a daily price erosion (an ETF
        #   expense-ratio NAV drag, not a realized sale, so it triggers no capital-gains event).
        # txn_fee = proportional cost charged on every buy and every sell.
        cfg = WINDOWS[window]
        self.window = window; self.cfg = cfg; self.shelter = shelter; self.rebin = rebin
        self.gf = gold_fee; self.tc = txn_fee
        self._track = False           # when True, runners record self._shist (share path) for the re-bin
        self.adj = pd.read_csv(cfg["adj"], index_col=0, parse_dates=True)
        self.yld = pd.read_csv(cfg["yld"], index_col=0, parse_dates=True)
        self.ASSETS = list(self.adj.columns); self.A = len(self.ASSETS)
        self.col = cfg["cols"]; self.roles = set(self.col)
        self.idx = self.adj.index; self.T = len(self.idx)
        self.P0 = self.adj.iloc[0].values.astype(float)
        self.rets = self.adj.pct_change().dropna().values.astype(float)
        self.yrow = self.yld.reindex(self.idx).fillna(0.0)[self.ASSETS].values[1:]
        self.N = len(self.rets)
        self.mon = np.array([d.month for d in self.idx]); self.yr = np.array([d.year for d in self.idx])
        col2role = {v: k for k, v in self.col.items()}
        self.RATE = np.array([ROLE_RATE[col2role[a]] for a in self.ASSETS]) * (1.0 - shelter)
        self.CG = CG * (1.0 - shelter)
        self.gold_i = self.ASSETS.index(self.col["GOLD"]) if "GOLD" in self.roles else None
        self.gold_decay = (1.0 - self.gf / 252.0) ** np.arange(self.T)  # cumulative daily gold management-fee drag
        cd = []; ym = set()
        for i, d in enumerate(self.idx):
            if d.day >= 15 and (d.year, d.month) not in ym: cd.append(i); ym.add((d.year, d.month))
        self.cd = np.array(cd); self.cdset = set(self.cd.tolist()); self.cpos = {int(c): k for k, c in enumerate(self.cd)}
        self.set_salary(salary_fn)
        self.trans = [int(next(i for i in self.cd if self.idx[i] >= pd.Timestamp(t))) for t in cfg["transitions"]]
        self.prev12 = np.clip(np.searchsorted(self.idx.values, (self.idx - pd.DateOffset(years=1)).values, side="right") - 1, 0, self.T - 1)
        self._build_div_cadence()

    # -- dividend cadence (bootstrap re-bin) ---------------------------------
    def _build_div_cadence(self):
        """Precompute the income re-bin cadence from the actual yield history (shared logic in
        div_rebin.build_div_cadence). Used only by the bootstrap (synth); the actual backtest
        keeps its true ex-dates."""
        self.div_K, self._yr_vals, self._yr_pos, self._dep_pos = build_div_cadence(
            self.yld, self.idx, self.yr, self.ASSETS)

    def run_rebin(self, base, P, Y):
        """Two-pass received-preserving dividend re-bin for a single strategy run. Pass 1 records
        the share path with the raw dividends; pass 2 runs on dividends re-spaced to a realistic
        even cadence with each year's received total held fixed (div_rebin.respace_received).
        Returns the pass-2 metrics."""
        if not self.rebin:
            return base(P, Y)
        self._track = True; base(P, Y); self._track = False        # pass 1: capture self._shist
        Yadj = respace_received(Y, P, self._shist, self._yr_vals, self._yr_pos, self._dep_pos)
        return base(P, Yadj)                                       # pass 2: scored run

    def set_salary(self, sf):
        self.camt = np.array([sf(self.idx[i]) * 0.20 / 12.0 for i in self.cd]); self.INV = self.camt.sum()
        self.cyr = np.array([(self.idx[i] - self.idx[self.cd[0]]).days / 365.25 for i in self.cd])
        self.ty = (self.idx[-1] - self.idx[self.cd[0]]).days / 365.25
        self.term_mask = np.asarray(self.idx >= (self.idx[-1] - pd.DateOffset(years=5)))  # final 5 years

    # -- role helpers --------------------------------------------------------
    def supports(self, strat): return set(STRATEGIES[strat]).issubset(self.roles)
    def wv(self, strat):
        v = np.zeros(self.A)
        for role, x in STRATEGIES[strat].items(): v[self.ASSETS.index(self.col[role])] = x
        return v
    def wvec(self, d):
        """Build a weight vector from an arbitrary {role: weight} dict (generalizes wv)."""
        v = np.zeros(self.A)
        for role, x in d.items(): v[self.ASSETS.index(self.col[role])] = x
        return v
    def phases(self, glide):
        names = GLIDES[glide]
        return [(int(self.cd[0]), self.wv(names[0])), (self.trans[0], self.wv(names[1])), (self.trans[1], self.wv(names[2]))]

    # -- bootstrap -----------------------------------------------------------
    def make_idx(self, rng):
        out = np.empty(self.T - 1, dtype=int); f = 0
        while f < self.T - 1:
            s = int(rng.integers(0, self.N)); L = int(rng.integers(22, 601)); k = min(L, self.N - s, (self.T - 1) - f)
            out[f:f + k] = np.arange(s, s + k); f += k
        return out
    def synth(self, ix):
        P = np.empty((self.T, self.A)); P[0] = self.P0; P[1:] = self.P0 * np.cumprod(1.0 + self.rets[ix], axis=0)
        if self.gold_i is not None: P[:, self.gold_i] *= self.gold_decay   # gold management-fee NAV drag
        Y = np.empty((self.T, self.A)); Y[0] = 0.0; Y[1:] = self.yrow[ix]
        return P, Y                              # raw resampled yields; the income re-bin is applied in-run
    def paths(self, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        return [self.synth(self.make_idx(rng)) for _ in range(n)]
    def actual(self):
        Pa = self.adj.values.astype(float)
        if self.gold_i is not None:
            Pa = Pa.copy(); Pa[:, self.gold_i] *= self.gold_decay         # gold management-fee NAV drag
        return Pa, self.yld.reindex(self.idx).fillna(0.0)[self.ASSETS].values

    # -- metrics -------------------------------------------------------------
    def metrics(self, val):
        fv = val[-1]
        cfa = np.append(-self.camt, fv); cft = np.append(self.cyr, self.ty)
        try: irr = brentq(lambda r: np.sum(cfa * (1 + r) ** (-cft)), -0.5, 0.6)
        except Exception: irr = np.nan
        pk = np.maximum.accumulate(val); dd = (val - pk) / np.where(pk > 0, pk, 1)
        tr = int(dd.argmin()); mdd = float(dd[tr]); peak = int(val[:tr + 1].argmax())
        ptt = (self.idx[tr] - self.idx[peak]).days / 365.25
        aft = np.where(val[tr:] >= val[peak])[0]
        if len(aft):
            rec = tr + int(aft[0]); dur = (self.idx[rec] - self.idx[peak]).days / 365.25; recov = 1.0; resid = 0.0
        else:
            dur = (self.idx[-1] - self.idx[peak]).days / 365.25; recov = 0.0; resid = (val[-1] - val[peak]) / val[peak]
        # Subtract net (post-fee) cash, not gross camt: val[t] already reflects the buy-fee reduction
        # because runners add camt*(1-tc) to shares. Using gross camt here would double-charge the fee
        # -- once in val[t], again as a phantom return drag -- biasing daily TWR (and Sharpe) down.
        c = np.zeros(self.T); c[self.cd] = self.camt * (1.0 - self.tc)
        tw = np.zeros(self.T); nz = val[:-1] > 0
        tw[1:][nz] = (val[1:][nz] - c[1:][nz]) / val[:-1][nz] - 1; tw = tw[1:]
        cum = np.cumprod(1 + tw); n = len(tw)
        ta = cum[-1] ** (252 / n) - 1; vol = tw.std(ddof=1) * np.sqrt(252); sharpe = (ta - 0.03) / vol
        # terminal max DD: worst drawdown whose TROUGH lands in the final 5 years, measured from the
        # running high-water mark (peak may sit before the window). Trough windowed, peak is not.
        tmdd = float(dd[self.term_mask].min()) if self.term_mask.any() else float("nan")
        return np.array([fv, irr, mdd, sharpe, dur, recov, ptt, resid, tmdd])

    def _pay_income(self, sh, P_t, basis, incyr, py, paid, cgyr, t_yr):
        """Sell shares pro-rata per leg to pay last year's income tax. The forced sale realizes
        capital gains pro-rata per leg (signed -- losses are kept, not clipped, so they offset
        gains via the loss-carryforward logic in _pay_cg); booked into cgyr[t_yr] for next year.
        Returns the updated share and basis vectors."""
        if py in incyr and py not in paid:
            due = self.RATE * incyr[py]
            sell = np.minimum(sh, due / ((1.0 - self.tc) * np.where(P_t > 0, P_t, 1)))
            frac = np.where(sh > 0, sell / np.where(sh > 0, sh, 1.0), 0.0)
            gain = float((frac * ((1 - self.tc) * sh * P_t - basis)).sum())    # signed; (1-tc) applies sell fee
            cgyr[t_yr] = cgyr.get(t_yr, 0.0) + gain
            sh = sh - sell; basis = basis * (1.0 - frac); paid.add(py)
        return sh, basis

    def _pay_cg(self, sh, P_t, basis, cgyr, py, paidcg, t_yr, loss_carryforward):
        """Pay last year's net realized capital gains, applying any accumulated loss carryforward.
        Policy (US-tax §1211/§1212 approximation):
          net = cgyr[py] - loss_carryforward         # signed; current-year position offset by past unused losses
          if net > 0: pay CG * net;    new carry = 0
          if net <=0: no CG;           new carry = max(0, -net - 3000)
                     # $3k of net loss is notionally consumed by the salary offset (no engine credit
                     # because salary tax isn't modelled); the rest carries forward to future years.
        The forced sale to pay CG tax itself realizes additional signed gain, booked into cgyr[t_yr].
        Returns (sh, basis, cgtax_paid, new_loss_carryforward)."""
        cgtax = 0.0
        if py in cgyr and py not in paidcg:
            net = cgyr[py] - loss_carryforward
            if net > 0:
                due = self.CG * net; tv = (sh * P_t).sum()
                if due > 0 and tv > 0:
                    frac = min(1.0, due / ((1.0 - self.tc) * tv))
                    gain = float((frac * ((1 - self.tc) * sh * P_t - basis)).sum())  # signed; (1-tc) applies sell fee
                    cgyr[t_yr] = cgyr.get(t_yr, 0.0) + gain
                    sh = sh * (1 - frac); basis = basis * (1 - frac); cgtax = min(due, tv)
                loss_carryforward = 0.0
            else:
                loss_carryforward = max(0.0, -net - 3000.0)
            paidcg.add(py)
        return sh, basis, cgtax, loss_carryforward

    # -- strategy runners ----------------------------------------------------
    def run_static(self, P, Y, w, soft):
        A, T, cd, cdset, mon, yr, cpos, camt = self.A, self.T, self.cd, self.cdset, self.mon, self.yr, self.cpos, self.camt
        act = w > 0; sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T)
        incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0; loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:
                sh, basis = self._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct, loss_carryforward = self._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t], loss_carryforward); cgtax += ct
            if t in cdset:
                a = camt[cpos[t]]; pv = sh * P[t]
                if soft:
                    tot = pv[act].sum()
                    if tot <= 0: ww = w.copy()
                    else:
                        u = act & (pv <= w * tot); ww = np.where(u, w, 0.0); ww = w.copy() if ww.sum() <= 0 else ww / ww.sum()
                else: ww = w
                sh = sh + a * ww * (1.0 - self.tc) / P[t]; basis = basis + a * ww
            val[t] = (sh * P[t]).sum()
        if self._track: self._shist = shist
        self.last_cgtax = cgtax; self.last_val = val
        tvf = (sh * P[-1]).sum(); self.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(self.A)
        return self.metrics(val)

    def run_static_hard(self, P, Y, w):
        """Static target w with an ANNUAL HARD REBALANCE on March 15 (the March contribution date):
        sell/buy every leg back to w, realizing capital gains taxed the following January. Income
        tax and realized-gain tax are both settled in January; the rebalance itself is in March.
        Mirrors the hard-reset accounting in run_contglide('hard'), with a constant target."""
        A, T, cd, cdset, mon, yr, cpos, camt = self.A, self.T, self.cd, self.cdset, self.mon, self.yr, self.cpos, self.camt
        sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T)
        incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0; loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:
                sh, basis = self._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct, loss_carryforward = self._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t], loss_carryforward); cgtax += ct
            if t in cdset and mon[t] == 3:                                  # annual hard rebalance, March 15
                cv = sh * P[t]; tv = cv.sum()
                if tv > 0:
                    tgt0 = tv * w
                    fee = self.tc * np.abs(tgt0 - cv).sum()                  # 0.1% on the sell leg + the buy leg
                    realized = np.sum(np.where((cv > 0) & (tgt0 < cv), (1 - tgt0 / np.where(cv > 0, cv, 1)) * ((1 - self.tc) * cv - basis), 0.0))
                    cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + realized
                    nval = (tv - fee) * w; sh = np.where(nval > 0, nval / P[t], 0.0)
                    basis = np.where(cv <= 0, tgt0, np.where(tgt0 <= cv, basis * (tgt0 / np.where(cv > 0, cv, 1)), basis + (tgt0 - cv)))
            if t in cdset:
                a = camt[cpos[t]]
                sh = sh + a * w * (1.0 - self.tc) / P[t]; basis = basis + a * w
            val[t] = (sh * P[t]).sum()
        if self._track: self._shist = shist
        self.last_cgtax = cgtax; self.last_val = val
        tvf = (sh * P[-1]).sum(); self.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(self.A)
        return self.metrics(val)

    def run_glide(self, P, Y, phases, reset):
        A, T, cd, cdset, mon, yr, cpos, camt = self.A, self.T, self.cd, self.cdset, self.mon, self.yr, self.cpos, self.camt
        pidx = [p[0] for p in phases]
        def wat(i):
            w = phases[0][1]
            for j, ww in phases:
                if i >= j: w = ww
            return w
        sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T); incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0; loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:
                sh, basis = self._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct, loss_carryforward = self._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t], loss_carryforward); cgtax += ct
            if reset == "hard" and t in pidx and t != phases[0][0]:
                w = wat(t); cv = sh * P[t]; tv = cv.sum(); tgt0 = tv * w
                fee = self.tc * np.abs(tgt0 - cv).sum()                       # 0.1% on the sell leg + the buy leg
                realized = np.sum(np.where((cv > 0) & (tgt0 < cv), (1 - tgt0 / np.where(cv > 0, cv, 1)) * ((1 - self.tc) * cv - basis), 0.0))
                cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + realized
                nval = (tv - fee) * w; sh = np.where(nval > 0, nval / P[t], 0.0)
                basis = np.where(cv <= 0, tgt0, np.where(tgt0 <= cv, basis * (tgt0 / np.where(cv > 0, cv, 1)), basis + (tgt0 - cv)))
            if t in cdset:
                w = wat(t); a = camt[cpos[t]]; pv = sh * P[t]; tot = pv[w > 0].sum()
                if tot <= 0: ww = w.copy()
                else:
                    u = (w > 0) & (pv <= w * tot); ww = np.where(u, w, 0.0); ww = w.copy() if ww.sum() <= 0 else ww / ww.sum()
                sh = sh + a * ww * (1.0 - self.tc) / P[t]; basis = basis + a * ww
            val[t] = (sh * P[t]).sum()
        if self._track: self._shist = shist
        self.last_cgtax = cgtax; self.last_val = val
        tvf = (sh * P[-1]).sum(); self.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(self.A)
        return self.metrics(val)

    def run_contglide(self, P, Y, w0, w1, reset, end=None):
        """Continuous (Vanguard-style) glide: target weights interpolate linearly from w0 to w1
        over the horizon by calendar time, reaching w1 at `end` (default the window end) and holding
        it after. soft = steer contributions only, never sell; hard = rebalance to the year's target
        each January (realizes gains, taxed) -- a faithful taxable TDF. `end` (a Timestamp) lets a
        shorter-horizon investor run a glide that completes at their retirement, not at year 46."""
        A, T, cd, cdset, mon, yr, cpos, camt = self.A, self.T, self.cd, self.cdset, self.mon, self.yr, self.cpos, self.camt
        t0 = self.cd[0]; span = max(1.0, ((self.idx[-1] if end is None else end) - self.idx[t0]).days)
        def wat(t):
            frac = min(1.0, max(0.0, (self.idx[t] - self.idx[t0]).days / span))
            return w0 + (w1 - w0) * frac
        sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T); incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0; loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:
                sh, basis = self._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct, loss_carryforward = self._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t], loss_carryforward); cgtax += ct
                if reset == "hard":                                            # annual rebalance to the glide target
                    w = wat(t); cv = sh * P[t]; tv = cv.sum(); tgt0 = tv * w
                    if tv > 0:
                        fee = self.tc * np.abs(tgt0 - cv).sum()                 # 0.1% on the sell leg + the buy leg
                        realized = np.sum(np.where((cv > 0) & (tgt0 < cv), (1 - tgt0 / np.where(cv > 0, cv, 1)) * ((1 - self.tc) * cv - basis), 0.0))
                        cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + realized
                        nval = (tv - fee) * w; sh = np.where(nval > 0, nval / P[t], 0.0)
                        basis = np.where(cv <= 0, tgt0, np.where(tgt0 <= cv, basis * (tgt0 / np.where(cv > 0, cv, 1)), basis + (tgt0 - cv)))
            if t in cdset:
                w = wat(t); a = camt[cpos[t]]; pv = sh * P[t]
                if reset == "hard":
                    ww = w
                else:
                    tot = pv[w > 0].sum()
                    if tot <= 0: ww = w.copy()
                    else:
                        u = (w > 0) & (pv <= w * tot); ww = np.where(u, w, 0.0); ww = w.copy() if ww.sum() <= 0 else ww / ww.sum()
                sh = sh + a * ww * (1.0 - self.tc) / P[t]; basis = basis + a * ww
            val[t] = (sh * P[t]).sum()
        if self._track: self._shist = shist
        self.last_cgtax = cgtax; self.last_val = val
        tvf = (sh * P[-1]).sum(); self.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(self.A)
        return self.metrics(val)

    def run_gem(self, P, Y, rebal):
        A, T, cd, cdset, mon, yr, cpos, camt, prev12 = self.A, self.T, self.cd, self.cdset, self.mon, self.yr, self.cpos, self.camt, self.prev12
        US, INTL, BOND, RF = (self.ASSETS.index(self.col[r]) for r in ("STOCK", "INTL", "BOND", "CASH"))
        held = None; lots = []; cgyr = {}; incyr = {}; paid = set(); paidcg = set(); val = np.zeros(T); sh = 0.0
        loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track and held is not None: shist[t, held] = sh
            if held is not None:
                incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)); incyr[yr[t]][held] += Y[t][held] * sh * P[t][held]
            if t in cdset and mon[t] == 1:
                py = yr[t] - 1
                if held is not None and py in incyr and py not in paid:
                    due = (self.RATE * incyr[py]).sum(); cur = sh * P[t][held]
                    if due > 0 and cur > 0:
                        s = min(sh, due / ((1.0 - self.tc) * P[t][held]))
                        if lots:
                            tot = sum(l[1] for l in lots); fr = min(1.0, s / tot) if tot > 0 else 0
                            basis_sold = fr * sum(l[2] for l in lots)
                            cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + ((1 - self.tc) * s * P[t][held] - basis_sold)
                            for l in lots: l[1] -= l[1] * fr; l[2] -= l[2] * fr
                        sh -= s
                    paid.add(py)
                if held is not None and py in cgyr and py not in paidcg:
                    net = cgyr[py] - loss_carryforward                        # apply prior unused losses
                    if net > 0:
                        due = self.CG * net; cur = sh * P[t][held]
                        if due > 0 and cur > 0:
                            s = min(sh, due / ((1.0 - self.tc) * P[t][held]))
                            if lots:
                                tot = sum(l[1] for l in lots); fr = min(1.0, s / tot) if tot > 0 else 0
                                basis_sold = fr * sum(l[2] for l in lots)
                                cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + ((1 - self.tc) * s * P[t][held] - basis_sold)
                                for l in lots: l[1] -= l[1] * fr; l[2] -= l[2] * fr
                            sh -= s
                        loss_carryforward = 0.0
                    else:
                        loss_carryforward = max(0.0, -net - 3000.0)           # $3k notionally consumed; rest carries
                    paidcg.add(py)
            if mon[t] in rebal or held is None:
                if t in cdset or held is None:
                    u = P[t][US] / P[prev12[t]][US] - 1; it = P[t][INTL] / P[prev12[t]][INTL] - 1; rf = P[t][RF] / P[prev12[t]][RF] - 1
                    tgt = (US if u >= it else INTL) if u > rf else BOND
                    if held is None: held = tgt
                    elif tgt != held:
                        Ph = P[t][held]; g = sh * Ph * (1 - self.tc) - sum(l[2] for l in lots)
                        cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + g
                        proceeds = sh * Ph * (1.0 - self.tc)                  # sell-leg fee
                        inv = proceeds * (1.0 - self.tc)                      # buy-leg fee on the redeploy
                        sh = inv / P[t][tgt]; lots = [[t, sh, inv]]; held = tgt
            if t in cdset:
                a = camt[cpos[t]]; q = a * (1.0 - self.tc) / P[t][held]; sh += q; lots.append([t, q, a])                  # basis = gross contribution (US tax: cost includes commissions)
            val[t] = sh * P[t][held] if held is not None else 0.0
        if self._track: self._shist = shist
        self.last_val = val; self.last_w = np.zeros(self.A)
        if held is not None: self.last_w[held] = 1.0
        return self.metrics(val)

    def run_signal(self, P, Y, mode):
        """Trend-following / mean-reversion contribution overlays on stock/gold/bond.
        Each month the contribution is split: a HARD signal chunk is placed directly on the
        trailing-12-month winner and/or loser (ignoring current weight, never selling), and the
        REMAINDER (including any signal chunk that does not fire) is soft-allocated toward equal
        thirds (1/3 each) on the updated holdings -- a common pot.
          TF   : 25% hard to the winner if its 12m return > +1% (else that 25% joins the pot).
          MR   : 25% hard to the loser  if its 12m return < -1% (else joins the pot).
          TFMR : 12.5% hard to winner (if >+1%) + 12.5% hard to loser (if <-1%); each unfired
                 half joins the pot (it does NOT cross to the other signal).
        Soft (never sells on signal) -> no rebalance-driven gains, but the January income-tax
        sale still realizes (and books) capital gains.  mode in {"TF","MR","TFMR"}."""
        A, T, cdset, mon, yr, cpos, camt, prev12 = self.A, self.T, self.cdset, self.mon, self.yr, self.cpos, self.camt, self.prev12
        sgb = [self.ASSETS.index(self.col[r]) for r in ("STOCK", "GOLD", "BOND")]
        tf_amt = 0.25 if mode == "TF" else (0.125 if mode == "TFMR" else 0.0)
        mr_amt = 0.25 if mode == "MR" else (0.125 if mode == "TFMR" else 0.0)
        base = np.zeros(A)
        for i in sgb: base[i] = 1.0 / 3.0          # equal-thirds soft target on stock/gold/bond
        act = base > 0
        sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T)
        incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0; loss_carryforward = 0.0
        shist = np.zeros((T, A)) if self._track else None
        for t in range(T):
            if self._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:
                sh, basis = self._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct, loss_carryforward = self._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t], loss_carryforward); cgtax += ct
            if t in cdset:
                a = camt[cpos[t]]; soft = a
                r = np.array([P[t][i] / P[prev12[t]][i] - 1 for i in sgb])   # trailing 12m total return
                wi = int(r.argmax()); li = int(r.argmin())
                if tf_amt > 0 and r[wi] > 0.01:                              # hard to winner
                    q = tf_amt * a * (1.0 - self.tc) / P[t][sgb[wi]]; sh[sgb[wi]] += q
                    basis[sgb[wi]] += tf_amt * a; soft -= tf_amt * a
                if mr_amt > 0 and r[li] < -0.01:                            # hard to loser
                    q = mr_amt * a * (1.0 - self.tc) / P[t][sgb[li]]; sh[sgb[li]] += q
                    basis[sgb[li]] += mr_amt * a; soft -= mr_amt * a
                pv = sh * P[t]; tot = pv[act].sum()                         # soft-allocate the pot on updated holdings
                if tot <= 0: ww = base.copy()
                else:
                    u = act & (pv <= base * tot); ww = np.where(u, base, 0.0); ww = base.copy() if ww.sum() <= 0 else ww / ww.sum()
                sh = sh + soft * ww * (1.0 - self.tc) / P[t]; basis = basis + soft * ww
            val[t] = (sh * P[t]).sum()
        if self._track: self._shist = shist
        self.last_cgtax = cgtax; self.last_val = val
        tvf = (sh * P[-1]).sum(); self.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(self.A)
        return self.metrics(val)

    # -- build the runnable field for this window ----------------------------
    def field(self):
        """Returns [(label, fn(P,Y)->metrics)] for every strategy this window supports."""
        R = []
        for nm in STRATEGIES:
            if not self.supports(nm): continue
            w = self.wv(nm)
            R.append((f"{nm} soft", lambda P, Y, w=w: self.run_static(P, Y, w, True)))
            R.append((f"{nm} no-reb", lambda P, Y, w=w: self.run_static(P, Y, w, False)))
            R.append((f"{nm} hard", lambda P, Y, w=w: self.run_static_hard(P, Y, w)))
        for g in GLIDES:
            if not all(self.supports(n) for n in GLIDES[g]): continue
            ph = self.phases(g)
            for r in ("soft", "hard"):
                R.append((f"{g} {r}", lambda P, Y, ph=ph, r=r: self.run_glide(P, Y, ph, r)))
        if all(r in self.roles for r in ("STOCK", "INTL", "BOND", "CASH")):
            R.append(("GEM monthly", lambda P, Y: self.run_gem(P, Y, frozenset(range(1, 13)))))
            R.append(("GEM quarterly", lambda P, Y: self.run_gem(P, Y, frozenset({1, 4, 7, 10}))))
        if all(r in self.roles for r in ("STOCK", "GOLD", "BOND")):
            for mode, label in [("TF", "TrendFollow"), ("MR", "MeanRevert"), ("TFMR", "TF+MR")]:
                R.append((label, lambda P, Y, m=mode: self.run_signal(P, Y, m)))
        return R


# ---------------------------------------------------------------------------
# CLI: run a window's whole field (actual + 1,000-path MC), print like the report.
# ---------------------------------------------------------------------------
def recstr(p): return f"{p[4]:.1f}y" if p[5] == 1 else f"{p[7] * 100:.0f}% at end"

def run_window(window, n=1000, seed=42, shelter=0.0):
    import time; t0 = time.time()
    e = Engine(window, shelter=shelter); R = e.field(); Pa, Ya = e.actual()
    tag = "taxfree" if shelter >= 1.0 else "taxable"
    print(f"=== WINDOW {window} [{tag}]  ({e.idx[0].date()}..{e.idx[-1].date()}, {e.ty:.1f}y)  contributed ${e.INV/1e3:,.0f}k ===")
    print(f"{'Strategy':16s} {'Final':>7s} {'IRR':>6s} {'Sharpe':>7s} {'MaxDD':>7s}")
    act = {}
    for nm, fn in R:
        m = fn(Pa, Ya); act[nm] = m
        print(f"{nm:16s} ${m[0]/1e6:>5.2f}M {m[1]*100:>5.1f}% {m[3]:>7.3f} {m[2]*100:>6.1f}%")
    paths = e.paths(n, seed)
    data = {nm: np.array([e.run_rebin(fn, P, Y) for P, Y in paths]) for nm, fn in R}
    np.save(os.path.join(RESULTS, f"core_{window}_{tag}.npy"), {"names": [n for n, _ in R], "data": data, "act": act}, allow_pickle=True)
    def tbl(title, col, q):
        print("\n=== " + title + " ===")
        print(f"{'Strategy':16s} {'Final':>7s} {'IRR':>6s} {'MaxDD':>7s} {'pk>tr':>6s} {'recovery':>12s} {'Sharpe':>7s}")
        for nm, _ in R:
            Rr = data[nm]; p = Rr[np.argsort(Rr[:, col])[int(round(q * n))]]
            print(f"{nm:16s} ${p[0]/1e6:>5.2f}M {p[1]*100:>5.1f}% {p[2]*100:>6.1f}% {p[6]:>5.1f}y {recstr(p):>12s} {p[3]:>7.2f}")
    tbl("Median by return", 1, 0.50)
    tbl("10th percentile by return", 1, 0.10)
    tbl("10th percentile by max drawdown", 2, 0.10)
    print(f"\nelapsed {time.time()-t0:.0f}s")

def compare_rebin(window, n=400, seed=42, shelter=0.0):
    """Quantify the dividend re-bin's effect: run the whole field with rebin ON (the method) vs
    OFF (original engines' raw resampled dividend dates), same window/seed/shelter. ON and OFF
    share identical price paths (only dividend Y-cadence differs), so paths are paired. Reports,
    per strategy: change in the distribution percentiles the report tiers approximate (P50/P10 of
    final wealth, P10 of max drawdown) and the paired per-path final-wealth change."""
    tag = "taxfree" if shelter >= 1.0 else "taxable"
    e = Engine(window, shelter=shelter, rebin=True); R = e.field(); paths = e.paths(n, seed)
    on = {nm: np.array([e.run_rebin(fn, P, Y) for P, Y in paths]) for nm, fn in R}   # received-preserving
    off = {nm: np.array([fn(P, Y) for P, Y in paths]) for nm, fn in R}               # raw resampled dates
    out = {True: (R, on), False: (R, off)}
    print(f"=== REBIN ON vs OFF | window {window} [{tag}], n={n} (ON=received-preserving, OFF=raw dates) ===")
    print(f"{'Strategy':16s} {'P50 d%':>7s} {'P10 d%':>7s} {'DD10 dpp':>9s} {'medPathd%':>10s} {'maxPathd%':>10s}")
    wF = 0.0; wDD = 0.0
    for nm, _ in R:
        on, off = out[True][1][nm], out[False][1][nm]
        d50 = (np.percentile(on[:, 0], 50) / np.percentile(off[:, 0], 50) - 1) * 100
        d10 = (np.percentile(on[:, 0], 10) / np.percentile(off[:, 0], 10) - 1) * 100
        ddd = (np.percentile(on[:, 2], 10) - np.percentile(off[:, 2], 10)) * 100  # maxDD pp
        pp = np.abs(on[:, 0] / off[:, 0] - 1) * 100                                # paired per-path
        wF = max(wF, abs(d50), abs(d10)); wDD = max(wDD, abs(ddd))
        print(f"{nm:16s} {d50:>+6.2f} {d10:>+6.2f} {ddd:>+8.2f} {np.median(pp):>9.2f} {pp.max():>9.2f}")
    print(f"\nlargest |percentile wealth change| = {wF:.2f}%   largest |P10 maxDD change| = {wDD:.2f} pp")

def verify_div(window, n=200, seed=42):
    """Recheck the received-preserving re-bin on real synthetic share paths (30/30/40 soft):
      (a) per asset-year, the re-binned RECEIVED dividend equals what was received raw (preserved);
      (b) a year with received dividend comes out at the asset's event count K (not 8, not sampled);
      (c) a year that received nothing stays zero;
      (d) sanity: bonds (monthly payers) essentially never produce a zero-receipt year."""
    e = Engine(window, rebin=True); rng = np.random.default_rng(seed)
    base = dict(e.field())["30/30/40 soft"]
    full = list(e._yr_vals[1:-1]); A = e.A
    max_inc_err = 0.0; bad_count = 0
    zero_cells = np.zeros(A); ev_total = np.zeros(A); cells = 0
    for _ in range(n):
        P, Y = e.synth(e.make_idx(rng))
        e._track = True; base(P, Y); e._track = False; sh = e._shist
        Yadj = respace_received(Y, P, sh, e._yr_vals, e._yr_pos, e._dep_pos)
        for yv in full:
            pos = e._yr_pos[yv]; slots = e._dep_pos[yv]
            for a in range(A):
                rec_raw = float((Y[pos, a] * sh[pos, a] * P[pos, a]).sum())
                rec_reb = float((Yadj[pos, a] * sh[pos, a] * P[pos, a]).sum())
                if rec_raw != 0.0: max_inc_err = max(max_inc_err, abs(rec_reb / rec_raw - 1.0))
                nev = int((Yadj[pos, a] != 0.0).sum())
                if slots[a] is not None and rec_raw != 0.0 and nev > len(slots[a]): bad_count += 1
                zero_cells[a] += (nev == 0); ev_total[a] += nev
            cells += 1
    print(f"=== VERIFY received re-bin | window {window}, {n} paths (30/30/40 soft), {len(full)} full years/path ===")
    print(f"  (a) max |received change| per asset-year : {max_inc_err:.2e}   (expect ~0)")
    print(f"  (b) received years above event count K   : {bad_count}        (expect 0)")
    print(f"  (c)/(d) per-asset mean events/yr and zero-year fraction:")
    print(f"      {'asset':8s} {'K':>3s} {'mean ev/yr':>11s} {'zero-yr frac':>13s}")
    for a, nm in enumerate(e.ASSETS):
        print(f"      {nm:8s} {int(e.div_K[a]):>3d} {ev_total[a]/cells:>11.2f} {zero_cells[a]/cells:>13.3f}")

if __name__ == "__main__":
    win = sys.argv[1] if len(sys.argv) > 1 else "1995"
    args = sys.argv[2:]
    shelter = 1.0 if "taxfree" in args else 0.0
    if "compare" in args:
        compare_rebin(win, shelter=shelter)
    elif "verify" in args:
        verify_div(win)
    else:
        run_window(win, shelter=shelter)
