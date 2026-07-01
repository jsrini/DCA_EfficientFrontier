"""Appendix H -- long-window Monte Carlo (1962-2026 return universe).

Compares two strategies under block-bootstrap Monte Carlo drawn from a longer return history than
the report's main 1980-2026 base: the target-date glide TDF 90/10->30/70 (hard) vs. the
stocks/gold/bonds stepped glide glide D (50/25/25 -> 40/30/30 -> 30/30/40, hard). Both are 46-year
DCA accumulations; the only thing changed vs. the report's field is the return-generation universe.

Data: dca_adj_div_60full.csv / dca_yield_div_60full.csv, built by build_60_extension.py.
  STOCK  1962-2026  (Yahoo ^GSPC + Shiller div yield + VFINX-equivalent ER, pre-1980; VFINX after)
  BOND   1962-2026  (FRED DGS10 -> total return + VBMFX-equivalent ER, pre-1980; VBMFX after)
  GOLD   1975-2026  (LBMA PM fix)                                     -- placeholder pre-1975

Bootstrap: standard block-bootstrap (same block-size distribution as the main engine), with one
appendix-specific fixup -- any drawn return index whose source date is pre-1975 is redrawn (for the
GOLD column only) from the 1975+ range. Cross-asset correlation is lost on that slice of gold, but
gold wasn't investable for a US retail investor until 1975-01-01 (US private gold ownership
legalized).

Runs both regimes (taxable + tax-free), 1,000 paths, seed 42, matching the report's convention.
Emits results/longwindow_appendix.tex for inclusion in wealth_report.tex.

Run:  python3 longwindow_appendix.py
"""
import os, numpy as np, pandas as pd
import dca_core as c

GOLD_VALID_START = pd.Timestamp("1975-01-01")   # US private gold ownership legalized here
EXTENDED_ADJ = "dca_adj_div_60full.csv"         # built by build_60_extension.py
EXTENDED_YLD = "dca_yield_div_60full.csv"
EXTENDED_CPI = "dca_cpi_60full.csv"             # daily CPI (ffill from monthly CPIAUCSL)

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


class LongWindowEngine(c.Engine):
    """dca_core.Engine variant for the appendix: reuses every runner and metric unchanged, but
      (a) contributions and the calendar stay on the report's 1980-2026 46-year schedule, and
      (b) the block-bootstrap draws from a WIDER return pool -- 1962-2026 for stocks/bonds and
          1975-2026 for gold -- by prepending pre-1980 returns to self.rets/self.yrow.
    Every runner still sees the 46-year self.idx/self.T; only the source pool for make_idx is
    larger. The pre-1975 gold fixup swaps out any index whose source date is pre-1975 (for the
    gold column only) with a random draw from the 1975+ range."""

    def __init__(self, window="1980", **kwargs):
        super().__init__(window, **kwargs)                     # 1980-2026 calendar, 46-year DCA, nominal salary
        # Load the extended series and prepend the pre-1980 daily returns to the bootstrap pool.
        adj = pd.read_csv(EXTENDED_ADJ, index_col=0, parse_dates=True)
        yld = pd.read_csv(EXTENDED_YLD, index_col=0, parse_dates=True)
        splice_date = self.idx[0]                              # 1980-01-15 (first day of 1980+ pool)
        pre = adj.index < splice_date
        assert pre.sum() > 0, f"no pre-{splice_date.date()} rows in {EXTENDED_ADJ}"
        # Extended CSV has three columns (STOCK/BOND/GOLD); the 1980+ engine has A>=5 columns.
        # Build zero-return blocks for the unused columns; overlay real returns for the shared ones.
        pre_rets = np.zeros((pre.sum() - 1, self.A))           # -1 because rets is diff-based
        pre_yrow = np.zeros((pre.sum() - 1, self.A))
        pre_adj = adj.loc[pre]
        pre_yld = yld.loc[pre]
        for col_name in adj.columns:                            # STOCK, BOND, GOLD
            if col_name not in self.ASSETS: continue
            j = self.ASSETS.index(col_name)
            pre_rets[:, j] = pre_adj[col_name].pct_change().dropna().values
            pre_yrow[:, j] = pre_yld[col_name].values[1:]       # align with rets (skip day 0)
        # Prepend: extended pool = pre-1980 + original 1980-2026
        self.rets = np.vstack([pre_rets, self.rets])
        self.yrow = np.vstack([pre_yrow, self.yrow])
        self.N = len(self.rets)
        # Gold substitution boundary: first index in the extended rets whose "source date" (the
        # DAY-END of the return) is >= 1975-01-01. Below this boundary, gold data is placeholder
        # (0 return) and the bootstrap must re-draw for gold from the 1975+ range.
        pre_dates = pre_adj.index[1:]                           # source dates for pre_rets
        pre_1975_count = int((pre_dates < GOLD_VALID_START).sum())
        self.gold_ret_valid_start = pre_1975_count              # first valid gold-return index
        self._gold_valid_ix = np.arange(self.gold_ret_valid_start, self.N)

        # Daily CPI inflation series aligned with self.rets. Built in two segments so the splice-day
        # transition (pre_adj's last date -> self.idx[0]) does not introduce a spurious inflation row.
        # inflation[k] is the fraction CPI change from day k to day k+1, matching self.rets[k].
        cpi_full = pd.read_csv(EXTENDED_CPI, index_col=0, parse_dates=True)["CPI"]
        pre_cpi = cpi_full.reindex(pre_adj.index).ffill().bfill().values     # 4522 pts, matches pre_adj
        post_cpi = cpi_full.reindex(self.idx).ffill().bfill().values         # 11923 pts, matches self.idx
        pre_infl = (pre_cpi[1:] - pre_cpi[:-1]) / pre_cpi[:-1]               # 4521 pts, matches pre_rets
        post_infl = (post_cpi[1:] - post_cpi[:-1]) / post_cpi[:-1]           # 11922 pts, matches original rets
        self.inflation = np.concatenate([pre_infl, post_infl])
        assert len(self.inflation) == self.N, f"inflation misalign: {len(self.inflation)} vs N={self.N}"
        # Actual 1980-01-15 to 2026-06-01 CPI ratio (~4.3x). Used to convert per-path real wealth
        # from path-start (1980) dollars to real 2026 dollars: real_2026 = nominal * actual/path_cpi.
        # For the actual historical path (path_cpi == actual), this reduces to identity -- nominal
        # 2026 dollars == real 2026 dollars, correct.
        self.actual_cpi_ratio = float(post_cpi[-1] / post_cpi[0])

    def paths(self, n=1000, seed=42):
        """Block bootstrap in NOMINAL returns (so tax accounting is on nominal dividends/gains,
        matching real US tax law). Gold column uses ix_gold (pre-1975 slots substituted from
        1975+). For each path also computes the cumulative inflation factor drawn from the same
        block sequence -- used by run_regime to deflate terminal wealth into real dollars for
        reporting.

        Returns list of (P, Y, cpi_factor) triples. cpi_factor > 1 means the path experienced
        net inflation; real_terminal_wealth = nominal_terminal_wealth / cpi_factor."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(n):
            ix = self.make_idx(rng)
            ix_gold = ix
            invalid = ix < self.gold_ret_valid_start
            if invalid.any():
                ix_gold = ix.copy()
                ix_gold[invalid] = rng.choice(self._gold_valid_ix, size=int(invalid.sum()))
            cpi_factor = float(np.prod(1.0 + self.inflation[ix]))
            P, Y = self._synth_pergold(ix, ix_gold)
            out.append((P, Y, cpi_factor))
        return out

    def _synth_pergold(self, ix, ix_gold):
        """Nominal path construction: same as Engine.synth but gold column uses ix_gold instead of
        ix (pre-1975 substitution). Preserves the gold NAV drag."""
        A, T = self.A, self.T
        rets = self.rets[ix].copy()
        rets[:, self.gold_i] = self.rets[ix_gold, self.gold_i]
        P = np.empty((T, A)); P[0] = self.P0
        P[1:] = self.P0 * np.cumprod(1.0 + rets, axis=0)
        P[:, self.gold_i] *= self.gold_decay
        Y = np.empty((T, A)); Y[0] = 0.0
        Y[1:] = self.yrow[ix].copy()
        Y[1:, self.gold_i] = self.yrow[ix_gold, self.gold_i]
        return P, Y


# -- strategy runners: TDF 90/10->30/70 hard and glide D hard -------------------------------------
def _tdf_hard(e):
    """TDF 90/10->30/70 hard: continuous glide from 90% stocks / 10% bonds to 30% stocks / 70% bonds,
    hard-rebalanced. Reaches the endpoint at the horizon."""
    w0 = e.wvec({"STOCK": 0.90, "BOND": 0.10})
    w1 = e.wvec({"STOCK": 0.30, "BOND": 0.70})
    return lambda P, Y: e.run_contglide(P, Y, w0, w1, "hard")


def _glD_hard(e):
    """glide D (hard): stepped 50/25/25 -> 40/30/30 -> 30/30/40 across the two phase transitions."""
    return lambda P, Y: e.run_glide(P, Y, e.phases("glD"), "hard")


def run_regime(shelter, tag):
    print(f"\n=== 1962-2026 long window, {tag} (shelter={shelter}) ===")
    e = LongWindowEngine(shelter=shelter)
    Pa, Ya = e.actual()
    fns = {"TDF 90/10->30/70 hard": _tdf_hard(e), "glD hard": _glD_hard(e)}
    print(f"{'Strategy':22s} {'Actual FV':>10s} {'IRR':>6s} | 1000-path MC medRealW / nominalTermPain")
    act_out = {}; mc_out = {}
    paths = e.paths(1000, 42)                          # shared across strategies (same seed, same blocks)
    cpi_factors = np.array([cf for _, _, cf in paths])
    for nm, fn in fns.items():
        m_act = fn(Pa, Ya); act_out[nm] = m_act
        M = np.array([e.run_rebin(fn, P, Y) for P, Y, _ in paths])
        # Convert nominal terminal wealth (col 0) to real 2026 dollars, per path. The path's own
        # cumulative bootstrap inflation deflates to path-start (1980) dollars; multiplying by the
        # actual 1980-to-2026 CPI ratio anchors to 2026 real dollars. For the actual historical
        # path (cpi_factor == actual_cpi_ratio), this reduces to identity -- nominal 2026 wealth
        # IS real 2026 wealth for the actual path, since 2026 is the reference year.
        # Terminal pain (col 8) stays nominal -- that's the psychological drawdown the investor sees.
        M[:, 0] = M[:, 0] * (e.actual_cpi_ratio / cpi_factors)
        mc_out[nm] = M
        medW = float(np.median(M[:, 0])) / 1e6
        tp = float(np.percentile(M[:, 8], 10)) * 100
        print(f"{nm:22s} ${m_act[0]/1e6:>7.2f}M {m_act[1]*100:>5.1f}% | ${medW:>5.2f}M / {tp:+.1f}%")
    np.save(os.path.join(RESULTS, f"longwindow_{tag}.npy"),
            {"act": act_out, "mc": mc_out}, allow_pickle=True)
    return act_out, mc_out


def emit_appendix_tex(taxable, taxfree):
    """Emit results/longwindow_appendix.tex -- the appendix body as a self-contained LaTeX fragment."""
    def _fmt(name, M_tx, M_tf):
        medW_tx = float(np.median(M_tx[:, 0])) / 1e6
        tp_tx   = float(np.percentile(M_tx[:, 8], 10)) * 100
        medW_tf = float(np.median(M_tf[:, 0])) / 1e6
        tp_tf   = float(np.percentile(M_tf[:, 8], 10)) * 100
        return f"{name} & \\${medW_tx:.2f}M & ${tp_tx:+.1f}\\%$ & \\${medW_tf:.2f}M & ${tp_tf:+.1f}\\%$"
    rows = [
        _fmt("TDF 90/10$\\to$30/70 (hard)", taxable[1]["TDF 90/10->30/70 hard"], taxfree[1]["TDF 90/10->30/70 hard"]),
        _fmt("glide D (hard)",              taxable[1]["glD hard"],              taxfree[1]["glD hard"]),
    ]
    # Match the eff_taxable.tex convention: rows separated by " \\\\\n", no trailing \\ on the last
    # row AND no trailing newline (blank final line breaks the tabular that \input inlines into).
    with open(os.path.join(RESULTS, "longwindow_rows.tex"), "w") as f:
        f.write(" \\\\\n".join(rows))
    print(f"wrote results/longwindow_rows.tex ({len(rows)} rows)")


if __name__ == "__main__":
    tx = run_regime(0.0, "taxable")
    tf = run_regime(1.0, "taxfree")
    emit_appendix_tex(tx, tf)
