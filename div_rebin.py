"""Shared received-preserving dividend-cadence re-bin for the DCA Monte Carlo.

Single source of truth used by dca_core.py and the per-window report engines
(dca_tax_mc.py, dca_tax_46yr.py, dca_tax_80field.py), so the cadence treatment
cannot drift between the tables.

Method (per asset, per calendar year of a synthetic path):
  1. received = sum over the year of (sampled yield x shares held x synthetic price)
     -- the actual dollars of dividend the strategy RECEIVED that year, on its own share path;
  2. amortize that received total evenly across the asset's realistic number of events for the
     year, as equal-dollar payments;
  3. recompute the yield on each event day as (received / K) / (shares held x price) there.
The year's received dividend is preserved exactly; the event count/cadence is normalized so a
resampled year can no longer carry 8 events (or 0). Because the received total is what the annual
dividend-income tax is levied on, preserving it makes the re-bin tax-neutral.

Receipt depends on the share balance, which depends on the strategy, so the re-bin is applied
in-run as a two-pass: pass 1 records the share path on the raw dividends; pass 2 re-runs on the
re-spaced yields (see run_rebin in each engine / dca_core). A year that sampled no dividends has
zero received and is left at zero -- a zero-dividend year is allowed (real for e.g. QQQ 1999-2002,
gold always).
"""
import numpy as np


def build_div_cadence(yld, idx, yr, assets):
    """Precompute the deposit cadence from the ACTUAL yield history.

    Returns (div_K, yr_vals, yr_pos, dep_pos):
      div_K[a]        : median nonzero-yield-day count per full year for asset a (its event count);
      yr_vals         : sorted list of calendar years in the window;
      yr_pos[yv]      : day-index positions belonging to year yv;
      dep_pos[yv][a]  : evenly-spaced positions to deposit asset a's events that year
                        (None if the asset pays nothing, i.e. div_K == 0).
    """
    A = len(assets)
    ynz = yld.reindex(idx).fillna(0.0)[assets].ne(0.0)
    cnt = ynz.groupby(yr).sum()
    full = cnt.index[1:-1] if len(cnt) > 2 else cnt.index   # drop partial first/last years
    div_K = cnt.loc[full].median().round().astype(int).values
    yr_vals = sorted(set(yr.tolist()))
    yr_pos = {}
    dep_pos = {}
    for yv in yr_vals:
        pos = np.where(yr == yv)[0]
        yr_pos[yv] = pos
        posv = pos[pos > 0]   # never deposit on day 0 (no shares held yet)
        slots = []
        for a in range(A):
            Ka = int(div_K[a])
            if Ka <= 0 or len(posv) == 0:
                slots.append(None)
                continue
            ke = min(len(posv), max(1, int(round(Ka * len(posv) / 252.0))))   # prorate partial years
            slots.append(np.unique(posv[np.linspace(0, len(posv) - 1, ke).round().astype(int)]))
        dep_pos[yv] = slots
    return div_K, yr_vals, yr_pos, dep_pos


def respace_received(Y, P, shist, yr_vals, yr_pos, dep_pos):
    """Received-preserving cadence normalization. Returns a NEW (T, A) yield array; Y is unchanged.

    Y     : (T, A) synthetic daily yields (real yields sampled for the path).
    P     : (T, A) synthetic compounded price path (P0 * cumprod(1+returns)).
    shist : (T, A) share balance held on each day under the strategy (from pass 1).

    For each asset/year, the dividend RECEIVED = sum(Y*shist*P) over the year is laid back onto the
    asset's realistic number of event days as equal-dollar payments, and the per-day yield is backed
    out from the share balance and price there. The year's received dividend is held fixed.
    """
    A = Y.shape[1]
    Yadj = np.zeros_like(Y)
    for yv in yr_vals:
        pos = yr_pos[yv]
        slots = dep_pos[yv]
        for a in range(A):
            sel = slots[a]
            if sel is None:                                   # no cadence (gold): leave as drawn
                Yadj[pos, a] = Y[pos, a]
                continue
            received = float((Y[pos, a] * shist[pos, a] * P[pos, a]).sum())
            if received == 0.0:                               # sampled no dividends -> stays zero
                continue
            base = shist[sel, a] * P[sel, a]
            valid = base > 0
            if not valid.any():                               # no shares on any event day -> leave raw
                Yadj[pos, a] = Y[pos, a]
                continue
            Yadj[sel[valid], a] = (received / int(valid.sum())) / base[valid]
    return Yadj
