"""Parallel recompute of the 1980 MC stores (tpb_* and core_*), both tax regimes.

Produces byte-for-byte the same arrays as `terminal_pain_b.py` + `dca_core.run_window` (same seed-42
paths, same run_rebin compute) but fans the strategies across CPU cores. The paths list is built once
per regime in the parent and inherited copy-on-write by forked workers (run_rebin never writes to P/Y),
so there is no per-worker path duplication and no pickling of the strategy lambdas.

  tpb_{win}_{tag}.npy : {names, data:{nm:{M,Wt}}, cols}  -- field() + the four TDF glides (gen_tables2)
  core_{win}_{tag}.npy: {names, data:{nm:M}, act}        -- field() only, plus the actual backtest (compute_stats)

Usage: python3 parallel_recompute.py [1980]
"""
import os, sys, time, numpy as np, multiprocessing as mp
from dca_core import Engine, RESULTS

WIN = sys.argv[1] if len(sys.argv) > 1 else "1980"
N = 1000
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 42
SUF = "" if SEED == 42 else f"_s{SEED}"   # non-default seeds write seed-tagged files; the seed-42 stores are never overwritten
WORKERS = max(1, min(18, (os.cpu_count() or 4) - 2))
COLS = "fv,irr,fullDD,sharpe,dur,recov,ptt,resid,termDD"

_E = None; _R = None; _PATHS = None   # set per regime in the parent; inherited by forked workers


def build_field(e):
    """field() + the four TDF continuous glides, exactly as terminal_pain_b.py assembles them."""
    R = e.field()
    nfield = len(R)
    if all(r in e.roles for r in ("STOCK", "BOND")):
        sb = (e.wvec({"STOCK": 0.9, "BOND": 0.1}), e.wvec({"STOCK": 0.3, "BOND": 0.7}))
        t30 = e.wvec({"STOCK": 0.3, "GOLD": 0.3, "BOND": 0.4}) if "GOLD" in e.roles else None
        R = R + [("TDF 90>30 soft", lambda P, Y: e.run_contglide(P, Y, sb[0], sb[1], "soft")),
                 ("TDF 90>30 hard", lambda P, Y: e.run_contglide(P, Y, sb[0], sb[1], "hard"))]
        if t30 is not None:
            R = R + [("TDF 90>3030 soft", lambda P, Y: e.run_contglide(P, Y, sb[0], t30, "soft")),
                     ("TDF 90>3030 hard", lambda P, Y: e.run_contglide(P, Y, sb[0], t30, "hard"))]
    return R, nfield


def _work(i):
    nm, fn = _R[i]
    M = np.empty((N, 9)); Wt = np.empty((N, _E.A))
    for k, (P, Y) in enumerate(_PATHS):
        M[k] = _E.run_rebin(fn, P, Y); Wt[k] = _E.last_w
    return i, nm, M, Wt


def run_regime(window, shelter):
    global _E, _R, _PATHS
    tag = "taxfree" if shelter >= 1.0 else "taxable"
    t0 = time.time()
    _E = Engine(window, shelter=shelter)
    _R, nfield = build_field(_E)
    _PATHS = _E.paths(N, SEED)
    Pa, Ya = _E.actual()
    act = {nm: fn(Pa, Ya) for nm, fn in _R[:nfield]}          # actual backtest (true ex-dates), field() only
    with mp.Pool(WORKERS) as pool:
        out = pool.map(_work, range(len(_R)))
    out.sort(key=lambda r: r[0])
    names = [nm for _, nm, _, _ in out]
    tpb = {"names": names, "data": {nm: {"M": M, "Wt": Wt} for _, nm, M, Wt in out}, "cols": COLS}
    np.save(os.path.join(RESULTS, f"tpb_{window}_{tag}{SUF}.npy"), tpb, allow_pickle=True)
    fnames = names[:nfield]
    core = {"names": fnames, "data": {nm: tpb["data"][nm]["M"] for nm in fnames}, "act": act}
    np.save(os.path.join(RESULTS, f"core_{window}_{tag}{SUF}.npy"), core, allow_pickle=True)
    print(f"  {window} {tag}: {len(_R)} strategies ({nfield} field + {len(_R)-nfield} TDF), "
          f"{N} paths, {WORKERS} workers, {time.time()-t0:.0f}s")
    # spot-check a headline number
    m = tpb["data"]["30/30/40 soft"]["M"]
    print(f"    30/30/40 soft medW=${np.median(m[:,0])/1e6:.2f}M  termDD10={np.percentile(m[:,8],10)*100:.1f}%")


if __name__ == "__main__":
    print(f"parallel recompute, window {WIN}, {WORKERS} workers")
    for sh in (0.0, 1.0):
        run_regime(WIN, sh)
    print("done; wrote tpb_/core_ for both regimes")
