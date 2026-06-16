"""Horizon robustness: does the efficient set change if the accumulation is shorter than 46 years?

Monte Carlo paths are drawn from the full 1980--2026 pool, as elsewhere. Each strategy is evaluated at
several accumulation horizons (HORIZONS). Two cases, handled differently:

  * Static mixes hold fixed weights, so the first H years of a 46-year path IS an H-year run -- we run
    once on the full path and slice it at every horizon (cheap).
  * Glides are horizon-specific by construction: a 46-year 90/10->30/70 glide is only ~2/3 of the way
    to its endpoint at year 31, so truncating it measures a half-finished glide, not the glide a
    31-year investor would run. Each glide is therefore rebuilt to COMPLETE at the horizon (continuous:
    reach the endpoint at H; stepped: transitions at the same fractions of [start, H]) and run once per
    horizon.

The within-cap (terminal pain no deeper than -30%) efficient frontier is then computed at each horizon,
for each account. The dividend re-bin is a verified no-op (tax-neutral; see `dca_core.py 1995
verify`/`compare`), so we run the plain single-pass backtest, not the two-pass run_rebin. Results cache
to results/horizon_raw.npy; --reuse re-reports without recomputing when the cache covers HORIZONS.

Run: python3 horizon_robust.py [--reuse]
"""
import os, sys, numpy as np, pandas as pd, multiprocessing as mp
from dca_core import Engine, GLIDES

N, SEED = 1000, 42
HORIZONS = [31, 35, 38, 40, 44, 46]      # accumulation lengths to evaluate (years)
LIM = 30.0
VERSION = "glide-per-horizon-v1"         # bumped when the methodology changes; a cache from a different
                                         # version is NOT resumed (guards against stale/buggy results)
WORKERS = max(1, min(18, (os.cpu_count() or 4) - 2))
_E = _TASKS = _PATHS = _CUTS = None


def cut(e, yrs):
    T_cut = min(int(np.searchsorted(e.idx.values, np.datetime64(e.idx[0] + pd.DateOffset(years=yrs)))), e.T)
    ts = int(np.searchsorted(e.idx.values, np.datetime64(e.idx[T_cut - 1] - pd.DateOffset(years=5))))
    return T_cut, ts


def static_field(e):
    return [(nm, fn) for nm, fn in e.field() if not nm.startswith("gl")]   # glA-D are glides; rest are static


def glide_specs(e):
    specs = []
    if all(r in e.roles for r in ("STOCK", "BOND")):
        sb0 = e.wvec({"STOCK": 0.9, "BOND": 0.1}); sb1 = e.wvec({"STOCK": 0.3, "BOND": 0.7})
        t30 = e.wvec({"STOCK": 0.3, "GOLD": 0.3, "BOND": 0.4}) if "GOLD" in e.roles else None
        ends = [(sb1, "TDF 90>30")] + ([(t30, "TDF 90>3030")] if t30 is not None else [])
        for w1, lbl in ends:
            for r in ("soft", "hard"):
                specs.append((f"{lbl} {r}", ("cont", sb0, w1, r)))
    for g in GLIDES:
        if not all(e.supports(n) for n in GLIDES[g]): continue
        for r in ("soft", "hard"):
            specs.append((f"{g} {r}", ("step", g, r)))
    return specs


def scaled_phases(e, g, end_t):
    """Stepped-glide phases rescaled so the transitions fall at the same fractions of [start, end_t]
    as of the full window -- the glide completes by the horizon, not at year 46."""
    base = e.phases(g); cd0 = base[0][0]; d0 = e.idx[cd0]
    full = max(1, (e.idx[-1] - d0).days); span = (e.idx[end_t] - d0).days
    out = [(cd0, base[0][1])]
    for ix, w in base[1:]:
        fr = (e.idx[ix] - d0).days / full
        sd = np.datetime64(d0 + pd.Timedelta(days=int(round(fr * span))))
        out.append((int(np.searchsorted(e.idx.values, sd)), w))
    return out


def run_glide(e, spec, P, Y, end_t):
    if spec[0] == "cont":
        _, w0, w1, r = spec
        return e.run_contglide(P, Y, w0, w1, r, end=e.idx[end_t])
    _, g, r = spec
    return e.run_glide(P, Y, scaled_phases(e, g, end_t), r)


def _work(i):
    kind, nm, payload = _TASKS[i]
    W = {h: np.empty(N) for h in HORIZONS}; D = {h: np.empty(N) for h in HORIZONS}
    for k, (P, Y) in enumerate(_PATHS):
        if kind == "stat":                                   # one run; every horizon is a slice of it
            payload(P, Y); val = _E.last_val
            pk = np.maximum.accumulate(val); dd = (val - pk) / np.where(pk > 0, pk, 1)
            for h in HORIZONS:
                T_cut, ts = _CUTS[h]; W[h][k] = val[T_cut - 1]; D[h][k] = dd[ts:T_cut].min()
        else:                                                # glide: a horizon-specific run per horizon
            for h in HORIZONS:
                T_cut, ts = _CUTS[h]
                run_glide(_E, payload, P, Y, T_cut - 1); val = _E.last_val
                pk = np.maximum.accumulate(val[:T_cut]); dd = (val[:T_cut] - pk) / np.where(pk > 0, pk, 1)
                W[h][k] = val[T_cut - 1]; D[h][k] = dd[ts:T_cut].min()
    return i, nm, {h: (float(np.median(W[h])), float(np.percentile(D[h], 10))) for h in HORIZONS}


def run_regime(window, shelter, tag, res, RAW):
    """Compute (or resume) one tax regime, checkpointing the cache after every strategy so a crash
    never costs more than the strategy in flight. Strategies already present in `res[tag]` are skipped."""
    global _E, _TASKS, _PATHS, _CUTS
    _E = Engine(window, shelter=shelter)
    _TASKS = ([("stat", nm, fn) for nm, fn in static_field(_E)] +
              [("glide", nm, sp) for nm, sp in glide_specs(_E)])
    _PATHS = _E.paths(N, SEED); _CUTS = {h: cut(_E, h) for h in HORIZONS}
    done = res.setdefault(tag, {h: {} for h in HORIZONS})
    todo = [i for i in range(len(_TASKS)) if _TASKS[i][1] not in done[HORIZONS[0]]]
    if not todo:
        print(f"  {tag}: all {len(_TASKS)} strategies already cached", flush=True); return
    print(f"  {tag}: computing {len(todo)} of {len(_TASKS)} strategies", flush=True)
    with mp.Pool(WORKERS) as pool:
        for _, nm, summ in pool.imap_unordered(_work, todo):
            for h in HORIZONS: done[h][nm] = summ[h]
            np.save(RAW, res, allow_pickle=True)              # checkpoint after every strategy
            print(f"    [{tag}] {len(done[HORIZONS[0]])}/{len(_TASKS)}  {nm}", flush=True)


def frontier(stats):   # within-cap efficient names, sorted by W/TP
    pts = [(nm, w / 1e6, p * 100) for nm, (w, p) in stats.items()]   # (nm, wealth$M, pain% neg)
    def dom(p): return any((q[1] >= p[1] and q[2] >= p[2]) and (q[1] > p[1] or q[2] > p[2]) for q in pts if q[0] != p[0])
    eff = [p for p in pts if not dom(p) and abs(p[2]) <= LIM]
    return sorted(eff, key=lambda p: -(p[1] / abs(p[2]) if p[2] else 1e9))


DISP = {"TDF 90>30 hard": r"TDF 90/10$\to$30/70 hard", "TDF 90>3030 hard": r"TDF 90/10$\to$30/30/40 hard",
        "TDF 90>30 soft": r"TDF 90/10$\to$30/70 soft", "TDF 90>3030 soft": r"TDF 90/10$\to$30/30/40 soft",
        "All Weather no-reb": "All Weather (no rebal)", "All Weather soft": "All Weather (soft)",
        "All Weather hard": "All Weather (hard)", "glA hard": "glide A (hard)", "glA soft": "glide A (soft)",
        "glD soft": "glide D (soft)", "glD hard": "glide D (hard)", "PermPort hard": "Permanent Portfolio (hard)",
        "PermPort soft": "Permanent Portfolio (soft)", "60/40 hard": "60/40 (hard)",
        "50/25/25 soft": "50/25/25 (soft)", "40/25/35 soft": "40/25/35 (soft)", "30/30/40 hard": "30/30/40 (hard)"}


if __name__ == "__main__":
    REG = [("taxable", 0.0), ("taxfree", 1.0)]
    RAW = "results/horizon_raw.npy"
    res = {}
    if os.path.exists(RAW) and "--fresh" not in sys.argv:
        cached = np.load(RAW, allow_pickle=True).item()
        if cached.get("_version") == VERSION:                 # only resume a cache built by THIS methodology
            res = cached
        else:
            print("cache is from a different methodology version -> recomputing", flush=True)
    res.setdefault("_version", VERSION)
    complete = all(tag in res and len(res[tag][HORIZONS[0]]) for tag, _ in REG)
    if "--reuse" in sys.argv and complete:
        print("reusing cache (no recompute)", flush=True)
    else:
        for tag, sh in REG:                                   # resumes; checkpoints after each strategy
            run_regime("1980", sh, tag, res, RAW)

    for tag, _ in REG:
        print(f"\n=== {tag}: within-cap efficient set by accumulation length ===")
        for h in HORIZONS:
            eff = frontier(res[tag][h])
            print(f"  {h}yr: " + ", ".join(f"{e[0]} ({e[1]:.2f}M,{e[2]:.0f}%)" for e in eff))

    # viability drift: pick within the -30% cap at a chosen retirement horizon -- does terminal pain
    # breach the cap if the same investor instead runs all the way to year 46?
    for tag, _ in REG:
        print(f"\n=== {tag}: do within-cap picks breach -30% by year 46? ===")
        for h in [y for y in HORIZONS if y != 46]:
            effh = [nm for nm, *_ in frontier(res[tag][h])]
            br = [nm for nm in effh if abs(res[tag][46].get(nm, (0, 0))[1] * 100) > LIM]
            detail = "; ".join(f"{nm} {res[tag][h][nm][1]*100:.0f}%->{res[tag][46][nm][1]*100:.0f}%" for nm in br)
            print(f"  pick at {h}yr: {len(br)} of {len(effh)} breach" + (f"  ({detail})" if br else ""))

    # diversification study: do the stocks/gold/bonds mixes stay within the -30% viable cap at every
    # horizon? (viable = terminal pain no deeper than -30%; separate from being on the efficient frontier)
    DIVERSIFIED = ["30/30/40", "40/30/30", "40/25/35"]
    for tag, _ in REG:
        print(f"\n=== {tag}: diversified stocks/gold/bonds mixes by horizon -- median wealth $M / terminal pain % (viable cap -30%) ===")
        print("  " + "strategy".ljust(16) + "".join(f"{h}yr".rjust(11) for h in HORIZONS) + "   viable-all?")
        for base in DIVERSIFIED:
            for nm in [n for n in res[tag][HORIZONS[0]] if n == base or n.startswith(base + " ")]:
                cells = "".join(f"{res[tag][h][nm][0]/1e6:6.2f}/{res[tag][h][nm][1]*100:3.0f}%" for h in HORIZONS)
                worst = min(res[tag][h][nm][1] * 100 for h in HORIZONS)
                print(f"  {nm:16s}{cells}   {'yes' if abs(worst) <= LIM else 'NO'}")

    # appendix rows: union of efficient strategies across horizons, marking presence per horizon/account
    eff_sets = {(tag, h): {nm for nm, *_ in frontier(res[tag][h])} for tag, _ in REG for h in HORIZONS}
    seen = []
    for tag, _ in REG:
        for h in HORIZONS:
            [seen.append(n) for n in eff_sets[(tag, h)] if n not in seen]
    rows = []
    for nm in seen:
        cells = " & ".join("$\\checkmark$" if nm in eff_sets[(tag, h)] else "" for tag, _ in REG for h in HORIZONS)
        rows.append(f"{DISP.get(nm, nm)} & {cells}")
    with open("results/horizon_rows.tex", "w") as fh:
        fh.write(" \\\\\n".join(rows))
    print(f"\nwrote results/horizon_rows.tex ({len(seen)} strategies, {len(HORIZONS)} horizons x 2 accounts)")

    # diversified-mix wealth/pain table for the appendix (taxable account; rebalanced soft/hard only)
    divrows = []
    for base in ["30/30/40", "40/30/30", "40/25/35", "50/25/25"]:
        for var in ("soft", "hard", "no-reb"):
            nm = f"{base} {var}"
            if nm not in res["taxable"][HORIZONS[0]]: continue
            cells = " & ".join(f"${res['taxable'][h][nm][0]/1e6:.1f}$/${res['taxable'][h][nm][1]*100:.0f}$" for h in HORIZONS)
            divrows.append(f"{base} {var} & {cells}")
    with open("results/horizon_div_rows.tex", "w") as fh:
        fh.write(" \\\\\n".join(divrows))
    print(f"wrote results/horizon_div_rows.tex ({len(divrows)} rows)")
