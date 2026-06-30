"""Verify the tax accounting fix (gain on tax-funding sales is booked into next year's cgyr).

Runs three checks:
  1. Tax-free invariance. With shelter=1.0 the engine sets RATE = CG = 0, so the fix is a no-op.
     Re-runs the field and asserts the realized actual final wealth matches the cached store.
  2. Pre/post wealth delta. If results/core_1980_*.npy.prefix backups exist (the saved pre-fix MC
     stores), prints per-strategy median-wealth deltas for both regimes and the global summary.
     Skips with a notice if the backups have been removed.
  3. Within-cap efficient set. Prints the current efficient frontier (W/TP order, terminal pain
     <= 30%) for both regimes, the same set the report's Section "The Efficient Frontier" inputs.

Run:  python3 verify_tax_fix.py
"""
import os, sys, numpy as np
import dca_core as c

RESULTS = "results"
WINDOW = "1980"
LIM = 30.0


def check_taxfree_invariance():
    print("=== 1. Tax-free invariance (shelter=1.0 should match cached store exactly) ===")
    path = os.path.join(RESULTS, f"core_{WINDOW}_taxfree.npy")
    if not os.path.exists(path):
        print(f"  SKIP: {path} not found"); return None
    cached = np.load(path, allow_pickle=True).item()
    e = c.Engine(WINDOW, shelter=1.0); R = e.field(); Pa, Ya = e.actual()
    worst = 0.0; worst_nm = None
    for nm, fn in R:
        live = fn(Pa, Ya)[0]; ref = cached["act"][nm][0]
        if abs(live - ref) > worst:
            worst = abs(live - ref); worst_nm = nm
    ok = worst < 1.0
    print(f"  max abs delta = ${worst:.4f}  ({'PASS' if ok else 'FAIL'}, worst on {worst_nm})")
    return ok


def check_delta_vs_prefix():
    print("\n=== 2. Median wealth delta vs pre-fix backup ===")
    any_run = False
    for tag in ("taxable", "taxfree"):
        cur = os.path.join(RESULTS, f"core_{WINDOW}_{tag}.npy")
        bak = cur + ".prefix"
        if not (os.path.exists(cur) and os.path.exists(bak)):
            print(f"  SKIP {tag}: missing {bak}"); continue
        any_run = True
        o = np.load(bak, allow_pickle=True).item()
        n = np.load(cur, allow_pickle=True).item()
        names = [nm for nm in n["names"] if nm in o["data"]]
        rows = []
        for nm in names:
            om = float(np.median(o["data"][nm][:, 0]))
            nm_ = float(np.median(n["data"][nm][:, 0]))
            rows.append((nm, om, nm_, (nm_ - om) / om * 100 if om else 0.0))
        rows.sort(key=lambda r: r[3])
        print(f"\n  -- {tag} --")
        print(f"  {'Strategy':24s} {'oldMedW':>10s} {'newMedW':>10s} {'dWealth%':>9s}")
        for r in rows:
            print(f"  {r[0]:24s} ${r[1]/1e6:>7.2f}M  ${r[2]/1e6:>7.2f}M  {r[3]:>+7.2f}%")
        dlt = [r[3] for r in rows]
        print(f"  {tag} dWealth range: {min(dlt):+.2f}% to {max(dlt):+.2f}%  "
              f"(median {float(np.median(dlt)):+.2f}%)")
    if not any_run:
        print("  No prefix backups found; create them with `cp core_*.npy core_*.npy.prefix` "
              "BEFORE rerunning the MC to enable this check.")


def check_within_cap_frontier():
    print("\n=== 3. Within-cap efficient set (terminal pain <= 30%, ranked by W/TP) ===")
    for tag in ("taxable", "taxfree"):
        path = os.path.join(RESULTS, f"tpb_{WINDOW}_{tag}.npy")
        if not os.path.exists(path):
            print(f"  SKIP {tag}: {path} not found"); continue
        d = np.load(path, allow_pickle=True).item()
        names, data = d["names"], d["data"]
        pts = []
        for nm in names:
            entry = data[nm]
            medW = float(np.median(entry["M"][:, 0])) / 1e6
            termPain = float(np.percentile(entry["M"][:, 8], 10)) * 100   # col 8 = terminal-window MDD
            pts.append((nm, termPain, medW))
        front = []
        for p in pts:
            dom = any(q != p and q[2] >= p[2] and q[1] >= p[1] and (q[2] > p[2] or q[1] > p[1])
                      for q in pts)
            if not dom:
                front.append(p)
        within = [p for p in front if abs(p[1]) <= LIM]
        within.sort(key=lambda p: -(p[2] / abs(p[1]) if p[1] else float("inf")))
        print(f"\n  -- {tag}: {len(within)} strategies on the within-cap frontier --")
        print(f"  {'Strategy':24s} {'MedW':>8s} {'TermPain':>9s} {'W/TP':>7s}")
        for p in within:
            wtp = p[2] / abs(p[1]) if p[1] else float("inf")
            print(f"  {p[0]:24s} ${p[2]:>5.2f}M {p[1]:>+7.1f}%  {wtp:>6.3f}")


if __name__ == "__main__":
    ok = check_taxfree_invariance()
    check_delta_vs_prefix()
    check_within_cap_frontier()
    sys.exit(0 if (ok is None or ok) else 1)
