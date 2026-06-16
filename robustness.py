"""Robustness appendices for the report, on the unified 1980--2026 engine (dca_core, taxable account,
all fees). Two checks, both on the realized backtest:

  (1) INCOME PATH -- re-run a representative set of strategies under four income trajectories that hold
      the SAME lifetime contribution but differ in WHEN it arrives: base (the report's rising-salary
      path), flat (equal every month), front-loaded (more early), late-bloomer (more late). The level
      of wealth shifts with the timing (earlier money compounds longer), but the question is whether
      the ORDERING of strategies is stable.

  (2) LAZY INVESTOR -- the disciplined investor contributes every month; the lazy investor skips 25% of
      monthly contributions at random (N patterns). Reports disciplined vs lazy final wealth, the ratio,
      a 10--90th-percentile band across patterns, and max drawdown.

Run: python3 robustness.py
"""
import numpy as np
from dca_core import Engine

e = Engine("1980", shelter=0.0)            # taxable account, fee'd
Pa, Ya = e.actual()
base_camt = e.camt.copy(); base_total = base_camt.sum(); nm = len(base_camt)

# representative strategies: the efficient set + common references (label, runner)
def soft(w):  return lambda: e.run_static(Pa, Ya, w, True)
def hard(w):  return lambda: e.run_static_hard(Pa, Ya, w)
REPS = [
    ("30/30/40 soft",    soft(e.wv("30/30/40"))),
    ("All Weather soft", soft(e.wv("All Weather"))),
    ("PermPort hard",    hard(e.wv("PermPort"))),
    ("60/40 soft",       soft(e.wv("60/40"))),
    ("60/40 hard",       hard(e.wv("60/40"))),
    ("80/20 soft",       soft(e.wv("80/20"))),
    ("Aggressive soft",  soft(e.wv("Aggressive"))),
]

# ---- (1) income path ----
def ramp(up):
    w = np.arange(1, nm + 1, dtype=float)
    if not up: w = w[::-1]
    return base_total * w / w.sum()
TRAJ = [("base", base_camt), ("flat", np.full(nm, base_total / nm)),
        ("front-loaded", ramp(False)), ("late-bloomer", ramp(True))]

print("=== (1) INCOME-PATH ROBUSTNESS (1980--2026, taxable, realized) ===")
print(f"each trajectory contributes the same lifetime total (${base_total/1e3:,.0f}k); only the timing differs")
print(f"{'Strategy':18s} " + " ".join(f"{t[0]:>13s}" for t in TRAJ) + "   rank stable?")
wealth = {}
for nmv, camt in TRAJ:
    e.camt = camt
    wealth[nmv] = {lab: float(fn()[0]) for lab, fn in REPS}
e.camt = base_camt
order_ref = [lab for lab, _ in sorted(wealth["base"].items(), key=lambda kv: -kv[1])]
for lab, _ in REPS:
    cells = " ".join(f"${wealth[t[0]][lab]/1e6:>11.2f}M" for t in TRAJ)
    print(f"{lab:18s} {cells}")
def _disp(lab): return "Permanent Portfolio (hard)" if lab == "PermPort hard" else lab
with open("results/income_rows.tex", "w") as _fh:          # rows for the report's income-path table
    _fh.write(" \\\\\n".join(_disp(lab) + " & " + " & ".join(f"{wealth[t[0]][lab]/1e6:.2f}" for t in TRAJ)
                             for lab, _ in REPS))

print("\nordering of these strategies under each trajectory (by final wealth):")
for nmv, _ in TRAJ:
    order = [lab for lab, _ in sorted(wealth[nmv].items(), key=lambda kv: -kv[1])]
    print(f"  {nmv:13s}: " + " > ".join(order))

# ---- (2) lazy investor ----
SKIP, N = 0.25, 500
rng = np.random.default_rng(42)
print(f"\n=== (2) LAZY INVESTOR -- skips {SKIP*100:.0f}% of monthly contributions at random, {N} patterns ===")
print(f"{'Strategy':18s} {'disciplined / maxDD':>22s}   {'lazy mean / ratio [10-90th] / maxDD':>40s}")
_lazy_rows = []
for lab, fn in REPS:
    e.camt = base_camt
    md = fn(); disc_f, disc_dd = float(md[0]), float(md[2])
    fin = np.empty(N); dd = np.empty(N)
    for i in range(N):
        e.camt = base_camt * (rng.random(nm) >= SKIP)
        m = fn(); fin[i] = m[0]; dd[i] = m[2]
    e.camt = base_camt
    r = fin / disc_f
    print(f"{lab:18s} ${disc_f/1e6:5.2f}M / {disc_dd*100:5.1f}%        "
          f"${fin.mean()/1e6:5.2f}M / {r.mean():.3f} [{np.percentile(r,10):.3f},{np.percentile(r,90):.3f}] / {dd.mean()*100:5.1f}%")
    _lazy_rows.append(f"{_disp(lab)} & \\${disc_f/1e6:.2f}M & $-{abs(disc_dd)*100:.0f}\\%$ & "
                      f"\\${fin.mean()/1e6:.2f}M & {r.mean():.2f} [{np.percentile(r,10):.2f}, {np.percentile(r,90):.2f}] & "
                      f"$-{abs(dd.mean())*100:.0f}\\%$")
with open("results/lazy_rows.tex", "w") as _fh:            # rows for the report's lazy-investor table
    _fh.write(" \\\\\n".join(_lazy_rows))
