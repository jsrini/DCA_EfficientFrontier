"""Report-quality wealth-vs-terminal-pain frontier from a saved tpb_*.npy run.
x = terminal pain (10th-pct final-5yr max DD, %), increasing rightward (Markowitz convention);
y = median final wealth ($M); ideal corner = TOP-LEFT. Dots colored by full max DD. Only the
efficient-frontier points (plus 30/30/40 as a reference) are labelled, with leader lines and
vertical de-collision, so the figure stays clean.
Usage: python3 plot_frontier.py [1995|1980] [taxable|taxfree]"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

win = sys.argv[1] if len(sys.argv) > 1 else "1995"
tag = sys.argv[2] if len(sys.argv) > 2 else "taxable"
# optional seed token like "s7" -> read/write the seed-tagged store (leaves the seed-42 files alone)
suf = next((f"_{a}" for a in sys.argv if a.startswith("s") and a[1:].isdigit()), "")
BASE = os.path.dirname(os.path.abspath(__file__)) + "/"

if "glidetable" in sys.argv:   # rows for the report's young-risk glide table (both accounts)
    GL = [("TDF 90>30 hard",   r"Glide 90/10$\to$30/70, hard (stock/bond)"),
          ("TDF 90>30 soft",   r"Glide 90/10$\to$30/70, soft (never sells)"),
          ("TDF 90>3030 hard", r"Glide 90/10$\to$30/30/40, hard (gold endpoint)"),
          ("30/30/40 soft",    r"Hold 30/30/40 throughout (soft)")]
    def _stat(tg, nm):
        M = np.load(f"{BASE}results/tpb_{win}_{tg}.npy", allow_pickle=True).item()["data"][nm]["M"]
        return np.median(M[:, 0]) / 1e6, np.percentile(M[:, 8], 10) * 100
    _grows = []
    for nm, lab in GL:
        fw, fp = _stat("taxfree", nm); tw, tp = _stat("taxable", nm)
        _grows.append(f"{lab} & \\${fw:.2f}M & $-{abs(fp):.0f}\\%$ & \\${tw:.2f}M & $-{abs(tp):.0f}\\%$")
    with open(f"{BASE}results/glide_rows.tex", "w") as _fh:
        _fh.write(" \\\\\n".join(_grows))
    print("wrote results/glide_rows.tex"); sys.exit(0)

d = np.load(f"{BASE}results/tpb_{win}_{tag}{suf}.npy", allow_pickle=True).item()
names, data = d["names"], d["data"]

if "stability" in sys.argv:
    # Resample the 1,000 paths with replacement (paired across strategies) B times; each resample is
    # an independent draw equivalent to a fresh seed. Recompute median wealth + terminal pain, rebuild
    # the within-cap frontier, tally how often each strategy is on it. High % = robust, not seed-luck.
    LIM = 30.0; B = 1000
    rng = np.random.default_rng(0)
    Ms = [(nm, data[nm]["M"]) for nm in names]
    n = len(Ms[0][1]); incl = {nm: 0 for nm in names}
    for _ in range(B):
        bi = rng.integers(0, n, size=n)
        stats = [(nm, np.percentile(M[bi, 8], 10) * 100, np.median(M[bi, 0]) / 1e6) for nm, M in Ms]
        for nm, pain, w in stats:
            if pain < -LIM:
                continue                                  # outside the cap, not a within-cap frontier candidate
            dom = any((q[1] >= pain and q[2] >= w) and (q[1] > pain or q[2] > w)
                      for q in stats if q[0] != nm)
            if not dom:
                incl[nm] += 1
    print(f"=== {win} {tag}: within-cap frontier-inclusion freq over {B} path-resamples (proxy for changing the seed) ===")
    for nm in sorted(names, key=lambda x: -incl[x]):
        if incl[nm] > 0:
            print(f"  {nm:22s} {incl[nm] / B * 100:5.1f}%")
    sys.exit(0)

pts = []   # (name, pain[neg %], median_wealth $M, full_dd[neg %, 10th], median_irr %, median_sharpe)
for nm in names:
    M = data[nm]["M"]
    pts.append((nm, np.percentile(M[:, 8], 10) * 100, np.median(M[:, 0]) / 1e6,
                np.percentile(M[:, 2], 10) * 100, np.median(M[:, 1]) * 100, np.median(M[:, 3])))

def dominated(p):
    return any((q[2] >= p[2] and q[1] >= p[1]) and (q[2] > p[2] or q[1] > p[1])
               for q in pts if q[0] != p[0])
front = sorted([p for p in pts if not dominated(p)], key=lambda p: p[1])

# --- textual ranking companion to the diagram (identical metrics): median wealth / |terminal pain|,
#     partitioned at the -LIM% terminal-pain tolerance, on-frontier strategies flagged ---
LIM = 30.0
onf = {p[0] for p in front}
ratio = lambda p: p[2] / (abs(p[1]) if p[1] else 1e9)
ranked = sorted(pts, key=lambda p: (abs(p[1]) > LIM, -ratio(p)))
print(f"\n=== {win} {tag}: median wealth / |terminal pain| (MC, same basis as the frontier) ===")
print(f"{'Strategy':22s} {'MedWealth':>9s} {'IRR':>6s} {'Sharpe':>6s} {'TermPain':>8s} {'W/TP':>6s} front")
_div = False
for p in ranked:
    if abs(p[1]) > LIM and not _div:
        print(f"--- beyond -{LIM:.0f}% terminal-pain limit ---"); _div = True
    print(f"{p[0]:22s} ${p[2]:7.2f}M {p[4]:5.1f}% {p[5]:6.2f} {p[1]:7.1f}% {ratio(p):6.3f} "
          + ("  yes" if p[0] in onf else ""))

# --- rows for the report's efficient-frontier table: within-cap frontier members, sorted by W/TP ---
_DISP = {"TDF 90>30 hard": r"TDF 90/10$\to$30/70 hard", "All Weather no-reb": "All Weather (no rebal)",
         "All Weather soft": "All Weather (soft)", "glD soft": "glide D (soft)",
         "PermPort hard": "Permanent Portfolio (hard)", "60/40 hard": "60/40 (hard)"}
_eff = sorted([p for p in front if abs(p[1]) <= LIM], key=lambda p: -ratio(p))
with open(f"{BASE}results/eff_{tag}.tex", "w") as _fh:
    _fh.write(" \\\\\n".join(f"{(_DISP[p[0]] if p[0] in _DISP else p[0])} & \\${p[2]:.2f}M & "
                            f"$-{abs(p[1]):.0f}\\%$ & {ratio(p):.3f}" for p in _eff))

def clean(s):
    return (s.replace("TDF 90>3030", "glide 90/10→30/30/40")
             .replace("TDF 90>30", "glide 90/10→30/70")
             .replace(">", "→"))

fig, ax = plt.subplots(figsize=(11, 7))
sc = ax.scatter([-p[1] for p in pts], [p[2] for p in pts], c=[p[3] for p in pts],
                cmap="RdYlGn", s=60, zorder=3, edgecolors="0.35", linewidths=0.4)
cb = fig.colorbar(sc, ax=ax)
cb.set_label("full max drawdown over the whole horizon (10th-pct, %) — green shallow, red deep")
ax.plot([-p[1] for p in front], [p[2] for p in front], "-", color="crimson", lw=2.0, zorder=4,
        label="efficient frontier")

xlo, xhi = ax.get_xlim(); ylo, yhi = ax.get_ylim()
gap = (yhi - ylo) * 0.045
tolabel = list(front)
if not any(p[0] == "30/30/40 soft" for p in front):
    tolabel += [p for p in pts if p[0] == "30/30/40 soft"]
tolabel = sorted(tolabel, key=lambda p: -p[1])    # left (less pain) to right
placed = []
for p in tolabel:
    x, y = -p[1], p[2]; ly = y
    for py in placed:
        if abs(ly - py) < gap:
            ly = py + gap
    placed.append(ly)
    onf = any(p[0] == f[0] for f in front)
    ax.annotate(clean(p[0]), (x, y), xytext=(x + (xhi - xlo) * 0.015, ly), fontsize=8,
                color=("crimson" if onf else "0.45"), weight=("bold" if onf else "normal"),
                va="center", arrowprops=dict(arrowstyle="-", color=("crimson" if onf else "0.6"), lw=0.5))
ax.text(xlo + (xhi - xlo) * 0.03, yhi - (yhi - ylo) * 0.05, "ideal\n(most wealth,\nleast pain)",
        fontsize=9, color="0.5", style="italic", va="top")
ax.set_xlabel("terminal pain (risk) — 10th-pct drawdown in the final 5 years (%);  ◀ less    more ▶")
ax.set_ylabel("median final wealth ($M)")
acct = "tax-free (Roth/401k)" if tag == "taxfree" else "taxable"
ax.set_title(f"Wealth vs terminal pain — {win}–2026, {acct}\nred = efficient frontier (the menu); dots colored by full-horizon drawdown")
ax.grid(True, alpha=0.3); ax.legend(loc="lower right")
out = f"{BASE}results/frontier_{win}_{tag}{suf}.png"
fig.tight_layout(); fig.savefig(out, dpi=140)
print(f"wrote {out}  |  frontier: " + ", ".join(f"{p[0]}({p[1]:.0f}%,${p[2]:.2f}M)" for p in front))
