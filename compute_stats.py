"""Read saved MC arrays (no simulation re-run), compute per-strategy distribution stats and
paired-bootstrap / Calmar-pairwise tests for ALL strategies, append to results_record.md."""
import os, numpy as np

_R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FILES = [("1980","taxable",os.path.join(_R,"core_1980_taxable.npy")),
         ("1980","taxfree",os.path.join(_R,"core_1980_taxfree.npy"))]
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results_record.md")
L = []
def w(s=""): L.append(s)
def calmar(R): return R[:,1]/np.where(np.abs(R[:,2])>1e-9, np.abs(R[:,2]), np.nan)

w("\n\n=================================================================")
w("## DISTRIBUTION STATS + PAIRED TESTS (computed from saved arrays; read-only)")
w("=================================================================")
w("Each column below is an INDEPENDENT statistic across the 1000 paths (a row is NOT one path).")
w("medX = median across paths; p10X = 10th percentile across paths; Calmar = per-path IRR/|maxDD|.")

for win, regime, path in FILES:
    d = np.load(path, allow_pickle=True).item(); data = d["data"]; names = d["names"]
    w(f"\n### {win} / {regime} — per-strategy distribution stats (all strategies)")
    w("| Strategy | medFinal | medIRR | medSharpe | p10Sharpe | medMaxDD | p10MaxDD | medCalmar |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|")
    for nm in names:
        R = data[nm]; cal = calmar(R)
        w(f"| {nm} | ${np.median(R[:,0])/1e6:.2f}M | {np.median(R[:,1])*100:.1f}% | "
          f"{np.median(R[:,3]):.3f} | {np.percentile(R[:,3],10):.3f} | {np.median(R[:,2])*100:.1f}% | "
          f"{np.percentile(R[:,2],10)*100:.1f}% | {np.nanmedian(cal):.3f} |")

# paired bootstrap, 1980 taxable, every soft strategy vs three references
d = np.load(os.path.join(_R, "core_1980_taxable.npy"), allow_pickle=True).item(); data = d["data"]; names = d["names"]
soft = [n for n in names if n.endswith("soft")]
rng = np.random.default_rng(7); B = 5000
def series(nm, kind):
    R = data[nm]
    return {"final": R[:,0], "sharpe": R[:,3], "calmar": calmar(R)}[kind]
def test(a, b, kind):
    da = series(a, kind) - series(b, kind); n = len(da)
    bidx = rng.integers(0, n, size=(B, n)); bs = np.nanmean(da[bidx], 1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); win = np.nanmean(series(a,kind) > series(b,kind))*100
    return np.nanmean(da), lo, hi, win, ("SIG" if (lo>0 or hi<0) else "n.s.")

for ref in ["30/30/40 soft", "PermPort soft", "50/25/25 soft"]:
    w(f"\n### 1980 taxable — paired bootstrap: every soft strategy MINUS {ref} (95% CI of mean diff, 5000 resamples)")
    w("| Strategy − ref | Final mean ($k) | Final 95%CI | F>% | Sharpe mean | Sharpe 95%CI | S>% | Calmar mean | Calmar 95%CI | C>% |")
    w("|---|--:|--|--:|--:|--|--:|--:|--|--:|")
    for nm in soft:
        if nm == ref: continue
        fm,flo,fhi,fw,fs = test(nm, ref, "final")
        sm,slo,shi,sw,ss = test(nm, ref, "sharpe")
        cm,clo,chi,cw,cs = test(nm, ref, "calmar")
        w(f"| {nm.replace(' soft','')} − {ref.replace(' soft','')} | "
          f"{fm/1e3:+.0f} {fs} | [{flo/1e3:+.0f},{fhi/1e3:+.0f}] | {fw:.0f}% | "
          f"{sm:+.4f} {ss} | [{slo:+.4f},{shi:+.4f}] | {sw:.0f}% | "
          f"{cm:+.4f} {cs} | [{clo:+.4f},{chi:+.4f}] | {cw:.0f}% |")

with open(OUT, "w") as _fh:                    # overwrite: the record reflects the current run only
    _fh.write("\n".join(L))
print(f"wrote {len(L)} lines to {OUT}")
