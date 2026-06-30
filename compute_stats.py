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
        w(f"| {nm} | ${np.nanmedian(R[:,0])/1e6:.2f}M | {np.nanmedian(R[:,1])*100:.1f}% | "
          f"{np.nanmedian(R[:,3]):.3f} | {np.nanpercentile(R[:,3],10):.3f} | {np.nanmedian(R[:,2])*100:.1f}% | "
          f"{np.nanpercentile(R[:,2],10)*100:.1f}% | {np.nanmedian(cal):.3f} |")

# Paired bootstrap, 1980 taxable, EVERY strategy in the field (soft, no-reb, hard, GEM, signals)
# tested against the three reference soft baselines. The reference set spans the diversified soft
# (30/30/40), the calm soft (PermPort), and the stocks-tilted soft (50/25/25) so every comparison
# is against a strategy that's in the report's prose.
#
# SIG is Benjamini-Hochberg-controlled across the WHOLE family below (every strategy x 3 references
# x 3 metrics) at a 5% false-discovery rate, so the flags account for the multiple comparisons
# rather than tagging each 95% CI in isolation (which would expect ~5% false positives).
d = np.load(os.path.join(_R, "core_1980_taxable.npy"), allow_pickle=True).item(); data = d["data"]; names = d["names"]
REFS = ["30/30/40 soft", "PermPort soft", "50/25/25 soft"]; METRICS = ["final", "sharpe", "calmar"]
rng = np.random.default_rng(7); B = 5000; FDR = 0.05
def series(nm, kind):
    R = data[nm]
    return {"final": R[:,0], "sharpe": R[:,3], "calmar": calmar(R)}[kind]
def test(a, b, kind):
    da = series(a, kind) - series(b, kind); n = len(da)
    bidx = rng.integers(0, n, size=(B, n)); bs = np.nanmean(da[bidx], 1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5]); win = np.nanmean(series(a,kind) > series(b,kind))*100
    p = min(1.0, 2.0 * min(np.nanmean(bs <= 0), np.nanmean(bs >= 0)))   # two-sided percentile-bootstrap p
    return np.nanmean(da), lo, hi, win, p

# pass 1: run every test in the family and collect p-values (rng order unchanged: ref, strategy, metric)
res = {}
for ref in REFS:
    for nm in names:
        if nm == ref: continue
        for kind in METRICS:
            res[(ref, nm, kind)] = test(nm, ref, kind)
pv = sorted(r[4] for r in res.values()); M = len(pv)
pcrit = max((p for k, p in enumerate(pv, 1) if p <= k / M * FDR), default=-1.0)   # BH critical p
def sigtag(p): return "SIG" if p <= pcrit else "n.s."

# Audit block -- summary stats so the per-reference tables can be sanity-checked at a glance
sig_n  = sum(1 for r in res.values() if r[4] <= pcrit)
ns_n   = M - sig_n
nan_m  = sum(1 for r in res.values() if np.isnan(r[0]))
nan_ci = sum(1 for r in res.values() if np.isnan(r[1]) or np.isnan(r[2]))
nan_p  = sum(1 for r in res.values() if np.isnan(r[4]))
patho_sig_zero  = [k for k, r in res.items() if r[4] <= pcrit and r[1] <= 0 <= r[2]]
patho_ns_excl   = [k for k, r in res.items() if r[4] >  pcrit and not (r[1] <= 0 <= r[2])]
sign_mm         = [k for k, r in res.items() if not np.isnan(r[0]) and (r[0] > 0) != ((r[1] + r[2]) / 2 > 0)]
w(f"\n### 1980 taxable — paired-test family audit")
w(f"- Tests in family: {M} (every strategy x {len(REFS)} references x {len(METRICS)} metrics, minus self-comparisons)")
w(f"- BH critical p at FDR {FDR:.0%}: {pcrit:.5f}")
w(f"- SIG: {sig_n} ({sig_n/M*100:.1f}%); n.s.: {ns_n} ({ns_n/M*100:.1f}%)")
w(f"- NaN counts -- mean: {nan_m}, CI: {nan_ci}, p-value: {nan_p}")
w(f"- Pathology -- SIG with 95%CI containing 0: {len(patho_sig_zero)}; n.s. with 95%CI excluding 0: {len(patho_ns_excl)}; sign mismatch (mean vs CI midpoint): {len(sign_mm)}")
w(f"- Effect size by metric (mean diff: median / 10th-pct / 90th-pct):")
for kind in METRICS:
    arr = [r[0] for k, r in res.items() if k[2] == kind]
    w(f"  - {kind}: {np.median(arr):+.4f} / {np.percentile(arr,10):+.4f} / {np.percentile(arr,90):+.4f}  (n={len(arr)})")

# pass 2: render each reference's table, tagging significance from the family-wide BH threshold
for ref in REFS:
    w(f"\n### 1980 taxable — paired bootstrap: every strategy MINUS {ref} (mean diff, 5000 resamples; SIG = BH-corrected at FDR {FDR:.0%} over the {M}-test family)")
    w("| Strategy − ref | Final mean ($k) | Final 95%CI | F>% | Sharpe mean | Sharpe 95%CI | S>% | Calmar mean | Calmar 95%CI | C>% |")
    w("|---|--:|--|--:|--:|--|--:|--:|--|--:|")
    for nm in names:
        if nm == ref: continue
        fm,flo,fhi,fw,fp = res[(ref, nm, "final")]
        sm,slo,shi,sw,sp = res[(ref, nm, "sharpe")]
        cm,clo,chi,cw,cp = res[(ref, nm, "calmar")]
        w(f"| {nm} − {ref.replace(' soft','')} | "
          f"{fm/1e3:+.0f} {sigtag(fp)} | [{flo/1e3:+.0f},{fhi/1e3:+.0f}] | {fw:.0f}% | "
          f"{sm:+.4f} {sigtag(sp)} | [{slo:+.4f},{shi:+.4f}] | {sw:.0f}% | "
          f"{cm:+.4f} {sigtag(cp)} | [{clo:+.4f},{chi:+.4f}] | {cw:.0f}% |")

with open(OUT, "w") as _fh:                    # overwrite: the record reflects the current run only
    _fh.write("\n".join(L))
print(f"wrote {len(L)} lines to {OUT}")
