"""Calendar-year time-weighted returns for stock/bond vs diversified strategies, 1980--2026, on the
unified engine (dca_core). Tax-free basis (shelter=1) so the figures show the strategies' own
performance without tax-payment artifacts in individual years. Prints a per-year table and writes a
bar-chart of a stock/bond mix vs a diversified mix, highlighting down years for the stock/bond mix.

Run: python3 calendar_compare.py
"""
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dca_core import Engine

e = Engine("1980", shelter=0.0)   # taxable basis, to match the copied Calendar-Year Returns table
Pa, Ya = e.actual()
yr = e.yr
cflow = np.zeros(e.T); cflow[e.cd] = e.camt

def yearly_twr(fn):
    fn(); val = e.last_val
    tw = np.zeros(e.T); nz = val[:-1] > 0
    tw[1:][nz] = (val[1:][nz] - cflow[1:][nz]) / val[:-1][nz] - 1.0
    return {int(y): float(np.prod(1.0 + tw[yr == y]) - 1.0) for y in sorted(set(yr.tolist()))}

STRATS = [
    ("60/40",       lambda: e.run_static(Pa, Ya, e.wv("60/40"), True)),
    ("70/30",       lambda: e.run_static(Pa, Ya, e.wv("70/30"), True)),
    ("30/30/40",    lambda: e.run_static(Pa, Ya, e.wv("30/30/40"), True)),
    ("All Weather", lambda: e.run_static(Pa, Ya, e.wv("All Weather"), True)),
    ("PermPort",    lambda: e.run_static(Pa, Ya, e.wv("PermPort"), True)),
]
res = {nm: yearly_twr(fn) for nm, fn in STRATS}
years = sorted(res["60/40"].keys())

print("Calendar-year time-weighted return (%), tax-free basis, 1980--2026")
print("Year  " + "".join(f"{nm:>12s}" for nm, _ in STRATS))
for y in years:
    mark = "  <-- 60/40 down" if res["60/40"][y] < 0 else ""
    print(f"{y}  " + "".join(f"{res[nm][y]*100:>11.1f} " for nm, _ in STRATS) + mark)

# down years for 60/40
dn = [y for y in years if res["60/40"][y] < 0]
print(f"\n60/40 down years: {dn}")
for y in dn:
    print(f"  {y}: 60/40 {res['60/40'][y]*100:+.1f}%  30/30/40 {res['30/30/40'][y]*100:+.1f}%  "
          f"All Weather {res['All Weather'][y]*100:+.1f}%  PermPort {res['PermPort'][y]*100:+.1f}%")

# raw leg returns (year over year; gold is price, no fee) + 30/30/40 TWR -> report calendar table rows
adj_raw = e.adj.values
si, gi, bi = (e.ASSETS.index(e.col[r]) for r in ("STOCK", "GOLD", "BOND"))
eoy = {y: int(np.where(yr == y)[0][-1]) for y in years}
def fp(v): return f"{v*100:.1f}" if v >= 0 else f"$-{abs(v)*100:.1f}$"   # 47-yr table: pos plain, neg in math
def fm(v): return f"${v*100:.1f}$"                                        # down-years table: all in math
cal = {}
prev = 0
for k, y in enumerate(years):
    i0 = 0 if k == 0 else prev
    legret = lambda leg: adj_raw[eoy[y], leg] / adj_raw[i0, leg] - 1.0
    cal[y] = (legret(si), legret(gi), legret(bi), res["30/30/40"][y])
    prev = eoy[y]
def crow(y): s, g, b, m = cal[y]; return f"{y} & {fp(s)} & {fp(g)} & {fp(b)} & {fp(m)}"
with open("results/calendar_rows_left.tex", "w") as fh:
    fh.write(" \\\\\n".join(crow(y) for y in years if y <= 2002))
with open("results/calendar_rows_right.tex", "w") as fh:
    fh.write(" \\\\\n".join(crow(y) for y in years if y >= 2003))
with open("results/downyears_rows.tex", "w") as fh:        # 60/40 down years: Year & Gold & 60/40 & 30/30/40
    fh.write(" \\\\\n".join(f"{y} & {fm(cal[y][1])} & {fm(res['60/40'][y])} & {fm(res['30/30/40'][y])}" for y in dn))
print("wrote results/calendar_rows_{left,right}.tex and downyears_rows.tex")

# asset-class drawdowns over the window (daily total return) -- substantiates the bond/diversification
# discussion: stocks suffered severe multi-year drawdowns in-sample; intermediate bonds did not.
print("\nAsset-class worst drawdown (daily total return), 1980--2026:")
print(f"  {'asset':6s} {'worst DD':>9s}  {'peak':>10s} {'trough':>10s} {'underwater':>11s}")
DDLAB = {"STOCK": "US stocks", "BOND": "Intermediate bonds", "LONGT": "Long Treasuries",
         "GOLD": "Gold", "REIT": "REITs"}
dd_rows = []
for role in ("STOCK", "BOND", "LONGT", "GOLD", "REIT"):
    if role not in e.col: continue
    s = adj_raw[:, e.ASSETS.index(e.col[role])].astype(float)
    pk = np.maximum.accumulate(s); dd = (s - pk) / pk
    tr = int(dd.argmin()); pkidx = int(s[:tr + 1].argmax())
    rec = np.where(s[tr:] >= s[pkidx])[0]
    uw = ((e.idx[tr + int(rec[0])] if len(rec) else e.idx[-1]) - e.idx[pkidx]).days / 365.25
    tail = "" if len(rec) else " (not recovered)"
    print(f"  {role:6s} {dd[tr]*100:8.1f}%  {str(e.idx[pkidx].date()):>10s} {str(e.idx[tr].date()):>10s} {uw:9.1f}y{tail}")
    uwtex = f"{uw:.0f} yr" + ("" if len(rec) else "$^{*}$")
    dd_rows.append(f"{DDLAB[role]} & ${dd[tr]*100:.0f}\\%$ & {e.idx[pkidx].year}--{e.idx[tr].year} & {uwtex}")
with open("results/asset_dd_rows.tex", "w") as fh:
    fh.write(" \\\\\n".join(dd_rows))
print("wrote results/asset_dd_rows.tex")

# chart: 60/40 vs 30/30/40 by year
fig, ax = plt.subplots(figsize=(13, 5))
x = np.arange(len(years)); w = 0.4
ax.bar(x - w/2, [res["60/40"][y]*100 for y in years], w, label="60/40 (stocks/bonds)", color="#c0504d")
ax.bar(x + w/2, [res["30/30/40"][y]*100 for y in years], w, label="30/30/40 (diversified, incl. gold)", color="#4f81bd")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([str(y) for y in years], rotation=90, fontsize=7)
ax.set_ylabel("time-weighted return (%)")
ax.set_title("Calendar-year return: 60/40 vs diversified 30/30/40, 1980--2026 (tax-free basis)")
ax.legend(); ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig("results/calendar_60_40_vs_div.png", dpi=140)
print("\nwrote results/calendar_60_40_vs_div.png")
