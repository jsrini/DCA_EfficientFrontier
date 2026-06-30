"""Standalone (NOT one of the report's 56) wind-down strategy: 100% US stock until five years
before retirement, then a scheduled glide to cash.

  * Contributions (20% of salary, the report's DCA premise) buy 100% US stock -- the STOCK leg, which
    on the 1980--2026 basis is VFINX (Vanguard 500 Index, an S&P 500 tracker; see dca_build80.py) --
    for the entire accumulation, INCLUDING the wind-down (we keep buying stocks throughout).
  * On March 15 of the year five years before retirement, sell 50% of the stock holding into the CASH
    leg. On March 15 of each of the next five years, sell a further slice so the position liquidates in
    equal 10%-of-original steps: as a fraction of the stock STILL held that March that is
    0.50, 0.20, 0.25, 1/3, 0.50, 1.00 -- the last fully converts the remaining stock to cash.
  * Taxable account realizes capital gains on every sale (cost basis tracked, CG taxed the following
    January, exactly as run_static_hard does); the tax-free account (shelter=1) waives it.

Reuses dca_core.Engine for the data, tax accrual, dividend re-bin, bootstrap, and metrics, so the
numbers sit on the report's own basis (1980--2026, seed 42, 1,000 paths, received-preserving re-bin).
It does NOT touch dca_core.py or the report's results/, so the 56-strategy field is unchanged.

Run: python3 winddown_strategy.py
"""
import os, sys
import numpy as np
from dca_core import Engine

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
N, SEED = 1000, 42   # SEED overridable on the command line (e.g. `python3 winddown_strategy.py 7`)


# The three wind-downs, on equal footing. Each holds 100% US stock until five years before retirement,
# then sells the stock down to cash over six March 15s. They SHARE the same staging of the stock still
# held -- the first sale, then 20%, 25%, 1/3, 50%, 100% (the last fully liquidates) -- and differ only
# in (a) the size of the FIRST sale and (b) where contributions go once the wind-down has begun.
#   (label, first-sale fraction, leg that receives contributions after the first sale)
WINDDOWNS = [
    ("sell 50%, contribute to stock", 0.50, "STOCK"),   # the original
    ("sell 60%, contribute to stock", 0.60, "STOCK"),
    ("sell 40%, contribute to cash",  0.40, "CASH"),
]
WD_TAIL = [0.20, 0.25, 1.0 / 3.0, 0.50, 1.00]           # common staging after the first sale


def make_winddown(e, first_frac, inflow_role, retire_year=None):
    """Return a base(P,Y)->metrics runner for one staged wind-down on engine `e`. Contributions buy
    stock until the first sale, then the `inflow_role` leg. Mirrors Engine.run_static_hard accounting.
    `retire_year` anchors the sale schedule (first sale five years before it); defaults to the data
    end, like Engine.run_contglide's `end` arg, so a shorter-horizon investor can re-anchor it."""
    si = e.ASSETS.index(e.col["STOCK"])
    ci = e.ASSETS.index(e.col["CASH"])
    ii = e.ASSETS.index(e.col[inflow_role])             # contribution destination after the first sale
    A, T, cdset, mon, yr, cpos, camt = e.A, e.T, e.cdset, e.mon, e.yr, e.cpos, e.camt
    if retire_year is None:
        retire_year = int(e.idx[-1].year)
    sale_years = list(range(retire_year - 5, retire_year + 1))
    sched = dict(zip(sale_years, [first_frac] + WD_TAIL))
    first_year = sale_years[0]

    def run(P, Y):
        sh = np.zeros(A); basis = np.zeros(A); val = np.zeros(T)
        incyr = {}; cgyr = {}; paid = set(); paidcg = set(); cgtax = 0.0
        shist = np.zeros((T, A)) if e._track else None
        sold = False                                                         # has the first sale happened yet?
        for t in range(T):
            if e._track: shist[t] = sh
            incyr[yr[t]] = incyr.get(yr[t], np.zeros(A)) + Y[t] * sh * P[t]
            if t in cdset and mon[t] == 1:                                    # settle last year's taxes
                sh, basis = e._pay_income(sh, P[t], basis, incyr, yr[t] - 1, paid, cgyr, yr[t])
                sh, basis, ct = e._pay_cg(sh, P[t], basis, cgyr, yr[t] - 1, paidcg, yr[t]); cgtax += ct
            if t in cdset and mon[t] == 3 and yr[t] in sched:                # wind-down sale, March 15
                f = sched[yr[t]]; sv = sh[si] * P[t][si]
                if sv > 0 and f > 0:
                    cgyr[yr[t]] = cgyr.get(yr[t], 0.0) + f * max(0.0, sv - basis[si])
                    proceeds = f * sv * (1.0 - e.tc)                          # sell-leg fee on the stock
                    sh[si] -= f * sh[si]; basis[si] -= f * basis[si]
                    buy = proceeds * (1.0 - e.tc)                             # buy-leg fee into cash
                    sh[ci] += buy / P[t][ci]; basis[ci] += buy
                if yr[t] == first_year: sold = True                          # contributions switch leg after it
            if t in cdset:                                                    # contribution: stock, then inflow leg
                dest = ii if sold else si
                a = camt[cpos[t]]; sh[dest] += a * (1.0 - e.tc) / P[t][dest]; basis[dest] += a * (1.0 - e.tc)
            val[t] = (sh * P[t]).sum()
        if e._track: e._shist = shist
        e.last_cgtax = cgtax; e.last_val = val
        tvf = (sh * P[-1]).sum(); e.last_w = (sh * P[-1]) / tvf if tvf > 0 else np.zeros(A)
        return e.metrics(val)

    return run, sched


def soft_static(e, wdict):
    """A soft (never-sell) static mix on the same engine, for a reference baseline."""
    w = e.wvec(wdict)
    return lambda P, Y: e.run_static(P, Y, w, True)


# Path-coherent reporting, identical to the report's Appendix G (gen_tables2.py): each MC column is a
# SINGLE bootstrap path selected at a percentile, and every metric in that column is read from that
# same path. metric cols: 0 fv, 1 irr, 2 fullDD, 3 sharpe, 4 dur, 5 recov, 6 ptt, 7 resid, 8 termDD.
def _rec(p): return f"{p[4]:.1f}y" if p[5] == 1 else f"{p[7]*100:.0f}% at end"
def _cal(p): return p[1] / abs(p[2]) if abs(p[2]) > 1e-9 else float("nan")
METRICS = [
    ("Wealth",         lambda p: f"${p[0]/1e6:.2f}M"),
    ("IRR",            lambda p: f"{p[1]*100:.1f}%"),
    ("Max DD (full)",  lambda p: f"{p[2]*100:.1f}%"),
    ("Final-5yr DD",   lambda p: f"{p[8]*100:.1f}%"),
    ("Peak-to-trough", lambda p: f"{p[6]:.1f}y"),
    ("Recovery",       lambda p: _rec(p)),
    ("Sharpe",         lambda p: f"{p[3]:.2f}"),
    ("Calmar",         lambda p: f"{_cal(p):.2f}"),
]


def select_reps(M, act):
    """The four report columns, path-coherent, exactly as gen_tables2.py selects them."""
    n = len(M)
    med = M[np.argsort(M[:, 1])[int(round(0.50 * n))]]   # MC median-by-return  (sorted on IRR, col 1)
    r10 = M[np.argsort(M[:, 1])[int(round(0.10 * n))]]   # MC 10th-pct-by-return
    d10 = M[np.argsort(M[:, 2])[int(round(0.10 * n))]]   # MC 10th-pct-by-maxDD (sorted on full DD, col 2)
    tp = np.percentile(M[:, 8], 10)                       # strategy-level terminal pain
    return [act, med, r10, d10], tp


def report_block(name, M, act):
    """Print one strategy's path-coherent block (stdout)."""
    reps, tp = select_reps(M, act)
    print(f"\n  {name}   (terminal pain {tp*100:.0f}%)")
    print(f"  {'Metric':16s} {'Actual':>10s} {'MC med/ret':>12s} {'MC 10th/ret':>12s} {'MC 10th/DD':>12s}")
    for lab, fn in METRICS:
        print(f"  {lab:16s} " + " ".join(f"{fn(r):>12s}" for r in reps))


# -- LaTeX emitters: Appendix H table + body macros, in the report's own style (gen_tables2.py) -------
def _lrec(p): return f"{p[4]:.1f}y" if p[5] == 1 else f"${p[7]*100:.0f}\\%$\\,end"
def _lcal(p): return p[1] / abs(p[2]) if abs(p[2]) > 1e-9 else float("nan")
LATEX_METRICS = [
    ("Wealth",         lambda p: f"${p[0]/1e6:.2f}$M"),
    ("IRR",            lambda p: f"{p[1]*100:.1f}\\%"),
    ("Max DD (full)",  lambda p: f"${p[2]*100:.1f}\\%$"),
    ("Final-5yr DD",   lambda p: f"${p[8]*100:.1f}\\%$"),
    ("Peak-to-trough", lambda p: f"{p[6]:.1f}y"),
    ("Recovery",       _lrec),
    ("Sharpe",         lambda p: f"{p[3]:.2f}"),
    ("Calmar",         lambda p: f"{_lcal(p):.2f}"),
]
_HDR = ("Strategy & Metric & Actual & \\makecell{MC median\\\\by return} & \\makecell{MC 10th pct\\\\by return} "
        "& \\makecell{MC 10th pct\\\\by max DD} \\\\")


def _latex_block(name, M, act):
    reps, tp = select_reps(M, act)
    namecell = f"\\makecell[l]{{{name}\\\\{{\\scriptsize term.\\ pain ${tp*100:.0f}\\%$}}}}"
    rows = []
    for i, (lab, fn) in enumerate(LATEX_METRICS):
        vals = " & ".join(fn(r) for r in reps)
        head = f"\\multirow{{8}}{{*}}{{{namecell}}}" if i == 0 else ""
        term = " \\\\*" if i < len(LATEX_METRICS) - 1 else " \\\\ \\hline"
        rows.append(f"{head} & {lab} & {vals}{term}")
    return "\n".join(rows)


def _latex_table(treg, blocks):
    title = ("\\toprule \\multicolumn{6}{@{}l}{\\normalsize\\textbf{Wind-down to cash \\textemdash\\ 1980, "
             + treg + "}} \\\\ \\midrule " + _HDR + " \\midrule")
    out = [f"\\paragraph{{1980, {treg}.}} {{\\scriptsize The figure under each strategy name is its "
           "\\emph{terminal pain}, the 10th-percentile final-5yr drawdown across all 1{,}000 paths. Each "
           "strategy spans eight rows, one per metric. The three wind-downs differ only in the size of "
           "the first sale and where contributions go afterward; 100\\% US stock (soft) is the all-stock "
           "baseline they de-risk from.}",
           "{\\scriptsize\\renewcommand{\\arraystretch}{1.05}",
           "\\begin{longtable}{@{}ll cccc@{}}",
           title + " \\endfirsthead",
           title + " \\endhead",
           "\\bottomrule \\endfoot"]
    out += blocks
    out += ["\\end{longtable}}", ""]
    return out


def write_report_tex(results):
    """Write the body macros and the Appendix H table from the primary-seed run."""
    wt = {"taxable": results["TAXABLE"], "tax-free": results["TAX-FREE"]}
    # body macros: median final wealth ($M) and terminal pain (|%|). The Section 8 prose cites the
    # original wind-down (sell 50%, the first block) and the all-stock baseline (the last block).
    def med_wealth(M): return f"{np.median(M[:, 0])/1e6:.2f}"
    def pain(M): return f"{abs(np.percentile(M[:, 8], 10)*100):.0f}"
    wd_tx, st_tx = wt["taxable"][0], wt["taxable"][-1]      # (name, mc, act)
    wd_tf, st_tf = wt["tax-free"][0], wt["tax-free"][-1]
    macros = {
        "WDtaxWealth": med_wealth(wd_tx[1]), "WDtaxPain": pain(wd_tx[1]),
        "WDtfWealth": med_wealth(wd_tf[1]), "WDtfPain": pain(wd_tf[1]),
        "WDstockTaxWealth": med_wealth(st_tx[1]), "WDstockTaxPain": pain(st_tx[1]),
        "WDstockTfWealth": med_wealth(st_tf[1]), "WDstockTfPain": pain(st_tf[1]),
    }
    # how many of the three wind-downs are EFFICIENT within the -30% cap, per account: non-dominated on
    # median wealth vs terminal pain against the 56-field + the three variants, AND terminal pain <= 30%.
    # This is the report's own frontier test (plot_frontier.py), so the Section 8 / Key Findings claim is
    # reproducible rather than asserted. tpb_*.npy are the same stores the frontier and Appendix G use.
    def eff_within_cap(tpb_tag, blocks):
        fld = np.load(os.path.join(RESULTS, f"tpb_1980_{tpb_tag}.npy"), allow_pickle=True).item()
        pts = [(nm, np.percentile(fld["data"][nm]["M"][:, 8], 10) * 100, np.median(fld["data"][nm]["M"][:, 0]) / 1e6)
               for nm in fld["names"]]
        wd = [(name, np.percentile(mc[:, 8], 10) * 100, np.median(mc[:, 0]) / 1e6) for name, mc, _ in blocks[:-1]]
        allp = pts + wd
        def dominated(p):
            return any((q[2] >= p[2] and q[1] >= p[1]) and (q[2] > p[2] or q[1] > p[1]) for q in allp if q[0] != p[0])
        return sum(1 for p in wd if not dominated(p) and abs(p[1]) <= 30.0)
    macros["WDtaxEffN"] = str(eff_within_cap("taxable", wt["taxable"]))
    macros["WDtfEffN"] = str(eff_within_cap("taxfree", wt["tax-free"]))
    with open(os.path.join(RESULTS, "winddown_macros.tex"), "w") as fh:
        fh.write("% Auto-generated by winddown_strategy.py -- do not edit by hand.\n")
        for k, v in macros.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    # Section 8 summary table: each strategy's terminal pain and median wealth, both accounts.
    # Row order: all-stock baseline, then the three wind-downs. blocks are (name, mc, act).
    def srow(label, i):
        tx, tf = wt["taxable"][i][1], wt["tax-free"][i][1]
        return (f"{label} & $-{pain(tx)}\\%$ & \\${med_wealth(tx)} M & "
                f"$-{pain(tf)}\\%$ & \\${med_wealth(tf)} M")
    srows = [srow("100\\% US stock", -1), srow("Wind-down, sell 50\\%", 0),
             srow("Wind-down, sell 60\\%", 1), srow("Wind-down, sell 40\\%", 2)]
    with open(os.path.join(RESULTS, "winddown_summary_rows.tex"), "w") as fh:
        fh.write(" \\\\\n".join(srows))
    out = ["% Auto-generated by winddown_strategy.py from a fresh seed-42 run -- do not edit by hand.", ""]
    for treg in ("taxable", "tax-free"):
        blocks = [_latex_block(name.replace("%", "\\%"), mc, act) for name, mc, act in wt[treg]]
        out += _latex_table(treg, blocks)
    with open(os.path.join(RESULTS, "winddown_appendixH.tex"), "w") as fh:
        fh.write("\n".join(out))


def run_account(shelter, tag, seed=SEED):
    e = Engine("1980", shelter=shelter)
    paths = e.paths(N, seed)
    Pa, Ya = e.actual()
    print(f"\n=== {tag} account (shelter={shelter:.0f})  1980--2026, {N} paths, seed {seed} ===")
    blocks = []                                             # (name, mc, act) -- three wind-downs, then baseline
    for desc, ff, role in WINDDOWNS:
        run, sched = make_winddown(e, ff, role)
        act = run(Pa, Ya); mc = np.array([e.run_rebin(run, P, Y) for P, Y in paths])
        name = f"Wind-down ({desc})"
        print(f"  schedule -- {name}: " +
              ", ".join(f"{y}:{f*100:.0f}%" for y, f in sched.items()))
        blocks.append((name, mc, act))
    stock_run = soft_static(e, {"STOCK": 1.0})              # all-stock baseline the wind-downs de-risk from
    blocks.append(("100% US stock (soft)",
                   np.array([e.run_rebin(stock_run, P, Y) for P, Y in paths]), stock_run(Pa, Ya)))
    for name, mc, act in blocks:
        report_block(name, mc, act)
    return blocks


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else SEED
    results = {tag: run_account(shelter, tag, seed) for shelter, tag in [(0.0, "TAXABLE"), (1.0, "TAX-FREE")]}
    if seed == SEED:                                         # only the report's primary seed feeds the PDF
        write_report_tex(results)
        print("\nwrote results/winddown_macros.tex and results/winddown_appendixH.tex")
