#!/usr/bin/env python3
"""One-command pipeline for the report. Runs every stage in order and ends with wealth_report.pdf:

  1. data   build the 1980 dataset: fetch the fund legs (dca_build80), construct the long-Treasury /
            T-bill legs (build_rate_legs) and the REIT leg (build_reit), then merge them into the
            single 8-leg file (dca_build80 merge). Needs network (Yahoo + LBMA) and reit_monthly.csv.
  2. mc     run the Monte Carlo for both tax regimes (parallel_recompute). ~40 min; the slow stage.
  3. tables Appendix F strategy tables -> wealth_tables.tex (gen_tables2).
  4. stats  per-strategy distribution stats + paired tests -> results_record.md (compute_stats).
  5. figs   the two frontier diagrams (plot_frontier) and the calendar 60/40-vs-30/30/40 chart
            (calendar_compare); the income-path and lazy-investor robustness tables (robustness);
            the horizon-robustness table -> results/horizon_rows.tex (horizon_robust).
  6. pdf    compile wealth_report.tex -> wealth_report.pdf (twice, for the table of contents).

Usage:
  python3 make_report.py              full run
  python3 make_report.py --no-fetch   skip stage 1 (reuse the committed CSVs)
  python3 make_report.py --skip-mc    skip stage 2 (reuse the existing results/ stores)
  python3 make_report.py --seed-check run the second-seed robustness check (seed 7) and print the
                                      seed-7 efficient sets for comparison against eff_*.tex. This is
                                      a separate ~40-min Monte Carlo, off by default; it backs the
                                      report's "unchanged under a second random seed" claim.

Note: the report's in-body tables (the efficient-frontier sets, the calendar table, the income-path
and lazy-investor tables) are transcribed from the stage outputs above; stage 5 prints the current
numbers so they can be checked. Only the full-field table (wealth_tables.tex) and the figures are
pulled into the PDF automatically.
"""
import subprocess, sys, time

NOFETCH = "--no-fetch" in sys.argv
SKIPMC = "--skip-mc" in sys.argv
SEEDCHECK = "--seed-check" in sys.argv


def run(desc, *cmd):
    print(f"\n=== {desc} ===\n    {' '.join(cmd)}", flush=True)
    t = time.time()
    subprocess.run(list(cmd), check=True)
    print(f"    [{time.time() - t:.0f}s]", flush=True)


t0 = time.time()

if not NOFETCH:
    run("1. data: fetch fund legs (1980 base)", "python3", "dca_build80.py")
    run("1. data: construct long-Treasury and T-bill legs", "python3", "build_rate_legs.py")
    run("1. data: construct REIT leg", "python3", "build_reit.py")
    run("1. data: merge legs into the unified 8-leg 1980 file", "python3", "dca_build80.py", "merge")

if not SKIPMC:
    run("2. Monte Carlo: 1980, both tax regimes (slow, ~40 min)", "python3", "parallel_recompute.py", "1980")

if SEEDCHECK:
    # Second-seed robustness: re-run the whole field under seed 7 (writes the seed-tagged _s7 stores),
    # then print the seed-7 within-cap efficient sets so they can be compared against the seed-42 sets
    # in results/eff_taxable.tex / eff_taxfree.tex. Backs the report's "unchanged under a second seed".
    run("2b. seed check: Monte Carlo, seed 7 (slow, ~40 min)", "python3", "parallel_recompute.py", "1980", "7")
    run("2b. seed check: seed-7 efficient set, taxable", "python3", "plot_frontier.py", "1980", "taxable", "s7")
    run("2b. seed check: seed-7 efficient set, tax-free", "python3", "plot_frontier.py", "1980", "taxfree", "s7")

run("3. tables: Appendix F (wealth_tables.tex)", "python3", "gen_tables2.py")
run("3. macros: body-prose numbers (results/prose_macros.tex)", "python3", "gen_prose_macros.py")
run("4. stats: results_record.md", "python3", "compute_stats.py")
run("5. figure: efficient frontier, taxable", "python3", "plot_frontier.py", "1980", "taxable")
run("5. figure: efficient frontier, tax-free", "python3", "plot_frontier.py", "1980", "taxfree")
run("5. table: young-risk glide table", "python3", "plot_frontier.py", "1980", "glidetable")
run("5. table: individual-stock drawdowns (Yahoo)", "python3", "build_stock_dd.py", *(("--no-fetch",) if NOFETCH else ()))
run("5. table: terminal-pain figures for the Collins rebuttal", "python3", "collins_pain.py")
run("5. table: wind-down-to-cash strategy (Section 8 close + Appendix G)", "python3", "winddown_strategy.py")
run("5. appendix: long-window bootstrap (Appendix H, 1962-2026)", "python3", "longwindow_appendix.py")
run("5. figure + tables: calendar 60/40 vs diversified 30/30/40", "python3", "calendar_compare.py")
run("5. robustness: income-path + lazy investor", "python3", "robustness.py")
run("5. robustness: horizon (44/40-yr efficient sets)", "python3", "horizon_robust.py", "--fresh")
run("6. compile pass 1", "pdflatex", "-interaction=nonstopmode", "wealth_report.tex")
run("6. compile pass 2", "pdflatex", "-interaction=nonstopmode", "wealth_report.tex")

print(f"\n=== DONE in {(time.time()-t0)/60:.1f} min -> wealth_report.pdf ===")
