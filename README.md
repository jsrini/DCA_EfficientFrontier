# DCA Efficient Frontier

A Monte Carlo study of dollar-cost-averaging strategies over 1980–2026. It puts 56 portfolios
(static allocations, age-based glides, and momentum/signal overlays) through a block-bootstrap
Monte Carlo in a fully taxable and a fully tax-free account, and ranks them on median final wealth
versus **terminal pain** (the 10th-percentile drawdown over the final five years before retirement).

The full write-up is **`wealth_report.pdf`**.

## Build

Run from inside this folder:

```
python3 make_report.py             # full pipeline -> wealth_report.pdf
python3 make_report.py --no-fetch  # reuse the committed data (skip the network fetch)
python3 make_report.py --skip-mc   # reuse the existing results/ (skip the ~40-min Monte Carlo)
python3 make_report.py --seed-check # second-seed robustness run (seed 7), separate ~40-min Monte Carlo
```

The report rests its conclusions on the strategies that stay efficient under a second random seed.
`--seed-check` reproduces that: it re-runs the full Monte Carlo under seed 7 and prints the seed-7
efficient sets to compare against the seed-42 sets in `results/eff_taxable.tex` / `eff_taxfree.tex`.
It is off by default because it is a second ~40-minute run.

The pipeline fetches the fund data, builds the 1980 dataset, runs the Monte Carlo, generates the
tables and figures into `results/`, and compiles the report. `dca_files_readme.md` is the file map.

## Disclaimer

This is a backtest and thought experiment, not financial advice. See the disclaimer in the report.
It relies on AI-written code and public data (Yahoo Finance, the LBMA gold benchmark, the FTSE Nareit
index); reproduce any figure before relying on it.
