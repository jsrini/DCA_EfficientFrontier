"""Build the individual-stock drawdown data for the Market-Concentration section.

Fetches split-adjusted daily closes (and Cisco's dividend-adjusted close, for the total-return
note) for a set of large-cap stocks and the Nasdaq Composite from Yahoo, caches them to a local
CSV, and writes:
  results/stock_dd_rows.tex   -- the drawdown table rows (Cisco all-time; the rest deepest in 7yr)
  results/stock_dd_macros.tex -- every figure the section's prose cites, as \\newcommand macros

Everything in the table and every recovery/decline figure in the prose is COMPUTED from the cached
price series, so it is checkable. The only inputs that cannot be derived from a price series are the
index market-cap / weight figures (how big the Nasdaq-100 is relative to the whole US market, and the
top-ten S&P weight); those are kept as clearly-sourced constants in CONC below and emitted alongside,
labelled as cited rather than computed.

Run:  python3 build_stock_dd.py            (fetch fresh, refresh the local cache)
      python3 build_stock_dd.py --no-fetch (reuse the committed stock_dd_prices.csv)
"""
import os, sys, json, urllib.request, urllib.parse, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
CACHE = os.path.join(HERE, "stock_dd_prices.csv")          # committed local copy of the fetched data
NOFETCH = "--no-fetch" in sys.argv

# Cisco is shown from its all-time peak; the rest are each company's deepest fall in the trailing 7 yr.
CISCO = "CSCO"
RECENT = ["META", "NFLX", "NVDA", "AMZN", "GOOGL", "MSFT", "AAPL"]
NAME = {"CSCO": "Cisco", "META": "Meta", "NFLX": "Netflix", "NVDA": "Nvidia",
        "AMZN": "Amazon", "GOOGL": "Alphabet", "MSFT": "Microsoft", "AAPL": "Apple"}
# Dot-com decline of the Nasdaq-100, measured on QQQ -- the investable form of the index this section
# is about, the report's own US-technology leg, trading since 1999-03-10 (so it spans the 2000 peak).
QQQ = "QQQ"
# Japan's Nikkei 225 -- the counterexample to "the market always comes back": a major developed-market
# index that took decades, not years, to regain its peak. Computed from the index level (close).
N225 = "^N225"
RECENT_YEARS = 7

# Index concentration -- NOT derivable from price series; cited constants (single source of truth).
CONC = dict(
    ndx_mktcap_T=40.0,      # Nasdaq-100 aggregate market cap, $ trillion (stockanalysis.com)
    us_mktcap_T=69.0,       # total US market value, $ trillion (Siblis Research)
    ndx_share_2000=33.0,    # Nasdaq-100 was ~a third of the US market at the 2000 peak (%)
    top10_now=41.0,         # ten largest stocks as a share of the S&P 500 today (%)
    top10_2000=27.0,        # the same share at the 2000 peak (%)
)
CONC_SRC = (r"Nasdaq-100 aggregate market cap from stockanalysis.com; total US market value from "
            r"Siblis Research; top-ten S\&P 500 weight from RBC Wealth Management, "
            r"\emph{The Great Narrowing}, and CFA Institute, \emph{Market Concentration and Lost "
            r"Decades} (2025).")


def fetch(t):
    """Yahoo daily: split-adjusted close, and dividend+split-adjusted close (total return)."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(t)}"
           f"?period1=-630000000&period2=9999999999&interval=1d&events=splits")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    fh = urllib.request.urlopen(req, timeout=60)
    try:
        j = json.load(fh)
    finally:
        fh.close()
    r = j["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    cl = pd.Series(r["indicators"]["quote"][0]["close"], index=ts)
    ad = pd.Series(r["indicators"]["adjclose"][0]["adjclose"], index=ts)
    cl = cl[~cl.index.duplicated(keep="last")].dropna()
    ad = ad[~ad.index.duplicated(keep="last")].dropna()
    return cl, ad


def load_prices():
    """Return (close_df, cisco_tr): split-adjusted closes for every ticker, plus Cisco total return.
    Fetches from Yahoo and rewrites the local cache, unless --no-fetch reuses the committed CSV."""
    if NOFETCH:
        if not os.path.exists(CACHE):
            sys.exit(f"--no-fetch but {CACHE} is missing; run once without it to build the cache.")
        df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        return df.drop(columns=["CSCO_TR"]), df["CSCO_TR"].dropna()
    tickers = [CISCO] + RECENT + [QQQ, N225]
    closes = {}
    cisco_tr = None
    for t in tickers:
        cl, ad = fetch(t)
        closes[t] = cl
        if t == CISCO:
            cisco_tr = ad.rename("CSCO_TR")
        print(f"  fetched {t:6s} {len(cl):5d} days {cl.index[0].date()}..{cl.index[-1].date()}")
    df = pd.DataFrame(closes).sort_index()
    out = df.join(cisco_tr, how="left")
    out.to_csv(CACHE)
    print(f"  cached -> {os.path.relpath(CACHE, HERE)} ({len(out)} rows)")
    return df, cisco_tr


def episode(close, start=None):
    """Deepest peak-to-trough drawdown of `close` (optionally restricted to dates >= start for the
    trough search). Returns (peak_date, trough_date, decline_frac, recover_date_or_None). Recovery is
    searched over the full series after the trough, back to the pre-trough peak price."""
    s = close.dropna()
    w = s[s.index >= start] if start is not None else s
    run = w.cummax()
    dd = w / run - 1.0
    trough = dd.idxmin()
    peak = w.loc[:trough].idxmax()
    peak_px = s.loc[peak]
    after = s[s.index > trough]
    regate = after[after >= peak_px]
    recover = regate.index[0] if len(regate) else None
    return peak, trough, float(s.loc[trough] / peak_px - 1.0), recover


def yrs(peak, recover, last):
    return (recover - peak).days / 365.25 if recover is not None else None


def row(ticker, peak, trough, decl, recover, last):
    """One LaTeX table row: Company & Peak & Trough & Decline & Time to recover."""
    rec = (f"{yrs(peak, recover, last):.1f} yr" if recover is not None
           else "underwater$^{*}$")
    return (f"{NAME[ticker]} & {peak.strftime('%b %Y')} & {trough.strftime('%b %Y')} & "
            f"$-{abs(decl)*100:.0f}\\%$ & {rec}")


def main():
    print("building individual-stock drawdowns" + (" (cached)" if NOFETCH else " (fetching)"))
    close, cisco_tr = load_prices()
    last = close.index[-1]
    recent_start = last - pd.DateOffset(years=RECENT_YEARS)

    # Cisco: all-time. The rest: deepest fall whose trough lands in the trailing 7 years.
    ep = {CISCO: episode(close[CISCO])}
    for t in RECENT:
        ep[t] = episode(close[t], start=recent_start)

    # rows: Cisco first, then the seven sorted by depth of decline (deepest first)
    ordered = [CISCO] + sorted(RECENT, key=lambda t: ep[t][2])
    rows = [row(t, *ep[t], last) for t in ordered]
    with open(os.path.join(RESULTS, "stock_dd_rows.tex"), "w") as fh:
        fh.write(" \\\\\n".join(rows))
    print("  wrote results/stock_dd_rows.tex")

    # Cisco total-return recovery (note that it recovered earlier with dividends reinvested)
    cp, ct, cdecl, crec = ep[CISCO]
    tr_pk = cisco_tr.loc[:ct].idxmax(); tr_pkpx = cisco_tr.loc[tr_pk]
    tr_after = cisco_tr[cisco_tr.index > ct]; tr_reg = tr_after[tr_after >= tr_pkpx]
    cisco_tr_year = tr_reg.index[0].year if len(tr_reg) else None

    # the seven recent declines: range cited in the prose
    rec_decls = [abs(ep[t][2]) * 100 for t in RECENT]

    # dot-com decline of the Nasdaq-100, measured on QQQ; recovery searched over the full series
    q = close[QQQ].dropna()
    ip = q[:"2000-12"].idxmax()
    it = q.loc[ip:"2003"].idxmin()
    ndx_decl = abs(q.loc[it] / q.loc[ip] - 1) * 100
    q_after = q[(q.index > it) & (q >= q.loc[ip])]
    irec_date = q_after.index[0] if len(q_after) else None
    ndx_recover_yrs = (irec_date - ip).days / 365.25 if irec_date is not None else None

    # Japan: the Nikkei 225's post-1989 decline and the decades it stayed below the peak (computed)
    n = close[N225].dropna()
    jp = n[:"1990"].idxmax()
    jt = n.loc[jp:].idxmin()
    jp_decl = abs(n.loc[jt] / n.loc[jp] - 1) * 100
    n_after = n[(n.index > jt) & (n >= n.loc[jp])]
    jp_rec = n_after.index[0] if len(n_after) else None
    jp_recover_yrs = (jp_rec - jp).days / 365.25 if jp_rec is not None else None

    # concentration: 58% derives from the two cited market caps; the "erase" figures are that share
    # (or the 2000 share) times the COMPUTED dot-com decline.
    ndx_share_now = CONC["ndx_mktcap_T"] / CONC["us_mktcap_T"] * 100
    erase_now = ndx_share_now / 100 * ndx_decl
    erase_2000 = CONC["ndx_share_2000"] / 100 * ndx_decl

    M = {
        "CiscoDecline": f"{abs(cdecl)*100:.0f}",
        "CiscoRecoverYears": f"{int(yrs(cp, crec, last))}" if crec is not None else r"still underwater",
        "CiscoTRRecoverYear": str(cisco_tr_year) if cisco_tr_year else "--",
        "BigTechMinDecline": f"{min(rec_decls):.0f}",
        "BigTechMaxDecline": f"{max(rec_decls):.0f}",
        "NasdaqDotcomDecline": f"{ndx_decl:.0f}",
        "NasdaqDotcomRecoverYears": f"{ndx_recover_yrs:.0f}" if ndx_recover_yrs else "--",
        "JapanPeakYear": str(jp.year),
        "JapanTroughYear": str(jt.year),
        "JapanRecoverYear": str(jp_rec.year) if jp_rec is not None else "--",
        "JapanDecline": f"{jp_decl:.0f}",
        "JapanRecoverYears": f"{jp_recover_yrs:.0f}" if jp_recover_yrs else "--",
        "NdxMktcap": f"{CONC['ndx_mktcap_T']:.0f}",
        "UsMktcap": f"{CONC['us_mktcap_T']:.0f}",
        "NdxShareNow": f"{ndx_share_now:.0f}",
        "NdxShareThen": f"{CONC['ndx_share_2000']:.0f}",
        "TopTenNow": f"{CONC['top10_now']:.0f}",
        "TopTenThen": f"{CONC['top10_2000']:.0f}",
        "EraseNow": f"{erase_now:.0f}",
        "EraseThen": f"{erase_2000:.0f}",
        "ConcAsOf": last.strftime("%B %Y"),
        "ConcSources": CONC_SRC,
        "JapanSource": (r"Nikkei~225 daily close (Nikkei Inc.); the index first closed back above its "
                        r"29~December~1989 record on 22~February~2024."),
    }
    with open(os.path.join(RESULTS, "stock_dd_macros.tex"), "w") as fh:
        fh.write("% Auto-generated by build_stock_dd.py -- do not edit by hand.\n")
        for k, v in M.items():
            fh.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
    print("  wrote results/stock_dd_macros.tex")

    print(f"\n  Cisco: {abs(cdecl)*100:.0f}% peak {cp.date()} trough {ct.date()} "
          f"recovered {crec.date() if crec is not None else 'NOT YET'} "
          f"({yrs(cp, crec, last):.1f} yr); total-return recovered {cisco_tr_year}")
    print(f"  recent seven declines: {min(rec_decls):.0f}%..{max(rec_decls):.0f}%")
    print(f"  Nasdaq dot-com: -{ndx_decl:.0f}% peak {ip.date()} trough {it.date()} "
          f"recovered {irec_date.date() if irec_date is not None else 'NOT YET'} "
          f"({ndx_recover_yrs:.1f} yr)")
    print(f"  Nasdaq-100 share now {ndx_share_now:.0f}% -> erase {erase_now:.0f}% "
          f"(2000 share {CONC['ndx_share_2000']:.0f}% -> {erase_2000:.0f}%)")
    print(f"  Japan Nikkei: -{jp_decl:.0f}% peak {jp.date()} trough {jt.date()} "
          f"recovered {jp_rec.date() if jp_rec is not None else 'NOT YET'} ({jp_recover_yrs:.1f} yr)")


if __name__ == "__main__":
    main()
