"""Build the REIT leg once and save it to reit_series.csv.

Same shape and splice philosophy as every other leg (adjusted total-return level + a per-day income
yield; a non-taxable continuation splice), but a different construction because there is no daily
REIT source back to 1980:

  1980-01-15 .. FRESX inception (1986-11-14)
        from the FTSE Nareit ALL EQUITY REITs monthly TOTAL-RETURN index (reit_monthly.csv,
        history to 1971 -- the standard property-REIT benchmark). Each month's total return is
        distributed geometrically across that month's trading days (off the 80full calendar), and
        the month's income-return component is split out per day as the taxable yield. A REIT
        index-fund expense ratio is applied as a daily NAV drag (configurable), the same modelling
        choice used for gold and the constructed Treasury legs.
        CAVEAT: the source is MONTHLY, so the daily path in this stub is interpolated -- it does not
        carry real day-to-day REIT volatility (it smooths within each month). No daily REIT series
        exists before FRESX (1986), so this is unavoidable; it is the one leg whose pre-fund daily
        path is synthetic at the daily frequency, and it understates short-horizon REIT volatility
        and drawdowns in 1980-1986. The other legs all had genuine daily sources.

  1986-11-14 .. present
        the real daily FRESX -> VGSIX adjusted-close chain (the same proxy splice the report's 1995
        REIT leg already uses), total-return NAVs with the funds' fees already net, and their own
        dividends as the taxable income.

REIT income is taxed at the equity rate (25%), matching ROLE_RATE["REIT"].

Sources: reit_monthly.csv (Nareit monthly history, downloaded manually -- FRED/Nareit are blocked
from the build host) + Yahoo for FRESX/VGSIX.
  reit_monthly.csv from: https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls
"""
import json, urllib.request, numpy as np, pandas as pd

START   = "1980-01-15"
END     = "2026-06-01"
REIT_ER = 0.0070          # REIT index-fund expense ratio applied as NAV drag on the pre-fund stub (configurable; ~FRESX)
NAREIT  = "reit_monthly.csv"
CAL     = "dca_adj_div_80full.csv"   # trading-day calendar to interpolate the monthly index onto
SPLICE  = ["FRESX", "VGSIX"]         # daily fund chain spliced after the stub (oldest first)


def fetch(t):
    """Yahoo adj-close + per-ex-date dividend yield, same idiom as the other builders."""
    url = (f"https://query1.finance.yahoo.com/v8/finance/chart/{t}"
           "?period1=-630000000&period2=9999999999&interval=1d&events=div,splits")
    j = json.load(urllib.request.urlopen(
        urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60))
    r = j["chart"]["result"][0]
    ts = pd.to_datetime(r["timestamp"], unit="s").normalize()
    cl = pd.Series(r["indicators"]["quote"][0]["close"], index=ts)
    ad = pd.Series(r["indicators"]["adjclose"][0]["adjclose"], index=ts)
    cl = cl[~cl.index.duplicated(keep="last")].dropna()
    ad = ad[~ad.index.duplicated(keep="last")].dropna()
    yl = pd.Series(0.0, index=cl.index)
    for v in r.get("events", {}).get("dividends", {}).values():
        d = pd.to_datetime(int(v["date"]), unit="s").normalize()
        k = cl.index[cl.index <= d]
        if len(k):
            yl[k[-1]] += v["amount"] / cl.loc[k[-1]]
    return ad, yl


def splice_funds(tickers):
    """Splice the daily fund chain (oldest first) into one adj-close + yield series (continuation)."""
    data = [fetch(t) for t in tickers]
    ad, yl = data[0]
    for e_ad, e_yl in data[1:]:
        j = e_ad.index[0]; sc = ad.asof(j) / e_ad.iloc[0] if e_ad.iloc[0] else 1.0
        ad = pd.concat([ad[ad.index < j], e_ad * sc]).sort_index()
        yl = pd.concat([yl[yl.index < j], e_yl]).sort_index()
    ad = ad[~ad.index.duplicated(keep="last")]; yl = yl[~yl.index.duplicated(keep="last")]
    return ad, yl


def parse_nareit():
    """All Equity REITs monthly total return (col 22) and income return (col 26), as decimals."""
    raw = pd.read_csv(NAREIT, skiprows=8, header=None, dtype=str)
    dt = pd.to_datetime(raw[0].str.strip(), format="%b-%y")
    def num(c): return pd.to_numeric(raw[c].str.replace(",", "", regex=False), errors="coerce") / 100.0
    df = pd.DataFrame({"tr": num(22).values, "inc": num(26).values}, index=dt).dropna()
    return df.sort_index()


def build_stub(monthly, cal, er):
    """Distribute each month's total return / income across that month's trading days (geometric),
    net of the fund expense ratio. Returns (level, income-yield) on the calendar days."""
    days = cal[(cal >= pd.Timestamp(START))]
    lvl = []; inc = []; idx = []
    cur = 1.0
    ym = pd.Series(monthly.index).dt.to_period("M")
    mmap = {p: (monthly.iloc[i]["tr"], monthly.iloc[i]["inc"]) for i, p in enumerate(ym)}
    by_month = {}
    for d in days:
        by_month.setdefault(d.to_period("M"), []).append(d)
    for p, dlist in by_month.items():
        if p not in mmap:        # months past the last index value: stop the stub here
            continue
        tr, inc_m = mmap[p]
        k = len(dlist)
        f_tot = (1.0 + tr) ** (1.0 / k) - 1.0           # geometric daily share of the month TR
        f_tot = (1.0 + f_tot) * (1.0 - er / 252.0) - 1.0  # apply expense-ratio NAV drag
        d_inc = inc_m / k                                # even daily split of the month income
        for d in dlist:
            cur *= (1.0 + f_tot)
            lvl.append(cur); inc.append(d_inc); idx.append(d)
    return pd.Series(lvl, index=pd.DatetimeIndex(idx)), pd.Series(inc, index=pd.DatetimeIndex(idx))


def main():
    cal = pd.read_csv(CAL, index_col=0, parse_dates=True).index
    monthly = parse_nareit()
    print(f"Nareit All-Equity monthly: {monthly.index[0].date()}..{monthly.index[-1].date()} "
          f"({len(monthly)} months); avg div-income {monthly['inc'].mean()*12*100:.1f}%/yr")

    fund_ad, fund_yl = splice_funds(SPLICE)
    splice_dt = fund_ad.index[0]
    print(f"fund chain {'+'.join(SPLICE)}: {splice_dt.date()}..{fund_ad.index[-1].date()}")

    cal_stub = cal[cal < splice_dt]
    lvl, inc = build_stub(monthly, cal_stub, REIT_ER)

    fund = fund_ad.loc[splice_dt:]
    sc = lvl.iloc[-1] / fund.iloc[0]
    fund_lvl = fund * sc
    fund_inc = fund_yl.reindex(fund.index).fillna(0.0)

    adj = pd.concat([lvl, fund_lvl]).sort_index(); adj = adj[~adj.index.duplicated(keep="last")]
    yld = pd.concat([inc, fund_inc]).sort_index(); yld = yld[~yld.index.duplicated(keep="last")]
    yld = yld.reindex(adj.index).fillna(0.0)

    out = pd.DataFrame({"REIT": adj, "REIT_yield": yld}).loc[START:END]
    out.to_csv("reit_series.csv")

    # ---- validation ----
    yrs = (lvl.index[-1] - lvl.index[0]).days / 365.25
    f10 = fund_lvl.loc[:fund_lvl.index[0] + pd.DateOffset(years=10)]
    fyrs = (f10.index[-1] - f10.index[0]).days / 365.25
    sp = out["REIT"].pct_change().loc[splice_dt]
    print(f"\nbuilt {out.index[0].date()}..{out.index[-1].date()} {len(out)} days; splice {splice_dt.date()} (FRESX)")
    print(f"  stub 1980->1986 total-return CAGR: {(lvl.iloc[-1]/lvl.iloc[0])**(1/yrs)-1:.2%}")
    print(f"  real FRESX first-decade CAGR (net of fee): {(f10.iloc[-1]/f10.iloc[0])**(1/fyrs)-1:.2%}")
    print(f"  splice-day 1-step move: {sp*100:.3f}% (≈0 = clean); ER drag {REIT_ER*100:.2f}%/yr; NaN? {out.isna().any().any()}")
    print(f"  CAVEAT: 1980-1986 daily path interpolated from monthly (smooths within-month vol)")
    print(f"  wrote reit_series.csv")


if __name__ == "__main__":
    main()
