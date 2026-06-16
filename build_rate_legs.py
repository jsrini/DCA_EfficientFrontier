"""Build the constructed-rate Treasury legs (LONGT and CASH) ONCE and save them to CSV.

Both legs share one construction (this is the generalization of the earlier long-Treasury-only
builder): a constant-maturity Treasury total return spliced, as a non-taxable continuation (a
"tax-free exchange", like every other proxy splice here), into the real fund once it exists.

  LONGT  30-yr long Treasury.  Pre-fund segment from ^TYX (30-yr CMT yield, Yahoo, back to 1977),
         spliced into real VUSTX from its 1986-05-19 inception.
  CASH   short Treasury / T-bills.  Pre-fund segment from ^IRX (13-week T-bill rate, Yahoo, back to
         1960), spliced into real VFISX from its 1991-10-28 inception.

Pre-fund construction (per leg): each day hold a constant-maturity par Treasury struck at the prior
day's yield and reprice it at today's yield, so effective duration self-adjusts to the rate level of
the era (this matters in the high-rate early 1980s). Coupon accrual is the taxable income component,
price change is capital, and the matching fund's expense ratio is applied as a daily NAV drag -- the
same modelling choice used for gold (LBMA spot + GLD fee). A 13-week bill is carry-dominated
(near-zero duration), so CASH essentially tracks the rolled T-bill rate net of the fund fee.

Post-inception: the real fund's adjusted close (total-return NAV, fee already net) with its own
dividends as taxable income.

Output: longt_series.csv and cash_series.csv. Each has an adjusted total-return level (the role name)
and a per-day income yield (taxed annually by the engines) -- the same shape every other leg uses.
The series are built once and reused; they are NOT rebuilt at run time.

Sources are Yahoo only (FRED is blocked from the build host); ^TYX/^IRX are the same constant-maturity
series FRED publishes (DGS30 / TB3MS).
"""
import json, urllib.request, numpy as np, pandas as pd

START = "1980-01-15"
END   = "2026-06-01"

# role -> (rate ticker, constant maturity in years, splice fund, fund expense ratio applied as NAV drag)
LEGS = {
    "LONGT": dict(rate="^TYX", maturity=30.0,  fund="VUSTX", er=0.0020, out="longt_series.csv"),
    "CASH":  dict(rate="^IRX", maturity=0.25,  fund="VFISX", er=0.0020, out="cash_series.csv"),
}


def fetch(t):
    """Yahoo adj-close + per-ex-date dividend yield, same idiom as dca_build80.py."""
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


def par_price(y, c, m):
    """Price per 1 face of a semiannual coupon bond: coupon c, yield y (annual decimals), maturity m yrs.
    At c == y this is exactly 1 (par). For a short bill (small m) this is carry-dominated."""
    n = max(1, int(round(m * 2)))
    i = y / 2.0
    if abs(i) < 1e-12:
        return 1.0
    disc = (1.0 + i) ** (-n)
    ann = (1.0 - disc) / i
    return (c / 2.0) * ann + disc


def build_stub(yld, maturity, er):
    """Daily total-return level + per-day income yield for a rolled constant-maturity par Treasury,
    net of the fund expense ratio. yld: pd.Series of the yield in decimal, daily."""
    d = yld.sort_index()
    idx = d.index
    yv = d.values
    dt = np.diff(idx.view("int64")) / (365.0 * 24 * 3600 * 1e9)  # year fractions between obs
    lvl = np.empty(len(idx)); lvl[0] = 1.0
    inc = np.zeros(len(idx))
    for k in range(1, len(idx)):
        c0 = yv[k - 1]                                   # coupon of the par bond held = prior yield
        price = par_price(yv[k], c0, maturity - dt[k - 1])
        coupon = c0 * dt[k - 1]
        gross = price + coupon - 1.0                     # total return vs yesterday's par (=1)
        net = (1.0 + gross) * (1.0 - er * dt[k - 1]) - 1.0
        lvl[k] = lvl[k - 1] * (1.0 + net)
        inc[k] = coupon
    return pd.Series(lvl, index=idx), pd.Series(inc, index=idx)


def build_leg(role, cfg):
    print(f"\n=== {role}: {cfg['rate']} stub -> splice into {cfg['fund']} ===")
    rate_raw, _ = fetch(cfg["rate"])
    fund_ad, fund_yl = fetch(cfg["fund"])

    # rate tickers are quoted in percent (e.g. 13.5 = 13.5%); convert to decimal, auto-detecting a x10 quote.
    med = float(rate_raw.median())
    scale = 1000.0 if med > 25 else 100.0
    rate = (rate_raw / scale).loc[START:END]
    print(f"  {cfg['rate']} raw median {med:.2f} -> /{scale:.0f}; "
          f"yield range {rate.min()*100:.2f}%..{rate.max()*100:.2f}%")

    splice_dt = fund_ad.index[0]
    stub_yld = rate[rate.index < splice_dt]
    lvl, inc = build_stub(stub_yld, cfg["maturity"], cfg["er"])

    fund = fund_ad.loc[splice_dt:]
    sc = lvl.iloc[-1] / fund.iloc[0]                     # continue the constructed level (no realized gain)
    fund_lvl = fund * sc
    fund_inc = fund_yl.reindex(fund.index).fillna(0.0)

    adj = pd.concat([lvl, fund_lvl]).sort_index()
    adj = adj[~adj.index.duplicated(keep="last")]
    yld = pd.concat([inc, fund_inc]).sort_index()
    yld = yld[~yld.index.duplicated(keep="last")].reindex(adj.index).fillna(0.0)

    out = pd.DataFrame({role: adj, f"{role}_yield": yld}).loc[START:END]
    out.to_csv(cfg["out"])

    # ---- validation prints (checkable) ----
    yrs = (lvl.index[-1] - lvl.index[0]).days / 365.25
    f10 = fund_lvl.loc[:fund_lvl.index[0] + pd.DateOffset(years=10)]
    fyrs = (f10.index[-1] - f10.index[0]).days / 365.25
    sp_ret = out[role].pct_change().loc[splice_dt]
    print(f"  built {out.index[0].date()}..{out.index[-1].date()} {len(out)} days; splice {splice_dt.date()} ({cfg['fund']})")
    print(f"  stub total-return CAGR: {(lvl.iloc[-1]/lvl.iloc[0])**(1/yrs)-1:.2%}; "
          f"stub avg income yield: {inc.sum()/yrs*100:.2f}%/yr")
    print(f"  real {cfg['fund']} first-decade CAGR (net of its fee): {(f10.iloc[-1]/f10.iloc[0])**(1/fyrs)-1:.2%}")
    print(f"  splice-day 1-step move: {sp_ret*100:.3f}% (≈0 = clean continuation); "
          f"ER drag {cfg['er']*100:.2f}%/yr; NaN? {out.isna().any().any()}")
    print(f"  wrote {cfg['out']}")


def main():
    for role, cfg in LEGS.items():
        build_leg(role, cfg)


if __name__ == "__main__":
    main()
