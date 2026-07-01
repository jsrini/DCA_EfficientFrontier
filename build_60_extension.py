"""Build the 1962-2026 base for the long-window appendix (Appendix H).

Extends the report's 1980-2026 base (dca_adj_div_80full.csv / dca_yield_div_80full.csv) backwards
to 1962-01-02 for the STOCK and BOND legs. Gold stays NaN pre-1975 -- the bootstrap in dca_core
handles the pre-1975 gold fixup (blocks touching pre-1975 draw their gold slice from 1975+).

Constructed pre-1980 series:
  STOCK: Yahoo ^GSPC daily close + monthly Shiller S&P dividend yield overlay + VFINX-equivalent
         expense ratio drag (0.14%/yr, daily-compounded). The 1980+ series in dca_adj_div_80full is
         VFINX, whose NAV already reflects the fund's ER, so we mirror that on the constructed side.
  BOND:  FRED DGS10 daily 10-yr Treasury yield -> daily total return via price change (from yield
         change, 10y duration approximation) + coupon accrual (yield/252) -- VBMFX-equivalent ER
         drag (0.15%/yr). The 1980+ series is VBMFX (net of ER).
  GOLD:  LBMA gold PM fix daily 1975-01-02 through 1980-01-14, spliced onto the 1980+ series. Days
         pre-1975 are filled with the 1975-01-02 value (yields 0 daily return); the bootstrap in
         dca_core detects pre-1975 gold indices and re-draws them from the 1975+ range.

The extended CSVs are dca_adj_div_60full.csv and dca_yield_div_60full.csv, in the same wide format
as the 1980+ files (columns = ASSETS; index = daily dates). The 1980+ portion is spliced in
unchanged so downstream results match the report exactly on that segment.

Run:  python3 build_60_extension.py
"""
import io, json, os, urllib.request
import numpy as np, pandas as pd, yfinance as yf

BASE = os.path.dirname(os.path.abspath(__file__))
SPLICE = "1980-01-15"                       # first day of the report's 1980+ VFINX series
START = "1962-01-02"                        # first daily DGS10 date
CPI_START = "1962-01-01"                    # CPI series backdrop; monthly since 1947 on FRED
CPI_END   = "2026-06-01"                    # match the report's data end
GOLD_START = "1975-01-01"                   # per the appendix spec: gold universe begins 1975
VFINX_ER = 0.0014                           # VFINX current expense ratio (S&P 500 index)
VBMFX_ER = 0.0015                           # VBMFX current expense ratio (total bond index)


def fetch_gspc(start, end):
    """Yahoo ^GSPC daily close price, [start, end)."""
    d = yf.download("^GSPC", start=start, end=end, progress=False, auto_adjust=False)
    close = d["Close"]["^GSPC"] if isinstance(d.columns, pd.MultiIndex) else d["Close"]
    return close.dropna().rename("SPX")


def fetch_dgs10(start, end):
    """FRED DGS10 daily 10-yr Treasury yield (percent), forward-filled through NaN days (holidays)."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS10&cosd={start}&coed={end}"
    csv = urllib.request.urlopen(url).read()
    df = pd.read_csv(io.BytesIO(csv), parse_dates=["observation_date"])
    df = df.set_index("observation_date")["DGS10"].astype(float).ffill()
    return df.rename("DGS10")


def fetch_shiller_div_yield():
    """Shiller monthly S&P dividend yield 1962-1980 = Dividend / SP500 (annualized, in decimal).
    Uses the datahub.io mirror of Shiller's ie_data spreadsheet (columns Date/SP500/Dividend)."""
    url = "https://datahub.io/core/s-and-p-500/r/data.csv"
    csv = urllib.request.urlopen(url, timeout=30).read()
    df = pd.read_csv(io.BytesIO(csv), parse_dates=["Date"]).set_index("Date")
    y = (df["Dividend"] / df["SP500"]).dropna()   # annual yield fraction, monthly cadence
    return y.rename("SP_DIV_YIELD")


def build_stock_pre1980(splice_date):
    """Daily total return series for pre-1980 stocks: ^GSPC price + monthly dividend + ER drag."""
    px = fetch_gspc(START, splice_date)
    dy = fetch_shiller_div_yield()
    # Convert Shiller's monthly annualized dividend yield to a per-day coupon-equivalent, applied to
    # every trading day (piecewise-constant within a month). ffill THEN bfill so a trading day earlier
    # than the first Shiller month (rare but happens at the very start) still gets the nearest yield.
    day_yield = dy.reindex(px.index, method="ffill").fillna(method="bfill") / 252.0
    daily_price_ret = px.pct_change().fillna(0.0)
    daily_er_drag = VFINX_ER / 252.0
    daily_tr = daily_price_ret + day_yield - daily_er_drag                 # per-day total return, net of ER
    # Reconstruct a price index at whatever starting level (only ratios matter): begin at 100.0
    tr_index = 100.0 * (1.0 + daily_tr).cumprod()
    return tr_index.rename("STOCK"), day_yield.rename("STOCK")             # (price series, per-day yield series)


def fetch_cpi(start, end):
    """FRED CPIAUCSL monthly (CPI-U, all urban consumers, all items). Returns monthly Series
    indexed by first-of-month dates. Used by the long-window appendix to deflate nominal returns
    to real (2026-dollar) returns."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPIAUCSL&cosd={start}&coed={end}"
    csv = urllib.request.urlopen(url).read()
    df = pd.read_csv(io.BytesIO(csv), parse_dates=["observation_date"])
    return df.set_index("observation_date")["CPIAUCSL"].astype(float).rename("CPI")


def fetch_lbma_gold(start, end):
    """LBMA gold PM fix daily USD/oz, [start, end). Matches dca_build80's LBMA fetch."""
    req = urllib.request.Request("https://prices.lbma.org.uk/json/gold_pm.json",
                                 headers={"User-Agent": "Mozilla/5.0"})
    g = json.load(urllib.request.urlopen(req, timeout=40))
    s = pd.Series({pd.Timestamp(r["d"]): r["v"][0] for r in g if r["v"] and r["v"][0]}).sort_index()
    return s.loc[start:end].rename("GOLD")


def build_gold_pre1980(splice_date):
    """Daily gold price 1975-01-02 through splice_date (LBMA PM fix). No ER drag: matches dca_build80,
    which passes GOLD through unadjusted (the report's Engine applies GOLD_FEE=0.4%/yr as a NAV drag
    at run time -- see dca_core.py:gold_decay -- so we do NOT double-charge here)."""
    return fetch_lbma_gold(GOLD_START, splice_date)


def build_bond_pre1980(splice_date):
    """Daily total return series for pre-1980 bonds: DGS10 -> price + coupon, minus ER drag.
    10y duration is a fixed approximation (real duration of a 10y par bond is ~8y, but 10y is the
    common convention for 10y-Treasury bond-fund proxies)."""
    y = fetch_dgs10(START, splice_date) / 100.0                             # decimal (e.g. 0.04)
    daily_yield_change = y.diff().fillna(0.0)
    D = 10.0                                                                 # duration approximation
    daily_price_ret = -D * daily_yield_change                                # price effect of yield change
    daily_coupon = y.shift(1).fillna(y.iloc[0]) / 252.0                      # coupon accrual on yesterday's yield
    daily_er_drag = VBMFX_ER / 252.0
    daily_tr = daily_price_ret + daily_coupon - daily_er_drag
    tr_index = 100.0 * (1.0 + daily_tr).cumprod()
    return tr_index.rename("BOND"), daily_coupon.rename("BOND")


def splice(pre_series, post_series):
    """Splice a pre-1980 total-return index onto a 1980+ adjusted-close series by matching levels
    at the splice date. Returns a single continuous series indexed by both periods' dates."""
    if pre_series.empty or post_series.empty:
        raise ValueError("cannot splice empty series")
    # Scale pre so its last observation equals post's first observation
    scale = post_series.iloc[0] / pre_series.iloc[-1]
    pre_scaled = pre_series * scale
    # Drop the overlap (pre's last day == post's first day if aligned)
    combined = pd.concat([pre_scaled.iloc[:-1], post_series]).sort_index()
    return combined[~combined.index.duplicated(keep="last")]


def main():
    print("Fetching pre-1980 sources...")
    stock_pre, stock_yld_pre = build_stock_pre1980(SPLICE)
    bond_pre, bond_yld_pre = build_bond_pre1980(SPLICE)
    gold_pre = build_gold_pre1980(SPLICE)                              # 1975-01-02 through splice
    print(f"  STOCK pre: {len(stock_pre)} days, {stock_pre.index[0].date()} to {stock_pre.index[-1].date()}")
    print(f"  BOND  pre: {len(bond_pre)} days, {bond_pre.index[0].date()} to {bond_pre.index[-1].date()}")
    print(f"  GOLD  pre: {len(gold_pre)} days, {gold_pre.index[0].date()} to {gold_pre.index[-1].date()}")

    print("Loading existing 1980+ series...")
    adj80 = pd.read_csv(os.path.join(BASE, "dca_adj_div_80full.csv"), index_col=0, parse_dates=True)
    yld80 = pd.read_csv(os.path.join(BASE, "dca_yield_div_80full.csv"), index_col=0, parse_dates=True)
    print(f"  1980+ file has {len(adj80)} rows, columns: {list(adj80.columns)}")

    print("Splicing STOCK, BOND, GOLD (level-matched at splice date)...")
    stock_spliced = splice(stock_pre, adj80["STOCK"])
    bond_spliced = splice(bond_pre,  adj80["BOND"])
    gold_spliced = splice(gold_pre,  adj80["GOLD"])

    # Trading-day calendar: pre-1980 uses ^GSPC's dates, post-1980 uses adj80's dates. This keeps
    # holidays consistent with the report (NYSE business days). Days where only one leg had data
    # (e.g. DGS10 rate published on a stock holiday) are excluded so we never have NaN on a trading day.
    pre_idx = stock_pre.index[stock_pre.index < pd.Timestamp(SPLICE)]
    idx = pre_idx.union(adj80.index).sort_values()
    adj = pd.DataFrame(index=idx, columns=adj80.columns, dtype=float)
    adj.loc[adj80.index] = adj80.reindex(adj.index).loc[adj80.index]
    adj["STOCK"] = stock_spliced.reindex(idx).ffill()
    adj["BOND"] = bond_spliced.reindex(idx).ffill()
    # GOLD: real prices 1975-01-02 onward; pre-1975 filled with the first observed value so daily
    # returns are 0 there. The bootstrap in dca_core detects pre-1975 gold indices and re-draws them
    # from the 1975+ range (per the appendix spec).
    gold_full = gold_spliced.reindex(idx).ffill()
    first_gold = gold_full.dropna().iloc[0]
    adj["GOLD"] = gold_full.fillna(first_gold)                           # placeholder for pre-1975
    # Other pre-1980 legs (TECH, INTL, LONGT, CASH, REIT) remain NaN -- unused by the appendix strategies.
    print(f"  Extended adj: {len(adj)} rows, {adj.index[0].date()} to {adj.index[-1].date()}")

    # Yield frame: pre-1980 STOCK/BOND from constructed sources; other legs zero pre-1980; 1980+ from yld80
    yld = pd.DataFrame(index=idx, columns=yld80.columns, dtype=float).fillna(0.0)
    yld.loc[yld80.index] = yld80.reindex(yld.index).loc[yld80.index]
    # Overwrite pre-1980 STOCK/BOND with constructed yields (post-1980 kept from yld80)
    pre_idx = idx[idx < pd.Timestamp(SPLICE)]
    yld.loc[pre_idx, "STOCK"] = stock_yld_pre.reindex(pre_idx).fillna(0.0)
    yld.loc[pre_idx, "BOND"] = bond_yld_pre.reindex(pre_idx).fillna(0.0)

    # Slim down to just the three columns the appendix strategies use. dca_core.Engine reads every
    # column of the CSV as an asset; leaving TECH/INTL/LONGT/CASH/REIT in would force role lookups
    # for legs the 1962 window doesn't configure.
    keep = ["STOCK", "BOND", "GOLD"]
    adj[keep].to_csv(os.path.join(BASE, "dca_adj_div_60full.csv"))
    yld[keep].to_csv(os.path.join(BASE, "dca_yield_div_60full.csv"))
    print(f"wrote dca_adj_div_60full.csv ({len(adj)} rows x {len(keep)} cols), dca_yield_div_60full.csv (same)")

    # CPI on the same daily calendar: fetch monthly CPI-U from FRED, forward-fill to daily so every
    # trading day has a defined price level. dca_cpi_60full.csv stores the daily CPI series that
    # longwindow_appendix.py uses to deflate nominal returns into real 2026-dollar terms.
    print("Fetching CPI (FRED CPIAUCSL, monthly)...")
    cpi_m = fetch_cpi(CPI_START, CPI_END)
    cpi = cpi_m.reindex(idx, method="ffill").rename("CPI")
    cpi.to_csv(os.path.join(BASE, "dca_cpi_60full.csv"))
    print(f"  CPI: {cpi.index[0].date()} to {cpi.index[-1].date()}, "
          f"first={cpi.iloc[0]:.2f}, last={cpi.iloc[-1]:.2f}, "
          f"CAGR {(cpi.iloc[-1]/cpi.iloc[0])**(365.25/(cpi.index[-1]-cpi.index[0]).days)-1:.2%}")
    # Quick sanity: annualized STOCK return over the full window
    ret = (adj["STOCK"].iloc[-1] / adj["STOCK"].iloc[0]) ** (365.25 / (adj.index[-1] - adj.index[0]).days) - 1
    print(f"  extended STOCK CAGR {adj.index[0].year}-{adj.index[-1].year}: {ret*100:.2f}%")
    retb = (adj["BOND"].iloc[-1] / adj["BOND"].iloc[0]) ** (365.25 / (adj.index[-1] - adj.index[0]).days) - 1
    print(f"  extended BOND  CAGR {adj.index[0].year}-{adj.index[-1].year}: {retb*100:.2f}%")


if __name__ == "__main__":
    main()
