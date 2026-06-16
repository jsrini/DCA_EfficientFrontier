import json,urllib.request,numpy as np,pandas as pd
from functools import reduce
def fetch(t):
    url=f'https://query1.finance.yahoo.com/v8/finance/chart/{t}?period1=-630000000&period2=9999999999&interval=1d&events=div,splits'
    j=json.load(urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60))
    r=j['chart']['result'][0]; ts=pd.to_datetime(r['timestamp'],unit='s').normalize()
    cl=pd.Series(r['indicators']['quote'][0]['close'],index=ts); ad=pd.Series(r['indicators']['adjclose'][0]['adjclose'],index=ts)
    cl=cl[~cl.index.duplicated(keep='last')].dropna(); ad=ad[~ad.index.duplicated(keep='last')].dropna()
    yl=pd.Series(0.0,index=cl.index)
    for v in r.get('events',{}).get('dividends',{}).values():
        d=pd.to_datetime(int(v['date']),unit='s').normalize(); k=cl.index[cl.index<=d]
        if len(k): yl[k[-1]]+=v['amount']/cl.loc[k[-1]]
    return ad,yl
def sadj(e,l):
    j=l.index[0]; sc=l.iloc[0]/e.asof(j); out=pd.concat([e[e.index<j]*sc,l]).sort_index(); return out[~out.index.duplicated(keep='last')]
def syld(e,l):
    j=l.index[0]; out=pd.concat([e[e.index<j],l]).sort_index(); return out[~out.index.duplicated(keep='last')]
def leg(tickers):
    data=[fetch(t) for t in tickers]
    ad=reduce(sadj,[d[0] for d in data]); yl=reduce(syld,[d[1] for d in data]); return ad,yl
def build_base():
    """Fetch the five fund-based legs (STOCK/TECH/INTL/GOLD/BOND) and write the 1980 base files."""
    print('fetching legs...')
    S_a,S_y=leg(['VFINX'])
    T_a,T_y=leg(['VFINX','RYOCX','QQQ'])
    I_a,I_y=leg(['SCINX','PRITX','VTRIX','VGTSX'])
    B_a,B_y=leg(['FGOVX','VBMFX'])
    g=json.load(urllib.request.urlopen(urllib.request.Request('https://prices.lbma.org.uk/json/gold_pm.json',headers={'User-Agent':'Mozilla/5.0'}),timeout=40))
    G_a=pd.Series({pd.Timestamp(r['d']):r['v'][0] for r in g if r['v'] and r['v'][0]}).sort_index()
    adj=pd.DataFrame({'STOCK':S_a,'TECH':T_a,'INTL':I_a,'GOLD':G_a,'BOND':B_a}).sort_index().ffill().loc['1980-01-15':'2026-06-01'].dropna()
    idx=adj.index
    yl=pd.DataFrame({'STOCK':S_y,'TECH':T_y,'INTL':I_y,'GOLD':pd.Series(0.0,index=G_a.index),'BOND':B_y}).reindex(idx).fillna(0.0)[['STOCK','TECH','INTL','GOLD','BOND']]
    adj.to_csv('dca_adj_div_80full.csv'); yl.to_csv('dca_yield_div_80full.csv')
    print('built',idx[0].date(),'..',idx[-1].date(),len(idx),'days')
    print('INTL switch check -- 1981 intl yld %.2f%%; TECH 1985 == STOCK 1985 returns?'%(yl['INTL']['1981'].sum()*100))
    print('  TECH vs STOCK corr 1985-1993 (should be ~1, US proxy):',round(adj['TECH'].pct_change()['1985':'1993'].corr(adj['STOCK'].pct_change()['1985':'1993']),3))
    print('  TECH vs STOCK corr 2000-2010 (should diverge, QQQ):',round(adj['TECH'].pct_change()['2000':'2010'].corr(adj['STOCK'].pct_change()['2000':'2010']),3))

def merge_legs():
    """Join the constructed legs (LONGT, CASH from build_rate_legs.py; REIT from build_reit.py) into the
    base, producing the single 8-leg 1980 basis every 1980 engine reads. No re-fetch: reads the existing
    base + the leg CSVs and reindexes each leg onto the base calendar (ffill). This is the unified file
    that retires dca_adj_div_46.csv -- one derivation per asset."""
    adj=pd.read_csv('dca_adj_div_80full.csv',index_col=0,parse_dates=True)
    yl =pd.read_csv('dca_yield_div_80full.csv',index_col=0,parse_dates=True)
    for role,f in [('LONGT','longt_series.csv'),('CASH','cash_series.csv'),('REIT','reit_series.csv')]:
        s=pd.read_csv(f,index_col=0,parse_dates=True)
        adj[role]=s[role].reindex(adj.index).ffill().bfill()
        yl[role]=s['%s_yield'%role].reindex(adj.index).fillna(0.0)
    cols=['STOCK','TECH','INTL','GOLD','BOND','LONGT','CASH','REIT']
    adj=adj[cols]; yl=yl[cols]
    assert not adj.isna().any().any(),'NaN in unified adj'
    adj.to_csv('dca_adj_div_80full.csv'); yl.to_csv('dca_yield_div_80full.csv')
    print('merged -> 8-leg 1980 basis:',list(adj.columns),'|',len(adj),'rows | NaN?',bool(adj.isna().any().any()))

if __name__=='__main__':
    import sys
    if 'merge' in sys.argv: merge_legs()
    else: build_base()
