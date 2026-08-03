FEATURES = {


# ==========================================================
# Treasury rates
# ==========================================================

"fed_funds": {
    "source": "FRED",
    "series": "DFF",
    "category": "rate",
    "frequency": "daily",
    "units": "percent",
    "timezone": "America/New_York",
    "release_lag_minutes": None,
    "fill_policy": "none",
},


"1y_treasury": {
    "source": "FRED",
    "series": "DGS1",
    "category": "treasury",
    "frequency": "daily",
    "units": "percent",
    "timezone": "America/New_York",
},


"2y_treasury": {
    "source": "FRED",
    "series": "DGS2",
    "category": "treasury",
    "frequency": "daily",
    "units": "percent",
},


"5y_treasury": {
    "source": "FRED",
    "series": "DGS5",
    "category": "treasury",
    "frequency": "daily",
    "units": "percent",
},


"10y_treasury": {
    "source": "FRED",
    "series": "DGS10",
    "category": "treasury",
    "frequency": "daily",
    "units": "percent",
},


"30y_treasury": {
    "source": "FRED",
    "series": "DGS30",
    "category": "treasury",
    "frequency": "daily",
    "units": "percent",
},



# ==========================================================
# Real rates
# ==========================================================

"tips10y": {
    "source": "FRED",
    "series": "DFII10",
    "category": "real_rate",
    "frequency": "daily",
    "units": "percent",
},



# ==========================================================
# Corporate credit
# ==========================================================

"aaa_oas": {
    "source": "FRED",
    "series": "BAMLC0A1CAAA",
    "category": "credit",
    "frequency": "daily",
    "units": "percent",
},


"aa_oas": {
    "source": "FRED",
    "series": "BAMLC0A2CAA",
    "category": "credit",
    "frequency": "daily",
    "units": "percent",
},


"a_oas": {
    "source": "FRED",
    "series": "BAMLC0A3CA",
    "category": "credit",
    "frequency": "daily",
    "units": "percent",
},


"bbb_oas": {
    "source": "FRED",
    "series": "BAMLC0A4CBBB",
    "category": "credit",
    "frequency": "daily",
    "units": "percent",
},


"hy_oas": {
    "source": "FRED",
    "series": "BAMLH0A0HYM2",
    "category": "credit",
    "frequency": "daily",
    "units": "percent",
},



# ==========================================================
# Funding
# ==========================================================

"sofr": {
    "source": "FRED",
    "series": "SOFR",
    "category": "funding",
    "frequency": "daily",
    "units": "percent",
},


"ted_spread": {
    "source": "FRED",
    "series": "TEDRATE",
    "category": "funding",
    "frequency": "daily",
    "units": "percent",
},


"commercial_paper_3m": {
    "source": "FRED",
    "series": "CPF3M",
    "category": "funding",
    "frequency": "daily",
    "units": "percent",
},



# ==========================================================
# Liquidity
# ==========================================================

"fed_assets": {
    "source": "FRED",
    "series": "WALCL",
    "category": "liquidity",
    "frequency": "weekly",
    "units": "million_usd",
},


"bank_reserves": {
    "source": "FRED",
    "series": "TOTRESNS",
    "category": "liquidity",
    "frequency": "monthly",
    "units": "million_usd",
},


"m2": {
    "source": "FRED",
    "series": "M2SL",
    "category": "money",
    "frequency": "monthly",
    "units": "billion_usd",
},


"dollar_index": {
    "source": "FRED",
    "series": "DTWEXBGS",
    "category": "currency",
    "frequency": "daily",
    "units": "index",
},



# ==========================================================
# Market
# ==========================================================

"sp500": {
    "source": "FRED",
    "series": "SP500",
    "category": "equity",
    "frequency": "daily",
    "units": "index",
},


"vix": {
    "source": "FRED",
    "series": "VIXCLS",
    "category": "volatility",
    "frequency": "daily",
    "units": "index",
},



# ==========================================================
# Commodities
# ==========================================================

"wti_oil": {
    "source": "FRED",
    "series": "DCOILWTICO",
    "category": "commodity",
    "frequency": "daily",
    "units": "usd_per_barrel",
},


"brent_oil": {
    "source": "FRED",
    "series": "DCOILBRENTEU",
    "category": "commodity",
    "frequency": "daily",
    "units": "usd_per_barrel",
},


"gold_spot": {
    "source": "TwelveData",
    "symbol": "XAU/USD",
    "category": "commodity",
    "frequency": "daily",
    "units": "usd_per_oz",
},



# ==========================================================
# Real estate
# ==========================================================

"case_shiller": {
    "source": "FRED",
    "series": "CSUSHPINSA",
    "category": "housing",
    "frequency": "monthly",
    "units": "index",
},


"fhfa_price": {
    "source": "FRED",
    "series": "USSTHPI",
    "category": "housing",
    "frequency": "quarterly",
    "units": "index",
},


"housing_starts": {
    "source": "FRED",
    "series": "HOUST",
    "category": "housing",
    "frequency": "monthly",
    "units": "thousands",
},


"building_permits": {
    "source": "FRED",
    "series": "PERMIT",
    "category": "housing",
    "frequency": "monthly",
    "units": "thousands",
},


"mortgage_30y": {
    "source": "FRED",
    "series": "MORTGAGE30US",
    "category": "housing",
    "frequency": "weekly",
    "units": "percent",
},

}