"""
fsv_dictionary.py

Feature Dictionary with Release Metadata.

Each feature includes:
    - source: Data source (FRED, TwelveData, etc.)
    - series/symbol: Series ID or symbol
    - category: Feature category
    - units: Units of measurement
    - timezone: Source timezone
    - release_timeframe: How often data is released (minute, hourly, daily, weekly, biweekly, monthly, quarterly)
    - release_lag_days: Fixed calendar days added after release_timeframe
    - release_time: Theoretical time of day data is released (HH:MM in local time)
    - fill_policy: Forward fill policy (none, forward)
    - description: Human-readable description
    - use_vintage_data: Whether historical actual release dates are available
    - vintage_source: Source of vintage data (FRED, etc.)
"""

FEATURES = {

    # ==========================================================
    # Interest Rates
    # ==========================================================

    "fed_funds": {
        "source": "FRED",
        "series": "DFF",
        "category": "rate",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "08:00",
        "fill_policy": "none",
        "description": "Effective Federal Funds Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "1y_treasury": {
        "source": "FRED",
        "series": "DGS1",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "1-Year Treasury Constant Maturity Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "2y_treasury": {
        "source": "FRED",
        "series": "DGS2",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "2-Year Treasury Constant Maturity Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "5y_treasury": {
        "source": "FRED",
        "series": "DGS5",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "5-Year Treasury Constant Maturity Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "10y_treasury": {
        "source": "FRED",
        "series": "DGS10",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "10-Year Treasury Constant Maturity Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "30y_treasury": {
        "source": "FRED",
        "series": "DGS30",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "30-Year Treasury Constant Maturity Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==========================================================
    # Real Rates
    # ==========================================================

    "tips10y": {
        "source": "FRED",
        "series": "DFII10",
        "category": "real_rate",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "10-Year Treasury Inflation-Indexed Security",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==========================================================
    # Corporate Credit
    # ==========================================================

    "aaa_oas": {
        "source": "FRED",
        "series": "BAMLC0A1CAAA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "ICE BofA AAA Option-Adjusted Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "aa_oas": {
        "source": "FRED",
        "series": "BAMLC0A2CAA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "ICE BofA AA Option-Adjusted Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "a_oas": {
        "source": "FRED",
        "series": "BAMLC0A3CA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "ICE BofA A Option-Adjusted Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "bbb_oas": {
        "source": "FRED",
        "series": "BAMLC0A4CBBB",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "ICE BofA BBB Option-Adjusted Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "hy_oas": {
        "source": "FRED",
        "series": "BAMLH0A0HYM2",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "ICE BofA High Yield Option-Adjusted Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==========================================================
    # Funding
    # ==========================================================

    "sofr": {
        "source": "FRED",
        "series": "SOFR",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "08:00",
        "fill_policy": "none",
        "description": "Secured Overnight Financing Rate",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "ted_spread": {
        "source": "FRED",
        "series": "TEDRATE",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:00",
        "fill_policy": "none",
        "description": "TED Spread",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "commercial_paper_3m": {
        "source": "FRED",
        "series": "CPF3M",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:00",
        "fill_policy": "none",
        "description": "3-Month Commercial Paper Rate",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==========================================================
    # Liquidity
    # ==========================================================

    "fed_assets": {
        "source": "FRED",
        "series": "WALCL",
        "category": "liquidity",
        "units": "million_usd",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 1,
        "release_time": "16:30",
        "fill_policy": "forward",
        "description": "Federal Reserve Total Assets (Weekly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "bank_reserves": {
        "source": "FRED",
        "series": "TOTRESNS",
        "category": "liquidity",
        "units": "million_usd",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 30,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Depository Institution Reserves (Monthly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "m2": {
        "source": "FRED",
        "series": "M2SL",
        "category": "money",
        "units": "billion_usd",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 30,
        "release_time": "16:30",
        "fill_policy": "forward",
        "description": "M2 Money Stock (Monthly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "dollar_index": {
        "source": "FRED",
        "series": "DTWEXBGS",
        "category": "currency",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "Trade Weighted US Dollar Index",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==========================================================
    # Equity / Market
    # ==========================================================

    "sp500": {
        "source": "FRED",
        "series": "SP500",
        "category": "equity",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "S&P 500 Index",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "vix": {
        "source": "FRED",
        "series": "VIXCLS",
        "category": "volatility",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "CBOE Volatility Index (VIX)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==========================================================
    # Commodities
    # ==========================================================

    "wti_oil": {
        "source": "FRED",
        "series": "DCOILWTICO",
        "category": "commodity",
        "units": "usd_per_barrel",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "WTI Spot Crude Oil Price",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "brent_oil": {
        "source": "FRED",
        "series": "DCOILBRENTEU",
        "category": "commodity",
        "units": "usd_per_barrel",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "Brent Spot Crude Oil Price",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    "gold_spot": {
        "source": "TwelveData",
        "symbol": "XAU/USD",
        "category": "commodity",
        "units": "usd_per_oz",
        "timezone": "UTC",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "17:00",
        "fill_policy": "none",
        "description": "Gold Spot Price (24/7 Market)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==========================================================
    # Housing / Real Estate
    # ==========================================================

    "case_shiller": {
        "source": "FRED",
        "series": "CSUSHPINSA",
        "category": "housing",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 60,
        "release_time": "09:00",
        "fill_policy": "forward",
        "description": "Case-Shiller Home Price Index (Monthly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "fhfa_price": {
        "source": "FRED",
        "series": "USSTHPI",
        "category": "housing",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "quarterly",
        "release_lag_days": 60,
        "release_time": "09:00",
        "fill_policy": "forward",
        "description": "FHFA House Price Index (Quarterly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "housing_starts": {
        "source": "FRED",
        "series": "HOUST",
        "category": "housing",
        "units": "thousands",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 21,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Housing Starts (Monthly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "building_permits": {
        "source": "FRED",
        "series": "PERMIT",
        "category": "housing",
        "units": "thousands",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 21,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Building Permits (Monthly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    "mortgage_30y": {
        "source": "FRED",
        "series": "MORTGAGE30US",
        "category": "housing",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 0,
        "release_time": "12:00",
        "fill_policy": "forward",
        "description": "30-Year Fixed Mortgage Rate (Weekly)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

}