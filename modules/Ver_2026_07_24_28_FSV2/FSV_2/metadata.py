"""
metadata.py

Financial State Vector (FSV) Raw Feature Metadata

This file contains ONLY raw external feature definitions.

No:
    - forward filling
    - interpolation
    - normalization
    - derived features

The availability engine will later use:
    - observation_time_definition
    - publish_time
    - release_delay
    - timezone

to calculate true information availability.

All timestamps will eventually be converted to UTC.
"""


FEATURES = {


# ==========================================================
# Interest Rates
# ==========================================================


"fed_funds": {

    "source": "FRED",
    "series": "DFF",

    "category": "rate",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "08:00",

    "release_delay": {
        "type": "next_business_day"
    },

    "publication_lag_days": 1,

    "observation_time_definition":
        "daily_effective_rate",

    "revision": True,

    "forward_fill": False,

    "description":
        "Effective Federal Funds Rate",
},



"1m_treasury": {

    "source": "FRED",
    "series": "DGS1MO",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "1-Month Treasury Constant Maturity Rate",
},



"3m_treasury": {

    "source": "FRED",
    "series": "DGS3MO",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "3-Month Treasury Constant Maturity Rate",
},



"6m_treasury": {

    "source": "FRED",
    "series": "DGS6MO",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "6-Month Treasury Constant Maturity Rate",
},



"1y_treasury": {

    "source": "FRED",
    "series": "DGS1",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "1-Year Treasury Constant Maturity Rate",
},



"2y_treasury": {

    "source": "FRED",
    "series": "DGS2",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "2-Year Treasury Constant Maturity Rate",
},



"3y_treasury": {

    "source": "FRED",
    "series": "DGS3",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "3-Year Treasury Constant Maturity Rate",
},



"5y_treasury": {

    "source": "FRED",
    "series": "DGS5",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "5-Year Treasury Constant Maturity Rate",
},



"7y_treasury": {

    "source": "FRED",
    "series": "DGS7",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "7-Year Treasury Constant Maturity Rate",
},



"10y_treasury": {

    "source": "FRED",
    "series": "DGS10",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "10-Year Treasury Constant Maturity Rate",
},



"20y_treasury": {

    "source": "FRED",
    "series": "DGS20",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "20-Year Treasury Constant Maturity Rate",
},



"30y_treasury": {

    "source": "FRED",
    "series": "DGS30",

    "category": "treasury",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "30-Year Treasury Constant Maturity Rate",
},


"tips10y": {

    "source": "FRED",
    "series": "DFII10",

    "category": "real_rate",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "10-Year Treasury Inflation Indexed Real Yield",
},



"aaa": {

    "source": "FRED",
    "series": "AAA",

    "category": "corporate_rate",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_average",

    "revision": False,

    "forward_fill": False,

    "description":
        "Moody's Seasoned AAA Corporate Bond Yield",
},



"baa": {

    "source": "FRED",
    "series": "BAA",

    "category": "corporate_rate",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_average",

    "revision": False,

    "forward_fill": False,

    "description":
        "Moody's Seasoned BAA Corporate Bond Yield",
},



"sofr": {

    "source": "FRED",
    "series": "SOFR",

    "category": "funding",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "08:00",

    "release_delay": {
        "type": "next_business_day"
    },

    "publication_lag_days": 1,

    "observation_time_definition":
        "overnight_rate",

    "revision": True,

    "forward_fill": False,

    "description":
        "Secured Overnight Financing Rate",
},



# ==========================================================
# Credit Spreads
# ==========================================================


"aaa_oas": {

    "source": "FRED",
    "series": "BAMLC0A1CAAA",

    "category": "credit",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "ICE BofA AAA Option Adjusted Spread",
},



"aa_oas": {

    "source": "FRED",
    "series": "BAMLC0A2CAA",

    "category": "credit",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "ICE BofA AA Option Adjusted Spread",
},



"a_oas": {

    "source": "FRED",
    "series": "BAMLC0A3CA",

    "category": "credit",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "ICE BofA A Option Adjusted Spread",
},



"bbb_oas": {

    "source": "FRED",
    "series": "BAMLC0A4CBBB",

    "category": "credit",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "ICE BofA BBB Option Adjusted Spread",
},



"hy_oas": {

    "source": "FRED",
    "series": "BAMLH0A0HYM2",

    "category": "credit",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "ICE BofA High Yield Option Adjusted Spread",
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

    "timezone": "America/New_York",

    "publish_time": "16:30",

    "release_delay": {
        "type": "weekly"
    },

    "publication_lag_days": 1,

    "observation_time_definition":
        "weekly_balance_sheet",

    "revision": True,

    "forward_fill": False,

    "description":
        "Federal Reserve Total Assets",
},



"bank_reserves": {

    "source": "FRED",
    "series": "TOTRESNS",

    "category": "liquidity",

    "frequency": "monthly",

    "units": "million_usd",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "monthly"
    },

    "publication_lag_days": 30,

    "observation_time_definition":
        "monthly_balance",

    "revision": True,

    "forward_fill": False,

    "description":
        "Depository Institution Reserves",
},



"m2": {

    "source": "FRED",
    "series": "M2SL",

    "category": "money",

    "frequency": "monthly",

    "units": "billion_usd",

    "timezone": "America/New_York",

    "publish_time": "16:30",

    "release_delay": {
        "type": "monthly"
    },

    "publication_lag_days": 30,

    "observation_time_definition":
        "monthly_money_stock",

    "revision": True,

    "forward_fill": False,

    "description":
        "M2 Money Stock",
},



"dollar_index": {

    "source": "FRED",
    "series": "DTWEXBGS",

    "category": "currency",

    "frequency": "daily",

    "units": "index",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "Trade Weighted US Dollar Index",
},



"ted_spread": {

    "source": "FRED",
    "series": "TEDRATE",

    "category": "funding",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_average",

    "revision": False,

    "forward_fill": False,

    "description":
        "TED Spread",
},



"commercial_paper_3m": {

    "source": "FRED",
    "series": "CPF3M",

    "category": "funding",

    "frequency": "daily",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_average",

    "revision": False,

    "forward_fill": False,

    "description":
        "3-Month Commercial Paper Rate",
},

# ==========================================================
# Equity / Volatility
# ==========================================================


"sp500": {

    "source": "FRED",
    "series": "SP500",

    "category": "equity",

    "frequency": "daily",

    "units": "index",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "S&P500 Index",
},



"vix": {

    "source": "FRED",
    "series": "VIXCLS",

    "category": "volatility",

    "frequency": "daily",

    "units": "index",

    "timezone": "America/New_York",

    "publish_time": "16:15",

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "market_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "CBOE Volatility Index",
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

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_spot_price",

    "revision": False,

    "forward_fill": False,

    "description":
        "WTI Spot Crude Oil Price",
},



"brent_oil": {

    "source": "FRED",
    "series": "DCOILBRENTEU",

    "category": "commodity",

    "frequency": "daily",

    "units": "usd_per_barrel",

    "timezone": "America/New_York",

    "publish_time": None,

    "release_delay": {
        "type": "same_day"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_spot_price",

    "revision": False,

    "forward_fill": False,

    "description":
        "Brent Spot Crude Oil Price",
},



"gold_spot": {

    "source": "TwelveData",

    "symbol": "XAU/USD",

    "category": "commodity",

    "frequency": "daily",

    "units": "usd_per_oz",

    "timezone": "UTC",

    "publish_time": None,

    "release_delay": {
        "type": "continuous_market"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "daily_close",

    "revision": False,

    "forward_fill": False,

    "description":
        "Gold Spot Price",
},



# ==========================================================
# Housing
# ==========================================================


"case_shiller": {

    "source": "FRED",
    "series": "CSUSHPINSA",

    "category": "housing",

    "frequency": "monthly",

    "units": "index",

    "timezone": "America/New_York",

    "publish_time": "09:00",

    "release_delay": {
        "type": "monthly",
        "value": 2,
        "unit": "month"
    },

    "publication_lag_days": 60,

    "observation_time_definition":
        "monthly_home_price_index",

    "revision": True,

    "forward_fill": False,

    "description":
        "Case-Shiller Home Price Index",
},



"fhfa_price": {

    "source": "FRED",
    "series": "USSTHPI",

    "category": "housing",

    "frequency": "quarterly",

    "units": "index",

    "timezone": "America/New_York",

    "publish_time": "09:00",

    "release_delay": {
        "type": "quarterly"
    },

    "publication_lag_days": 60,

    "observation_time_definition":
        "quarterly_house_price_index",

    "revision": True,

    "forward_fill": False,

    "description":
        "FHFA House Price Index",
},



"housing_starts": {

    "source": "FRED",
    "series": "HOUST",

    "category": "housing",

    "frequency": "monthly",

    "units": "thousand_units",

    "timezone": "America/New_York",

    "publish_time": "08:30",

    "release_delay": {
        "type": "monthly",
        "value": 3,
        "unit": "week"
    },

    "publication_lag_days": 21,

    "observation_time_definition":
        "monthly_construction_activity",

    "revision": True,

    "forward_fill": False,

    "description":
        "Housing Starts",
},



"building_permits": {

    "source": "FRED",
    "series": "PERMIT",

    "category": "housing",

    "frequency": "monthly",

    "units": "thousand_units",

    "timezone": "America/New_York",

    "publish_time": "08:30",

    "release_delay": {
        "type": "monthly",
        "value": 3,
        "unit": "week"
    },

    "publication_lag_days": 21,

    "observation_time_definition":
        "monthly_building_authorizations",

    "revision": True,

    "forward_fill": False,

    "description":
        "Building Permits",
},



"mortgage_30y": {

    "source": "FRED",
    "series": "MORTGAGE30US",

    "category": "housing",

    "frequency": "weekly",

    "units": "percent",

    "timezone": "America/New_York",

    "publish_time": "12:00",

    "release_delay": {
        "type": "weekly"
    },

    "publication_lag_days": 0,

    "observation_time_definition":
        "weekly_mortgage_survey",

    "revision": False,

    "forward_fill": False,

    "description":
        "30-Year Fixed Mortgage Rate",
},



}