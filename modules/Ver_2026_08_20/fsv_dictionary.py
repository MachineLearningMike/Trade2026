"""
fsv_dictionary.py

Feature Dictionary with Release Metadata
========================================

Core catalog: FEATURES (versioned in this module).
Runtime catalog: get_features(config) merges config['macro_features_extra']
and applies features_to_include / features_to_exclude.

Consumed by:
    fsv_01_downloader.py     — which series to pull (source / series / symbol)
    fsv_02_find_value_intervals.py — when a value becomes public (lag + time)
    fsv_03_query_value_age.py      — decay half-life from release_timeframe

How availability is computed (fsv_02)
-------------------------------------
    release_datetime ≈ observation_date + release_lag_days  at  release_time
                       (in feature timezone; US business-day adjust unless
                        continuous_market / non-US)

    fsv_03 converts those feature-local release intervals to UTC before
    filling the minute grid that aligns with Binance (also UTC).

    next_release_datetime = release of the following observation
                            (last obs → open-ended far-future sentinel)

    If use_vintage_data and ``{feature}__first_release`` exists in raw CSV,
    that first-publication date OVERRIDES release_lag_days (clock still from
    release_time when the vintage stamp is date-only).

Field reference
---------------
    source              "FRED" | "TwelveData" | …
    series              FRED series id (FRED only)
    symbol              Market symbol (TwelveData only)
    category            Logical group (rate, treasury, credit, …)
    units               percent | index | million_usd | …
    timezone            IANA tz for release_time (usually America/New_York)
    release_timeframe   daily | weekly | monthly | quarterly | continuous_market
                        → also drives decay half-life minutes in fsv_03
    release_lag_days    Calendar days after observation/period date before
                        the value is treated as known (heuristic fallback)
    release_time        HH:MM local wall clock of official/conservative print
    fill_policy         "none" | "forward" — documentation only today;
                        interval model already holds value until next_release
    description         Human-readable note (source + schedule rationale)
    use_vintage_data    If True, fsv_01 fetches ALFRED first-release dates
    vintage_source      "FRED" when ALFRED applies

Caveats
-------
    * Times/lags are best-effort official alignments, NOT a full econ calendar
      (no "last Tuesday of month", limited holiday logic).
    * Prefer slightly LATER availability when uncertain → less lookahead risk.
    * FRED can lag the primary source by minutes–hours; some entries pad for that.
"""

FEATURES = {

    # ==================================================================
    # Interest Rates
    # ==================================================================

    # ------------------------------------------------------------------
    # fed_funds  |  FRED:DFF  |  Effective federal funds rate
    # Primary public path on FRED is Board H.15 (~16:15 ET), not NY Fed
    # EFFR (~09:00 ET). Observation date T is typically first on H.15 at T+1.
    # lag=1, time=16:15  →  available T+1 16:15 ET
    # ------------------------------------------------------------------
    "fed_funds": {
        "source": "FRED",
        "series": "DFF",                    # H.15 effective funds (daily)
        "category": "rate",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,              # T+1 vs observation date
        "release_time": "16:15",            # H.15 post time (ET)
        "fill_policy": "none",
        "description": (
            "Effective Federal Funds Rate (H.15/DFF). "
            "Public ~T+1 at 16:15 ET via H.15 (NY Fed EFFR is ~09:00 ET)."
        ),
        "use_vintage_data": True,           # ALFRED first-release preferred
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # Treasury constant-maturity yields (DGS*) — Fed H.15
    # Same-day print ~16:15 ET for the trade/observation date.
    # lag=0, time=16:15 for all tenors below.
    # Underlying quotes ~15:30 ET; H.15 is the standardized public release.
    # ------------------------------------------------------------------

    # 1y_treasury | FRED:DGS1 | 1-Year CMT
    "1y_treasury": {
        "source": "FRED",
        "series": "DGS1",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,              # same calendar day as obs date
        "release_time": "16:15",            # H.15
        "fill_policy": "none",
        "description": "1-Year Treasury Constant Maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # 2y_treasury | FRED:DGS2 | 2-Year CMT
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
        "description": "2-Year Treasury Constant Maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # 5y_treasury | FRED:DGS5 | 5-Year CMT
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
        "description": "5-Year Treasury Constant Maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # 10y_treasury | FRED:DGS10 | 10-Year CMT (common macro factor)
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
        "description": "10-Year Treasury Constant Maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # 30y_treasury | FRED:DGS30 | 30-Year CMT
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
        "description": "30-Year Treasury Constant Maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Real Rates
    # ==================================================================

    # tips10y | FRED:DFII10 | 10Y TIPS real yield (also H.15 family)
    # Same schedule as nominal CMT: lag=0, 16:15 ET
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
        "description": "10-Year TIPS constant maturity (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Corporate Credit — ICE BofA Option-Adjusted Spreads (via FRED)
    # End-of-day vendor construction; FRED ingest timing is not a hard
    # official clock. Conservative no-lookahead: T+1 09:00 ET.
    # use_vintage_data=False (no reliable ALFRED path for these).
    # ==================================================================

    # aaa_oas | FRED:BAMLC0A1CAAA | ICE BofA AAA US Corporate OAS
    "aaa_oas": {
        "source": "FRED",
        "series": "BAMLC0A1CAAA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,              # conservative vs same-day EOD
        "release_time": "09:00",
        "fill_policy": "none",
        "description": "ICE BofA AAA OAS (conservative: T+1 09:00 ET on FRED)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # aa_oas | FRED:BAMLC0A2CAA
    "aa_oas": {
        "source": "FRED",
        "series": "BAMLC0A2CAA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "09:00",
        "fill_policy": "none",
        "description": "ICE BofA AA OAS (conservative: T+1 09:00 ET on FRED)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # a_oas | FRED:BAMLC0A3CA
    "a_oas": {
        "source": "FRED",
        "series": "BAMLC0A3CA",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "09:00",
        "fill_policy": "none",
        "description": "ICE BofA A OAS (conservative: T+1 09:00 ET on FRED)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # bbb_oas | FRED:BAMLC0A4CBBB
    "bbb_oas": {
        "source": "FRED",
        "series": "BAMLC0A4CBBB",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "09:00",
        "fill_policy": "none",
        "description": "ICE BofA BBB OAS (conservative: T+1 09:00 ET on FRED)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # hy_oas | FRED:BAMLH0A0HYM2 | High yield OAS (risk-sentiment proxy)
    "hy_oas": {
        "source": "FRED",
        "series": "BAMLH0A0HYM2",
        "category": "credit",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "09:00",
        "fill_policy": "none",
        "description": "ICE BofA HY OAS (conservative: T+1 09:00 ET on FRED)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==================================================================
    # Funding / money markets
    # ==================================================================

    # ------------------------------------------------------------------
    # sofr | FRED:SOFR | Secured Overnight Financing Rate
    # NY Fed publishes rate for prior business day ~08:00 ET each morning.
    # lag=1, time=08:00
    # ------------------------------------------------------------------
    "sofr": {
        "source": "FRED",
        "series": "SOFR",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "08:00",            # NY Fed SOFR print
        "fill_policy": "none",
        "description": "SOFR (NY Fed ~08:00 ET for prior business day)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # ted_spread | FRED:TEDRATE
    # Historically LIBOR–T-bill; series is legacy / often sparse on FRED.
    # Excluded by default in config features_to_exclude — kept for completeness.
    # Conservative T+1 afternoon.
    # ------------------------------------------------------------------
    "ted_spread": {
        "source": "FRED",
        "series": "TEDRATE",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "TED Spread (legacy/sparse on FRED; conservative T+1 H.15-like)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ------------------------------------------------------------------
    # commercial_paper_3m | FRED:CPF3M | 3M AA financial CP
    # Short FRED publication lag; treat as T+1 ~16:15.
    # Often excluded in config (features_to_exclude).
    # ------------------------------------------------------------------
    "commercial_paper_3m": {
        "source": "FRED",
        "series": "CPF3M",
        "category": "funding",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "3-Month AA Financial Commercial Paper (conservative T+1 ~16:15 ET)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==================================================================
    # Liquidity / money stock / USD
    # ==================================================================

    # ------------------------------------------------------------------
    # fed_assets | FRED:WALCL | Fed total assets
    # H.4.1 weekly: Thursday ~16:30 ET, Wednesday close levels.
    # Obs date is typically Wednesday → lag_days=1, time=16:30.
    # (Weekday-of-week is NOT encoded — only lag+clock.)
    # ------------------------------------------------------------------
    "fed_assets": {
        "source": "FRED",
        "series": "WALCL",
        "category": "liquidity",
        "units": "million_usd",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 1,              # Wed level → Thu print
        "release_time": "16:30",            # H.4.1
        "fill_policy": "forward",           # hold between weekly prints
        "description": "Fed total assets WALCL (H.4.1 Thu ~16:30 ET; Wed levels)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # bank_reserves | FRED:TOTRESNS | Depository institution reserves
    # Monthly; lag varies. ~35 calendar days is a mid-range heuristic.
    # Prefer ALFRED first-release when downloaded.
    # ------------------------------------------------------------------
    "bank_reserves": {
        "source": "FRED",
        "series": "TOTRESNS",
        "category": "liquidity",
        "units": "million_usd",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 35,             # heuristic after month label date
        "release_time": "16:00",
        "fill_policy": "forward",
        "description": "Reserves of depository institutions (monthly; ~35d lag heuristic)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # m2 | FRED:M2SL | M2 money stock
    # H.6 release: often ~3–4 weeks after month-end, afternoon ET.
    # lag≈25d, time=16:30 heuristic (vintage preferred).
    # ------------------------------------------------------------------
    "m2": {
        "source": "FRED",
        "series": "M2SL",
        "category": "money",
        "units": "billion_usd",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 25,
        "release_time": "16:30",            # typical H.6 afternoon window
        "fill_policy": "forward",
        "description": "M2 money stock (H.6; ~25d lag / 16:30 ET heuristic)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # dollar_index | FRED:DTWEXBGS | Trade-weighted broad USD (goods)
    # Fed series; FRED often delayed vs trade date. Conservative T+1 10:00.
    # ------------------------------------------------------------------
    "dollar_index": {
        "source": "FRED",
        "series": "DTWEXBGS",
        "category": "currency",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "10:00",
        "fill_policy": "none",
        "description": "Trade-weighted USD broad (goods); conservative T+1 10:00 ET",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==================================================================
    # Equity / volatility
    # ==================================================================

    # ------------------------------------------------------------------
    # sp500 | FRED:SP500
    # Cash session closes 16:00 ET; FRED update often later same evening.
    # lag=0, time=18:00 pads FRED ingest without waiting until T+1.
    # ------------------------------------------------------------------
    "sp500": {
        "source": "FRED",
        "series": "SP500",
        "category": "equity",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "18:00",            # conservative same-day FRED time
        "fill_policy": "none",
        "description": "S&P 500 (cash close 16:00; FRED often ~18:00 ET same day)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ------------------------------------------------------------------
    # vix | FRED:VIXCLS | CBOE Volatility Index
    # Settlement / close around 16:00–16:15 ET window.
    # ------------------------------------------------------------------
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
        "description": "CBOE VIX close (~16:15 ET settlement window)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==================================================================
    # Commodities
    # ==================================================================

    # ------------------------------------------------------------------
    # wti_oil | FRED:DCOILWTICO | WTI spot Cushing (EIA)
    # Often public next business morning (~10:30 ET), not same-day close.
    # lag=1, time=10:30
    # ------------------------------------------------------------------
    "wti_oil": {
        "source": "FRED",
        "series": "DCOILWTICO",
        "category": "commodity",
        "units": "usd_per_barrel",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "10:30",            # EIA-style morning window
        "fill_policy": "none",
        "description": "WTI spot Cushing (EIA; conservative T+1 ~10:30 ET)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ------------------------------------------------------------------
    # brent_oil | FRED:DCOILBRENTEU | Brent spot Europe
    # Same conservative next-morning treatment as WTI.
    # ------------------------------------------------------------------
    "brent_oil": {
        "source": "FRED",
        "series": "DCOILBRENTEU",
        "category": "commodity",
        "units": "usd_per_barrel",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 1,
        "release_time": "10:30",
        "fill_policy": "none",
        "description": "Brent spot (EIA Europe; conservative T+1 ~10:30 ET)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ------------------------------------------------------------------
    # gold_spot | TwelveData:XAU/USD | Continuous spot FX-style gold
    # No official daily "print". continuous_market → fsv_02 skips US
    # business-day shifting; release_time 00:00 UTC = daily bar boundary.
    # ------------------------------------------------------------------
    "gold_spot": {
        "source": "TwelveData",
        "symbol": "XAU/USD",                # not a FRED series
        "category": "commodity",
        "units": "usd_per_oz",
        "timezone": "UTC",
        "release_timeframe": "continuous_market",
        "release_lag_days": 0,
        "release_time": "00:00",            # daily bar end (UTC)
        "fill_policy": "none",
        "description": "Gold spot XAU/USD (continuous; daily bar available at 00:00 UTC next)",
        "use_vintage_data": False,
        "vintage_source": None,
    },

    # ==================================================================
    # Housing / Real Estate
    # ==================================================================

    # ------------------------------------------------------------------
    # case_shiller | FRED:CSUSHPINSA | S&P CoreLogic Case-Shiller NSA
    # ~2-month publication lag; 09:00 ET (often last Tuesday — not encoded).
    # lag=60, time=09:00
    # ------------------------------------------------------------------
    "case_shiller": {
        "source": "FRED",
        "series": "CSUSHPINSA",
        "category": "housing",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 60,             # ~2 calendar months
        "release_time": "09:00",
        "fill_policy": "forward",
        "description": "Case-Shiller NSA national HPI (~2-month lag, 09:00 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # fhfa_price | FRED:USSTHPI | FHFA house price index
    # Quarterly; roughly ~2 months after quarter. Often excluded in config.
    # ------------------------------------------------------------------
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
        "description": "FHFA US house price index (quarterly; ~60d / 09:00 heuristic)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # housing_starts | FRED:HOUST | Census housing starts
    # New Residential Construction: usually mid-month ~08:30 ET (~16–18 days).
    # ------------------------------------------------------------------
    "housing_starts": {
        "source": "FRED",
        "series": "HOUST",
        "category": "housing",
        "units": "thousands",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 18,             # mid-month heuristic
        "release_time": "08:30",            # Census standard print
        "fill_policy": "forward",
        "description": "Housing starts (Census ~08:30 ET, ~mid-month / 18d lag)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # building_permits | FRED:PERMIT | Census building permits
    # Same Census release family / schedule as housing starts.
    # ------------------------------------------------------------------
    "building_permits": {
        "source": "FRED",
        "series": "PERMIT",
        "category": "housing",
        "units": "thousands",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 18,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Building permits (Census ~08:30 ET, ~mid-month / 18d lag)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ------------------------------------------------------------------
    # mortgage_30y | FRED:MORTGAGE30US | Freddie Mac PMMS 30Y fixed
    # Official: Thursdays 12:00 ET (week ends prior Wednesday).
    # lag=0 on the survey week date + 12:00; weekday rule not encoded.
    # ------------------------------------------------------------------
    "mortgage_30y": {
        "source": "FRED",
        "series": "MORTGAGE30US",
        "category": "housing",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 0,
        "release_time": "12:00",            # Freddie Mac PMMS noon ET
        "fill_policy": "forward",
        "description": "30Y fixed mortgage (Freddie PMMS Thu 12:00 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Short rates & curve (FRED)
    # ==================================================================

    # 3m_treasury | FRED:DGS3MO | 3-Month CMT (H.15)
    "3m_treasury": {
        "source": "FRED",
        "series": "DGS3MO",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "3-Month Treasury CMT (H.15 ~16:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # t10y2y | FRED:T10Y2Y | 10Y–2Y spread (same-day H.15 family)
    "t10y2y": {
        "source": "FRED",
        "series": "T10Y2Y",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "10Y minus 2Y Treasury yield spread (FRED T10Y2Y)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # t10y3m | FRED:T10Y3M | 10Y–3M spread
    "t10y3m": {
        "source": "FRED",
        "series": "T10Y3M",
        "category": "treasury",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "daily",
        "release_lag_days": 0,
        "release_time": "16:15",
        "fill_policy": "none",
        "description": "10Y minus 3M Treasury yield spread (FRED T10Y3M)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Labor & activity (FRED / BLS schedule — conservative lags)
    # ==================================================================

    # unemployment | FRED:UNRATE | Civilian unemployment rate
    # Employment Situation: usually first Friday ~08:30 ET for prior month.
    "unemployment": {
        "source": "FRED",
        "series": "UNRATE",
        "category": "labor",
        "units": "percent",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 8,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": (
            "Unemployment rate (BLS Employment Situation ~first Friday 08:30 ET). "
            "lag≈8d conservative; prefer ALFRED first-release when available."
        ),
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # nonfarm_payrolls | FRED:PAYEMS | All employees, total nonfarm
    "nonfarm_payrolls": {
        "source": "FRED",
        "series": "PAYEMS",
        "category": "labor",
        "units": "thousands",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 8,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Nonfarm payrolls (BLS PAYEMS, Employment Situation ~08:30 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # initial_claims | FRED:ICSA | Initial jobless claims (weekly)
    # DOL: Thursdays 08:30 ET for week ending prior Saturday.
    "initial_claims": {
        "source": "FRED",
        "series": "ICSA",
        "category": "labor",
        "units": "number",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 5,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Initial unemployment claims (DOL ICSA, Thu 08:30 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Inflation & real activity (FRED)
    # ==================================================================

    # cpi | FRED:CPIAUCSL | CPI all urban consumers (index)
    # BLS CPI usually mid-month (~10–15d after month-end) 08:30 ET.
    "cpi": {
        "source": "FRED",
        "series": "CPIAUCSL",
        "category": "inflation",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 15,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "CPI-U SA (CPIAUCSL); ~mid-month 08:30 ET, lag≈15d",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # pce_price | FRED:PCEPI | PCE price index
    # BEA personal income/outlays ~end of following month.
    "pce_price": {
        "source": "FRED",
        "series": "PCEPI",
        "category": "inflation",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 30,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "PCE price index (BEA PCEPI); lag≈30d conservative",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # industrial_production | FRED:INDPRO
    "industrial_production": {
        "source": "FRED",
        "series": "INDPRO",
        "category": "activity",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 16,
        "release_time": "09:15",
        "fill_policy": "forward",
        "description": "Industrial production (Fed G.17 INDPRO ~09:15 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # retail_sales | FRED:RSXFS | Advance retail sales ex food services
    "retail_sales": {
        "source": "FRED",
        "series": "RSXFS",
        "category": "activity",
        "units": "million_usd",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 16,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Retail sales ex food (Census RSXFS ~mid-month 08:30 ET)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # real_gdp | FRED:GDPC1 | Real GDP (quarterly)
    "real_gdp": {
        "source": "FRED",
        "series": "GDPC1",
        "category": "activity",
        "units": "billion_usd",
        "timezone": "America/New_York",
        "release_timeframe": "quarterly",
        "release_lag_days": 30,
        "release_time": "08:30",
        "fill_policy": "forward",
        "description": "Real GDP (BEA GDPC1); advance ~1 month after quarter",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # ==================================================================
    # Sentiment & stress (FRED)
    # ==================================================================

    # umich_sentiment | FRED:UMCSENT | University of Michigan consumer sentiment
    "umich_sentiment": {
        "source": "FRED",
        "series": "UMCSENT",
        "category": "sentiment",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "monthly",
        "release_lag_days": 14,
        "release_time": "10:00",
        "fill_policy": "forward",
        "description": (
            "U.Mich consumer sentiment (UMCSENT); lag≈14d conservative"
        ),
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

    # financial_stress | FRED:STLFSI4 | St. Louis Fed Financial Stress Index
    "financial_stress": {
        "source": "FRED",
        "series": "STLFSI4",
        "category": "stress",
        "units": "index",
        "timezone": "America/New_York",
        "release_timeframe": "weekly",
        "release_lag_days": 3,
        "release_time": "16:00",
        "fill_policy": "forward",
        "description": "St. Louis Fed Financial Stress Index (STLFSI4)",
        "use_vintage_data": True,
        "vintage_source": "FRED",
    },

}


# ═══════════════════════════════════════════════════════════
# RUNTIME CATALOG: core FEATURES + config extras + filters
# ═══════════════════════════════════════════════════════════

from typing import Any, Dict, Mapping, Optional, Iterable

_META_REQUIRED = (
    "source",
    "category",
    "units",
    "timezone",
    "release_timeframe",
    "release_lag_days",
    "release_time",
)


def validate_feature_meta(name: str, meta: Mapping[str, Any]) -> None:
    """Raise ValueError if a feature meta dict is incomplete / inconsistent."""
    if not isinstance(meta, Mapping):
        raise ValueError(f"Feature {name!r}: meta must be a mapping, got {type(meta)}")
    missing = [k for k in _META_REQUIRED if k not in meta]
    if missing:
        raise ValueError(f"Feature {name!r}: missing required fields {missing}")
    source = meta.get("source")
    if source == "FRED" and not meta.get("series"):
        raise ValueError(f"Feature {name!r}: FRED entries require 'series'")
    if source == "TwelveData" and not meta.get("symbol"):
        raise ValueError(f"Feature {name!r}: TwelveData entries require 'symbol'")


def merge_feature_catalog(
    extras: Optional[Mapping[str, Mapping[str, Any]]] = None,
    *,
    override_core: bool = False,
) -> Dict[str, dict]:
    """
    Core FEATURES plus optional extras.

    - New names are appended after core (insertion order of ``extras``).
    - If a name already exists in core: raise unless override_core=True.
    """
    catalog: Dict[str, dict] = {k: dict(v) for k, v in FEATURES.items()}
    if not extras:
        return catalog

    for name, meta in extras.items():
        validate_feature_meta(name, meta)
        meta_copy = dict(meta)
        if name in catalog and not override_core:
            raise ValueError(
                f"macro_features_extra name {name!r} collides with core FEATURES. "
                f"Rename it or set override_core=True."
            )
        catalog[name] = meta_copy
    return catalog


def filter_feature_catalog(
    catalog: Mapping[str, dict],
    *,
    features_to_include: Optional[Iterable[str]] = None,
    features_to_exclude: Optional[Iterable[str]] = None,
) -> Dict[str, dict]:
    """
    Apply optional whitelist then exclude list (stable order preserved).

    - features_to_include None/empty → keep all keys in catalog order
    - unknown include names are ignored (with no error) so configs stay soft
    """
    exclude = set(features_to_exclude or [])
    if features_to_include:
        include = set(features_to_include)
        ordered = {k: catalog[k] for k in catalog if k in include and k not in exclude}
        return ordered
    return {k: v for k, v in catalog.items() if k not in exclude}


def get_features(
    config: Optional[Mapping[str, Any]] = None,
    *,
    apply_filters: bool = True,
    override_core: bool = False,
) -> Dict[str, dict]:
    """
    Active feature catalog for download / intervals / model layout.

    Parameters
    ----------
    config :
        Optional training config. Reads:
          - macro_features_extra : dict[name → meta]  (config-only FRED/etc.)
          - features_to_include  : optional whitelist of base names
          - features_to_exclude  : optional denylist
          - macro_override_core  : if True, extras may replace core names
    apply_filters :
        If True (default), apply include/exclude. Use False in fsv_01/02 so
        raw CSV still contains every merged series even if model excludes some.
    """
    extras = None
    if config is not None:
        extras = config.get("macro_features_extra") or None
        override_core = bool(config.get("macro_override_core", override_core))

    catalog = merge_feature_catalog(extras, override_core=override_core)

    if not apply_filters or config is None:
        return catalog

    return filter_feature_catalog(
        catalog,
        features_to_include=config.get("features_to_include"),
        features_to_exclude=config.get("features_to_exclude"),
    )
