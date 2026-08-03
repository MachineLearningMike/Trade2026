"""
FSV Timestamp Configuration

Defines when the daily financial state vector is evaluated.

All aligned features are transformed into:
    information available BEFORE this timestamp.
"""


PREDICTION_TIME = {

    # Decision time
    "hour": 8,
    "minute": 0,

    # Local market timezone
    "timezone":
        "America/New_York",

}


# How far back we allow stale values
#
# Example:
# monthly housing data should not be considered
# valid forever.

MAX_STALENESS_DAYS = {

    "daily": 5,

    "weekly": 14,

    "monthly": 60,

    "quarterly": 120,

}