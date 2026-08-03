"""
metadata.py

Feature metadata utilities for the Feature State Vector (FSV).

Each feature should contain:

    name
    source
    series/symbol
    category
    frequency
    units
    timezone
    publish_time
    release_delay
    decay_period
    forward_fill
    description

The downloader only collects raw observations.
Later stages use this metadata to calculate:
    - publication time
    - feature age
    - decay weight
"""


from dataclasses import dataclass
from typing import Optional



@dataclass
class FeatureMetadata:
    """
    Metadata definition of one FSV feature.
    """


    # ------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------

    name: str

    source: str


    # FRED:
    #     series = FRED series id
    #
    # TwelveData:
    #     symbol = market symbol

    series: Optional[str] = None

    symbol: Optional[str] = None



    # ------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------

    category: str = "unknown"


    frequency: str = (
        "daily"
    )


    units: str = (
        "unknown"
    )



    # ------------------------------------------------------------
    # Time information
    # ------------------------------------------------------------

    timezone: str = (
        "UTC"
    )


    # Actual release time.
    #
    # Example:
    #   CPI:
    #       08:30 America/New_York
    #
    # Market data:
    #       None
    #

    publish_time: Optional[str] = None



    # Delay between observation period
    # and availability.
    #
    # Example:
    #
    # M2:
    #   observed January
    #   released February
    #
    release_delay_days: Optional[int] = None



    # ------------------------------------------------------------
    # Decay model
    # ------------------------------------------------------------

    # Characteristic decay period τ.
    #
    # decay_weight =
    #
    #       exp(-age / decay_period)
    #

    decay_period_minutes: Optional[int] = None



    # ------------------------------------------------------------
    # Processing rules
    # ------------------------------------------------------------

    forward_fill: bool = False


    description: str = ""



    def validate(self):

        """
        Validate metadata consistency.
        """


        if self.source == "FRED":

            if self.series is None:

                raise ValueError(
                    f"{self.name}: FRED feature "
                    "requires series id"
                )


        if self.source == "TwelveData":

            if self.symbol is None:

                raise ValueError(
                    f"{self.name}: TwelveData feature "
                    "requires symbol"
                )


        if self.frequency not in [
            "minute",
            "hourly",
            "daily",
            "weekly",
            "monthly",
            "quarterly",
        ]:

            raise ValueError(
                f"{self.name}: invalid frequency "
                f"{self.frequency}"
            )


        return True



    def to_dict(self):

        """
        Convert metadata to dictionary.
        """

        return {

            "name":
                self.name,

            "source":
                self.source,

            "series":
                self.series,

            "symbol":
                self.symbol,

            "category":
                self.category,

            "frequency":
                self.frequency,

            "units":
                self.units,

            "timezone":
                self.timezone,

            "publish_time":
                self.publish_time,

            "release_delay_days":
                self.release_delay_days,

            "decay_period_minutes":
                self.decay_period_minutes,

            "forward_fill":
                self.forward_fill,

            "description":
                self.description,
        }



####################################################################
# Helper functions
####################################################################


def validate_features(features):

    """
    Validate entire feature dictionary/list.
    """

    for feature in features:

        if isinstance(
            feature,
            FeatureMetadata
        ):

            feature.validate()

        else:

            raise TypeError(
                "Feature must be FeatureMetadata"
            )


    return True



def metadata_summary(features):

    """
    Print compact metadata table.
    """

    rows = []


    for f in features:

        rows.append(

            {

                "name":
                    f.name,

                "source":
                    f.source,

                "category":
                    f.category,

                "frequency":
                    f.frequency,

                "units":
                    f.units,

            }

        )


    return rows