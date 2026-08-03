"""
fsv_baseline_model.py

Financial State Vector Baseline Model

Purpose:
    Evaluate predictive power of FSV representation.

Input:
    fsv_training.csv

Outputs:
    fsv_baseline_results.csv
    fsv_baseline_feature_importance.csv

"""

import numpy as np
import pandas as pd


from sklearn.model_selection import train_test_split


from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler


from sklearn.linear_model import (
    LinearRegression,
    Ridge
)


from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)


from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)



# ==========================================================
# Configuration
# ==========================================================


INPUT_FILE = "fsv_training.csv"


RESULT_FILE = (
    "fsv_baseline_results.csv"
)


IMPORTANCE_FILE = (
    "fsv_baseline_feature_importance.csv"
)


START_DATE = "2014-01-01"


SEED = 42


TEST_SIZE = 0.2



# ==========================================================
# Load
# ==========================================================


print(
    "Loading training data..."
)


df = pd.read_csv(

    INPUT_FILE,

    parse_dates=[
        "date"
    ]

)


df = (

    df

    .sort_values(
        "date"
    )

    .reset_index(
        drop=True
    )

)



df = df[
    df.date >= START_DATE
]


print(
    "Rows after date filter:",
    len(df)
)


print(
    "Columns:",
    len(df.columns)
)



# ==========================================================
# Discover targets
# ==========================================================


TARGET_KEYWORDS = [

    "return",

    "volatility",

    "vol",

    "regime",

    "target"

]



TARGETS = [

    c

    for c in df.columns

    if any(

        k in c.lower()

        for k in TARGET_KEYWORDS

    )

]



print()

print(
    "Detected targets:"
)


for t in TARGETS:

    print(
        " ",
        t
    )



if len(TARGETS) == 0:

    raise RuntimeError(
        "No targets detected"
    )



# ==========================================================
# Features
# ==========================================================


DROP_COLUMNS = [

    "date"

]


FEATURES = [

    c

    for c in df.columns

    if c not in DROP_COLUMNS

    and c not in TARGETS

]



FEATURES = [

    c

    for c in FEATURES

    if pd.api.types.is_numeric_dtype(
        df[c]
    )

]



print()

print(
    "Features:",
    len(FEATURES)
)



# ==========================================================
# Models
# ==========================================================


def build_models():


    return {


        "LinearRegression":

        Pipeline(

            [

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "scaler",

                    StandardScaler()

                ),

                (
                    "model",

                    LinearRegression()

                )

            ]

        ),



        "Ridge":

        Pipeline(

            [

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "scaler",

                    StandardScaler()

                ),

                (
                    "model",

                    Ridge(
                        alpha=10.0
                    )

                )

            ]

        ),



        "RandomForest":

        Pipeline(

            [

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "model",

                    RandomForestRegressor(

                        n_estimators=300,

                        max_depth=5,

                        random_state=SEED,

                        n_jobs=-1

                    )

                )

            ]

        ),



        "GradientBoosting":

        Pipeline(

            [

                (
                    "imputer",

                    SimpleImputer(
                        strategy="median"
                    )

                ),

                (
                    "model",

                    GradientBoostingRegressor(

                        n_estimators=200,

                        learning_rate=0.03,

                        max_depth=3,

                        random_state=SEED

                    )

                )

            ]

        )

    }



# ==========================================================
# Training
# ==========================================================


results = []


importance_rows = []



for target in TARGETS:


    print()

    print(
        "=" * 60
    )

    print(
        "Target:",
        target
    )


    data = (

        df

        [

            df[target]
            .notna()

        ]

        .reset_index(
            drop=True
        )

    )


    if len(data) < 200:

        print(
            "Skip: insufficient samples"
        )

        continue



    X = data[FEATURES]


    y = data[target]



    print(
        "Samples:",
        len(data)
    )



    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=SEED

    )



    models = build_models()



    for name, model in models.items():


        print(
            " ",
            name
        )


        model.fit(

            X_train,

            y_train

        )



        pred = model.predict(
            X_test
        )



        mse = mean_squared_error(

            y_test,

            pred

        )



        results.append(

            {

                "target":

                    target,


                "model":

                    name,


                "samples":

                    len(data),


                "rmse":

                    np.sqrt(mse),


                "mae":

                    mean_absolute_error(

                        y_test,

                        pred

                    ),


                "r2":

                    r2_score(

                        y_test,

                        pred

                    )

            }

        )



        # ------------------------------------------
        # Feature importance
        # ------------------------------------------


        estimator = (

            model

            .named_steps[

                "model"

            ]

        )



        if hasattr(

            estimator,

            "feature_importances_"

        ):


            for f, v in zip(

                FEATURES,

                estimator.feature_importances_

            ):


                importance_rows.append(

                    {

                        "target":

                            target,


                        "model":

                            name,


                        "feature":

                            f,


                        "importance":

                            v

                    }

                )



# ==========================================================
# Save
# ==========================================================


result_df = pd.DataFrame(
    results
)


result_df = result_df.sort_values(

    [

        "target",

        "r2"

    ],

    ascending=[

        True,

        False

    ]

)



result_df.to_csv(

    RESULT_FILE,

    index=False

)



importance_df = pd.DataFrame(

    importance_rows

)



if len(importance_df):

    importance_df = (

        importance_df

        .sort_values(

            "importance",

            ascending=False

        )

    )



importance_df.to_csv(

    IMPORTANCE_FILE,

    index=False

)



print()

print(
    "=" * 60
)

print(
    "Baseline complete"
)

print(
    "Saved:",
    RESULT_FILE
)

print(
    "Saved:",
    IMPORTANCE_FILE
)

print(
    "=" * 60
)