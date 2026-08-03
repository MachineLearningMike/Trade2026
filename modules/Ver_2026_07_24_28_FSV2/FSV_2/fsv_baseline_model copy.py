"""
fsv_baseline_model.py

FSV Baseline Models

Purpose:
    Establish baseline predictability before deep learning.

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


START_DATE = (
    "2014-01-01"
)


SEED = 42


TEST_SIZE = 0.2



TARGETS = [

    "sp500_return_1d",

    "sp500_return_5d",

    "sp500_return_20d",

]



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



# ==========================================================
# Features
# ==========================================================


drop_cols = [

    "date"

]


features = [

    c

    for c in df.columns

    if c not in drop_cols

    and not c.startswith(
        "sp500_return"
    )

]



features = [

    c

    for c in features

    if pd.api.types.is_numeric_dtype(
        df[c]
    )

]



print(
    "Features:",
    len(features)
)



X = df[features]



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


    if target not in df.columns:

        print(
            "Skip missing target:",
            target
        )

        continue



    print()

    print(
        "="*50
    )

    print(
        "Regression:",
        target
    )



    data = df[
        df[target].notna()
    ].copy()



    X_target = data[features]

    y_target = data[target]



    print(
        "Samples:",
        len(data)
    )



    X_train, X_test, y_train, y_test = train_test_split(

        X_target,

        y_target,

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


        result = {


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



        results.append(
            result
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

                features,

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
    "="*60
)

print(
    "Baseline complete"
)

print(
    RESULT_FILE
)

print(
    IMPORTANCE_FILE
)

print(
    "="*60
)