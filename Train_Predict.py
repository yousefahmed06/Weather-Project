"""
Trains a temperature-forecasting model on weather_data_features.csv and
writes out predictions for the dashboard.

This is a script version of the "Temperature prediction model" section of
EDA_and_anomaly_detection.ipynb / WeatherFinalProject.ipynb. The modeling
logic (features, split, pipeline, model choice) is kept identical to the
notebook on purpose -- only the surrounding plumbing (I/O, logging, exit
codes) has been added so it can run unattended in CI.

Reads:
    weather_data_features.csv   (produced by process_weather.py)

Writes:
    linear_weather_model.pkl    - the trained model, retrained fresh each run
    weather_predictions.csv     - weather_data_features.csv + predicted_temp_3h,
                                   the file the dashboard reads

Run manually:
    python train_and_predict.py

In GitHub Actions, this runs as a step right after process_weather.py in the
same job (see the workflow file).

Exit codes:
    0 - success
    1 - unrecoverable problem (input file missing, too few rows to train/
        evaluate, or an essential column is missing)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

FEATURES_PATH = Path("weather_data_features.csv")
MODEL_PATH = Path("linear_weather_model.pkl")
PREDICTIONS_PATH = Path("weather_predictions.csv")

# Same feature lists as the notebook.
NUMERIC_FEATURES = [
    "temp_c", "feels_like_c", "temp_min_c", "temp_max_c", "pressure",
    "humidity", "wind_speed", "wind_deg", "wind_gust", "clouds_pct",
    "visibility", "pop", "rain_3h_mm", "snow_3h_mm", "hour", "month",
    "temp_range_c", "feels_like_delta_c", "temp_rolling_avg_9h",
]
CATEGORICAL_FEATURES = [
    "city", "weather_main", "part_of_day", "is_raining", "is_snowing", "is_daytime",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "target_temp_3h"

# Minimum rows needed before a train/test split + model fit is meaningful.
MIN_ROWS_TO_TRAIN = 20

# Columns the pipeline genuinely can't function without -- building the
# (city, next-reading) target and sorting the forecast timeline both depend
# on these directly, so there's no sensible default to fall back on.
ESSENTIAL_COLUMNS = ["forecast_datetime", "temp_c", "city"]

# process_weather.py's drop_uninformative_columns() removes any feature that
# ends up constant or all-null in a given run's batch (e.g. snow_3h_mm when
# no tracked city is snowing that cycle). That's the right call upstream --
# but it means the schema this script sees can legitimately shrink from run
# to run. These are the neutral fill-ins used to reconstruct a dropped
# column: the column not existing already meant "no signal", so filling
# with a constant is equivalent to what the drop was implying.
NUMERIC_FEATURE_DEFAULTS = {col: 0.0 for col in NUMERIC_FEATURES}
CATEGORICAL_FEATURE_DEFAULTS = {
    "city": "unknown",
    "weather_main": "unknown",
    "part_of_day": "unknown",
    "is_raining": False,
    "is_snowing": False,
    "is_daytime": True,
}


def load_features(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run process_weather.py first.")
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows from {path}")
    return df


def reconcile_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruct any ALL_FEATURES column that process_weather.py dropped as
    uninformative (constant/all-null) in this run, instead of hard-failing.
    The model just sees an uninformative constant feature -- exactly what
    it would have seen if the column had been kept as-is.
    """
    df = df.copy()
    for col in NUMERIC_FEATURES:
        if col not in df.columns:
            default = NUMERIC_FEATURE_DEFAULTS[col]
            print(f"  '{col}' missing (likely dropped upstream as constant/all-null) -> filling with {default}")
            df[col] = default
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            default = CATEGORICAL_FEATURE_DEFAULTS.get(col, "unknown")
            print(f"  '{col}' missing (likely dropped upstream as constant/all-null) -> filling with {default!r}")
            df[col] = default
    return df


def build_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the notebook's target-construction cells:
    sort by city/time, shift temp_c by -1 within each city to get the
    'next reading' as the 3h-ahead target, then drop rows with no target
    (the last reading per city, which has nothing to predict).
    """
    df = df.copy()
    df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"])
    df = df.sort_values(["city", "forecast_datetime"])

    df[TARGET] = df.groupby("city")["temp_c"].shift(-1)
    df = df.dropna(subset=[TARGET])

    return df


def make_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])


def train(df_train: pd.DataFrame) -> Pipeline:
    X = df_train[ALL_FEATURES]
    y = df_train[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = make_pipeline()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print("\n--- Holdout evaluation (Linear Regression) ---")
    print(f"  MAE : {mae:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  R2  : {r2:.3f}")

    return model


def main() -> int:
    try:
        df = load_features(FEATURES_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    missing_essential = [c for c in ESSENTIAL_COLUMNS if c not in df.columns]
    if missing_essential:
        print(f"ERROR: weather_data_features.csv is missing essential columns: {missing_essential}", file=sys.stderr)
        return 1

    missing_features = [c for c in ALL_FEATURES if c not in df.columns]
    if missing_features:
        print(f"{len(missing_features)} feature column(s) dropped upstream as uninformative -- reconciling:")
        df = reconcile_features(df)

    df_train = build_training_frame(df)
    print(f"{len(df_train):,} rows have a valid target after building (city, next-reading) pairs")

    if len(df_train) < MIN_ROWS_TO_TRAIN:
        print(
            f"ERROR: only {len(df_train)} trainable rows (need >= {MIN_ROWS_TO_TRAIN}). "
            "Let weather_data.csv accumulate more history before training.",
            file=sys.stderr,
        )
        return 1

    model = train(df_train)

    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}")

    # Predict on the full features file, same as the notebook's
    # "Predict New Data" section (re-reads the features file and scores it).
    X_all = df[ALL_FEATURES]
    df_predictions = df.copy()
    df_predictions["predicted_temp_3h"] = model.predict(X_all)

    df_predictions.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {len(df_predictions):,} rows with predictions to {PREDICTIONS_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
