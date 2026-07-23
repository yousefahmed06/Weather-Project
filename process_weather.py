"""
Validates, cleans, and feature-engineers weather_data.csv (produced by fetch_weather.py).

Reads:
    weather_data.csv

Writes:
    weather_data_clean.csv     - deduplicated, type-corrected, range-clipped
    weather_data_features.csv  - weather_data_clean.csv plus derived columns

Run manually:
    python process_weather.py

In GitHub Actions, this runs as a step right after fetch_weather.py in the
same job (see the workflow file).

Exit codes:
    0 - success (even if minor issues were found and auto-fixed)
    1 - unrecoverable problem (input file missing, or no usable rows survive
        cleaning). A non-zero exit fails the GitHub Actions run so you get
        notified instead of silently pushing empty/broken data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path("weather_data.csv")
CLEAN_PATH = Path("weather_data_clean.csv")
FEATURES_PATH = Path("weather_data_features.csv")

EXPECTED_COLUMNS = [
    "fetch_timestamp", "city", "country", "lat", "lon", "forecast_datetime",
    "temp_c", "feels_like_c", "temp_min_c", "temp_max_c",
    "pressure", "humidity", "wind_speed", "wind_deg", "wind_gust",
    "clouds_pct", "visibility", "pop", "rain_3h_mm", "snow_3h_mm",
    "weather_main", "weather_description", "part_of_day",
]

RANGE_CHECKS = {
    "humidity":   (0, 100),
    "clouds_pct": (0, 100),
    "pop":        (0, 1),
    "lat":        (-90, 90),
    "lon":        (-180, 180),
    "wind_speed": (0, None),
    "wind_gust":  (0, None),
    "pressure":   (800, 1100),  # generous bounds for sea-level-ish hPa readings
}


def load_raw(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run fetch_weather.py first.")
    df = pd.read_csv(path)
    print(f"Ingested {len(df):,} rows, {df['city'].nunique()} cities from {path}")
    return df


def validate(df: pd.DataFrame, ignore_missing: list[str] | None = None) -> dict:
    """Report-only: never modifies df.

    `ignore_missing` lists columns that are expected to be absent on purpose
    (e.g. dropped by drop_uninformative_columns) so they aren't flagged as
    a validation failure.
    """
    report = {}
    ignore_missing = set(ignore_missing or [])

    report["missing_columns"] = [
        c for c in EXPECTED_COLUMNS if c not in df.columns and c not in ignore_missing
    ]

    null_counts = df.isnull().sum()
    report["null_counts"] = null_counts[null_counts > 0].to_dict()

    report["duplicate_rows"] = int(df.duplicated().sum())

    violations = {}
    for col, (lo, hi) in RANGE_CHECKS.items():
        if col not in df.columns:
            continue
        mask = pd.Series(False, index=df.index)
        if lo is not None:
            mask |= df[col] < lo
        if hi is not None:
            mask |= df[col] > hi
        if mask.sum():
            violations[col] = int(mask.sum())
    report["range_violations"] = violations

    report["passed"] = not (
        report["missing_columns"] or report["duplicate_rows"] or report["range_violations"]
    )
    return report


def print_report(title: str, report: dict) -> None:
    print(f"\n--- {title} ---")
    for k, v in report.items():
        print(f"  {k}: {v}")


def drop_uninformative_columns(df: pd.DataFrame, keep: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """
    Dynamic filtering: drop columns that carry no signal (constant value or
    entirely null) instead of hardcoding which ones to remove. Columns in
    `keep` are never dropped, even if they happen to be constant/null in a
    given run (e.g. a single-city fetch where city is constant).

    Returns (df, dropped_columns) so callers can tell validate() these
    absences are intentional, not a sign of upstream schema drift.
    """
    df = df.copy()

    useless_cols = [
        col for col in df.columns
        if col not in keep and df[col].nunique(dropna=True) <= 1
    ]

    if useless_cols:
        print(f"Dropping uninformative columns (constant or all-null): {useless_cols}")
        df = df.drop(columns=useless_cols)

    return df, useless_cols


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"], errors="coerce")
    df["fetch_timestamp"] = pd.to_datetime(df["fetch_timestamp"], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows")

    before = len(df)
    df = df.dropna(subset=["forecast_datetime", "city"])
    print(f"Dropped {before - len(df)} rows missing city/forecast_datetime")

    before = len(df)
    df = df.sort_values("fetch_timestamp").drop_duplicates(
        subset=["city", "forecast_datetime"], keep="last"
    )
    print(f"Dropped {before - len(df)} duplicate (city, forecast_datetime) rows, kept most recent fetch")

    numeric_cols = [
        "temp_c", "feels_like_c", "temp_min_c", "temp_max_c", "pressure",
        "humidity", "wind_speed", "wind_gust", "clouds_pct", "pop",
        "rain_3h_mm", "snow_3h_mm", "lat", "lon",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["rain_3h_mm", "snow_3h_mm"]:
        df[col] = df[col].fillna(0)

    if "humidity" in df.columns:
        df["humidity"] = df["humidity"].fillna(df.groupby("city")["humidity"].transform("median"))
        df["humidity"] = df["humidity"].clip(0, 100)

    if "clouds_pct" in df.columns:
        df["clouds_pct"] = df["clouds_pct"].clip(0, 100)
    if "pop" in df.columns:
        df["pop"] = df["pop"].clip(0, 1)
    for col in ["wind_speed", "wind_gust"]:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)
    df["visibility"]= df["visibility"].fillna(10000).clip(lower=0)

    df, dropped_cols = drop_uninformative_columns(
        df,
        keep=["fetch_timestamp", "city", "country", "forecast_datetime", "lat", "lon"],
    )

    return df, dropped_cols


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["forecast_datetime"].dt.hour
    df["day_of_week"] = df["forecast_datetime"].dt.day_name()
    df["is_weekend"] = df["forecast_datetime"].dt.weekday >= 5
    df["month"] = df["forecast_datetime"].dt.month

    if "temp_max_c" in df.columns and "temp_min_c" in df.columns:
        df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]
    if "feels_like_c" in df.columns and "temp_c" in df.columns:
        df["feels_like_delta_c"] = df["feels_like_c"] - df["temp_c"]
    # rain_3h_mm / snow_3h_mm may have been dropped by drop_uninformative_columns
    # if they were constant (e.g. all zero - no rain/snow in this batch). That
    # itself means "never raining/snowing", so default accordingly instead of
    # erroring out.
    df["is_raining"] = df["rain_3h_mm"] > 0 if "rain_3h_mm" in df.columns else False
    df["is_snowing"] = df["snow_3h_mm"] > 0 if "snow_3h_mm" in df.columns else False
    df["is_daytime"] = (df["part_of_day"] == "d") if "part_of_day" in df.columns else np.nan

    df = df.sort_values(["city", "forecast_datetime"])

    if "temp_c" in df.columns:
        bins = [-np.inf, 0, 10, 20, 30, np.inf]
        labels = ["freezing", "cold", "mild", "warm", "hot"]
        df["temp_category"] = pd.cut(df["temp_c"], bins=bins, labels=labels)

        df["temp_rolling_avg_9h"] = (
            df.groupby("city")["temp_c"]
              .transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )

    return df


def main() -> int:
    try:
        df_raw = load_raw(RAW_PATH)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    pre_report = validate(df_raw)
    print_report("Validation (raw)", pre_report)
    if not pre_report["passed"]:
        print("\nIssues found above will be auto-fixed during cleaning.")

    df_clean, dropped_cols = clean(df_raw)

    if df_clean.empty:
        print("ERROR: no rows survived cleaning.", file=sys.stderr)
        return 1

    post_report = validate(df_clean, ignore_missing=dropped_cols)
    print_report("Validation (post-clean)", post_report)
    if not post_report["passed"]:
        # Cleaning should have resolved everything by this point. If it
        # hasn't, something upstream changed shape unexpectedly - fail loud
        # rather than push data that still fails its own checks.
        print("ERROR: data still fails validation after cleaning.", file=sys.stderr)
        return 1

    df_clean.to_csv(CLEAN_PATH, index=False)
    print(f"\nWrote {len(df_clean):,} cleaned rows to {CLEAN_PATH}")

    df_features = add_features(df_clean)
    df_features.to_csv(FEATURES_PATH, index=False)
    print(f"Wrote {len(df_features):,} feature-engineered rows to {FEATURES_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
