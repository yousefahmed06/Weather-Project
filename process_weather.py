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
    "fetch_timestamp", "city", "country", "forecast_datetime",
    "temp_c", "feels_like_c", "temp_min_c", "temp_max_c",
    "pressure", "humidity", "wind_speed", "wind_deg", "wind_gust",
    "clouds_pct", "visibility", "pop", "rain_3h_mm", "snow_3h_mm",
    "weather_main", "weather_description", "part_of_day",
]

RANGE_CHECKS = {
    "humidity":   (0, 100),
    "clouds_pct": (0, 100),
    "pop":        (0, 1),
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


def validate(df: pd.DataFrame) -> dict:
    """Report-only: never modifies df."""
    report = {}

    report["missing_columns"] = [c for c in EXPECTED_COLUMNS if c not in df.columns]

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


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["forecast_datetime"] = pd.to_datetime(df["forecast_datetime"], errors="coerce")
    df["fetch_timestamp"] = pd.to_datetime(df["fetch_timestamp"], errors="coerce")

    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact duplicate rows")

    before = len(df)
    df = df.dropna(subset=["forecast_datetime", "city"])
    print(f"Dropped {before - len(df)} rows missing city/forecast_datetime")

    numeric_cols = [
        "temp_c", "feels_like_c", "temp_min_c", "temp_max_c", "pressure",
        "humidity", "wind_speed", "wind_gust", "clouds_pct", "pop",
        "rain_3h_mm", "snow_3h_mm",
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

    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["forecast_datetime"].dt.hour
    df["day_of_week"] = df["forecast_datetime"].dt.day_name()
    df["is_weekend"] = df["forecast_datetime"].dt.weekday >= 5
    df["month"] = df["forecast_datetime"].dt.month

    df["temp_range_c"] = df["temp_max_c"] - df["temp_min_c"]
    df["feels_like_delta_c"] = df["feels_like_c"] - df["temp_c"]
    df["is_raining"] = df["rain_3h_mm"] > 0
    df["is_snowing"] = df["snow_3h_mm"] > 0
    df["is_daytime"] = df["part_of_day"] == "d"

    bins = [-np.inf, 0, 10, 20, 30, np.inf]
    labels = ["freezing", "cold", "mild", "warm", "hot"]
    df["temp_category"] = pd.cut(df["temp_c"], bins=bins, labels=labels)

    df = df.sort_values(["city", "forecast_datetime"])
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

    df_clean = clean(df_raw)

    if df_clean.empty:
        print("ERROR: no rows survived cleaning.", file=sys.stderr)
        return 1

    post_report = validate(df_clean)
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
