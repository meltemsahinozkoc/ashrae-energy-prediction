from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CATEGORICAL = ["site_id", "building_id", "primary_use", "meter"]

NUMERIC_DOWNCAST = {
    "building_id": "int16",
    "meter": "int8",
    "site_id": "int8",
    "square_feet": "float32",
    "year_built": "float32",
    "floor_count": "float32",
    "air_temperature": "float32",
    "cloud_coverage": "float32",
    "dew_temperature": "float32",
    "precip_depth_1_hr": "float32",
    "sea_level_pressure": "float32",
    "wind_direction": "float32",
    "wind_speed": "float32",
}


def _read(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, parse_dates=["timestamp"] if "timestamp" else None)


def load_raw(split: str = "train") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    meters = pd.read_csv(DATA_DIR / f"{split}.csv", parse_dates=["timestamp"])
    buildings = pd.read_csv(DATA_DIR / "building_metadata.csv")
    weather = pd.read_csv(DATA_DIR / f"weather_{split}.csv", parse_dates=["timestamp"])
    return meters, buildings, weather


def fill_weather(weather: pd.DataFrame) -> pd.DataFrame:
    weather = weather.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    cols = [c for c in weather.columns if c not in {"site_id", "timestamp"}]
    weather[cols] = (
        weather.groupby("site_id")[cols]
        .apply(lambda g: g.interpolate(limit_direction="both"))
        .reset_index(level=0, drop=True)
    )
    weather[cols] = weather[cols].fillna(weather[cols].median(numeric_only=True))
    return weather


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["weekday"] = ts.dt.weekday.astype("int8")
    df["month"] = ts.dt.month.astype("int8")
    df["is_weekend"] = (df["weekday"] >= 5).astype("int8")
    return df


def add_weather_features(df: pd.DataFrame, heat_thresh: float = 18.0) -> pd.DataFrame:
    t = df["air_temperature"]
    df["hdh"] = (heat_thresh - t).clip(lower=0).astype("float32")
    df["cdh"] = (t - heat_thresh).clip(lower=0).astype("float32")
    return df


def merge(meters: pd.DataFrame, buildings: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    df = meters.merge(buildings, on="building_id", how="left")
    df = df.merge(weather, on=["site_id", "timestamp"], how="left")
    for col, dt in NUMERIC_DOWNCAST.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    df["primary_use"] = df["primary_use"].astype("category")
    df["building_age"] = (df["timestamp"].dt.year - df["year_built"]).astype("float32")
    return df


def filter_train(df: pd.DataFrame) -> pd.DataFrame:
    bad = (df["site_id"] == 0) & (df["meter"] == 0) & (df["timestamp"] < "2016-05-21")
    return df.loc[~bad].reset_index(drop=True)


def build(split: str = "train") -> pd.DataFrame:
    meters, buildings, weather = load_raw(split)
    weather = fill_weather(weather)
    df = merge(meters, buildings, weather)
    df = add_time_features(df)
    df = add_weather_features(df)
    if split == "train":
        df = filter_train(df)
    return df


FEATURES = [
    "site_id",
    "building_id",
    "primary_use",
    "square_feet",
    "year_built",
    "floor_count",
    "building_age",
    "air_temperature",
    "cloud_coverage",
    "dew_temperature",
    "precip_depth_1_hr",
    "sea_level_pressure",
    "wind_direction",
    "wind_speed",
    "hdh",
    "cdh",
    "hour",
    "weekday",
    "month",
    "is_weekend",
]

CAT_FEATURES = ["site_id", "building_id", "primary_use"]


def rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred = np.clip(y_pred, 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))
