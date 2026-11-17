"""Load the data, clean it and build the features the model uses."""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EARTH_RADIUS_MILES = 3958.8

MIN_RPM = 1.30
MAX_RPM = 4.00

FEATURE_COLS = [
    "distance",
    "weight",
    "equipment",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "delta_lat",
    "delta_lon",
    "circuity",
    "weight_per_mile",
    "dayofweek",
    "is_weekend",
    "weight_missing",
    "weight_was_negative",
    "market_index_missing",
]
QUOTE_COLS = ["quote_signal"]
MARKET_COLS = ["market_index", "market_quote_prod"]
EQUIPMENT_TYPES = ["Dry Van", "Flatbed", "Reefer"]


def load_dataset(name: str) -> pd.DataFrame:
    """Read a CSV from the data folder, whether its name uses dashes or underscores."""
    stems = {name, name.replace("_", "-"), name.replace("-", "_")}
    for stem in stems:
        for folder in (ROOT / "data", ROOT):
            path = folder / f"{stem}.csv"
            if path.is_file():
                return pd.read_csv(path, parse_dates=["date"])
    raise FileNotFoundError(f"Could not find {name}.csv in data/ or project root")


def haversine_distance(lat1, lon1, lat2, lon2):
    """Straight-line distance in miles between two points on the globe."""
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    return 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(a))


def filter_target_outliers(df: pd.DataFrame, target_col="posted_rate", min_rpm=MIN_RPM, max_rpm=MAX_RPM):
    """Drop training rows whose price per mile is too low or too high to be real."""
    if target_col not in df.columns:
        return df
    rpm = df[target_col] / df["distance"].abs().clip(lower=1.0)
    keep = rpm.between(min_rpm, max_rpm)
    print(f"Filtered {(~keep).sum()} target outlier rows (rate per mile outside ${min_rpm:.2f}-${max_rpm:.2f})")
    return df[keep].copy()


def build_reference(train_df: pd.DataFrame, extra_feature_frames=()) -> dict:
    """Collect the numbers needed to clean any dataset the same way, like medians and city coordinates."""
    frames = [train_df, *extra_feature_frames]
    feats = pd.concat([f.drop(columns=["posted_rate"], errors="ignore") for f in frames], ignore_index=True)

    coords = {}
    for side in ("pickup", "delivery"):
        tbl = feats.groupby(side)[[f"{side}_lat", f"{side}_lon"]].mean()
        tbl.columns = ["lat", "lon"]
        coords.update(tbl.to_dict("index"))

    daily_market = feats.groupby(feats["date"].dt.normalize())["market_index"].mean().dropna()

    return {
        "weight_median": train_df["weight"].abs().median(),
        "quote_signal_median": train_df["quote_signal"].median(),
        "market_index_median": train_df["market_index"].median(),
        "daily_market": daily_market,
        "coords": coords,
    }


def _fill_market_index(dates: pd.Series, ref: dict) -> pd.Series:
    """Get the market index for each date, using recent days when a date is unknown."""
    daily = ref["daily_market"]
    filled = dates.dt.normalize().map(daily)
    if filled.isna().any():
        recent = daily.iloc[-14:].mean() if len(daily) else ref["market_index_median"]
        filled = filled.fillna(recent)
    return filled


def preprocess_data(df: pd.DataFrame, ref: dict) -> pd.DataFrame:
    """Clean one dataset and add all model features to it."""
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    if "weight" not in data.columns:
        data["weight"] = np.nan
    data["weight_missing"] = data["weight"].isna().astype(int)
    data["weight_was_negative"] = (data["weight"] < 0).astype(int)
    data["weight"] = data["weight"].abs().fillna(ref["weight_median"])
    data["distance"] = data["distance"].abs()

    if "market_index" not in data.columns:
        data["market_index"] = np.nan
    data["market_index_missing"] = data["market_index"].isna().astype(int)
    data["market_index"] = data["market_index"].fillna(_fill_market_index(data["date"], ref))

    if "quote_signal" not in data.columns:
        data["quote_signal"] = np.nan
    data["quote_signal"] = data["quote_signal"].fillna(ref["quote_signal_median"])

    for side in ("pickup", "delivery"):
        for axis in ("lat", "lon"):
            col = f"{side}_{axis}"
            lookup = data[side].map(lambda c, a=axis: ref["coords"].get(c, {}).get(a, np.nan))
            data[col] = data[col].fillna(lookup) if col in data.columns else lookup
    unknown = data["pickup_lat"].isna() | data["delivery_lat"].isna()
    if unknown.any():
        raise ValueError(f"No coordinates for cities: {sorted(set(data.loc[unknown, 'pickup']) | set(data.loc[unknown, 'delivery']))}")

    data["delta_lat"] = data["delivery_lat"] - data["pickup_lat"]
    data["delta_lon"] = data["delivery_lon"] - data["pickup_lon"]
    straight = haversine_distance(data["pickup_lat"], data["pickup_lon"], data["delivery_lat"], data["delivery_lon"])
    data["circuity"] = data["distance"] / straight.clip(lower=1.0)

    data["dayofweek"] = data["date"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    data["market_quote_prod"] = data["market_index"] * data["quote_signal"]
    data["weight_per_mile"] = data["weight"] / data["distance"].clip(lower=1.0)

    data["equipment"] = pd.Categorical(data["equipment"], categories=EQUIPMENT_TYPES)

    if "posted_rate" in data.columns:
        data["rate_per_mile"] = data["posted_rate"] / data["distance"]

    return data


if __name__ == "__main__":
    train = filter_target_outliers(load_dataset("train_test"))
    val = load_dataset("validation")
    dec = load_dataset("december_chart_inputs")
    ref = build_reference(train, [val])

    for name, frame in [("train", train), ("validation", val), ("december", dec)]:
        out = preprocess_data(frame, ref)
        nans = out[FEATURE_COLS].isna().sum().sum()
        print(f"{name:<11} rows={len(out):>6}  features={len(FEATURE_COLS)}  NaNs in features={nans}")

    dec_out = preprocess_data(dec, ref)
    print("\nDecember market_index (first 5 days):")
    print(dec_out[["date", "market_index", "pickup_lat", "delivery_lat"]].head().to_string(index=False))
