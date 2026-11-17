"""Explore the raw data and print what it shows."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "reports" / "figures"


def find_file(*candidates: str) -> Path:
    """Return the first of the given file paths that exists."""
    for name in candidates:
        path = ROOT / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of these files exist: {candidates}")


def load_data():
    """Load the training, validation and December files."""
    train = pd.read_csv(find_file("data/train-test.csv", "data/train_test.csv"), parse_dates=["date"])
    val = pd.read_csv(find_file("data/validation.csv"), parse_dates=["date"])
    dec = pd.read_csv(find_file("data/december-chart-inputs.csv", "data/december_chart_inputs.csv"))
    return train, val, dec


def section(title: str) -> None:
    """Print a section heading."""
    print(f"\n=== {title} ===")


def quality_report(df: pd.DataFrame, name: str) -> None:
    """Print missing values and other data problems in a dataset."""
    section(f"Data quality: {name}")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        print("No missing values")
    else:
        print(pd.DataFrame({"missing": missing, "pct": (missing / len(df) * 100).round(2)}))
    print(f"Negative weight rows:   {(df['weight'] < 0).sum()}")
    print(f"Zero weight rows:       {(df['weight'] == 0).sum()}")
    print(f"Non-positive distance:  {(df['distance'] <= 0).sum()}")
    print(f"Duplicate load_id:      {df['load_id'].duplicated().sum()}")
    print(f"Pickup == delivery:     {(df['pickup'] == df['delivery']).sum()}")


def run_eda() -> None:
    """Print the main findings about the data and save a few charts."""
    train, val, dec = load_data()

    section("Shapes")
    print(f"Train/Test:      {train.shape}")
    print(f"Validation:      {val.shape}")
    print(f"December inputs: {dec.shape}")

    section("Date ranges")
    print(f"Train:      {train['date'].min().date()} to {train['date'].max().date()}")
    print(f"Validation: {val['date'].min().date()} to {val['date'].max().date()}")
    print("-> Validation is strictly later than training: use a time-based split.")

    quality_report(train, "train")
    quality_report(val, "validation")

    section("Target: posted_rate")
    print(train["posted_rate"].describe().round(2))
    rpm = train["posted_rate"] / train["distance"]
    print("\nRate per mile percentiles:")
    print(rpm.quantile([0, 0.001, 0.01, 0.5, 0.99, 0.999, 1]).round(2))
    print("\nRate per mile distribution (label errors cluster at ~0.3x and ~3.5x normal):")
    print(pd.cut(rpm, [0, 0.5, 1.0, 1.3, 1.67, 3.2, 4.0, 8.0, 20.0]).value_counts().sort_index().to_string())
    lo, hi = 1.30, 4.00
    print(f"Rows outside ${lo:.2f}-${hi:.2f}/mile (dropped in training): {((rpm < lo) | (rpm > hi)).sum()}")

    section("Correlations with posted_rate")
    num_cols = ["distance", "weight", "market_index", "quote_signal", "posted_rate"]
    print(train[num_cols].corr()["posted_rate"].round(3))
    print("\nCorrelations with rate per mile:")
    print(train[num_cols[:-1]].corrwith(rpm).round(3))

    section("Categoricals")
    print("Equipment (train):")
    print(train["equipment"].value_counts())
    print(f"\nMean rate per mile by equipment:\n{rpm.groupby(train['equipment']).mean().round(3)}")
    print(f"\nUnique pickups: {train['pickup'].nunique()}, deliveries: {train['delivery'].nunique()}")
    seen = set(train["pickup"]) | set(train["delivery"])
    unseen = sorted((set(val["pickup"]) | set(val["delivery"])) - seen)
    print(f"Cities in validation but not in training ({len(unseen)}): {unseen}")
    coords_per_city = train.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max().max()
    print(f"Max distinct coordinates per city: {coords_per_city} (1 = fixed lookup)")

    section("Time patterns")
    month = train["date"].dt.to_period("M")
    print("Mean rate per mile by month:")
    print(rpm.groupby(month).mean().round(3))
    print("\nMean rate per mile by day of week (0=Mon):")
    print(rpm.groupby(train["date"].dt.dayofweek).mean().round(3))

    section("quote_signal regime check")
    print("Correlation of quote_signal with rate per mile, by month (train, label errors excluded):")
    ok = rpm.between(1.30, 4.00)
    print(train.loc[ok, "quote_signal"].groupby(month[ok]).corr(rpm[ok]).round(3).to_string())
    print("\nCorrelation of quote_signal with distance, by month (train + validation, no labels needed):")
    all_rows = pd.concat([train, val])
    q_dist = all_rows.groupby(all_rows["date"].dt.to_period("M")).apply(lambda g: g["quote_signal"].corr(g["distance"]))
    print(q_dist.round(3).to_string())
    print("-> quote_signal equals rate/mile in some months, its mirror in others, and is noise in Aug.")
    print("   Nov-Dec look like Aug (corr with distance ~0), so quote_signal is noise when it matters.")

    section("Market index drift (train + validation)")
    both = pd.concat([train, val])
    print(both.groupby(both["date"].dt.to_period("M"))["market_index"].mean().round(3))

    section("December chart inputs")
    missing_cols = sorted(set(val.columns) - set(dec.columns) - {"load_id"})
    print(f"Columns in validation but not in December inputs: {missing_cols}")
    lane = train[(train["pickup"] == dec["pickup"].iloc[0]) & (train["delivery"] == dec["delivery"].iloc[0])]
    print(f"Training rows on the December lane ({dec['pickup'].iloc[0]} -> {dec['delivery'].iloc[0]}): {len(lane)}")

    save_figures(train, rpm)


def save_figures(train: pd.DataFrame, rpm: pd.Series) -> None:
    """Save basic charts of price against distance and time."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(train["distance"], train["posted_rate"], s=3, alpha=0.3)
    ax.set(title="Posted rate vs distance", xlabel="Distance (miles)", ylabel="Posted rate ($)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_vs_distance.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rpm.clip(upper=6), bins=100)
    ax.set(title="Rate per mile (clipped at $6 for display)", xlabel="$ per mile", ylabel="Loads")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_per_mile_hist.png", dpi=150)
    plt.close(fig)

    monthly = rpm.groupby(train["date"].dt.to_period("M")).mean()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(monthly.index.astype(str), monthly.values, marker="o")
    ax.set(title="Mean rate per mile by month", xlabel="Month", ylabel="$ per mile")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "rate_per_mile_by_month.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved charts to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
