"""Use the trained model to make the final predictions and check them with the scorer."""

import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from preprocess import load_dataset, preprocess_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "freight_model.pkl"
PREDICTIONS_PATH = ROOT / "validation_predictions.csv"


def find_data_file(*names: str) -> Path:
    """Return the first of the given file names that exists in the data folder."""
    for name in names:
        path = ROOT / "data" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of these files exist in data/: {names}")


def predict_rates(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Predict the total price in dollars for each load."""
    features = preprocess_data(df, bundle["reference"])
    rpm = bundle["model"].predict(features[bundle["feature_cols"]])
    rates = rpm * features["distance"].to_numpy()
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Model produced non-finite or non-positive rates")
    return np.round(rates, 2)


def run_predictions() -> None:
    """Write the validation and December predictions, then run the scorer."""
    if not MODEL_PATH.is_file():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
    bundle = joblib.load(MODEL_PATH)

    print("=== Predicting validation loads ===")
    val = load_dataset("validation")
    val_pred = pd.DataFrame({"load_id": val["load_id"], "predicted_rate": predict_rates(bundle, val)})

    template = pd.read_csv(find_data_file("validation-predictions-template.csv", "validation_predictions_template.csv"))
    output = template[["load_id"]].merge(val_pred, on="load_id", how="left", validate="one_to_one")
    if output["predicted_rate"].isna().any():
        raise ValueError(f"{output['predicted_rate'].isna().sum()} template load_ids have no prediction")
    output.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Wrote {PREDICTIONS_PATH.name}: {len(output):,} rows, "
          f"mean ${output['predicted_rate'].mean():,.2f}")

    print("\n=== Predicting December chart inputs ===")
    dec_path = find_data_file("december-chart-inputs.csv", "december_chart_inputs.csv")
    dec_raw = pd.read_csv(dec_path)
    dec_raw["predicted_rate"] = predict_rates(bundle, load_dataset("december_chart_inputs"))
    dec_raw.to_csv(dec_path, index=False)
    print(f"Updated {dec_path.relative_to(ROOT)}: "
          f"${dec_raw['predicted_rate'].min():,.2f} - ${dec_raw['predicted_rate'].max():,.2f}")

    print("\n=== Running score.py ===")
    result = subprocess.run(
        [sys.executable, "score.py", "--predictions", PREDICTIONS_PATH.name,
         "--december-predictions", str(dec_path.relative_to(ROOT))],
        cwd=ROOT, capture_output=True, text=True,
    )
    print(result.stdout.strip())
    if result.returncode != 0:
        sys.exit(f"score.py failed:\n{result.stderr.strip() or result.stdout.strip()}")


if __name__ == "__main__":
    run_predictions()
