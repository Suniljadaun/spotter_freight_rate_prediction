"""Generate final predictions and run the scorer.

Run from the project root (after python src/train.py):
    python src/predict.py

Outputs:
    validation_predictions.csv                 load_id,predicted_rate for all 12,000 loads
    data/december-chart-inputs.csv             predicted_rate column filled (31 rows)
    scorer_results/candidate_december.png     created by score.py
"""
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
    for name in names:
        path = ROOT / "data" / name
        if path.is_file():
            return path
    raise FileNotFoundError(f"None of these files exist in data/: {names}")


def predict_rates(bundle: dict, df: pd.DataFrame) -> np.ndarray:
    """Clean + featurise with the training reference stats, predict rate per mile, convert to $."""
    features = preprocess_data(df, bundle["reference"])
    rpm = bundle["model"].predict(features[bundle["feature_cols"]])
    rates = rpm * features["distance"].to_numpy()
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Model produced non-finite or non-positive rates")
    return np.round(rates, 2)


def run_predictions() -> None:
    if not MODEL_PATH.is_file():
        sys.exit(f"Model not found at {MODEL_PATH}. Run: python src/train.py")
    bundle = joblib.load(MODEL_PATH)

    # 1. Validation predictions, written in the template's load_id order
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

    # 2. December chart inputs: fill predicted_rate in place, keeping the original columns and order
    print("\n=== Predicting December chart inputs ===")
    dec_path = find_data_file("december-chart-inputs.csv", "december_chart_inputs.csv")
    dec_raw = pd.read_csv(dec_path)  # re-read without date parsing so the file keeps its original format
    dec_raw["predicted_rate"] = predict_rates(bundle, load_dataset("december_chart_inputs"))
    dec_raw.to_csv(dec_path, index=False)
    print(f"Updated {dec_path.relative_to(ROOT)}: "
          f"${dec_raw['predicted_rate'].min():,.2f} - ${dec_raw['predicted_rate'].max():,.2f}")

    # 3. Run the scorer with the same Python interpreter
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
