"""Train the model, test it on later months, and save the final version."""

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from preprocess import FEATURE_COLS, MARKET_COLS, QUOTE_COLS, build_reference, filter_target_outliers, load_dataset, preprocess_data

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

FOLDS = [
    ("2025-07-01", "2025-08-01"),
    ("2025-08-01", "2025-09-01"),
    ("2025-09-01", "2025-10-01"),
    ("2025-10-01", "2025-11-01"),
]
VALIDATION_LIKE_FOLD = "test 2025-08"

LGB_PARAMS = dict(
    objective="huber",
    alpha=0.5,
    learning_rate=0.03,
    num_leaves=31,
    min_child_samples=40,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=1.0,
    n_estimators=3000,
    random_state=42,
    verbose=-1,
)


def rate_metrics(actual_rate, predicted_rate) -> dict:
    """Measure how far predicted prices are from the real ones."""
    actual_rate, predicted_rate = np.asarray(actual_rate), np.asarray(predicted_rate)
    return {
        "MAE": mean_absolute_error(actual_rate, predicted_rate),
        "RMSE": float(np.sqrt(mean_squared_error(actual_rate, predicted_rate))),
        "MAPE_%": float(np.mean(np.abs(predicted_rate - actual_rate) / actual_rate) * 100),
    }


class NaiveBaseline:
    """Simple benchmark that predicts the typical price per mile for each equipment type."""

    def fit(self, X, y, **_):
        """Learn the median price per mile for each equipment type."""
        self.rpm_ = y.groupby(X["equipment"], observed=True).median()
        self.global_ = y.median()
        return self

    def predict(self, X):
        """Predict price per mile from the equipment type."""
        return X["equipment"].map(self.rpm_).astype(float).fillna(self.global_).to_numpy()


def make_ridge():
    """Build a simple linear model to compare against."""
    numeric = [c for c in FEATURE_COLS if c != "equipment"]
    pre = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), ["equipment"])]
    )
    return make_pipeline(pre, Ridge(alpha=1.0))


def fit_lgb(X_tr, y_tr, w_tr, X_te=None, y_te=None, w_te=None, n_estimators=None):
    """Train a LightGBM model, stopping early if a test set is given."""
    params = dict(LGB_PARAMS)
    if n_estimators:
        params["n_estimators"] = n_estimators
    model = lgb.LGBMRegressor(**params)
    if X_te is not None:
        common = dict(sample_weight=w_tr, eval_sample_weight=[w_te], eval_metric="l1",
                      callbacks=[lgb.early_stopping(200, verbose=False)])
        try:
            model.fit(X_tr, y_tr, eval_X=(X_te,), eval_y=(y_te,), **common)
        except TypeError:
            model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], **common)
    else:
        model.fit(X_tr, y_tr, sample_weight=w_tr)
    return model


def train_model() -> None:
    """Run the full training process and save the model and its scores."""
    print("=== Loading and preprocessing ===")
    raw_train = filter_target_outliers(load_dataset("train_test"))
    val_features = load_dataset("validation")
    ref = build_reference(raw_train, [val_features])
    data = preprocess_data(raw_train, ref).sort_values("date").reset_index(drop=True)

    X, y, dist = data[FEATURE_COLS], data["rate_per_mile"], data["distance"]
    X_quote = data[FEATURE_COLS + QUOTE_COLS]
    X_mkt = data[FEATURE_COLS + MARKET_COLS]
    print(f"Rows after cleaning: {len(data):,}  |  features: {len(FEATURE_COLS)}")

    results, best_iters = [], []
    for start, end in FOLDS:
        tr = data["date"] < start
        te = (data["date"] >= start) & (data["date"] < end)
        label = f"test {start[:7]}"
        print(f"\n=== Fold: train < {start} ({tr.sum():,} rows) -> {label} ({te.sum():,} rows) ===")
        actual = data.loc[te, "posted_rate"]

        naive = NaiveBaseline().fit(X[tr], y[tr])
        ridge = make_ridge().fit(X[tr], y[tr], ridge__sample_weight=dist[tr])
        lgbm_quote = fit_lgb(X_quote[tr], y[tr], dist[tr], X_quote[te], y[te], dist[te])
        lgbm_mkt = fit_lgb(X_mkt[tr], y[tr], dist[tr], X_mkt[te], y[te], dist[te])
        lgbm = fit_lgb(X[tr], y[tr], dist[tr], X[te], y[te], dist[te])
        best_iters.append(lgbm.best_iteration_)

        candidates = [
            ("Naive (equipment rpm)", naive, X),
            ("Ridge", ridge, X),
            ("LightGBM + quote_signal", lgbm_quote, X_quote),
            ("LightGBM + market_index", lgbm_mkt, X_mkt),
            ("LightGBM (final)", lgbm, X),
        ]
        for name, model, feats in candidates:
            pred = model.predict(feats[te]) * dist[te]
            m = rate_metrics(actual, pred)
            results.append({"fold": label, "model": name, **m})
            print(f"  {name:<25} MAE ${m['MAE']:8.2f}  RMSE ${m['RMSE']:8.2f}  MAPE {m['MAPE_%']:5.2f}%")

    res_df = pd.DataFrame(results)
    summary = res_df.groupby("model", sort=False)[["MAE", "RMSE", "MAPE_%"]].mean()
    summary["Aug_MAE"] = res_df[res_df["fold"] == VALIDATION_LIKE_FOLD].set_index("model")["MAE"]
    print("\n=== Mean across folds (Aug_MAE = validation-like fold) ===")
    print(summary.round(2).to_string())

    importance = (
        pd.Series(lgbm.booster_.feature_importance("gain"), index=FEATURE_COLS)
        .sort_values(ascending=False)
        .pipe(lambda s: (s / s.sum() * 100).round(2))
    )
    print("\n=== LightGBM feature importance (% of gain) ===")
    print(importance.to_string())

    n_final = int(np.mean(best_iters) * 1.1)
    print(f"\n=== Refitting LightGBM on all data ({n_final} trees) ===")
    final_model = fit_lgb(X, y, dist, n_estimators=n_final)

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(
        {"model": final_model, "feature_cols": FEATURE_COLS, "reference": ref, "target": "rate_per_mile"},
        MODEL_DIR / "freight_model.pkl",
    )
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "metrics.json").write_text(
        json.dumps(
            {
                "folds": results,
                "mean": summary.round(3).reset_index().to_dict("records"),
                "feature_importance_pct": importance.to_dict(),
                "final_n_estimators": n_final,
            },
            indent=2,
        )
    )
    print(f"Saved {MODEL_DIR / 'freight_model.pkl'} and {REPORT_DIR / 'metrics.json'}")


if __name__ == "__main__":
    train_model()
