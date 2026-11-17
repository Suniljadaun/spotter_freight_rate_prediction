"""Compare other tree models with LightGBM on the same time-based test months."""

import time

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from preprocess import FEATURE_COLS, build_reference, filter_target_outliers, load_dataset, preprocess_data
from train import FOLDS, VALIDATION_LIKE_FOLD, fit_lgb, rate_metrics


def fit_xgb(X_tr, y_tr, w_tr, X_te, y_te, w_te):
    """Train an XGBoost model."""
    from xgboost import XGBRegressor

    model = XGBRegressor(
        objective="reg:squarederror",
        learning_rate=0.03, max_depth=6, min_child_weight=40, subsample=0.8, colsample_bytree=0.8,
        reg_lambda=1.0, n_estimators=3000, early_stopping_rounds=200, eval_metric="mae",
        enable_categorical=True, tree_method="hist", random_state=42,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr, eval_set=[(X_te, y_te)], sample_weight_eval_set=[w_te], verbose=False)
    return model


def fit_rf(X_tr, y_tr, w_tr, *_):
    """Train a random forest model."""
    X_tr = X_tr.assign(equipment=X_tr["equipment"].cat.codes)
    model = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=42)
    return model.fit(X_tr, y_tr, sample_weight=w_tr)


def fit_cat(X_tr, y_tr, w_tr, X_te, y_te, w_te):
    """Train a CatBoost model."""
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(loss_function="RMSE", learning_rate=0.1, depth=6, iterations=2000,
                              early_stopping_rounds=200, eval_metric="MAE", random_seed=42, verbose=False,
                              cat_features=["equipment"], allow_writing_files=False)
    to_str = lambda X: X.assign(equipment=X["equipment"].astype(str))
    model.fit(to_str(X_tr), y_tr, sample_weight=w_tr, eval_set=(to_str(X_te), y_te))
    model._needs_str = True
    return model


def predict(model, X):
    """Predict with any of the models, formatting the equipment column the way each one needs."""
    if isinstance(model, RandomForestRegressor):
        X = X.assign(equipment=X["equipment"].cat.codes)
    elif getattr(model, "_needs_str", False):
        X = X.assign(equipment=X["equipment"].astype(str))
    return model.predict(X)


def main() -> None:
    """Train every available model on each test month and print how they compare."""
    raw = filter_target_outliers(load_dataset("train_test"))
    ref = build_reference(raw, [load_dataset("validation")])
    data = preprocess_data(raw, ref).sort_values("date").reset_index(drop=True)
    X, y, dist = data[FEATURE_COLS], data["rate_per_mile"], data["distance"]

    fitters = {"LightGBM (final)": fit_lgb}
    for name, module, fit in [("XGBoost", "xgboost", fit_xgb), ("Random Forest", None, fit_rf), ("CatBoost", "catboost", fit_cat)]:
        try:
            if module:
                __import__(module)
            fitters[name] = fit
        except Exception as exc:
            print(f"{name} skipped: {type(exc).__name__}")

    rows = []
    for start, end in FOLDS:
        tr, te = data["date"] < start, (data["date"] >= start) & (data["date"] < end)
        label = f"test {start[:7]}"
        for name, fit in fitters.items():
            t0 = time.perf_counter()
            model = fit(X[tr], y[tr], dist[tr], X[te], y[te], dist[te])
            secs = time.perf_counter() - t0
            m = rate_metrics(data.loc[te, "posted_rate"], predict(model, X[te]) * dist[te])
            rows.append({"fold": label, "model": name, **m, "fit_sec": secs})
            print(f"{label}  {name:<17} MAE ${m['MAE']:7.2f}  RMSE ${m['RMSE']:7.2f}  ({secs:5.1f}s)")

    res = pd.DataFrame(rows)
    summary = res.groupby("model", sort=False)[["MAE", "RMSE", "MAPE_%", "fit_sec"]].mean()
    summary["Aug_MAE"] = res[res["fold"] == VALIDATION_LIKE_FOLD].set_index("model")["MAE"]
    print("\n=== Mean across 4 out-of-time folds ===")
    print(summary.round(2).to_string())


if __name__ == "__main__":
    main()
