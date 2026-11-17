"""Basic checks that the data pipeline and output files are correct."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from preprocess import (
    FEATURE_COLS, MARKET_COLS, QUOTE_COLS, MIN_RPM, MAX_RPM,
    build_reference, filter_target_outliers, load_dataset, preprocess_data,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    """Load and prepare the datasets once for all tests."""
    train = filter_target_outliers(load_dataset("train_test"))
    val = load_dataset("validation")
    dec = load_dataset("december_chart_inputs")
    ref = build_reference(train, [val])
    return train, val, dec, ref


def test_excluded_features_not_used():
    """The unreliable quote and market columns must not be model features."""
    for col in QUOTE_COLS + MARKET_COLS:
        assert col not in FEATURE_COLS


def test_outlier_filter_keeps_band(data):
    """Only rows with a believable price per mile are kept."""
    train = data[0]
    rpm = train["posted_rate"] / train["distance"]
    assert rpm.between(MIN_RPM, MAX_RPM).all()
    assert len(train) == 48_000 - 677


@pytest.mark.parametrize("which", [0, 1, 2])
def test_features_complete(data, which):
    """Every feature is filled in after cleaning, with no rows lost."""
    frame, ref = data[which], data[3]
    out = preprocess_data(frame, ref)
    assert len(out) == len(frame)
    assert out[FEATURE_COLS].isna().sum().sum() == 0


def test_weights_positive_and_flagged(data):
    """Weights end up positive, and bad or missing weights are flagged."""
    val, ref = data[1], data[3]
    out = preprocess_data(val, ref)
    assert (out["weight"] > 0).all()
    assert out["weight_was_negative"].sum() == (val["weight"] < 0).sum()
    assert out["weight_missing"].sum() == val["weight"].isna().sum()


def test_december_gets_coordinates(data):
    """December rows get city coordinates even though the file has none."""
    dec, ref = data[2], data[3]
    out = preprocess_data(dec, ref)
    assert out[["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]].nunique().eq(1).all()
    assert (out["circuity"] > 1).all()


def test_preprocess_does_not_mutate_input(data):
    """Cleaning does not change the original data."""
    val, ref = data[1], data[3]
    before = val.copy()
    preprocess_data(val, ref)
    pd.testing.assert_frame_equal(val, before)


@pytest.mark.skipif(not (ROOT / "validation_predictions.csv").is_file(), reason="run src/predict.py first")
def test_output_format():
    """The prediction file matches the required format and load order."""
    sub = pd.read_csv(ROOT / "validation_predictions.csv")
    template = pd.read_csv(next((ROOT / "data").glob("validation*template.csv")))
    assert list(sub.columns) == ["load_id", "predicted_rate"]
    assert sub["load_id"].tolist() == template["load_id"].tolist()
    assert np.isfinite(sub["predicted_rate"]).all() and (sub["predicted_rate"] > 0).all()
