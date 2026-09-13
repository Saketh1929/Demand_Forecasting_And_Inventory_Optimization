import pickle

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    DATA_PATH,
    FEATURE_COLUMNS_PATH,
    FORECAST_HORIZONS,
    LAG_FEATURES,
    PROCESSED_PATH,
    ROLLING_FEATURES,
    create_future_targets,
    fit_encoders,
    preprocess_input,
)


@pytest.fixture
def valid_row():
    raw = pd.read_csv(DATA_PATH)
    sample = raw.iloc[0]
    return {
        "store_id": sample["Store ID"],
        "product_id": sample["Product ID"],
        "category": sample["Category"],
        "region": sample["Region"],
        "weather_condition": sample["Weather Condition"],
        "inventory_level": 100,
        "units_sold": 100,
        "units_ordered": 100,
        "price": 100,
        "discount_rate": 10,
        "promotion_active": False,
        "competitor_pricing": 100,
        "epidemic": 0,
        "forecast_period": "1_week",
        "date": "2024-01-31",
    }


def test_preprocess_input_is_row_only_and_matches_feature_artifact(valid_row):
    features = preprocess_input(valid_row)
    feature_columns = pickle.load(FEATURE_COLUMNS_PATH.open("rb"))

    assert isinstance(features, np.ndarray)
    assert features.shape == (1, len(feature_columns))
    assert np.isfinite(features).all()


def test_preprocess_input_rejects_unknown_store(valid_row):
    valid_row["store_id"] = "UNKNOWN"

    with pytest.raises(ValueError, match="Unknown Store ID"):
        preprocess_input(valid_row)


def test_processed_csv_has_unique_tracking_and_model_columns():
    processed = pd.read_csv(PROCESSED_PATH)
    feature_columns = pickle.load(FEATURE_COLUMNS_PATH.open("rb"))

    assert not processed.columns.duplicated().any()
    assert processed.columns.tolist().count("forecast_horizon") == 1
    assert processed.columns.tolist().count("Target_Demand") == 1
    assert set(feature_columns).issubset(processed.columns)
    assert processed[feature_columns + ["Target_Demand"]].notna().all().all()


def test_future_targets_use_demand_not_units_sold():
    data = pd.DataFrame(
        {
            "Store ID": ["S1"] * 8,
            "Product ID": ["P1"] * 8,
            "Units Sold": [1] * 8,
            "Demand": list(range(10, 18)),
        }
    )

    targets = create_future_targets(data)
    one_day = targets[targets["forecast_period"] == "1_day"].iloc[0]
    one_week = targets[targets["forecast_period"] == "1_week"].iloc[0]

    assert one_day["Target_Demand"] == 11
    assert one_week["Target_Demand"] == sum(range(11, 18))
    assert set(FORECAST_HORIZONS.values()) == set(targets["forecast_horizon"])


def test_feature_names_include_historical_features():
    feature_columns = pickle.load(FEATURE_COLUMNS_PATH.open("rb"))

    assert set(LAG_FEATURES + ROLLING_FEATURES).issubset(feature_columns)
    assert "forecast_horizon" in feature_columns


def test_encoder_artifact_can_be_regenerated(tmp_path):
    raw = pd.read_csv(DATA_PATH)
    output_path = tmp_path / "encoders.pkl"
    encoders = fit_encoders(raw, output_path)

    assert output_path.exists()
    assert encoders["feature_order"] == pickle.load(output_path.open("rb"))["feature_order"]
