"""Test suite for recursive forecast_engine.py under M3 38-feature contract.

Verifies the 9 required M3 recursive forecasting integration criteria:
1. One-day prediction.
2. Multi-day recursive prediction using the same model.
3. Correct history update with predicted Demand.
4. Correct lag calculation from Demand.
5. Correct rolling calculation from Demand.
6. Feature count consistency (exactly 38).
7. Feature ordering consistency (matches FEATURE_ORDER).
8. No use of future actual demand.
9. Requested forecast horizon is respected.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.forecasting import ForecastEngine, ForecastResult, forecast_demand
from src.data_preprocessing import (
    BUSINESS_FEATURES,
    CALENDAR_FEATURES,
    FEATURE_ORDER,
    LAG_FEATURES,
    ONE_HOT_MAPPING,
    ROLLING_FEATURES,
    fit_encoders,
    preprocess_input,
)


class MockConstantModel:
    """Mock model predicting a constant demand value."""

    def __init__(self, constant_value: float = 42.0) -> None:
        self.constant_value = constant_value
        self.call_count = 0
        self.recorded_features: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.recorded_features.append(np.array(X, copy=True))
        return np.array([self.constant_value], dtype=float)


class MockDynamicModel:
    """Mock model whose prediction depends on lag_1 feature."""

    def __init__(self, feature_columns: list[str], lag_1_multiplier: float = 1.1) -> None:
        self.feature_columns = feature_columns
        self.lag_1_idx = feature_columns.index("lag_1")
        self.lag_1_multiplier = lag_1_multiplier
        self.call_count = 0
        self.recorded_features: list[np.ndarray] = []

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.call_count += 1
        self.recorded_features.append(np.array(X, copy=True))
        lag_1_val = X[0, self.lag_1_idx]
        return np.array([lag_1_val * self.lag_1_multiplier], dtype=float)


@pytest.fixture
def test_env(tmp_path):
    """Generate isolated mock dataset, fitted encoders, and feature columns artifact."""
    records = []
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    categories = {"P0001": "Electronics", "P0002": "Toys"}

    for store in ["S001", "S002"]:
        for prod in ["P0001", "P0002"]:
            for i, date in enumerate(dates):
                records.append(
                    {
                        "Date": date.strftime("%Y-%m-%d"),
                        "Store ID": store,
                        "Product ID": prod,
                        "Category": categories[prod],
                        "Region": "North" if store == "S001" else "South",
                        "Inventory Level": 200 - i,
                        "Units Sold": 20 + (i % 5),
                        "Units Ordered": 100,
                        "Price": 99.99,
                        "Discount": 5.0 if i % 7 == 0 else 0.0,
                        "Weather Condition": "Sunny" if i % 2 == 0 else "Rainy",
                        "Promotion": 1 if i % 6 == 0 else 0,
                        "Competitor Pricing": 95.0,
                        "Seasonality": "Winter",
                        "Epidemic": 0,
                        "Demand": 50.0 + i * 2.0,
                    }
                )
    df = pd.DataFrame(records)
    data_path = tmp_path / "sales_data.csv"
    df.to_csv(data_path, index=False)

    encoders_path = tmp_path / "encoders.pkl"
    feature_columns_path = tmp_path / "feature_columns.pkl"

    encoders = fit_encoders(df, encoders_path)
    with feature_columns_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

    obs_row = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "weather_condition": "Sunny",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-31",
    }
    history = df[(df["Store ID"] == "S001") & (df["Product ID"] == "P0001")].copy()

    return {
        "df": df,
        "data_path": data_path,
        "encoders_path": encoders_path,
        "feature_columns_path": feature_columns_path,
        "feature_cols": FEATURE_ORDER,
        "obs_row": obs_row,
        "history": history,
    }


def test_1_one_day_prediction(test_env):
    """Test 1: Single-step prediction produces valid ForecastResult."""
    model = MockConstantModel(constant_value=45.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    result = engine.forecast(
        current_row=test_env["obs_row"],
        horizon=1,
        history_df=test_env["history"],
    )

    assert isinstance(result, ForecastResult)
    assert len(result.predictions) == 1
    assert result.predictions[0] == 45.0
    assert result.total_demand == 45.0
    assert result.mean_daily_demand == 45.0
    assert len(result.forecast_df) == 1
    assert result.forecast_df["date"].iloc[0] == "2023-01-31"
    assert model.call_count == 1


def test_2_multiday_recursive_prediction_uses_same_model(test_env):
    """Test 2: Multi-day rollout calls the SAME single-step model recursively."""
    model = MockConstantModel(constant_value=30.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    result = engine.forecast(
        current_row=test_env["obs_row"],
        horizon=7,
        history_df=test_env["history"],
    )

    assert result.horizon == 7
    assert len(result.predictions) == 7
    assert model.call_count == 7
    assert result.total_demand == 30.0 * 7
    assert result.mean_daily_demand == 30.0
    assert len(result.forecast_df) == 7
    assert list(result.forecast_df["step"]) == list(range(1, 8))


def test_3_correct_history_update(test_env):
    """Test 3: History buffer is updated with predicted Demand after each step."""
    model = MockConstantModel(constant_value=77.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    initial_len = len(test_env["history"])
    result = engine.forecast(
        current_row=test_env["obs_row"],
        horizon=5,
        history_df=test_env["history"],
    )

    assert len(result.history_df) == initial_len + 5
    last_5_demands = result.history_df["Demand"].iloc[-5:].tolist()
    assert last_5_demands == [77.0] * 5


def test_4_correct_lag_calculation_in_rollout(test_env):
    """Test 4: Predicted demand at step 1 becomes lag_1 for step 2."""
    model = MockConstantModel(constant_value=99.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    engine.forecast(
        current_row=test_env["obs_row"],
        horizon=3,
        history_df=test_env["history"],
    )

    lag_1_idx = FEATURE_ORDER.index("lag_1")
    step2_features = model.recorded_features[1]
    assert step2_features[0, lag_1_idx] == 99.0

    step3_features = model.recorded_features[2]
    assert step3_features[0, lag_1_idx] == 99.0


def test_5_correct_rolling_calculation_in_rollout(test_env):
    """Test 5: Rolling window incorporates recursively predicted demand."""
    model = MockConstantModel(constant_value=100.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    engine.forecast(
        current_row=test_env["obs_row"],
        horizon=8,
        history_df=test_env["history"],
    )

    rolling_mean_7_idx = FEATURE_ORDER.index("rolling_mean_7")
    step8_features = model.recorded_features[7]
    assert np.isclose(step8_features[0, rolling_mean_7_idx], 100.0)


def test_6_feature_count_consistency(test_env):
    """Test 6: Feature matrix has exactly 38 features at every recursive step."""
    horizon = 14
    model = MockConstantModel(constant_value=40.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    engine.forecast(
        current_row=test_env["obs_row"],
        horizon=horizon,
        history_df=test_env["history"],
    )

    assert len(model.recorded_features) == horizon
    for step_idx, feat in enumerate(model.recorded_features):
        assert feat.shape == (1, 38), f"Step {step_idx+1} had shape {feat.shape}"
        assert np.isfinite(feat).all(), f"Step {step_idx+1} contained NaN or Inf"


def test_7_feature_ordering_consistency(test_env):
    """Test 7: Feature column order matches FEATURE_ORDER strictly at every step."""
    horizon = 5
    model = MockConstantModel(constant_value=35.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    result = engine.forecast(
        current_row=test_env["obs_row"],
        horizon=horizon,
        history_df=test_env["history"],
    )

    assert result.feature_columns == FEATURE_ORDER


def test_8_no_use_of_future_actual_demand(test_env):
    """Test 8: Rollout uses ONLY historical data <= t and simulated predictions."""
    history = test_env["history"].copy()
    cutoff_date = pd.Timestamp("2023-01-30")
    history = history[pd.to_datetime(history["Date"]) <= cutoff_date]

    hist_last = history.sort_values("Date")["Demand"].iloc[-1]
    model = MockDynamicModel(feature_columns=FEATURE_ORDER, lag_1_multiplier=1.1)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    result = engine.forecast(
        current_row=test_env["obs_row"],
        horizon=3,
        history_df=history,
    )

    expected = [hist_last * 1.1, hist_last * 1.1**2, hist_last * 1.1**3]
    for actual, exp in zip(result.predictions, expected):
        assert np.isclose(actual, exp, rtol=1e-4)


def test_9_requested_forecast_horizon_is_respected(test_env):
    """Test 9: Engine respects varying horizons (1, 3, 7, 14, 30)."""
    model = MockConstantModel(constant_value=50.0)
    engine = ForecastEngine(
        model=model,
        encoders_path=test_env["encoders_path"],
        feature_columns_path=test_env["feature_columns_path"],
        data_path=test_env["data_path"],
    )

    for h in [1, 3, 7, 14, 30]:
        res = engine.forecast(
            current_row=test_env["obs_row"],
            horizon=h,
            history_df=test_env["history"],
        )
        assert res.horizon == h
        assert len(res.predictions) == h
        assert len(res.forecast_df) == h
        assert res.forecast_df["step"].iloc[-1] == h