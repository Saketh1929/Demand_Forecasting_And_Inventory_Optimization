"""10 additional edge-case tests for Demand Forecasting codebase.

Edge-cases covered:
  EC-01  horizon=0 raises ValueError in ForecastEngine.forecast()
  EC-02  no model → ValueError (no model provided anywhere)
  EC-03  negative raw prediction is clamped to 0.0
  EC-04  future_business_schedule overrides exogenous features correctly
  EC-05  ForecastResult.to_dict() contains all expected keys and correct types
  EC-06  PreprocessResult.get() returns default for missing key
  EC-07  preprocess_input() with return_dict=True returns plain dict, not PreprocessResult
  EC-08  rolling_std_7 = 0 when all 7 history values are identical
  EC-09  forecast_demand() functional wrapper produces identical result to ForecastEngine
  EC-10  store_id/product_id missing from current_row raises ValueError
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    FEATURE_ORDER,
    PreprocessResult,
    fit_encoders,
    preprocess_input,
)
from src.forecasting import ForecastEngine, ForecastResult, forecast_demand


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_env(tmp_path):
    """Minimal isolated environment: 2 stores x 2 products x 30 days."""
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
    fit_encoders(df, encoders_path)
    with feature_columns_path.open("wb") as fh:
        pickle.dump(FEATURE_ORDER, fh)

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
        "obs_row": obs_row,
        "history": history,
    }


class _ConstantModel:
    """Minimal mock that always predicts a fixed value."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.value], dtype=float)


class _NegativeModel:
    """Minimal mock that always returns a negative prediction."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([-99.5], dtype=float)


# ---------------------------------------------------------------------------
# EC-01  horizon = 0 raises ValueError
# ---------------------------------------------------------------------------


def test_ec01_horizon_zero_raises_value_error(base_env):
    """ForecastEngine.forecast() must raise ValueError for horizon < 1."""
    engine = ForecastEngine(
        model=_ConstantModel(10.0),
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        engine.forecast(
            current_row=base_env["obs_row"],
            horizon=0,
            history_df=base_env["history"],
        )


# ---------------------------------------------------------------------------
# EC-02  no model provided anywhere -> ValueError
# ---------------------------------------------------------------------------


def test_ec02_no_model_raises_value_error(base_env):
    """ForecastEngine.forecast() must raise ValueError when no model is available."""
    engine = ForecastEngine(
        model=None,
        model_path=None,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    engine.model = None
    with pytest.raises(ValueError, match="No forecasting model"):
        engine.forecast(
            current_row=base_env["obs_row"],
            horizon=1,
            history_df=base_env["history"],
        )


# ---------------------------------------------------------------------------
# EC-03  negative raw prediction is clamped to 0.0
# ---------------------------------------------------------------------------


def test_ec03_negative_prediction_clamped_to_zero(base_env):
    """The engine must enforce a non-negative demand constraint (clamp to 0.0)."""
    engine = ForecastEngine(
        model=_NegativeModel(),
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    result = engine.forecast(
        current_row=base_env["obs_row"],
        horizon=3,
        history_df=base_env["history"],
    )
    assert all(p == 0.0 for p in result.predictions), (
        f"Expected all 0.0 from negative model, got {result.predictions}"
    )
    assert result.total_demand == 0.0
    assert result.mean_daily_demand == 0.0


# ---------------------------------------------------------------------------
# EC-04  future_business_schedule overrides exogenous features
# ---------------------------------------------------------------------------


def test_ec04_future_schedule_overrides_business_features(base_env):
    """Known future business schedule must override Price at the scheduled date.

    Design note: the engine mutates current_step_row in-place, so schedule
    overrides persist into subsequent unscheduled steps (carry-forward behaviour).
    This test verifies:
      1. The override IS applied on the scheduled step.
      2. When a second schedule entry exists it correctly re-overrides on its step.
      3. A step with NO schedule entry carries the previously mutated value forward.
    """
    recorded_features: list[np.ndarray] = []

    class _RecordingModel:
        def predict(self, X: np.ndarray) -> np.ndarray:
            recorded_features.append(X.copy())
            return np.array([10.0], dtype=float)

    price_idx = FEATURE_ORDER.index("Price")
    promo_idx = FEATURE_ORDER.index("Promotion")

    # Two schedule entries: step 1 (2023-01-31) and step 3 (2023-02-02)
    schedule = [
        {"date": "2023-01-31", "price": 200.0, "promotion": 1},
        {"date": "2023-02-02", "price": 50.0,  "promotion": 0},
    ]

    engine = ForecastEngine(
        model=_RecordingModel(),
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    engine.forecast(
        current_row=base_env["obs_row"],
        horizon=3,
        history_df=base_env["history"],
        future_business_schedule=schedule,
    )

    # Step 1  (2023-01-31): schedule applies -> Price=200, Promotion=1
    assert recorded_features[0][0, price_idx] == pytest.approx(200.0), (
        "Schedule Price override not applied to step 1"
    )
    assert recorded_features[0][0, promo_idx] == pytest.approx(1.0), (
        "Schedule Promotion override not applied to step 1"
    )

    # Step 2 (2023-02-01): no schedule entry -> value carried forward from step 1
    assert recorded_features[1][0, price_idx] == pytest.approx(200.0), (
        "Engine should carry forward last mutated Price to unscheduled step 2"
    )

    # Step 3 (2023-02-02): second schedule entry -> Price=50, Promotion=0
    assert recorded_features[2][0, price_idx] == pytest.approx(50.0), (
        "Second schedule entry Price not applied to step 3"
    )
    assert recorded_features[2][0, promo_idx] == pytest.approx(0.0), (
        "Second schedule entry Promotion not applied to step 3"
    )


# ---------------------------------------------------------------------------
# EC-05  ForecastResult.to_dict() keys and value types
# ---------------------------------------------------------------------------


def test_ec05_forecast_result_to_dict_structure(base_env):
    """ForecastResult.to_dict() must return all expected keys with correct types."""
    engine = ForecastEngine(
        model=_ConstantModel(25.0),
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    result = engine.forecast(
        current_row=base_env["obs_row"],
        horizon=7,
        history_df=base_env["history"],
    )
    d = result.to_dict()

    required_keys = {
        "forecast_df", "predictions", "total_demand",
        "mean_daily_demand", "horizon", "store_id",
        "product_id", "start_date", "end_date", "feature_columns",
    }
    assert required_keys.issubset(d.keys()), f"Missing keys: {required_keys - d.keys()}"
    assert isinstance(d["forecast_df"], pd.DataFrame)
    assert isinstance(d["predictions"], list)
    assert isinstance(d["total_demand"], float)
    assert isinstance(d["mean_daily_demand"], float)
    assert isinstance(d["horizon"], int)
    assert isinstance(d["store_id"], str)
    assert isinstance(d["product_id"], str)
    assert isinstance(d["feature_columns"], list)


# ---------------------------------------------------------------------------
# EC-06  PreprocessResult.get() returns default for missing key
# ---------------------------------------------------------------------------


def test_ec06_preprocess_result_get_returns_default():
    """PreprocessResult.get() must return the default for non-existent keys."""
    arr = np.ones((1, 38), dtype=float)
    result = PreprocessResult(arr, feature_columns=FEATURE_ORDER)

    sentinel = object()
    assert result.get("non_existent_key_xyz", sentinel) is sentinel
    # Existing structural keys must still work
    assert result.get("feature_columns") == FEATURE_ORDER
    assert result.get("metadata") == {}


# ---------------------------------------------------------------------------
# EC-07  preprocess_input(return_dict=True) returns plain dict
# ---------------------------------------------------------------------------


def test_ec07_preprocess_input_return_dict_true(base_env):
    """When return_dict=True, preprocess_input must return a plain dict, not PreprocessResult."""
    row = dict(base_env["obs_row"])
    history = base_env["history"]

    result = preprocess_input(
        row,
        history_df=history,
        return_dict=True,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"
    assert not isinstance(result, PreprocessResult), (
        "return_dict=True must NOT return a PreprocessResult"
    )
    assert "feature_vector" in result
    assert "feature_columns" in result
    assert "processed_features" in result
    assert "metadata" in result
    assert result["feature_vector"].shape == (1, 38)


# ---------------------------------------------------------------------------
# EC-08  rolling_std_7 = 0 when all 7 history values are identical
# ---------------------------------------------------------------------------


def test_ec08_rolling_std_zero_for_constant_history(base_env):
    """rolling_std_7 must be 0.0 when all 7 most recent demand values are identical."""
    constant_demand = 42.0
    dates = pd.date_range("2023-01-01", periods=14, freq="D")
    constant_history = pd.DataFrame(
        {
            "Date": dates,
            "Store ID": "S001",
            "Product ID": "P0001",
            "Demand": [constant_demand] * 14,
            "Category": "Electronics",
            "Region": "North",
            "Weather Condition": "Sunny",
            "Seasonality": "Winter",
            "Inventory Level": 100,
            "Units Sold": 10,
            "Units Ordered": 20,
            "Price": 99.99,
            "Discount": 0.0,
            "Promotion": 0,
            "Competitor Pricing": 95.0,
            "Epidemic": 0,
        }
    )

    row = {
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
        "date": "2023-01-15",
    }
    result = preprocess_input(
        row,
        history_df=constant_history,
        return_dict=True,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
    )

    rolling_std_7 = result["processed_features"]["rolling_std_7"]
    assert rolling_std_7 == pytest.approx(0.0), (
        f"Expected rolling_std_7=0.0 for constant history, got {rolling_std_7}"
    )


# ---------------------------------------------------------------------------
# EC-09  forecast_demand() wrapper matches ForecastEngine output
# ---------------------------------------------------------------------------


def test_ec09_forecast_demand_wrapper_identical_to_engine(base_env):
    """forecast_demand() functional wrapper must produce identical results to ForecastEngine."""
    model_a = _ConstantModel(33.0)
    model_b = _ConstantModel(33.0)

    engine = ForecastEngine(
        model=model_a,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )
    result_engine = engine.forecast(
        current_row=base_env["obs_row"],
        horizon=5,
        history_df=base_env["history"],
    )

    result_func = forecast_demand(
        model=model_b,
        current_row=base_env["obs_row"],
        horizon=5,
        history_df=base_env["history"],
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )

    assert result_engine.predictions == result_func.predictions
    assert result_engine.total_demand == result_func.total_demand
    assert result_engine.start_date == result_func.start_date
    assert result_engine.end_date == result_func.end_date
    assert result_engine.horizon == result_func.horizon


# ---------------------------------------------------------------------------
# EC-10  store_id or product_id missing from current_row raises ValueError
# ---------------------------------------------------------------------------


def test_ec10_missing_store_or_product_id_raises_value_error(base_env):
    """current_row without Store ID or Product ID must raise a descriptive ValueError."""
    engine = ForecastEngine(
        model=_ConstantModel(10.0),
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
        data_path=base_env["data_path"],
    )

    # Missing store_id
    row_no_store = {
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
    with pytest.raises(ValueError, match="Store ID"):
        engine.forecast(
            current_row=row_no_store,
            horizon=1,
            history_df=base_env["history"],
        )

    # Missing product_id
    row_no_product = {
        "store_id": "S001",
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
    with pytest.raises(ValueError, match="Product ID"):
        engine.forecast(
            current_row=row_no_product,
            horizon=1,
            history_df=base_env["history"],
        )


# ---------------------------------------------------------------------------
# EC-11  FEATURE_ORDER length is 38
# ---------------------------------------------------------------------------


def test_ec11_feature_order_length_is_38():
    assert len(FEATURE_ORDER) == 38


# ---------------------------------------------------------------------------
# EC-12  Clothing, East, Cloudy, Autumn are valid in preprocess_input
# ---------------------------------------------------------------------------


def test_ec12_real_data_values_are_valid(base_env):
    row = dict(base_env["obs_row"])
    row["category"] = "Clothing"
    row["region"] = "East"
    row["weather_condition"] = "Cloudy"
    row["seasonality"] = "Autumn"
    
    # Should not raise ValueError
    res = preprocess_input(
        row,
        history_df=base_env["history"],
        return_dict=True,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
    )
    assert res["processed_features"]["Category_Clothing"] == 1.0
    assert res["processed_features"]["Region_East"] == 1.0
    assert res["processed_features"]["Weather Condition_Cloudy"] == 1.0
    assert res["processed_features"]["Seasonality_Autumn"] == 1.0


# ---------------------------------------------------------------------------
# EC-13  Autumn months (9,10,11) map to Autumn
# ---------------------------------------------------------------------------


def test_ec13_autumn_months_map_to_autumn(base_env):
    row = dict(base_env["obs_row"])
    # 9 = September
    row["date"] = "2023-09-15"
    row.pop("seasonality", None) # Force auto-infer
    
    res = preprocess_input(
        row,
        history_df=base_env["history"],
        return_dict=True,
        encoders_path=base_env["encoders_path"],
        feature_columns_path=base_env["feature_columns_path"],
    )
    assert res["processed_features"]["Seasonality_Autumn"] == 1.0
    assert res["processed_features"]["Seasonality_Winter"] == 0.0


# ---------------------------------------------------------------------------
# EC-14  Deployment model saved by train() has NO feature_order_ attribute
# ---------------------------------------------------------------------------


def test_ec14_train_saves_model_without_feature_order(monkeypatch, tmp_path):
    import src.model_traning.train_model as tm
    import joblib
    
    # Mock paths
    mock_model_path = tmp_path / "best_model.pkl"
    monkeypatch.setattr(tm, "MODEL_PATH", mock_model_path)
    monkeypatch.setattr(tm, "REPORT_PATH", tmp_path / "report.md")
    monkeypatch.setattr(tm, "RESIDUAL_PATH", tmp_path / "res.txt")
    
    # Mock data to return a dummy dataframe with Demand
    import pandas as pd
    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=300),
        "Demand": [10.0] * 300
    })
    for col in FEATURE_ORDER:
        df[col] = 1.0
        
    monkeypatch.setattr(tm, "load_training_data", lambda: df)
    
    # Train
    tm.train()
    
    # Load model
    saved_model = joblib.load(mock_model_path)
    assert not hasattr(saved_model, "feature_order_"), "feature_order_ must not be set on deployment model"


# ---------------------------------------------------------------------------
# EC-15  load_training_data() raises ValueError if CSV is missing Demand column
# ---------------------------------------------------------------------------


def test_ec15_load_training_data_missing_demand_raises(monkeypatch, tmp_path):
    import src.model_traning.train_model as tm
    import pandas as pd
    
    mock_csv_path = tmp_path / "preprocessed_sales_data.csv"
    monkeypatch.setattr(tm, "PROCESSED_DATA_PATH", mock_csv_path)
    
    # Write CSV missing 'Demand'
    df = pd.DataFrame({"Date": ["2023-01-01"], "Target_Demand": [10.0]})
    df.to_csv(mock_csv_path, index=False)
    
    import pytest
    with pytest.raises(ValueError, match="missing the target column"):
        tm.load_training_data()

