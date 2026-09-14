"""Comprehensive test suite for M3 38-feature demand preprocessing contract.

Covers all 19 required verification criteria:
1. Exact feature count = 38.
2. Exact feature order.
3. Target definition is Demand without shift(-1).
4. Lag correctness: lag_1 = Demand(t-1), lag_7 = Demand(t-7), lag_14 = Demand(t-14).
5. Rolling-window correctness: Demand(t-7)...Demand(t-1) and Demand(t-14)...Demand(t-1).
6. No Units Sold, Inventory Level, or Units Ordered in feature matrix.
7. Calendar features derived from observation date in snake_case.
8. Store/Product encoding consistency (numeric label encoding).
9. One-hot encoding consistency.
10. preprocess_input() returns shape (1, 38).
11. preprocess_input() dual interface.
12. preprocess_input() lag alignment.
13. Missing business features rejection.
14. Unknown Store/Product ID rejection.
15. Inference feature order strictly equals training order.
16. Unknown Category rejection (e.g. 'Clothing' -> ValueError).
17. Unknown Region, Weather Condition, Seasonality rejection.
18. Insufficient history rejection (< 14 records -> ValueError without fallback).
19. Exact 14 days history succeeds.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import (
    BUSINESS_FEATURES,
    CALENDAR_FEATURES,
    DATA_PATH,
    ENCODERS_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_ORDER,
    GROUP_COLUMNS,
    LAG_FEATURES,
    ONE_HOT_MAPPING,
    PROCESSED_PATH,
    ROLLING_FEATURES,
    PreprocessResult,
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    encode_features,
    fit_encoders,
    prepare_data,
    preprocess_input,
)


@pytest.fixture
def mock_dataset():
    """Synthetic dataset with 2 stores, 2 products, and 30 daily observations each."""
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
                        "Units Ordered": 50,
                        "Price": 99.99 if prod == "P0001" else 49.99,
                        "Discount": 5.0 if i % 5 == 0 else 0.0,
                        "Weather Condition": "Sunny" if i % 2 == 0 else "Rainy",
                        "Promotion": 1 if i % 7 == 0 else 0,
                        "Competitor Pricing": 95.0,
                        "Seasonality": "Winter",
                        "Epidemic": 0,
                        "Demand": 100.0 + i * 2.0,
                    }
                )
    return pd.DataFrame(records)


def test_1_exact_feature_count_is_38():
    """Verify FEATURE_ORDER contains exactly 38 features (expanded to cover all real data values)."""
    assert len(FEATURE_ORDER) == 38, (
        f"Expected 38 features, got {len(FEATURE_ORDER)}: {FEATURE_ORDER}"
    )


def test_2_exact_feature_order():
    """Verify the 38 features match the M3 expanded contract covering all real data values."""
    expected = [
        # Identifiers
        "Store ID",
        "Product ID",
        # Business
        "Price",
        "Discount",
        "Promotion",
        "Competitor Pricing",
        "Epidemic",
        # Calendar
        "day_of_week",
        "month",
        "day_of_month",
        "week",
        "quarter",
        "year",
        "is_weekend",
        # Lags
        "lag_1",
        "lag_7",
        "lag_14",
        # Rolling
        "rolling_mean_7",
        "rolling_std_7",
        "rolling_mean_14",
        "rolling_std_14",
        # Category (5 — includes Clothing)
        "Category_Clothing",
        "Category_Electronics",
        "Category_Furniture",
        "Category_Groceries",
        "Category_Toys",
        # Region (4 — includes East)
        "Region_East",
        "Region_North",
        "Region_South",
        "Region_West",
        # Weather Condition (4 — includes Cloudy)
        "Weather Condition_Cloudy",
        "Weather Condition_Rainy",
        "Weather Condition_Snowy",
        "Weather Condition_Sunny",
        # Seasonality (4 — includes Autumn)
        "Seasonality_Autumn",
        "Seasonality_Spring",
        "Seasonality_Summer",
        "Seasonality_Winter",
    ]
    assert FEATURE_ORDER == expected


def test_3_target_definition_is_demand_without_shift_minus_one(mock_dataset, tmp_path):
    """Verify target is Demand(t) with no shift(-1) and one row per valid date."""
    csv_path = tmp_path / "sales.csv"
    mock_dataset.to_csv(csv_path, index=False)
    proc_csv = tmp_path / "preprocessed.csv"
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"

    df_out, feature_cols, encoders = prepare_data(
        data_path=csv_path,
        processed_path=proc_csv,
        encoders_path=enc_path,
        feature_columns_path=col_path,
    )

    assert "Demand" in df_out.columns
    assert "Target_Demand" not in df_out.columns
    assert "forecast_horizon" not in df_out.columns

    s1_p1_out = df_out[(df_out["Store ID"] == 0.0) & (df_out["Product ID"] == 0.0)].reset_index(drop=True)
    raw_s1_p1 = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")].reset_index(drop=True)
    assert s1_p1_out.loc[0, "Date"] == "2023-01-15"
    assert s1_p1_out.loc[0, "Demand"] == raw_s1_p1.loc[14, "Demand"]


def test_4_lag_correctness_from_demand_not_units_sold(mock_dataset):
    """Verify lag_1 = Demand(t-1), lag_7 = Demand(t-7), lag_14 = Demand(t-14)."""
    lags = create_lag_features(mock_dataset)
    s1_p1 = lags[(lags["Store ID"] == "S001") & (lags["Product ID"] == "P0001")].reset_index(drop=True)

    assert s1_p1.loc[14, "lag_1"] == s1_p1.loc[13, "Demand"]
    assert s1_p1.loc[14, "lag_1"] != s1_p1.loc[13, "Units Sold"]
    assert s1_p1.loc[14, "lag_7"] == s1_p1.loc[7, "Demand"]
    assert s1_p1.loc[14, "lag_14"] == s1_p1.loc[0, "Demand"]


def test_5_rolling_correctness_excludes_current_demand(mock_dataset):
    """Verify rolling windows use only Demand(t-7)...Demand(t-1) and Demand(t-14)...Demand(t-1)."""
    rolling = create_rolling_features(mock_dataset)
    s1_p1 = rolling[(rolling["Store ID"] == "S001") & (rolling["Product ID"] == "P0001")].reset_index(drop=True)

    expected_mean_7 = s1_p1.loc[7:13, "Demand"].mean()
    assert np.isclose(s1_p1.loc[14, "rolling_mean_7"], expected_mean_7)

    mean_including_current = s1_p1.loc[8:14, "Demand"].mean()
    assert not np.isclose(s1_p1.loc[14, "rolling_mean_7"], mean_including_current)

    expected_mean_14 = s1_p1.loc[0:13, "Demand"].mean()
    assert np.isclose(s1_p1.loc[14, "rolling_mean_14"], expected_mean_14)


def test_6_no_units_sold_or_inventory_in_feature_matrix():
    """Verify Units Sold, Inventory Level, and Units Ordered are excluded from FEATURE_ORDER."""
    assert "Units Sold" not in FEATURE_ORDER
    assert "units_sold" not in FEATURE_ORDER
    assert "Inventory Level" not in FEATURE_ORDER
    assert "inventory_level" not in FEATURE_ORDER
    assert "Units Ordered" not in FEATURE_ORDER
    assert "units_ordered" not in FEATURE_ORDER


def test_7_calendar_features_derived_from_date():
    """Verify 7 calendar features are derived from observation date in snake_case."""
    df = pd.DataFrame({"Date": ["2023-01-01", "2023-01-02"]})
    cal = create_calendar_features(df)

    assert cal.loc[0, "day_of_week"] == 6
    assert cal.loc[0, "month"] == 1
    assert cal.loc[0, "day_of_month"] == 1
    assert cal.loc[0, "is_weekend"] == 1

    assert cal.loc[1, "day_of_week"] == 0
    assert cal.loc[1, "month"] == 1
    assert cal.loc[1, "day_of_month"] == 2
    assert cal.loc[1, "is_weekend"] == 0


def test_8_store_product_numeric_encoding(mock_dataset, tmp_path):
    """Verify Store ID and Product ID are numerically encoded as floats/ints in feature matrix."""
    enc_path = tmp_path / "encoders.pkl"
    encoded_df, encoders = encode_features(mock_dataset, encoders_path=enc_path)

    assert np.issubdtype(encoded_df["Store ID"].dtype, np.number)
    assert np.issubdtype(encoded_df["Product ID"].dtype, np.number)
    assert set(encoded_df["Store ID"].unique()) == {0.0, 1.0}
    assert set(encoded_df["Product ID"].unique()) == {0.0, 1.0}


def test_9_one_hot_encoding_deterministic(mock_dataset, tmp_path):
    """Verify one-hot encoding columns match the authoritative 13 dummy indicators."""
    enc_path = tmp_path / "encoders.pkl"
    encoded_df, encoders = encode_features(mock_dataset, encoders_path=enc_path)

    expected_one_hot = [
        "Category_Electronics", "Category_Furniture", "Category_Groceries", "Category_Toys",
        "Region_North", "Region_South", "Region_West",
        "Weather Condition_Rainy", "Weather Condition_Snowy", "Weather Condition_Sunny",
        "Seasonality_Spring", "Seasonality_Summer", "Seasonality_Winter",
    ]
    for col in expected_one_hot:
        assert col in encoded_df.columns
        assert set(encoded_df[col].unique()).issubset({0, 1})


def test_10_preprocess_input_returns_shape_1_38(mock_dataset, tmp_path):
    """Verify preprocess_input returns a NumPy array with exact shape (1, 38)."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    encoders = fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
        "date": "2023-01-30",
    }
    history = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")]

    vec = preprocess_input(
        row,
        history_df=history,
        encoders_path=enc_path,
        feature_columns_path=col_path,
    )

    assert isinstance(vec, np.ndarray)
    assert vec.shape == (1, 38)
    assert np.isfinite(vec).all()
    assert vec.feature_columns == FEATURE_ORDER


def test_11_preprocess_input_dual_interface(mock_dataset, tmp_path):
    """Verify PreprocessResult acts as an ndarray and provides dict-style access."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    encoders = fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
        "date": "2023-01-30",
    }
    history = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")]

    res = preprocess_input(
        row,
        history_df=history,
        encoders_path=enc_path,
        feature_columns_path=col_path,
    )

    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 38)
    assert res["feature_columns"] == FEATURE_ORDER
    assert res["metadata"]["store_id"] == "S001"
    assert res["metadata"]["num_features"] == 38


def test_12_preprocess_input_lag_alignment(mock_dataset, tmp_path):
    """Verify inference lag features use Demand strictly before target date t."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    encoders = fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
        "date": "2023-01-30",
    }
    history = mock_dataset[
        (mock_dataset["Store ID"] == "S001")
        & (mock_dataset["Product ID"] == "P0001")
        & (pd.to_datetime(mock_dataset["Date"]) < pd.Timestamp("2023-01-30"))
    ]

    res = preprocess_input(
        row,
        history_df=history,
        encoders_path=enc_path,
        feature_columns_path=col_path,
    )

    hist_demands = history.sort_values("Date")["Demand"].tolist()
    assert res["processed_features"]["lag_1"] == hist_demands[-1]
    assert res["processed_features"]["lag_7"] == hist_demands[-7]
    assert res["processed_features"]["lag_14"] == hist_demands[-14]


def test_13_missing_business_features_rejection(mock_dataset, tmp_path):
    """Verify missing required business inputs raise informative ValueError."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

    row = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "date": "2023-01-30",
    }
    with pytest.raises(ValueError, match="Missing required business inputs"):
        preprocess_input(row, encoders_path=enc_path, feature_columns_path=col_path)


def test_14_unknown_store_or_product_rejection(mock_dataset, tmp_path):
    """Verify unknown Store ID or Product ID raises ValueError."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

    row = {
        "store_id": "UNKNOWN_STORE_999",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "weather_condition": "Sunny",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-30",
    }
    with pytest.raises(ValueError, match="Unknown Store ID"):
        preprocess_input(row, encoders_path=enc_path, feature_columns_path=col_path)


def test_15_inference_feature_order_strictly_equals_training_order(mock_dataset, tmp_path):
    """Verify inference feature order matches FEATURE_ORDER strictly."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
        "date": "2023-01-30",
    }
    history = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")]
    res = preprocess_input(row, history_df=history, encoders_path=enc_path, feature_columns_path=col_path)

    assert list(res.feature_columns) == FEATURE_ORDER
    assert len(res.feature_columns) == 38


def test_16_unknown_category_rejection(mock_dataset, tmp_path):
    """Verify unknown Category (e.g. 'Clothing') raises ValueError."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

    row = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Space",  # Unknown Category
        "region": "North",
        "weather_condition": "Sunny",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-30",
    }
    history = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")]

    with pytest.raises(ValueError, match="Unknown Category: 'Space'"):
        preprocess_input(row, history_df=history, encoders_path=enc_path, feature_columns_path=col_path)


def test_17_unknown_region_and_weather_rejection(mock_dataset, tmp_path):
    """Verify unknown Region, Weather Condition, and Seasonality raise informative ValueErrors."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

    history = mock_dataset[(mock_dataset["Store ID"] == "S001") & (mock_dataset["Product ID"] == "P0001")]

    # Unknown Region
    row_region = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "Mars",
        "weather_condition": "Sunny",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-30",
    }
    with pytest.raises(ValueError, match="Unknown Region: 'Mars'"):
        preprocess_input(row_region, history_df=history, encoders_path=enc_path, feature_columns_path=col_path)

    # Unknown Weather Condition
    row_weather = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "weather_condition": "Foggy",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-30",
    }
    with pytest.raises(ValueError, match="Unknown Weather Condition: 'Foggy'"):
        preprocess_input(row_weather, history_df=history, encoders_path=enc_path, feature_columns_path=col_path)

    # Unknown Seasonality
    row_season = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "weather_condition": "Sunny",
        "seasonality": "Monsoon",
        "price": 99.99,
        "discount_rate": 0.0,
        "promotion_active": 0,
        "competitor_pricing": 95.0,
        "epidemic": 0,
        "date": "2023-01-30",
    }
    with pytest.raises(ValueError, match="Unknown Seasonality: 'Monsoon'"):
        preprocess_input(row_season, history_df=history, encoders_path=enc_path, feature_columns_path=col_path)


def test_18_insufficient_history_rejection_prevents_leakage(mock_dataset, tmp_path):
    """Verify that absent or insufficient history (< 14 records) raises ValueError without fallback."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
        "date": "2023-01-30",
        "Demand": 999.0,  # Must NOT be used as fallback
    }

    # 1. No history provided at all (empty)
    with pytest.raises(ValueError, match="Insufficient history"):
        preprocess_input(
            row,
            history_df=pd.DataFrame(),
            encoders_path=enc_path,
            feature_columns_path=col_path,
            data_path=tmp_path / "non_existent.csv",
        )

    # 2. Only 5 historical observations (< 14 required)
    partial_history = mock_dataset.head(5).copy()
    with pytest.raises(ValueError, match="Insufficient history"):
        preprocess_input(
            row,
            history_df=partial_history,
            encoders_path=enc_path,
            feature_columns_path=col_path,
            data_path=tmp_path / "non_existent.csv",
        )


def test_19_exact_14_days_history_succeeds(mock_dataset, tmp_path):
    """Verify that exactly 14 prior observations satisfies the contract and computes valid features."""
    enc_path = tmp_path / "encoders.pkl"
    col_path = tmp_path / "feature_columns.pkl"
    fit_encoders(mock_dataset, enc_path)
    with col_path.open("wb") as f:
        pickle.dump(FEATURE_ORDER, f)

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
    # Exactly 14 records: 2023-01-01 through 2023-01-14
    history_14 = mock_dataset[
        (mock_dataset["Store ID"] == "S001")
        & (mock_dataset["Product ID"] == "P0001")
        & (pd.to_datetime(mock_dataset["Date"]) < pd.Timestamp("2023-01-15"))
    ].head(14)
    assert len(history_14) == 14

    res = preprocess_input(
        row,
        history_df=history_14,
        encoders_path=enc_path,
        feature_columns_path=col_path,
    )

    assert res.shape == (1, 38)
    assert np.isfinite(res).all()
    assert res["metadata"]["history_status"].startswith("complete (14 records")
