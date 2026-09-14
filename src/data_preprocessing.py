import numpy as np
import pandas as pd
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

CATEGORY_ORDER = ["Groceries", "Clothing", "Electronics", "Furniture", "Toys"]
REGION_ORDER = ["North", "South", "East", "West"]
WEATHER_ORDER = ["Sunny", "Cloudy", "Rainy", "Snowy"]
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]


def normalize_store_id(raw_store_id: Any) -> str:
    value = str(raw_store_id).strip()
    if value.lower().startswith("store "):
        try:
            store_number = int(value.split()[-1])
            return f"S{store_number:03d}"
        except ValueError:
            return value
    if value.upper().startswith("S") and value[1:].isdigit():
        return f"S{int(value[1:]):03d}"
    return value


def normalize_product_id(raw_product_id: Any) -> str:
    value = str(raw_product_id).strip()
    if value.upper().startswith("P") and value[1:].isdigit():
        return f"P{int(value[1:]):04d}"
    return value


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def infer_seasonality(date_value: Any) -> str:
    dt = pd.to_datetime(date_value)
    month = int(dt.month)
    if month in {12, 1, 2}:
        return "Winter"
    if month in {3, 4, 5}:
        return "Spring"
    if month in {6, 7, 8}:
        return "Summer"
    return "Autumn"


def extract_date_features(date_val: Any) -> Dict[str, int]:
    if isinstance(date_val, str):
        dt = pd.to_datetime(date_val)
    elif isinstance(date_val, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(date_val)
    else:
        dt = pd.to_datetime(str(date_val))

    return {
        "Year": int(dt.year),
        "Month": int(dt.month),
        "Day": int(dt.day),
        "DayOfWeek": int(dt.dayofweek),
        "WeekOfYear": int(dt.isocalendar().week),
        "Quarter": int(dt.quarter),
        "IsWeekend": int(dt.dayofweek >= 5),
    }


def _one_hot(value: str, allowed_values: Iterable[str]) -> Dict[str, float]:
    allowed = list(allowed_values)
    return {f"{value_name}_{option}": 1.0 if option == value else 0.0 for value_name, option in []}


def _one_hot_vector(value: str, prefix: str, allowed_values: Iterable[str]) -> list:
    options = list(allowed_values)
    return [1.0 if option == value else 0.0 for option in options]


def _store_code(store_id: str) -> float:
    clean = normalize_store_id(store_id)
    if clean.upper().startswith("S") and clean[1:].isdigit():
        return float(int(clean[1:]))
    return 1.0


def _product_code(product_id: str) -> float:
    clean = normalize_product_id(product_id)
    if clean.upper().startswith("P") and clean[1:].isdigit():
        return float(int(clean[1:]))
    return 1.0


def preprocess_input(row: Dict[str, Any]) -> Dict[str, Any]:
    raw_row = dict(row or {})
    date_value = raw_row.get("date") or datetime.now().strftime("%Y-%m-%d")
    date_features = extract_date_features(date_value)

    store_id = normalize_store_id(raw_row.get("store_id", "S001"))
    product_id = normalize_product_id(raw_row.get("product_id", "P0001"))
    category = str(raw_row.get("category", "Groceries"))
    region = str(raw_row.get("region", "North"))
    weather_condition = str(raw_row.get("weather_condition", "Sunny"))
    seasonality = str(raw_row.get("seasonality") or infer_seasonality(date_value))

    lag_1 = _safe_float(raw_row.get("lag_1", raw_row.get("units_sold", raw_row.get("demand", 0.0))), 0.0)
    lag_7 = _safe_float(raw_row.get("lag_7", raw_row.get("lag_1", lag_1)), lag_1)
    lag_14 = _safe_float(raw_row.get("lag_14", raw_row.get("lag_7", lag_7)), lag_7)
    rolling_mean_7 = _safe_float(raw_row.get("rolling_mean_7", raw_row.get("lag_1", lag_1)), lag_1)
    rolling_std_7 = _safe_float(raw_row.get("rolling_std_7", 0.0), 0.0)
    rolling_mean_14 = _safe_float(raw_row.get("rolling_mean_14", raw_row.get("lag_1", lag_1)), lag_1)
    rolling_std_14 = _safe_float(raw_row.get("rolling_std_14", 0.0), 0.0)

    processed_features = {
        "date": date_value,
        "store_id": store_id,
        "product_id": product_id,
        "category": category,
        "region": region,
        "weather_condition": weather_condition,
        "seasonality": seasonality,
        "Store_ID": _store_code(store_id),
        "Product_ID": _product_code(product_id),
        "Price": _safe_float(raw_row.get("price", raw_row.get("competitor_pricing", 25.0)), 25.0),
        "Discount": _safe_float(raw_row.get("discount", raw_row.get("discount_rate", 0.0)), 0.0),
        "Promotion": _safe_float(raw_row.get("promotion", raw_row.get("promotion_active", 0)), 0.0),
        "Competitor_Pricing": _safe_float(raw_row.get("competitor_pricing", 25.0), 25.0),
        "Epidemic": _safe_float(raw_row.get("epidemic", 0), 0.0),
        "lag_1": lag_1,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "rolling_mean_7": rolling_mean_7,
        "rolling_std_7": rolling_std_7,
        "rolling_mean_14": rolling_mean_14,
        "rolling_std_14": rolling_std_14,
    }
    processed_features.update(date_features)

    feature_names = [
        "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "Quarter", "IsWeekend",
        "Store_ID", "Product_ID", "Price", "Discount", "Promotion", "Competitor_Pricing", "Epidemic",
        "lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
    ]

    for option in CATEGORY_ORDER:
        feature_names.append(f"Category_{option}")
    for option in REGION_ORDER:
        feature_names.append(f"Region_{option}")
    for option in WEATHER_ORDER:
        feature_names.append(f"Weather_{option}")
    for option in SEASON_ORDER:
        feature_names.append(f"Seasonality_{option}")

    feature_values = [
        processed_features["Year"],
        processed_features["Month"],
        processed_features["Day"],
        processed_features["DayOfWeek"],
        processed_features["WeekOfYear"],
        processed_features["Quarter"],
        processed_features["IsWeekend"],
        processed_features["Store_ID"],
        processed_features["Product_ID"],
        processed_features["Price"],
        processed_features["Discount"],
        processed_features["Promotion"],
        processed_features["Competitor_Pricing"],
        processed_features["Epidemic"],
        processed_features["lag_1"],
        processed_features["lag_7"],
        processed_features["lag_14"],
        processed_features["rolling_mean_7"],
        processed_features["rolling_std_7"],
        processed_features["rolling_mean_14"],
        processed_features["rolling_std_14"],
    ]

    for option in CATEGORY_ORDER:
        feature_values.append(1.0 if option == category else 0.0)
    for option in REGION_ORDER:
        feature_values.append(1.0 if option == region else 0.0)
    for option in WEATHER_ORDER:
        feature_values.append(1.0 if option == weather_condition else 0.0)
    for option in SEASON_ORDER:
        feature_values.append(1.0 if option == seasonality else 0.0)

    if len(feature_values) != 38:
        raise ValueError(f"Expected 38 feature values but produced {len(feature_values)}")

    feature_vector = np.array(feature_values, dtype=float).reshape(1, -1)

    return {
        "raw_input": raw_row,
        "processed_features": processed_features,
        "feature_vector": feature_vector,
        "feature_names": feature_names,
        "used_encoder_pkl": False,
    }


def load_encoders():
    return None


__all__ = ["preprocess_input", "extract_date_features", "load_encoders", "normalize_store_id", "normalize_product_id"]
