from pathlib import Path
from typing import Any
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "preprocessed_sales_data.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.pkl"

REQUIRED_COLUMNS: list[str] = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
    "Epidemic", "Demand",
]

GROUP_COLUMNS: list[str] = ["Store ID", "Product ID"]

LAG_FEATURES: list[str] = ["lag_1", "lag_7", "lag_14"]
MAX_LAG: int = 14

ROLLING_FEATURES: list[str] = [
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
]
MAX_ROLLING_WINDOW: int = 14

MIN_HISTORY: int = MAX_ROLLING_WINDOW

BUSINESS_FEATURES: list[str] = [
    "Price", "Discount", "Promotion", "Competitor Pricing", "Epidemic",
]

CALENDAR_FEATURES: list[str] = [
    "day_of_week", "month", "day_of_month", "week", "quarter", "year", "is_weekend",
]

ONE_HOT_MAPPING: dict[str, list[str]] = {
    "Category": ["Clothing", "Electronics", "Furniture", "Groceries", "Toys"],
    "Region": ["East", "North", "South", "West"],
    "Weather Condition": ["Cloudy", "Rainy", "Snowy", "Sunny"],
    "Seasonality": ["Autumn", "Spring", "Summer", "Winter"],
}

FEATURE_ORDER: list[str] = [
    "Store ID",
    "Product ID",
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    "day_of_week",
    "month",
    "day_of_month",
    "week",
    "quarter",
    "year",
    "is_weekend",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    "Category_Clothing",
    "Category_Electronics",
    "Category_Furniture",
    "Category_Groceries",
    "Category_Toys",
    "Region_East",
    "Region_North",
    "Region_South",
    "Region_West",
    "Weather Condition_Cloudy",
    "Weather Condition_Rainy",
    "Weather Condition_Snowy",
    "Weather Condition_Sunny",
    "Seasonality_Autumn",
    "Seasonality_Spring",
    "Seasonality_Summer",
    "Seasonality_Winter",
]

FEATURE_COLUMNS = FEATURE_ORDER

INPUT_MAPPING: dict[str, str] = {
    "store_id": "Store ID",
    "product_id": "Product ID",
    "category": "Category",
    "region": "Region",
    "weather_condition": "Weather Condition",
    "inventory_level": "Inventory Level",
    "units_sold": "Units Sold",
    "units_ordered": "Units Ordered",
    "price": "Price",
    "discount_rate": "Discount",
    "discount": "Discount",
    "promotion_active": "Promotion",
    "promotion": "Promotion",
    "competitor_pricing": "Competitor Pricing",
    "epidemic": "Epidemic",
    "seasonality": "Seasonality",
    "date": "Date",
    "demand": "Demand",
}

def normalize_store_id(raw_store_id: Any) -> str:
    if raw_store_id is None:
        return ""
    value = str(raw_store_id).strip()
    if not value:
        return ""
    if value.lower().startswith("store "):
        try:
            return f"S{int(value.split()[-1]):03d}"
        except ValueError:
            return value
    return value

def normalize_product_id(raw_product_id: Any) -> str:
    if raw_product_id is None:
        return ""
    value = str(raw_product_id).strip()
    if not value:
        return ""
    if value.lower().startswith("product "):
        try:
            return f"P{int(value.split()[-1]):03d}"
        except ValueError:
            return value.upper()
    return value.upper()

def _calendar_features_for_date(date: pd.Timestamp) -> dict[str, int]:
    return {
        "day_of_week": int(date.dayofweek),
        "month": int(date.month),
        "day_of_month": int(date.day),
        "week": int(date.isocalendar().week),
        "quarter": int(date.quarter),
        "year": int(date.year),
        "is_weekend": int(date.dayofweek >= 5),
    }

def extract_date_features(date_value: Any) -> dict[str, Any]:
    date = pd.Timestamp(date_value) if date_value is not None and not pd.isna(date_value) else pd.Timestamp.now().normalize()
    return _calendar_features_for_date(pd.Timestamp(date).normalize())

def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    parsed_dates = pd.to_datetime(result["Date"], errors="raise")
    result["day_of_week"] = parsed_dates.dt.dayofweek
    result["month"] = parsed_dates.dt.month
    result["day_of_month"] = parsed_dates.dt.day
    result["week"] = parsed_dates.dt.isocalendar().week.astype(int)
    result["quarter"] = parsed_dates.dt.quarter
    result["year"] = parsed_dates.dt.year
    result["is_weekend"] = (result["day_of_week"] >= 5).astype(int)
    return result

def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped_demand = result.groupby(GROUP_COLUMNS, sort=False)["Demand"]
    result["lag_1"] = grouped_demand.shift(1)
    result["lag_7"] = grouped_demand.shift(7)
    result["lag_14"] = grouped_demand.shift(14)
    return result

def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    grouped_demand = result.groupby(GROUP_COLUMNS, sort=False)["Demand"]
    for window in (7, 14):
        result[f"rolling_mean_{window}"] = grouped_demand.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).mean()
        )
        result[f"rolling_std_{window}"] = grouped_demand.transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=window).std()
        )
    return result
