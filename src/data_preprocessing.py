"""Canonical demand preprocessing for M3 1-day-ahead XGBoost model training and inference.

The serving contract is:
    Observation at date t (exogenous factors & historical Demand)
        ↓
    preprocess_input(row, history_df=historical_demand)
        ↓
    PreprocessResult / NumPy feature vector (1, 34)
        ↓
    M3 1-day-ahead XGBoost model -> predicted Demand(t)
        ↓
    forecast_engine.py recursively generates multi-day rollouts
        ↓
    downstream agent -> inventory reorder & stockout risk

Key Architectural Rules & Temporal Alignment:
---------------------------------------------
1. Target Definition:
   Target = Demand(t).
   The model predicts Demand on date t using information available strictly
   before date t or independently known for date t.
2. Lag Features (Derived from Demand, NOT Units Sold):
   lag_1  = Demand(t-1)  [shift(1)]
   lag_7  = Demand(t-7)  [shift(7)]
   lag_14 = Demand(t-14) [shift(14)]
3. Rolling Features (Derived from historical Demand before date t):
   rolling_mean_7, rolling_std_7   -> computed over [t-7, t-1]
   rolling_mean_14, rolling_std_14 -> computed over [t-14, t-1]
   No target leakage: Demand(t) is strictly excluded from rolling windows.
4. Calendar Features:
   day_of_week, month, day_of_month, week, quarter, year, is_weekend.
   Derived directly from row's Date t in snake_case.
5. Business Features:
   Price, Discount, Promotion, Competitor Pricing, Epidemic.
   Exogenous variables known in advance. Endogenous variables (Units Sold,
   Inventory Level, Units Ordered) are completely excluded from model features.
6. Categorical & Identifier Encoding:
   Store ID and Product ID are numerically label-encoded (positions 1 and 2).
   Category, Region, Weather Condition, Seasonality are one-hot encoded into
   deterministic binary dummy indicators. Unknown categorical values are strictly rejected.
7. Exact Model Feature Contract:
   Exactly 34 features in authoritative FEATURE_ORDER.
8. History Requirement:
   At least 14 historical Demand observations are strictly required at inference time.
   Insufficient history raises a clear ValueError (zero fallback imputation).
"""

from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "preprocessed_sales_data.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "src" / "models" / "feature_columns.pkl"
ENCODERS_PATH = PROJECT_ROOT / "src" / "models" / "encoders.pkl"

REQUIRED_COLUMNS: list[str] = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
    "Epidemic", "Demand",
]

GROUP_COLUMNS: list[str] = ["Store ID", "Product ID"]

# Confirmed M3 Lag Specifications (from historical Demand)
LAG_FEATURES: list[str] = ["lag_1", "lag_7", "lag_14"]
MAX_LAG: int = 14

# Confirmed M3 Rolling Specifications (from historical Demand before date t)
ROLLING_FEATURES: list[str] = [
    "rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14",
]
MAX_ROLLING_WINDOW: int = 14

MIN_HISTORY: int = MAX_ROLLING_WINDOW

# Confirmed M3 Exogenous Business Features
BUSINESS_FEATURES: list[str] = [
    "Price", "Discount", "Promotion", "Competitor Pricing", "Epidemic",
]

# Confirmed M3 Calendar Features (snake_case)
CALENDAR_FEATURES: list[str] = [
    "day_of_week", "month", "day_of_month", "week", "quarter", "year", "is_weekend",
]

# Authoritative One-Hot Mapping Specifications
# All unique values present in the real sales_data.csv are included here.
# Adding a new category value here automatically expands FEATURE_ORDER and
# the inference validation — no other code changes are needed.
ONE_HOT_MAPPING: dict[str, list[str]] = {
    "Category": ["Clothing", "Electronics", "Furniture", "Groceries", "Toys"],
    "Region": ["East", "North", "South", "West"],
    "Weather Condition": ["Cloudy", "Rainy", "Snowy", "Sunny"],
    "Seasonality": ["Autumn", "Spring", "Summer", "Winter"],
}

# Authoritative 38-Feature Contract in Exact Order
# Generated from: 2 identifiers + 5 business + 7 calendar + 3 lags + 4 rolling
#                + 5 Category dummies + 4 Region dummies
#                + 4 Weather Condition dummies + 4 Seasonality dummies
# IMPORTANT: This list is the single source of truth for both training and inference.
# train_model.py imports it directly; do not define feature lists anywhere else.
FEATURE_ORDER: list[str] = [
    # Identifiers (label-encoded)
    "Store ID",
    "Product ID",
    # Exogenous business features
    "Price",
    "Discount",
    "Promotion",
    "Competitor Pricing",
    "Epidemic",
    # Calendar features (snake_case, derived from observation date t)
    "day_of_week",
    "month",
    "day_of_month",
    "week",
    "quarter",
    "year",
    "is_weekend",
    # Lag features (from historical Demand strictly before date t)
    "lag_1",
    "lag_7",
    "lag_14",
    # Rolling statistics (from historical Demand strictly before date t)
    "rolling_mean_7",
    "rolling_std_7",
    "rolling_mean_14",
    "rolling_std_14",
    # Category one-hot dummies (5 values: all present in sales_data.csv)
    "Category_Clothing",
    "Category_Electronics",
    "Category_Furniture",
    "Category_Groceries",
    "Category_Toys",
    # Region one-hot dummies (4 values: all present in sales_data.csv)
    "Region_East",
    "Region_North",
    "Region_South",
    "Region_West",
    # Weather Condition one-hot dummies (4 values: all present in sales_data.csv)
    "Weather Condition_Cloudy",
    "Weather Condition_Rainy",
    "Weather Condition_Snowy",
    "Weather Condition_Sunny",
    # Seasonality one-hot dummies (4 values: all present in sales_data.csv)
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

# In-memory cache for sales dataset to avoid repeated disk reads during inference
_CACHED_SALES_DF: pd.DataFrame | None = None


class PreprocessResult(np.ndarray):
    """Dual-interface container: acts as an np.ndarray for XGBoost and dict for agent."""

    feature_columns: list[str]
    processed_features: dict[str, float]
    metadata: dict[str, Any]

    def __new__(
        cls,
        array: np.ndarray,
        feature_columns: list[str] | None = None,
        processed_features: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PreprocessResult:
        obj = np.asarray(array, dtype=float).view(cls)
        obj.feature_columns = list(feature_columns or FEATURE_ORDER)
        obj.processed_features = dict(processed_features or {})
        obj.metadata = dict(metadata or {})
        return obj

    def __array_finalize__(self, obj: Any) -> None:
        if obj is None:
            return
        self.feature_columns = getattr(obj, "feature_columns", FEATURE_ORDER)
        self.processed_features = getattr(obj, "processed_features", {})
        self.metadata = getattr(obj, "metadata", {})

    @property
    def feature_vector(self) -> np.ndarray:
        return np.asarray(self, dtype=float)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_vector": np.asarray(self, dtype=float),
            "feature_columns": self.feature_columns,
            "processed_features": self.processed_features,
            "metadata": self.metadata,
        }

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            if key == "feature_vector":
                return np.asarray(self, dtype=float)
            if key == "feature_columns":
                return self.feature_columns
            if key == "processed_features":
                return self.processed_features
            if key == "metadata":
                return self.metadata
            if key in self.processed_features:
                return self.processed_features[key]
            raise KeyError(f"Key '{key}' not found in PreprocessResult.")
        return super().__getitem__(key)

    def keys(self) -> list[str]:
        return ["feature_vector", "feature_columns", "processed_features", "metadata"]

    def values(self) -> list[Any]:
        return [np.asarray(self, dtype=float), self.feature_columns, self.processed_features, self.metadata]

    def items(self) -> list[tuple[str, Any]]:
        return [(k, self[k]) for k in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def validate_data(df: pd.DataFrame, min_history: int = MIN_HISTORY) -> bool:
    """Validate schema, completeness, chronology, duplicates, and minimum history."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0].to_dict()
    if missing_values:
        raise ValueError(f"Required columns contain missing values: {missing_values}")

    try:
        parsed_dates = pd.to_datetime(df["Date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Date contains invalid values.") from error

    if df.duplicated(subset=[*GROUP_COLUMNS, "Date"]).any():
        raise ValueError("Duplicate Store ID/Product ID/Date rows found.")

    expected_order = df.assign(Date=parsed_dates).sort_values([*GROUP_COLUMNS, "Date"]).index
    if not expected_order.equals(df.index):
        raise ValueError("Data must be sorted by Store ID, Product ID, and Date.")

    group_sizes = df.groupby(GROUP_COLUMNS, sort=False).size()
    short_groups = group_sizes[group_sizes < min_history]
    if not short_groups.empty:
        examples = [f"{group}: {size}" for group, size in short_groups.head(5).items()]
        raise ValueError(
            f"Store-product groups need at least {min_history} rows for lag/rolling features; "
            f"short groups include {', '.join(examples)}."
        )
    return True


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features directly from observation Date t in snake_case."""
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
    """Add confirmed M3 lags from historical Demand.

    For row t predicting target Demand(t):
    - lag_1  = Demand(t-1)  -> shift(1)
    - lag_7  = Demand(t-7)  -> shift(7)
    - lag_14 = Demand(t-14) -> shift(14)
    """
    result = df.copy()
    grouped_demand = result.groupby(GROUP_COLUMNS, sort=False)["Demand"]
    result["lag_1"] = grouped_demand.shift(1)
    result["lag_7"] = grouped_demand.shift(7)
    result["lag_14"] = grouped_demand.shift(14)
    return result


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped rolling statistics (7 & 14 days) from historical Demand before date t.

    No future or current target leakage: Demand(t) is strictly excluded via shift(1).
    - rolling_mean_7, rolling_std_7   -> computed over [t-7, t-1]
    - rolling_mean_14, rolling_std_14 -> computed over [t-14, t-1]
    """
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


def fit_encoders(data: pd.DataFrame, output_path: Path = ENCODERS_PATH) -> dict[str, Any]:
    """Fit and persist deterministic label encoders (Store ID, Product ID) and one-hot schema."""
    store_ids = sorted(data["Store ID"].astype(str).unique())
    product_ids = sorted(data["Product ID"].astype(str).unique())

    encoders: dict[str, Any] = {
        "label": {
            "Store ID": {store: idx for idx, store in enumerate(store_ids)},
            "Product ID": {prod: idx for idx, prod in enumerate(product_ids)},
        },
        "one_hot": ONE_HOT_MAPPING,
        "feature_order": FEATURE_ORDER,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(encoders, file)
    return encoders


def encode_features(
    df: pd.DataFrame,
    encoders: dict[str, Any] | None = None,
    encoders_path: Path = ENCODERS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply deterministic label and one-hot encodings to the prepared dataset."""
    result = df.copy()
    if encoders is None:
        encoders = fit_encoders(result, encoders_path)

    # Encode Store ID and Product ID numerically
    for col in ("Store ID", "Product ID"):
        mapping = encoders["label"][col]
        values = result[col].astype(str)
        unknown = sorted(set(values) - set(mapping.keys()))
        if unknown:
            raise ValueError(f"Unknown {col} values during encoding: {unknown}")
        result[col] = values.map(mapping).astype(float)

    # Encode one-hot columns according to ONE_HOT_MAPPING (rejects unknown values)
    for col, categories in ONE_HOT_MAPPING.items():
        values = result[col].astype(str)
        unknown = sorted(set(values) - set(categories))
        if unknown:
            raise ValueError(f"Unknown {col} values during encoding: {unknown}. Expected one of {categories}")
        for category in categories:
            result[f"{col}_{category}"] = (values == category).astype(int)

    return result, encoders


def prepare_data(
    data_path: Path = DATA_PATH,
    processed_path: Path = PROCESSED_PATH,
    encoders_path: Path = ENCODERS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Generate the single-step 1-day-ahead training dataset and model artifacts for M3.

    Outputs:
        1. data/preprocessed_sales_data.csv: single dataset with Date + 34 features + Demand.
        2. models/encoders.pkl: label mappings for Store/Product and one-hot schema.
        3. models/feature_columns.pkl: exact ordered list of 34 model features.
    """
    raw = pd.read_csv(data_path)
    print(f"Original dataset shape: {raw.shape}")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    prepared = raw.sort_values([*GROUP_COLUMNS, "Date"]).reset_index(drop=True)
    validate_data(prepared)

    encoders = fit_encoders(prepared, encoders_path)

    prepared = create_calendar_features(prepared)
    prepared = create_lag_features(prepared)
    prepared = create_rolling_features(prepared)

    # Drop rows with boundary missing values:
    # First 14 rows per series (insufficient lag/rolling history before date t)
    # and any rows where target Demand is missing
    before_drop = len(prepared)
    boundary_cols = ["lag_14", "rolling_mean_14", "rolling_std_14", "Demand"]
    prepared = prepared.dropna(subset=boundary_cols).reset_index(drop=True)
    print(f"Rows dropped for boundary history/missing target: {before_drop - len(prepared)}")

    prepared, encoders = encode_features(prepared, encoders=encoders, encoders_path=encoders_path)

    # Verify all 34 features exist
    missing_features = [col for col in FEATURE_ORDER if col not in prepared.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    model_features = prepared[FEATURE_ORDER].apply(pd.to_numeric, errors="raise")
    if model_features.isna().any().any():
        raise ValueError("Final model features contain unexpected NaN values.")

    # Target is Demand
    output_columns = ["Date", *FEATURE_ORDER, "Demand"]
    output = prepared[output_columns].copy()
    output["Date"] = output["Date"].dt.strftime("%Y-%m-%d")

    dup_cols = output.columns[output.columns.duplicated()].tolist()
    if dup_cols:
        raise ValueError(f"Duplicate columns detected in prepared dataset: {dup_cols}")

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)

    output.to_csv(processed_path, index=False)
    with feature_columns_path.open("wb") as file:
        pickle.dump(FEATURE_ORDER, file)

    print(f"Final preprocessed dataset shape: {output.shape}")
    print(f"Total feature count: {len(FEATURE_ORDER)}")
    print(f"Feature columns: {FEATURE_ORDER}")
    print("Target column: Demand")
    return output, FEATURE_ORDER, encoders


def _load_encoders(encoders_path: Path) -> dict[str, Any]:
    if not encoders_path.exists():
        raise FileNotFoundError(
            f"Encoder artifact not found at {encoders_path}. Run 'python src/data_preprocessing.py' to generate it."
        )
    with encoders_path.open("rb") as file:
        return pickle.load(file)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(
            f"Required artifact not found at {path}. Run 'python src/data_preprocessing.py' to generate it."
        )
    with path.open("rb") as file:
        return pickle.load(file)


def _get_cached_sales_data(data_path: Path = DATA_PATH) -> pd.DataFrame | None:
    global _CACHED_SALES_DF
    if _CACHED_SALES_DF is not None:
        return _CACHED_SALES_DF
    if data_path.exists():
        try:
            df = pd.read_csv(data_path)
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            _CACHED_SALES_DF = df
            return _CACHED_SALES_DF
        except Exception as error:
            warnings.warn(f"Failed to cache sales data from {data_path}: {error}")
            return None
    return None


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


def _extract_history(
    store_id: str,
    product_id: str,
    target_date: pd.Timestamp,
    history_df: pd.DataFrame | np.ndarray | list[float] | None = None,
    data_path: Path = DATA_PATH,
) -> tuple[np.ndarray, str]:
    """Extract chronological historical Demand sequence strictly before target_date."""
    if history_df is not None:
        if isinstance(history_df, (list, np.ndarray)):
            arr = np.asarray(history_df, dtype=float).flatten()
            return arr, "passed_history_array"
        if isinstance(history_df, pd.Series):
            return history_df.dropna().to_numpy(dtype=float), "passed_history_series"
        if isinstance(history_df, pd.DataFrame) and not history_df.empty:
            source_df = history_df
            source_label = "passed_history_df"
        else:
            source_df = _get_cached_sales_data(data_path)
            source_label = "cached_disk_data"
    else:
        source_df = _get_cached_sales_data(data_path)
        source_label = "cached_disk_data"

    if source_df is None or source_df.empty:
        return np.array([], dtype=float), "no_history_available"

    store_col = "Store ID" if "Store ID" in source_df.columns else "store_id" if "store_id" in source_df.columns else None
    prod_col = "Product ID" if "Product ID" in source_df.columns else "product_id" if "product_id" in source_df.columns else None
    date_col = "Date" if "Date" in source_df.columns else "date" if "date" in source_df.columns else None
    demand_col = "Demand" if "Demand" in source_df.columns else "demand" if "demand" in source_df.columns else None

    # Handle history provided as a single Demand column without ID filtering
    if demand_col in source_df.columns and (store_col is None or prod_col is None):
        return source_df[demand_col].dropna().to_numpy(dtype=float), source_label

    if store_col not in source_df.columns or prod_col not in source_df.columns or demand_col not in source_df.columns:
        return np.array([], dtype=float), "missing_id_columns"

    subset = source_df[
        (source_df[store_col].astype(str) == str(store_id))
        & (source_df[prod_col].astype(str) == str(product_id))
    ]
    if subset.empty:
        return np.array([], dtype=float), "empty_group"

    subset = subset.copy()
    if date_col in subset.columns:
        if not pd.api.types.is_datetime64_any_dtype(subset[date_col]):
            subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")

        norm_target_date = pd.Timestamp(target_date).normalize()
        # Strictly prior to target_date (information before date t)
        prior = subset[subset[date_col].dt.normalize() < norm_target_date]
        if prior.empty:
            return np.array([], dtype=float), "no_prior_dates"
        series = prior.sort_values(date_col)[demand_col].dropna().to_numpy(dtype=float)
        return series, source_label
    else:
        series = subset[demand_col].dropna().to_numpy(dtype=float)
        return series, source_label


def preprocess_input(
    row: dict[str, Any],
    history_df: pd.DataFrame | np.ndarray | list[float] | None = None,
    return_dict: bool = False,
    encoders_path: Path = ENCODERS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    data_path: Path = DATA_PATH,
) -> PreprocessResult | dict[str, Any]:
    """Transform an inference observation at date t to predict Demand(t).

    Produces a NumPy feature vector of shape (1, 34) in exact FEATURE_ORDER.
    Strictly validates all categorical values and requires at least 14 days of prior Demand.
    """
    encoders = _load_encoders(encoders_path)
    feature_order = encoders.get("feature_order", FEATURE_ORDER)

    normalized_row: dict[str, Any] = {}
    for key, value in row.items():
        if key in INPUT_MAPPING:
            normalized_row[INPUT_MAPPING[key]] = value
        else:
            normalized_row[key] = value

    missing = [col for col in BUSINESS_FEATURES if col not in normalized_row]
    if missing:
        raise ValueError(f"Missing required business inputs: {missing}")

    raw_date = normalized_row.get("Date")
    if raw_date is None or pd.isna(raw_date):
        date_t = pd.Timestamp.now().normalize()
    else:
        try:
            date_t = pd.Timestamp(raw_date).normalize()
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid date value: {raw_date!r}") from error

    values: dict[str, Any] = {col: normalized_row[col] for col in BUSINESS_FEATURES}
    values.update(_calendar_features_for_date(date_t))

    store_id_str = str(normalized_row.get("Store ID", normalized_row.get("store_id", "")))
    product_id_str = str(normalized_row.get("Product ID", normalized_row.get("product_id", "")))

    if not store_id_str or not product_id_str:
        raise ValueError("Inference input must contain valid Store ID and Product ID.")

    # Encode Store ID and Product ID numerically
    if store_id_str not in encoders["label"]["Store ID"]:
        raise ValueError(f"Unknown Store ID: '{store_id_str}'")
    if product_id_str not in encoders["label"]["Product ID"]:
        raise ValueError(f"Unknown Product ID: '{product_id_str}'")

    values["Store ID"] = float(encoders["label"]["Store ID"][store_id_str])
    values["Product ID"] = float(encoders["label"]["Product ID"][product_id_str])

    # Validate and encode one-hot categories according to ONE_HOT_MAPPING
    # Rejects unknown categories for Category, Region, Weather Condition, Seasonality
    # 1. Category
    raw_cat = normalized_row.get("Category")
    if raw_cat is None or pd.isna(raw_cat) or str(raw_cat).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Category'. Expected one of {ONE_HOT_MAPPING['Category']}")
    cat_str = str(raw_cat).strip()
    if cat_str not in ONE_HOT_MAPPING["Category"]:
        raise ValueError(f"Unknown Category: '{cat_str}'. Expected one of {ONE_HOT_MAPPING['Category']}")

    # 2. Region
    raw_region = normalized_row.get("Region")
    if raw_region is None or pd.isna(raw_region) or str(raw_region).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Region'. Expected one of {ONE_HOT_MAPPING['Region']}")
    region_str = str(raw_region).strip()
    if region_str not in ONE_HOT_MAPPING["Region"]:
        raise ValueError(f"Unknown Region: '{region_str}'. Expected one of {ONE_HOT_MAPPING['Region']}")

    # 3. Weather Condition
    raw_weather = normalized_row.get("Weather Condition")
    if raw_weather is None or pd.isna(raw_weather) or str(raw_weather).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Weather Condition'. Expected one of {ONE_HOT_MAPPING['Weather Condition']}")
    weather_str = str(raw_weather).strip()
    if weather_str not in ONE_HOT_MAPPING["Weather Condition"]:
        raise ValueError(f"Unknown Weather Condition: '{weather_str}'. Expected one of {ONE_HOT_MAPPING['Weather Condition']}")

    # 4. Seasonality
    raw_seasonality = normalized_row.get("Seasonality")
    if raw_seasonality is None or pd.isna(raw_seasonality) or str(raw_seasonality).strip() == "":
        # Derive Seasonality from calendar month when not explicitly supplied.
        # Months 9/10/11 are Autumn — previously had a bug mapping them to Winter.
        month = date_t.month
        seasonality_str = (
            "Winter" if month in (12, 1, 2)
            else "Spring" if month in (3, 4, 5)
            else "Summer" if month in (6, 7, 8)
            else "Autumn"  # months 9, 10, 11
        )
    else:
        seasonality_str = str(raw_seasonality).strip()

    if seasonality_str not in ONE_HOT_MAPPING["Seasonality"]:
        raise ValueError(f"Unknown Seasonality: '{seasonality_str}'. Expected one of {ONE_HOT_MAPPING['Seasonality']}")

    # Construct one-hot features in values
    for category in ONE_HOT_MAPPING["Category"]:
        values[f"Category_{category}"] = float(cat_str == category)
    for region in ONE_HOT_MAPPING["Region"]:
        values[f"Region_{region}"] = float(region_str == region)
    for weather in ONE_HOT_MAPPING["Weather Condition"]:
        values[f"Weather Condition_{weather}"] = float(weather_str == weather)
    for season in ONE_HOT_MAPPING["Seasonality"]:
        values[f"Seasonality_{season}"] = float(seasonality_str == season)

    # Extract historical Demand strictly before date_t
    history_series, history_source = _extract_history(
        store_id=store_id_str,
        product_id=product_id_str,
        target_date=date_t,
        history_df=history_df,
        data_path=data_path,
    )

    if len(history_series) < MIN_HISTORY:
        raise ValueError(
            f"Insufficient history for Store ID '{store_id_str}' and Product ID '{product_id_str}' "
            f"before date {date_t.date()}. At least {MIN_HISTORY} historical Demand observations "
            f"are required to compute lag_14 and 14-day rolling statistics, but got {len(history_series)} "
            f"(history source: '{history_source}')."
        )

    # Historical demand series has at least MIN_HISTORY (14) observations
    history_status = f"complete ({len(history_series)} records via {history_source})"
    values["lag_1"] = float(history_series[-1])
    values["lag_7"] = float(history_series[-7])
    values["lag_14"] = float(history_series[-14])
    recent_7 = history_series[-7:]
    recent_14 = history_series[-14:]
    values["rolling_mean_7"] = float(np.mean(recent_7))
    values["rolling_std_7"] = float(np.std(recent_7, ddof=1) if len(recent_7) > 1 else 0.0)
    values["rolling_mean_14"] = float(np.mean(recent_14))
    values["rolling_std_14"] = float(np.std(recent_14, ddof=1) if len(recent_14) > 1 else 0.0)

    # Ensure all 34 model features exist
    missing_model_features = [col for col in feature_order if col not in values]
    if missing_model_features:
        raise ValueError(f"Could not construct model features: {missing_model_features}")

    # Build 2D numeric NumPy feature array
    try:
        numeric_row = [float(values[col]) for col in feature_order]
        feature_array = np.array([numeric_row], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Inference features must be numeric.") from error

    processed_features = {col: float(values[col]) for col in feature_order}
    metadata: dict[str, Any] = {
        "store_id": store_id_str,
        "product_id": product_id_str,
        "observation_date": str(date_t.date()),
        "target_date": str(date_t.date()),
        "history_status": history_status,
        "num_features": len(feature_order),
    }

    if return_dict:
        return {
            "feature_vector": feature_array,
            "feature_columns": feature_order,
            "processed_features": processed_features,
            "metadata": metadata,
        }

    return PreprocessResult(
        array=feature_array,
        feature_columns=feature_order,
        processed_features=processed_features,
        metadata=metadata,
    )


if __name__ == "__main__":
    prepare_data()
