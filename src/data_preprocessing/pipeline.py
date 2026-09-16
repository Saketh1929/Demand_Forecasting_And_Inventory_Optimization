from __future__ import annotations

import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data_preprocessing.features import (
    DATA_PATH, PROCESSED_PATH, FEATURE_COLUMNS_PATH, ENCODERS_PATH,
    GROUP_COLUMNS, MIN_HISTORY, BUSINESS_FEATURES, ONE_HOT_MAPPING,
    FEATURE_ORDER, INPUT_MAPPING,
    create_calendar_features, create_lag_features, create_rolling_features,
    _calendar_features_for_date
)
from src.data_preprocessing.validation import validate_data


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


def fit_encoders(data: pd.DataFrame, output_path: Path = ENCODERS_PATH) -> dict[str, Any]:
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
    result = df.copy()
    if encoders is None:
        encoders = fit_encoders(result, encoders_path)

    for col in ("Store ID", "Product ID"):
        mapping = encoders["label"][col]
        values = result[col].astype(str)
        unknown = sorted(set(values) - set(mapping.keys()))
        if unknown:
            raise ValueError(f"Unknown {col} values during encoding: {unknown}")
        result[col] = values.map(mapping).astype(float)

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
    raw = pd.read_csv(data_path)
    print(f"Original dataset shape: {raw.shape}")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    prepared = raw.sort_values([*GROUP_COLUMNS, "Date"]).reset_index(drop=True)
    validate_data(prepared)

    encoders = fit_encoders(prepared, encoders_path)

    prepared = create_calendar_features(prepared)
    prepared = create_lag_features(prepared)
    prepared = create_rolling_features(prepared)

    before_drop = len(prepared)
    boundary_cols = ["lag_14", "rolling_mean_14", "rolling_std_14", "Demand"]
    prepared = prepared.dropna(subset=boundary_cols).reset_index(drop=True)
    print(f"Rows dropped for boundary history/missing target: {before_drop - len(prepared)}")

    prepared, encoders = encode_features(prepared, encoders=encoders, encoders_path=encoders_path)

    missing_features = [col for col in FEATURE_ORDER if col not in prepared.columns]
    if missing_features:
        raise ValueError(f"Missing engineered features: {missing_features}")

    model_features = prepared[FEATURE_ORDER].apply(pd.to_numeric, errors="raise")
    if model_features.isna().any().any():
        raise ValueError("Final model features contain unexpected NaN values.")

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

def load_encoders(encoders_path: Path | str = ENCODERS_PATH) -> dict[str, Any]:
    return _load_encoders(Path(encoders_path))

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


def _extract_history(
    store_id: str,
    product_id: str,
    target_date: pd.Timestamp,
    history_df: pd.DataFrame | np.ndarray | list[float] | None = None,
    data_path: Path = DATA_PATH,
) -> tuple[np.ndarray, str]:
    if history_df is not None:
        if isinstance(history_df, (list, np.ndarray)):
            arr = np.asarray(history_df, dtype=float).flatten()
            return arr, "passed_history_array"
        if isinstance(history_df, pd.Series):
            return history_df.dropna().to_numpy(dtype=float), "passed_history_series"
        if isinstance(history_df, pd.DataFrame):
            if history_df.empty:
                return np.array([], dtype=float), "passed_history_df"
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

    if store_id_str not in encoders["label"]["Store ID"]:
        raise ValueError(f"Unknown Store ID: '{store_id_str}'")
    if product_id_str not in encoders["label"]["Product ID"]:
        raise ValueError(f"Unknown Product ID: '{product_id_str}'")

    values["Store ID"] = float(encoders["label"]["Store ID"][store_id_str])
    values["Product ID"] = float(encoders["label"]["Product ID"][product_id_str])

    raw_cat = normalized_row.get("Category")
    if raw_cat is None or pd.isna(raw_cat) or str(raw_cat).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Category'. Expected one of {ONE_HOT_MAPPING['Category']}")
    cat_str = str(raw_cat).strip()
    if cat_str not in ONE_HOT_MAPPING["Category"]:
        raise ValueError(f"Unknown Category: '{cat_str}'. Expected one of {ONE_HOT_MAPPING['Category']}")

    raw_region = normalized_row.get("Region")
    if raw_region is None or pd.isna(raw_region) or str(raw_region).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Region'. Expected one of {ONE_HOT_MAPPING['Region']}")
    region_str = str(raw_region).strip()
    if region_str not in ONE_HOT_MAPPING["Region"]:
        raise ValueError(f"Unknown Region: '{region_str}'. Expected one of {ONE_HOT_MAPPING['Region']}")

    raw_weather = normalized_row.get("Weather Condition")
    if raw_weather is None or pd.isna(raw_weather) or str(raw_weather).strip() == "":
        raise ValueError(f"Missing required categorical input: 'Weather Condition'. Expected one of {ONE_HOT_MAPPING['Weather Condition']}")
    weather_str = str(raw_weather).strip()
    if weather_str not in ONE_HOT_MAPPING["Weather Condition"]:
        raise ValueError(f"Unknown Weather Condition: '{weather_str}'. Expected one of {ONE_HOT_MAPPING['Weather Condition']}")

    raw_seasonality = normalized_row.get("Seasonality")
    if raw_seasonality is None or pd.isna(raw_seasonality) or str(raw_seasonality).strip() == "":
        month = date_t.month
        seasonality_str = (
            "Winter" if month in (12, 1, 2)
            else "Spring" if month in (3, 4, 5)
            else "Summer" if month in (6, 7, 8)
            else "Autumn"
        )
    else:
        seasonality_str = str(raw_seasonality).strip()

    if seasonality_str not in ONE_HOT_MAPPING["Seasonality"]:
        raise ValueError(f"Unknown Seasonality: '{seasonality_str}'. Expected one of {ONE_HOT_MAPPING['Seasonality']}")

    for category in ONE_HOT_MAPPING["Category"]:
        values[f"Category_{category}"] = float(cat_str == category)
    for region in ONE_HOT_MAPPING["Region"]:
        values[f"Region_{region}"] = float(region_str == region)
    for weather in ONE_HOT_MAPPING["Weather Condition"]:
        values[f"Weather Condition_{weather}"] = float(weather_str == weather)
    for season in ONE_HOT_MAPPING["Seasonality"]:
        values[f"Seasonality_{season}"] = float(seasonality_str == season)

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

    missing_model_features = [col for col in feature_order if col not in values]
    if missing_model_features:
        raise ValueError(f"Could not construct model features: {missing_model_features}")

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
