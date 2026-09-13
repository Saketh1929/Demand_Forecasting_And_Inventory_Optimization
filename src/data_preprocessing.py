"""Canonical demand preprocessing for training and model inference.

The serving contract is ``API row -> preprocess_input(row) -> NumPy feature
vector -> forecasting model -> predicted Demand``. This module does not
calculate inventory, reorder quantities, stockout risk, or API responses.

``Units Sold`` is used only for historical lag and rolling features. Future
``Demand`` is used only to build ``Target_Demand`` during training, preventing
future-target leakage into model inputs.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "sales_data.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "preprocessed_sales_data.csv"
FEATURE_COLUMNS_PATH = PROJECT_ROOT / "models" / "feature_columns.pkl"
ENCODERS_PATH = PROJECT_ROOT / "models" / "encoders.pkl"

FORECAST_HORIZONS = {
    "1_day": 1,
    "1_week": 7,
    "1_month": 30,
    "3_months": 90,
    "6_months": 180,
    "1_year": 365,
}

REQUIRED_COLUMNS = [
    "Date", "Store ID", "Product ID", "Category", "Region",
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Weather Condition", "Promotion", "Competitor Pricing", "Seasonality",
    "Epidemic", "Demand",
]
GROUP_COLUMNS = ["Store ID", "Product ID"]
LAG_FEATURES = [
    "lag_1_day", "lag_7_day", "lag_30_day", "lag_90_day", "lag_180_day", "lag_365_day",
]
ROLLING_FEATURES = ["rolling_mean_7", "rolling_std_7", "rolling_mean_14", "rolling_std_14"]
MAX_LAG = 365
MIN_HISTORY = MAX_LAG + 1
BUSINESS_FEATURES = [
    "Inventory Level", "Units Sold", "Units Ordered", "Price", "Discount",
    "Promotion", "Competitor Pricing", "Epidemic",
]
CALENDAR_FEATURES = [
    "Year", "Month", "Day", "DayOfWeek", "WeekOfYear", "Quarter", "IsWeekend",
]
LABEL_FEATURES = ["Store ID", "Product ID"]
ONE_HOT_FEATURES = ["Category", "Region", "Weather Condition", "Seasonality"]

INPUT_MAPPING = {
    "store_id": "Store ID", "product_id": "Product ID", "category": "Category",
    "region": "Region", "weather_condition": "Weather Condition",
    "inventory_level": "Inventory Level", "units_sold": "Units Sold",
    "units_ordered": "Units Ordered", "price": "Price", "discount_rate": "Discount",
    "promotion_active": "Promotion", "competitor_pricing": "Competitor Pricing",
    "epidemic": "Epidemic", "seasonality": "Seasonality", "date": "Date",
}


def validate_data(df: pd.DataFrame, min_history: int = MIN_HISTORY) -> bool:
    """Validate required schema, values, chronology, duplicates, and history."""
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
            f"Store-product groups need at least {min_history} rows for lag features; "
            f"short groups include {', '.join(examples)}."
        )
    return True


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the calendar features established in notebook 03."""
    result = df.copy()
    result["Date"] = pd.to_datetime(result["Date"], errors="raise")
    result["Year"] = result["Date"].dt.year
    result["Month"] = result["Date"].dt.month
    result["Day"] = result["Date"].dt.day
    result["DayOfWeek"] = result["Date"].dt.dayofweek
    result["WeekOfYear"] = result["Date"].dt.isocalendar().week.astype(int)
    result["Quarter"] = result["Date"].dt.quarter
    result["IsWeekend"] = (result["DayOfWeek"] >= 5).astype(int)
    return result


def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add grouped Units Sold lags without crossing store-product groups."""
    result = df.copy()
    grouped_units_sold = result.groupby(GROUP_COLUMNS, sort=False)["Units Sold"]
    for feature_name in LAG_FEATURES:
        lag = int(feature_name.split("_")[1])
        result[feature_name] = grouped_units_sold.shift(lag)
    return result


def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add shifted grouped rolling statistics from Units Sold."""
    result = df.copy()
    previous_units_sold = result.groupby(GROUP_COLUMNS, sort=False)["Units Sold"].shift(1)
    grouped_previous = previous_units_sold.groupby(
        [result["Store ID"], result["Product ID"]], sort=False
    )
    for window in (7, 14):
        result[f"rolling_mean_{window}"] = grouped_previous.transform(
            lambda values: values.rolling(window=window, min_periods=window).mean()
        )
        result[f"rolling_std_{window}"] = grouped_previous.transform(
            lambda values: values.rolling(window=window, min_periods=window).std()
        )
    return result


def create_future_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Create future Demand targets without exposing them as input features."""
    grouped_demand = df.groupby(GROUP_COLUMNS, sort=False)["Demand"]
    horizon_frames = []
    for forecast_period, horizon in FORECAST_HORIZONS.items():
        future_demand = grouped_demand.transform(
            lambda values: _future_sum(values, horizon)
        )
        horizon_df = df.copy()
        horizon_df["forecast_period"] = forecast_period
        horizon_df["forecast_horizon"] = horizon
        horizon_df["Target_Demand"] = future_demand
        horizon_frames.append(horizon_df)
    return pd.concat(horizon_frames, ignore_index=True)


def _future_sum(values: pd.Series, horizon: int) -> pd.Series:
    """Sum the next horizon observations while preserving the original index."""
    # Exclude the final shifted value, which has no future observation.
    shifted = values.iloc[1:].to_numpy(dtype=float)
    result = np.full(len(values), np.nan, dtype=float)
    if len(shifted) >= horizon:
        cumulative = np.concatenate(([0.0], np.cumsum(shifted)))
        result[: len(shifted) - horizon + 1] = (
            cumulative[horizon:] - cumulative[:-horizon]
        )
    return pd.Series(result, index=values.index)


def _feature_order(encoders: dict[str, Any]) -> list[str]:
    return [
        *BUSINESS_FEATURES,
        *CALENDAR_FEATURES,
        *LAG_FEATURES,
        *ROLLING_FEATURES,
        "forecast_horizon",
        "Store_ID_encoded",
        "Product_ID_encoded",
        *[
            f"{column}_{value}"
            for column in ONE_HOT_FEATURES
            for value in encoders["one_hot"][column]
        ],
    ]


def fit_encoders(data: pd.DataFrame, output_path: Path = ENCODERS_PATH) -> dict[str, Any]:
    """Fit and persist encoders regenerated by the training pipeline."""
    encoders: dict[str, Any] = {"label": {}, "one_hot": {}}
    for column in ("Store ID", "Product ID"):
        values = sorted(data[column].astype(str).unique())
        encoders["label"][column] = {value: index for index, value in enumerate(values)}
    for column in ("Category", "Region", "Weather Condition", "Seasonality"):
        encoders["one_hot"][column] = sorted(data[column].astype(str).unique())
    encoders["feature_order"] = _feature_order(encoders)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        pickle.dump(encoders, file)
    return encoders


def encode_features(
    df: pd.DataFrame,
    encoders: dict[str, Any] | None = None,
    encoders_path: Path = ENCODERS_PATH,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply the persisted label and one-hot mappings to a prepared dataset."""
    result = df.copy()
    if encoders is None:
        encoders = fit_encoders(result, encoders_path)

    for column in LABEL_FEATURES:
        mapping = encoders["label"][column]
        values = result[column].astype(str)
        unknown = sorted(set(values) - set(mapping))
        if unknown:
            raise ValueError(f"Unknown {column} values: {unknown}")
        result[f"{column.replace(' ', '_')}_encoded"] = values.map(mapping)

    for column in ONE_HOT_FEATURES:
        categories = encoders["one_hot"][column]
        values = result[column].astype(str)
        unknown = sorted(set(values) - set(categories))
        if unknown:
            raise ValueError(f"Unknown {column} values: {unknown}")
        for category in categories:
            result[f"{column}_{category}"] = (values == category).astype(int)

    return result, encoders


def prepare_data(
    data_path: Path = DATA_PATH,
    processed_path: Path = PROCESSED_PATH,
    encoders_path: Path = ENCODERS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """Run training preprocessing and regenerate all three model artifacts.

    Running ``python src/data_preprocessing.py`` writes the processed CSV,
    ``models/encoders.pkl``, and ``models/feature_columns.pkl``.
    """
    raw = pd.read_csv(data_path)
    print(f"Original dataset shape: {raw.shape}")
    raw["Date"] = pd.to_datetime(raw["Date"], errors="raise")
    prepared = raw.sort_values([*GROUP_COLUMNS, "Date"]).reset_index(drop=True)
    validate_data(prepared)

    prepared = create_calendar_features(prepared)
    prepared = create_lag_features(prepared)
    prepared = create_rolling_features(prepared)
    prepared = create_future_targets(prepared)

    before_drop = len(prepared)
    prepared = prepared.dropna(
        subset=[*LAG_FEATURES, *ROLLING_FEATURES, "Target_Demand"]
    ).reset_index(drop=True)
    print(f"Rows removed for unavailable history/future targets: {before_drop - len(prepared)}")

    prepared, encoders = encode_features(prepared, encoders_path=encoders_path)
    feature_columns = _feature_order(encoders)
    missing_features = [column for column in feature_columns if column not in prepared]
    if missing_features:
        raise ValueError(f"Could not create model features: {missing_features}")

    model_features = prepared[feature_columns].apply(pd.to_numeric, errors="raise")
    if model_features.isna().any().any():
        raise ValueError("Final model features contain missing values.")

    tracking_columns = ["Store ID", "Product ID", "Date", "forecast_period"]
    output_columns = list(dict.fromkeys(tracking_columns + feature_columns + ["Target_Demand"]))
    output = prepared[output_columns]
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    encoders_path.parent.mkdir(parents=True, exist_ok=True)
    feature_columns_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(processed_path, index=False)
    with feature_columns_path.open("wb") as file:
        pickle.dump(feature_columns, file)

    print(f"Final preprocessed dataset shape: {output.shape}")
    print(f"Final feature columns: {feature_columns}")
    print("Target column: Target_Demand")
    print(f"Processed CSV: {processed_path}")
    print(f"Encoder artifact: {encoders_path}")
    print(f"Feature-order artifact: {feature_columns_path}")
    return output, feature_columns, encoders


def _load_encoders(encoders_path: Path) -> dict[str, Any]:
    if not encoders_path.exists():
        raise FileNotFoundError(f"Encoder artifact not found at {encoders_path}.")
    with encoders_path.open("rb") as file:
        return pickle.load(file)


def _load_pickle(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required preprocessing artifact not found at {path}.")
    with path.open("rb") as file:
        return pickle.load(file)


def _calendar_features_for_date(date: pd.Timestamp) -> dict[str, int]:
    return {
        "Year": date.year,
        "Month": date.month,
        "Day": date.day,
        "DayOfWeek": date.dayofweek,
        "WeekOfYear": int(date.isocalendar().week),
        "Quarter": date.quarter,
        "IsWeekend": int(date.dayofweek >= 5),
    }


def preprocess_input(
    row: dict[str, Any],
) -> np.ndarray:
    """Transform one API row into the persisted training feature order."""
    forecast_period = row.get("forecast_period")
    forecast_horizon = row.get("forecast_horizon")
    encoders = _load_encoders(ENCODERS_PATH)
    feature_columns = _load_pickle(FEATURE_COLUMNS_PATH)
    row = {dataset_name: row[api_name] for api_name, dataset_name in INPUT_MAPPING.items() if api_name in row}
    required_inputs = [column for column in REQUIRED_COLUMNS if column not in {"Date", "Demand", "Seasonality"}]
    missing = [column for column in required_inputs if column not in row]
    if missing:
        raise ValueError(f"Missing input fields: {missing}")

    if forecast_period is not None:
        period = forecast_period
        if period not in FORECAST_HORIZONS:
            raise ValueError(f"Invalid forecast_period. Choose one of: {list(FORECAST_HORIZONS)}")
        horizon = FORECAST_HORIZONS[period]
    elif forecast_horizon is not None:
        try:
            horizon = int(forecast_horizon)
        except (TypeError, ValueError) as error:
            raise ValueError("forecast_horizon must be one of the supported horizon values.") from error
        if horizon not in FORECAST_HORIZONS.values():
            raise ValueError(f"forecast_horizon must be one of: {list(FORECAST_HORIZONS.values())}")
    else:
        raise ValueError("forecast_period or forecast_horizon is required.")

    try:
        date = pd.Timestamp(row.get("Date", pd.Timestamp.today().normalize()))
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid date value: {row.get('Date')!r}") from error

    values: dict[str, Any] = {column: row[column] for column in required_inputs}
    values.update(_calendar_features_for_date(date))
    seasonality = row.get("Seasonality")
    if seasonality is None or pd.isna(seasonality):
        seasonality = "Winter" if date.month in (12, 1, 2) else "Spring" if date.month in (3, 4, 5) else "Summer" if date.month in (6, 7, 8) else "Autumn"
    values["Seasonality"] = seasonality

    for column in LABEL_FEATURES:
        mapping = encoders["label"][column]
        category = str(values[column])
        if category not in mapping:
            raise ValueError(f"Unknown {column}: {category}")

    history = pd.read_csv(DATA_PATH)
    if not set([*GROUP_COLUMNS, "Date", "Units Sold"]).issubset(history.columns):
        raise ValueError("The raw sales dataset must contain Store ID, Product ID, Date, and Units Sold.")
    history = history[
        (history["Store ID"].astype(str) == str(values["Store ID"]))
        & (history["Product ID"].astype(str) == str(values["Product ID"]))
    ].copy()
    history["Date"] = pd.to_datetime(history["Date"], errors="raise")
    history = history[history["Date"] < date]
    demand_history = history.sort_values("Date")["Units Sold"].dropna().to_numpy(dtype=float)
    if len(demand_history) < MIN_HISTORY:
        raise ValueError(f"At least {MIN_HISTORY} historical Units Sold values are required for this Store/Product.")
    for name in LAG_FEATURES:
        values[name] = demand_history[-int(name.split("_")[1])]
    for window in (7, 14):
        recent = demand_history[-window:]
        values[f"rolling_mean_{window}"] = np.mean(recent)
        values[f"rolling_std_{window}"] = np.std(recent, ddof=1)
    values["forecast_horizon"] = horizon

    for column in LABEL_FEATURES:
        mapping = encoders["label"][column]
        category = str(values[column])
        values[f"{column.replace(' ', '_')}_encoded"] = mapping[category]
    for column in ONE_HOT_FEATURES:
        categories = encoders["one_hot"][column]
        category = str(values[column])
        if category not in categories:
            raise ValueError(f"Unknown {column}: {category}")
        for known_category in categories:
            values[f"{column}_{known_category}"] = int(category == known_category)

    missing_features = [column for column in feature_columns if column not in values]
    if missing_features:
        raise ValueError(f"Could not construct model features: {missing_features}")
    try:
        result = pd.DataFrame(
            [[values[column] for column in feature_columns]],
            columns=feature_columns,
        ).astype(float).to_numpy()
    except (TypeError, ValueError) as error:
        raise ValueError("Inference features must be numeric.") from error
    if result.shape != (1, len(feature_columns)):
        raise ValueError(
            f"Unexpected inference shape {result.shape}; expected (1, {len(feature_columns)})."
        )
    return result

if __name__ == "__main__":
    prepare_data()
