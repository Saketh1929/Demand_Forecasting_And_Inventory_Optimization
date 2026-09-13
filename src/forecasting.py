"""Recursive 1-day-ahead demand forecasting engine for M3 XGBoost model.

Architecture & Contract:
------------------------
1. Model:
   M3 trains ONE single-step XGBoost model predicting Demand(t):
   Target = Demand(t).
   There are NO separate models for 7, 30, 90, 180, 365 days.

2. Recursive Forecasting Rollout:
   Target date t (exogenous business factors & calendar)
       ↓
   preprocess_input(current_row, history_df=working_history)
       ↓
   NumPy feature vector of shape (1, 34) matching FEATURE_ORDER
       ↓
   M3 1-day-ahead XGBoost model -> predicts Demand(t)
       ↓
   Append predicted Demand(t) to working_history as Demand(t)
       ↓
   Advance target date to t+1, predict Demand(t+1)
       ↓
   Repeat recursively until requested forecast horizon H is reached

3. Demand Lags & Rolling Continuity:
   - For step 1 predicting target day t, lags are extracted from historical Demand:
     lag_1 = Demand(t-1), lag_7 = Demand(t-7), lag_14 = Demand(t-14).
   - When predicted Demand(t) is generated, it is appended to working_history.
   - For step 2 predicting day t+1, lag_1 automatically becomes the predicted Demand(t),
     and rolling statistics incorporate Demand(t) without future leakage.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from src.data_preprocessing import (
    BUSINESS_FEATURES,
    CALENDAR_FEATURES,
    DATA_PATH,
    ENCODERS_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_ORDER,
    INPUT_MAPPING,
    LAG_FEATURES,
    ONE_HOT_MAPPING,
    ROLLING_FEATURES,
    PreprocessResult,
    preprocess_input,
)


class ModelPredictor(Protocol):
    """Protocol for any demand forecasting model implementing .predict()."""

    def predict(self, X: np.ndarray) -> np.ndarray | list[float] | float:
        ...


@dataclass
class ForecastResult:
    """Structured result returned by the recursive forecasting engine."""

    forecast_df: pd.DataFrame
    predictions: list[float]
    total_demand: float
    mean_daily_demand: float
    horizon: int
    store_id: str
    product_id: str
    start_date: str
    end_date: str
    history_df: pd.DataFrame
    feature_columns: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert result to standard Python dictionary for downstream consumption."""
        return {
            "forecast_df": self.forecast_df,
            "predictions": self.predictions,
            "total_demand": self.total_demand,
            "mean_daily_demand": self.mean_daily_demand,
            "horizon": self.horizon,
            "store_id": self.store_id,
            "product_id": self.product_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "feature_columns": self.feature_columns,
        }


class ForecastEngine:
    """Recursive forecasting engine executing 1-day-ahead rollouts with 34 features."""

    def __init__(
        self,
        model: ModelPredictor | None = None,
        model_path: Path | str | None = None,
        encoders_path: Path = ENCODERS_PATH,
        feature_columns_path: Path = FEATURE_COLUMNS_PATH,
        data_path: Path = DATA_PATH,
    ) -> None:
        self.encoders_path = Path(encoders_path)
        self.feature_columns_path = Path(feature_columns_path)
        self.data_path = Path(data_path)
        self.model: ModelPredictor | None = model
        if self.model is None and model_path is not None:
            self.model = self._load_model(Path(model_path))

    @staticmethod
    def _load_model(path: Path) -> Any:
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at: {path}")
        with path.open("rb") as f:
            return pickle.load(f)

    def forecast(
        self,
        current_row: dict[str, Any],
        horizon: int = 7,
        history_df: pd.DataFrame | None = None,
        future_business_schedule: pd.DataFrame | list[dict[str, Any]] | None = None,
        model: ModelPredictor | None = None,
    ) -> ForecastResult:
        """Generate multi-day demand forecasts recursively using ONE 1-day-ahead model.

        Parameters
        ----------
        current_row : dict[str, Any]
            State of observation for the first forecast step t (store, product, date, business features).
        horizon : int, default 7
            Number of days to forecast ahead (H >= 1).
        history_df : pd.DataFrame | None, optional
            Historical sales/demand DataFrame containing Date, Store ID, Product ID, Demand.
            Must contain at least 14 days of history prior to the first forecast date.
        future_business_schedule : pd.DataFrame | list[dict[str, Any]] | None, optional
            Optional known exogenous schedule for future dates (e.g. planned price, promotions).
        model : ModelPredictor | None, optional
            Optional model override implementing predict(features).

        Returns
        -------
        ForecastResult
            Structured result containing daily forecast table, demand totals, and metadata.
        """
        if horizon < 1:
            raise ValueError(f"Forecast horizon must be a positive integer (>= 1), got {horizon}.")

        active_model = model or self.model
        if active_model is None:
            raise ValueError(
                "No forecasting model provided. Pass 'model' to forecast() or initialize ForecastEngine with a model."
            )

        norm_row: dict[str, Any] = {}
        for k, v in current_row.items():
            if k in INPUT_MAPPING:
                norm_row[INPUT_MAPPING[k]] = v
            else:
                norm_row[k] = v

        store_id = str(norm_row.get("Store ID", norm_row.get("store_id", "")))
        product_id = str(norm_row.get("Product ID", norm_row.get("product_id", "")))
        if not store_id or not product_id:
            raise ValueError("current_row must contain valid Store ID and Product ID.")

        # Resolve initial forecast date t
        raw_date = norm_row.get("Date", norm_row.get("date"))
        if raw_date is None or pd.isna(raw_date):
            target_date = pd.Timestamp.now().normalize()
        else:
            target_date = pd.Timestamp(raw_date).normalize()

        # Initialize recursive working history
        if history_df is not None and not history_df.empty:
            working_history = history_df.copy()
            date_col = "Date" if "Date" in working_history.columns else "date"
            if not pd.api.types.is_datetime64_any_dtype(working_history[date_col]):
                working_history[date_col] = pd.to_datetime(working_history[date_col], errors="coerce")
        else:
            working_history = pd.DataFrame(
                columns=["Date", "Store ID", "Product ID", "Demand"]
            )

        # Process future schedule if provided
        schedule_by_date: dict[str, dict[str, Any]] = {}
        if future_business_schedule is not None:
            if isinstance(future_business_schedule, pd.DataFrame):
                sched_records = future_business_schedule.to_dict(orient="records")
            else:
                sched_records = list(future_business_schedule)
            for entry in sched_records:
                entry_norm = {INPUT_MAPPING.get(k, k): v for k, v in entry.items()}
                d_val = entry_norm.get("Date", entry_norm.get("date"))
                if d_val is not None:
                    d_key = str(pd.Timestamp(d_val).date())
                    schedule_by_date[d_key] = entry_norm

        daily_forecasts: list[dict[str, Any]] = []
        predictions: list[float] = []
        current_step_row = dict(norm_row)

        last_preprocess_result: PreprocessResult | None = None

        # Recursive Rollout Loop
        for step in range(1, horizon + 1):
            target_date_str = str(target_date.date())
            current_step_row["Date"] = target_date_str

            # Apply future scheduled exogenous features if available
            if target_date_str in schedule_by_date:
                for b_col in BUSINESS_FEATURES:
                    if b_col in schedule_by_date[target_date_str]:
                        current_step_row[b_col] = schedule_by_date[target_date_str][b_col]
                for c_col in ONE_HOT_MAPPING.keys():
                    if c_col in schedule_by_date[target_date_str]:
                        current_step_row[c_col] = schedule_by_date[target_date_str][c_col]

            # 1. Feature Preprocessing via Canonical Contract (shape 1, 34)
            prep_result = preprocess_input(
                row=current_step_row,
                history_df=working_history,
                return_dict=False,
                encoders_path=self.encoders_path,
                feature_columns_path=self.feature_columns_path,
                data_path=self.data_path,
            )
            if not isinstance(prep_result, PreprocessResult):
                raise TypeError(f"Expected PreprocessResult from preprocess_input, got {type(prep_result)}")
            last_preprocess_result = prep_result

            # 2. Predict Demand(t) using the 1-day-ahead model
            feature_vector = prep_result.feature_vector
            raw_prediction = active_model.predict(feature_vector)

            if isinstance(raw_prediction, (np.ndarray, list)):
                pred_demand = float(raw_prediction[0])
            else:
                pred_demand = float(raw_prediction)

            # Enforce non-negative demand constraint
            pred_demand = max(0.0, pred_demand)
            predictions.append(pred_demand)

            # Record daily forecast entry
            daily_forecasts.append(
                {
                    "step": step,
                    "date": target_date_str,
                    "store_id": store_id,
                    "product_id": product_id,
                    "predicted_demand": pred_demand,
                }
            )

            # 3. Update Recursive History
            # Record predicted demand at target_date into working history
            new_history_row = pd.DataFrame(
                [
                    {
                        "Date": target_date,
                        "Store ID": store_id,
                        "Product ID": product_id,
                        "Demand": pred_demand,
                    }
                ]
            )
            working_history = pd.concat([working_history, new_history_row], ignore_index=True)

            # 4. Advance target date for step + 1
            target_date = target_date + pd.Timedelta(days=1)

        forecast_df = pd.DataFrame(daily_forecasts)
        total_demand = float(sum(predictions))
        mean_demand = float(np.mean(predictions)) if predictions else 0.0
        feature_columns = last_preprocess_result.feature_columns if last_preprocess_result is not None else []

        return ForecastResult(
            forecast_df=forecast_df,
            predictions=predictions,
            total_demand=total_demand,
            mean_daily_demand=mean_demand,
            horizon=horizon,
            store_id=store_id,
            product_id=product_id,
            start_date=daily_forecasts[0]["date"],
            end_date=daily_forecasts[-1]["date"],
            history_df=working_history,
            feature_columns=feature_columns,
        )


def forecast_demand(
    model: ModelPredictor,
    current_row: dict[str, Any],
    horizon: int = 7,
    history_df: pd.DataFrame | None = None,
    future_business_schedule: pd.DataFrame | list[dict[str, Any]] | None = None,
    encoders_path: Path = ENCODERS_PATH,
    feature_columns_path: Path = FEATURE_COLUMNS_PATH,
    data_path: Path = DATA_PATH,
) -> ForecastResult:
    """Functional interface for recursive demand forecasting."""
    engine = ForecastEngine(
        model=model,
        encoders_path=encoders_path,
        feature_columns_path=feature_columns_path,
        data_path=data_path,
    )
    return engine.forecast(
        current_row=current_row,
        horizon=horizon,
        history_df=history_df,
        future_business_schedule=future_business_schedule,
    )