import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.data_preprocessing import preprocess_input

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_model.pkl"


def resolve_forecast_horizon(forecast_period: str) -> int:
    mapping = {
        "1 week": 7,
        "1 month": 30,
        "3 months": 90,
        "6 months": 180,
        "1 year": 365,
    }
    if forecast_period is None:
        raise ValueError("forecast_period is required")
    try:
        return int(forecast_period)
    except (TypeError, ValueError):
        pass
    normalized = str(forecast_period).strip().lower()
    for key, value in mapping.items():
        if normalized == key.lower():
            return value
    raise ValueError(f"Unsupported forecast period: {forecast_period}")


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        import joblib
        return joblib.load(MODEL_PATH)
    except Exception:
        try:
            import pickle
            with open(MODEL_PATH, "rb") as handle:
                return pickle.load(handle)
        except Exception:
            return None


class ForecastEngine:
    def __init__(self, model=None, horizon_days: Optional[int] = None):
        self.model = model or load_model()
        self.horizon_days = horizon_days

    def _ensure_model(self):
        if self.model is None:
            raise FileNotFoundError(
                f"Trained model not found at {MODEL_PATH}. The backend requires the real ML model to be present."
            )

    def _predict_day(self, row: Dict[str, Any]) -> float:
        self._ensure_model()
        prepared = preprocess_input(row)
        prediction = self.model.predict(prepared["feature_vector"])
        if isinstance(prediction, np.ndarray):
            prediction = prediction.reshape(-1)
        value = float(np.asarray(prediction)[0])
        return max(0.0, value)

    def _step_row(self, row: Dict[str, Any], predicted_value: float) -> Dict[str, Any]:
        next_row = dict(row)
        date_value = pd.to_datetime(row.get("date") or datetime.now().strftime("%Y-%m-%d")) + pd.Timedelta(days=1)
        next_row["date"] = date_value.strftime("%Y-%m-%d")
        next_row["lag_1"] = predicted_value
        next_row["lag_7"] = float(row.get("lag_7", predicted_value))
        next_row["lag_14"] = float(row.get("lag_14", predicted_value))
        next_row["rolling_mean_7"] = float(row.get("rolling_mean_7", predicted_value))
        next_row["rolling_std_7"] = float(row.get("rolling_std_7", 0.0))
        next_row["rolling_mean_14"] = float(row.get("rolling_mean_14", predicted_value))
        next_row["rolling_std_14"] = float(row.get("rolling_std_14", 0.0))
        next_row["units_sold"] = predicted_value
        return next_row

    def forecast(self, row: Dict[str, Any], forecast_period: str = "1 month") -> Dict[str, Any]:
        self._ensure_model()
        horizon = resolve_forecast_horizon(forecast_period)
        if self.horizon_days is not None:
            horizon = self.horizon_days

        current_row = dict(row or {})
        current_row.setdefault("date", datetime.now().strftime("%Y-%m-%d"))
        current_row.setdefault("store_id", "S001")
        current_row.setdefault("product_id", "P0001")
        current_row.setdefault("category", "Groceries")
        current_row.setdefault("region", "North")
        current_row.setdefault("weather_condition", "Sunny")
        current_row.setdefault("seasonality", "Spring")
        current_row.setdefault("price", 25.0)
        current_row.setdefault("discount", 0.0)
        current_row.setdefault("promotion", 0)
        current_row.setdefault("competitor_pricing", 25.0)
        current_row.setdefault("epidemic", 0)
        current_row.setdefault("lag_1", current_row.get("units_sold", 0.0))
        current_row.setdefault("lag_7", current_row.get("lag_1", 0.0))
        current_row.setdefault("lag_14", current_row.get("lag_7", 0.0))
        current_row.setdefault("rolling_mean_7", current_row.get("lag_1", 0.0))
        current_row.setdefault("rolling_std_7", 0.0)
        current_row.setdefault("rolling_mean_14", current_row.get("lag_1", 0.0))
        current_row.setdefault("rolling_std_14", 0.0)

        daily_forecast: List[float] = []
        for _ in range(horizon):
            daily_value = self._predict_day(current_row)
            daily_forecast.append(daily_value)
            current_row = self._step_row(current_row, daily_value)

        total = float(sum(daily_forecast))
        return {
            "forecast_period": forecast_period,
            "horizon_days": horizon,
            "predicted_demand": total,
            "daily_forecast": daily_forecast,
            "model_source": "ml_model",
            "status": "success",
        }


def forecast_tool(preprocessed_data: Any, forecast_period: str = "1 month") -> Dict[str, Any]:
    raw_input = preprocessed_data
    if isinstance(preprocessed_data, dict):
        raw_input = preprocessed_data.get("raw_input", preprocessed_data)
    if not isinstance(raw_input, dict):
        raise TypeError("forecast_tool expects a dict-like raw input or preprocessed payload")

    engine = ForecastEngine()
    result = engine.forecast(raw_input, forecast_period=forecast_period)
    return {
        "predicted_demand": float(result["predicted_demand"]),
        "model_source": result["model_source"],
        "status": result["status"],
        "forecast_period": result["forecast_period"],
        "horizon_days": result["horizon_days"],
        "model_path": str(MODEL_PATH),
    }


__all__ = ["ForecastEngine", "forecast_tool", "load_model", "resolve_forecast_horizon"]
