from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Protocol
import numpy as np

from src.data_preprocessing import FEATURE_ORDER, PreprocessResult

def resolve_forecast_horizon(period: str | int) -> int:
    if isinstance(period, int):
        return max(1, period)
    if not isinstance(period, str):
        raise ValueError(f"Unsupported forecast period type: {type(period).__name__}")

    normalized = period.strip().lower()
    mapping = {
        "1 week": 7,
        "7 days": 7,
        "1 month": 30,
        "30 days": 30,
        "3 months": 90,
        "90 days": 90,
        "6 months": 180,
        "180 days": 180,
        "1 year": 365,
        "365 days": 365,
    }
    if normalized in mapping:
        return mapping[normalized]
    if normalized.endswith("days"):
        try:
            value = int(normalized.split()[0])
            return max(1, value)
        except ValueError:
            pass
    if normalized.endswith("months"):
        try:
            value = int(normalized.split()[0])
            return max(1, value * 30)
        except ValueError:
            pass
    raise ValueError(f"Unsupported forecast period: {period!r}")

def load_model(model_path: Path | str | None = None) -> Any | None:
    path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[2] / "models" / "best_model.pkl"
    if not path.exists():
        return None
    with path.open("rb") as file:
        return pickle.load(file)

def forecast_tool(
    features: np.ndarray | PreprocessResult | dict[str, Any],
    model_path: Path | str | None = None,
    forecast_period: str | int | None = None,
) -> float | dict[str, Any]:
    path = Path(model_path) if model_path is not None else Path(__file__).resolve().parents[2] / "models" / "best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Trained model not found at {path}")
    with path.open("rb") as file:
        model = pickle.load(file)

    if isinstance(features, dict):
        feature_vector = np.asarray(features.get("feature_vector", features), dtype=float)
    elif hasattr(features, "feature_vector"):
        feature_vector = features.feature_vector
    else:
        feature_vector = np.asarray(features, dtype=float)

    expected = len(FEATURE_ORDER)
    if feature_vector.ndim != 2 or feature_vector.shape != (1, expected):
        raise ValueError(f"forecast_tool() expects shape (1, {expected}), got {feature_vector.shape}")

    prediction = float(model.predict(feature_vector)[0])
    prediction = max(0.0, round(prediction, 1))
    if forecast_period is None:
        return prediction

    return {
        "predicted_demand": prediction,
        "status": "success",
        "model_source": str(path),
        "horizon_days": resolve_forecast_horizon(forecast_period),
    }

class ModelPredictor(Protocol):
    def predict(self, X: np.ndarray) -> np.ndarray | list[float] | float:
        ...
