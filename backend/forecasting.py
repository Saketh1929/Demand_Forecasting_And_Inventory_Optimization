"""Compatibility wrapper that points the backend to the authoritative ML forecasting implementation."""
from src.forecasting import ForecastEngine, forecast_tool, load_model, resolve_forecast_horizon

__all__ = ["ForecastEngine", "forecast_tool", "load_model", "resolve_forecast_horizon"]
