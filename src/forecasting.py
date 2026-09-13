"""
Forwarding wrapper pointing to backend.forecasting
"""
from backend.forecasting import forecast_tool, load_model, compute_heuristic_forecast

__all__ = ["forecast_tool", "load_model", "compute_heuristic_forecast"]
