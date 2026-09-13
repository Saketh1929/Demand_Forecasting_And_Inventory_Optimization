import os
import pickle
try:
    import joblib
except ImportError:
    joblib = None

import numpy as np
from backend.config import MODEL_PATH

def load_model():
    """
    Attempts to load best_model.pkl from disk. Returns None if missing or unpickling fails.
    """
    if os.path.exists(MODEL_PATH):
        try:
            if joblib is not None:
                return joblib.load(MODEL_PATH)
            with open(MODEL_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def compute_heuristic_forecast(features: dict) -> float:
    """
    Feature-based heuristic demand forecasting engine used as a seamless fallback
    when best_model.pkl is absent.
    Grounded in EDA drivers discovered in notebooks 01-03.
    """
    category = features.get("Category", "Groceries")
    promotion = features.get("Promotion", 0)
    is_weekend = features.get("IsWeekend", 0)
    seasonality = features.get("Seasonality", "Spring")
    weather = features.get("Weather_Condition", "Sunny")
    price = float(features.get("Price", 25.0))
    discount = float(features.get("Discount", 0.0))
    competitor_price = float(features.get("Competitor_Pricing", 25.0))
    epidemic = features.get("Epidemic", 0)

    # Category baseline daily demand
    category_baselines = {
        "Groceries": 140.0,
        "Clothing": 105.0,
        "Electronics": 75.0,
        "Toys": 85.0,
        "Furniture": 45.0
    }
    base_demand = category_baselines.get(category, 100.0)

    # Multipliers
    promo_mult = 1.25 if promotion == 1 else 1.0
    weekend_mult = 1.15 if is_weekend == 1 else 1.0

    season_mults = {"Winter": 1.20, "Summer": 1.10, "Autumn": 1.00, "Spring": 1.05}
    season_mult = season_mults.get(seasonality, 1.0)

    weather_mults = {"Sunny": 1.00, "Cloudy": 0.95, "Rainy": 0.85, "Snowy": 0.75}
    weather_mult = weather_mults.get(weather, 1.0)

    discount_mult = 1.0 + (discount / 100.0) * 0.8

    competitor_mult = 1.10 if price < competitor_price else (0.92 if price > competitor_price * 1.15 else 1.0)

    epidemic_mult = 1.0
    if epidemic == 1:
        if category == "Groceries":
            epidemic_mult = 1.30
        else:
            epidemic_mult = 0.75

    predicted = base_demand * promo_mult * weekend_mult * season_mult * weather_mult * discount_mult * competitor_mult * epidemic_mult

    return float(max(10, round(predicted, 2)))

def forecast_tool(preprocessed_data) -> dict:
    """
    Forecasting tool wrapper that loads ML model if available,
    or falls back to feature-based heuristic forecasting.
    Accepts:
    - Preprocessed dictionary from preprocess_input()
    - Raw feature dictionary
    - Direct 2D numpy array feature vector
    """
    feature_vector = None
    features_dict = {}

    if isinstance(preprocessed_data, np.ndarray):
        feature_vector = preprocessed_data
    elif isinstance(preprocessed_data, dict):
        if "feature_vector" in preprocessed_data:
            feature_vector = preprocessed_data.get("feature_vector")
            features_dict = preprocessed_data.get("processed_features", {})
        else:
            from backend.data_preprocessing import preprocess_input
            prep = preprocess_input(preprocessed_data)
            feature_vector = prep.get("feature_vector")
            features_dict = prep.get("processed_features", {})

    model = load_model()
    
    if model is not None and feature_vector is not None:
        try:
            if hasattr(model, "predict"):
                pred = model.predict(feature_vector)
                val = float(pred[0]) if isinstance(pred, (list, np.ndarray)) else float(pred)
                val = float(max(0, round(val, 2)))
                return {
                    "predicted_demand": val,
                    "model_source": "ml_model",
                    "model_path": str(MODEL_PATH),
                    "status": "success"
                }
        except Exception:
            pass

    # Heuristic Fallback
    heuristic_val = compute_heuristic_forecast(features_dict)
    return {
        "predicted_demand": heuristic_val,
        "model_source": "heuristic_fallback",
        "model_path": str(MODEL_PATH),
        "status": "fallback_applied"
    }
