import os
import pickle
try:
    import joblib
except ImportError:
    joblib = None

import pandas as pd
import numpy as np
from datetime import datetime
from backend.config import ENCODERS_PATH, CATEGORIES, REGIONS, WEATHER_CONDITIONS, SEASONS

def extract_date_features(date_val) -> dict:
    """
    Extracts temporal features from a date string or pandas Timestamp:
    Year, Month, Day, DayOfWeek, WeekOfYear, Quarter, IsWeekend.
    """
    if isinstance(date_val, str):
        dt = pd.to_datetime(date_val)
    elif isinstance(date_val, (datetime, pd.Timestamp)):
        dt = pd.Timestamp(date_val)
    else:
        dt = pd.to_datetime(str(date_val))

    day_of_week = dt.dayofweek
    is_weekend = 1 if day_of_week >= 5 else 0
    week_of_year = dt.isocalendar().week

    return {
        "Year": int(dt.year),
        "Month": int(dt.month),
        "Day": int(dt.day),
        "DayOfWeek": int(day_of_week),
        "WeekOfYear": int(week_of_year),
        "Quarter": int(dt.quarter),
        "IsWeekend": int(is_weekend)
    }

def load_encoders():
    """
    Attempts to load encoders.pkl if present on disk.
    """
    if os.path.exists(ENCODERS_PATH):
        try:
            if joblib is not None:
                return joblib.load(ENCODERS_PATH)
            with open(ENCODERS_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None
    return None

def preprocess_input(row: dict) -> dict:
    """
    Preprocesses raw API input parameters into feature array and structured feature dict.
    Gracefully falls back if model encoders are not found.
    """
    date_features = extract_date_features(row.get("date", datetime.now().strftime("%Y-%m-%d")))
    
    encoders = load_encoders()
    used_encoder_pkl = encoders is not None

    # Base feature dictionary
    features = {
        **date_features,
        "Store_ID": str(row.get("store_id", "S001")),
        "Product_ID": str(row.get("product_id", "P0001")),
        "Category": str(row.get("category", "Groceries")),
        "Region": str(row.get("region", "North")),
        "Inventory_Level": float(row.get("inventory_level", 100)),
        "Units_Sold": float(row.get("units_sold", 0)),
        "Units_Ordered": float(row.get("units_ordered", 0)),
        "Price": float(row.get("price", 25.0)),
        "Discount": float(row.get("discount", 0.0)),
        "Weather_Condition": str(row.get("weather_condition", "Sunny")),
        "Promotion": int(row.get("promotion", 0)),
        "Competitor_Pricing": float(row.get("competitor_pricing", 25.0)),
        "Seasonality": str(row.get("seasonality", "Spring")),
        "Epidemic": int(row.get("epidemic", 0)),
    }

    # Encode categorical features numerically for model consumption
    if used_encoder_pkl and isinstance(encoders, dict):
        try:
            encoded_cat = {}
            for cat_col in ["Category", "Region", "Weather_Condition", "Seasonality", "Store_ID", "Product_ID"]:
                if cat_col in encoders:
                    val = features[cat_col]
                    encoder = encoders[cat_col]
                    if hasattr(encoder, "transform"):
                        encoded_cat[f"{cat_col}_code"] = int(encoder.transform([val])[0])
                    elif isinstance(encoder, dict):
                        encoded_cat[f"{cat_col}_code"] = int(encoder.get(val, 0))
                    else:
                        encoded_cat[f"{cat_col}_code"] = 0
                else:
                    encoded_cat[f"{cat_col}_code"] = 0
        except Exception:
            used_encoder_pkl = False

    if not used_encoder_pkl:
        # Fallback deterministic categorical encoding index
        cat_map = {
            "Category": CATEGORIES,
            "Region": REGIONS,
            "Weather_Condition": WEATHER_CONDITIONS,
            "Seasonality": SEASONS
        }
        encoded_cat = {}
        for cat_col, options in cat_map.items():
            val = features[cat_col]
            encoded_cat[f"{cat_col}_code"] = options.index(val) if val in options else 0

        # Store_ID & Product_ID fallback index extraction
        try:
            encoded_cat["Store_ID_code"] = int(features["Store_ID"].replace("S", "")) if "S" in features["Store_ID"] else 1
        except ValueError:
            encoded_cat["Store_ID_code"] = 1
            
        try:
            encoded_cat["Product_ID_code"] = int(features["Product_ID"].replace("P", "")) if "P" in features["Product_ID"] else 1
        except ValueError:
            encoded_cat["Product_ID_code"] = 1

    features.update(encoded_cat)

    # Convert to numeric feature array
    feature_vector = np.array([
        features["Year"],
        features["Month"],
        features["Day"],
        features["DayOfWeek"],
        features["WeekOfYear"],
        features["Quarter"],
        features["IsWeekend"],
        features["Store_ID_code"],
        features["Product_ID_code"],
        features["Category_code"],
        features["Region_code"],
        features["Inventory_Level"],
        features["Price"],
        features["Discount"],
        features["Weather_Condition_code"],
        features["Promotion"],
        features["Competitor_Pricing"],
        features["Seasonality_code"],
        features["Epidemic"]
    ], dtype=float).reshape(1, -1)

    return {
        "raw_input": row,
        "processed_features": features,
        "feature_vector": feature_vector,
        "used_encoder_pkl": used_encoder_pkl
    }
