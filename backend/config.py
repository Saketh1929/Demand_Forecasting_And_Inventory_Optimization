import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
ENCODERS_PATH = MODELS_DIR / "encoders.pkl"

LOGS_DIR = BASE_DIR / "logs"
APPROVAL_LOG_PATH = LOGS_DIR / "approval_log.json"

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", 8000))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Categorical Option Constants for UI Dropdowns & Validation
STORES = ["S001", "S002", "S003", "S004", "S005"]
CATEGORIES = ["Groceries", "Clothing", "Electronics", "Furniture", "Toys"]
REGIONS = ["North", "South", "East", "West"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
PROMOTION_OPTIONS = [0, 1]
EPIDEMIC_OPTIONS = [0, 1]

class Config:
    BASE_DIR = BASE_DIR
    MODELS_DIR = MODELS_DIR
    MODEL_PATH = MODEL_PATH
    ENCODERS_PATH = ENCODERS_PATH
    LOGS_DIR = LOGS_DIR
    APPROVAL_LOG_PATH = APPROVAL_LOG_PATH
    HOST = HOST
    PORT = PORT
    GEMINI_API_KEY = GEMINI_API_KEY
    STORES = STORES
    CATEGORIES = CATEGORIES
    REGIONS = REGIONS
    WEATHER_CONDITIONS = WEATHER_CONDITIONS
    SEASONS = SEASONS
    PROMOTION_OPTIONS = PROMOTION_OPTIONS
    EPIDEMIC_OPTIONS = EPIDEMIC_OPTIONS
