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
API_KEY = os.getenv("API_KEY", "").strip()
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}

# Categorical Option Constants for UI Dropdowns & Validation
STORES = ["S001", "S002", "S003", "S004", "S005"]
CATEGORIES = ["Groceries", "Clothing", "Electronics", "Furniture", "Toys"]
REGIONS = ["North", "South", "East", "West"]
WEATHER_CONDITIONS = ["Sunny", "Cloudy", "Rainy", "Snowy"]
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
PROMOTION_OPTIONS = [0, 1]
EPIDEMIC_OPTIONS = [0, 1]

VALID_STORE_LABELS = [f"Store {i}" for i in range(1, 6)]
VALID_PRODUCT_IDS = [f"P{i:03d}" for i in range(1, 5)]
MODEL_INFO = {
    "name": "dfio-demand-forecast",
    "version": "1.0.0",
    "accuracy": 0.855,
}
APPROVED_MODEL_REGISTRY = {
    "dfio-demand-forecast": {
        "version": "1.0.0",
        "status": "approved",
        "approved_at": "2026-09-13T00:00:00Z",
    }
}
ALLOWED_ROLES = {"admin", "manager", "analyst"}
PROTECTED_ENDPOINT_POLICIES = {
    "/api/approve": {
        "methods": {"POST", "PUT", "PATCH", "DELETE"},
        "roles": {"admin", "manager"},
    }
}

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
    API_KEY = API_KEY
    ENABLE_AUTH = ENABLE_AUTH
    STORES = STORES
    CATEGORIES = CATEGORIES
    REGIONS = REGIONS
    WEATHER_CONDITIONS = WEATHER_CONDITIONS
    SEASONS = SEASONS
    PROMOTION_OPTIONS = PROMOTION_OPTIONS
    EPIDEMIC_OPTIONS = EPIDEMIC_OPTIONS
    VALID_STORE_LABELS = VALID_STORE_LABELS
    VALID_PRODUCT_IDS = VALID_PRODUCT_IDS
    MODEL_INFO = MODEL_INFO
    APPROVED_MODEL_REGISTRY = APPROVED_MODEL_REGISTRY
    ALLOWED_ROLES = ALLOWED_ROLES
    PROTECTED_ENDPOINT_POLICIES = PROTECTED_ENDPOINT_POLICIES
