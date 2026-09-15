import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("DFIO_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
FORECAST_ENDPOINT = f"{API_BASE_URL}/api/v1/forecasts"
AGENT_FORECAST_ENDPOINT = f"{API_BASE_URL}/api/forecast"
APPROVAL_ENDPOINT = f"{API_BASE_URL}/api/approve"
STATUS_ENDPOINT = f"{API_BASE_URL}/api/v1/model/status"

API_HEADERS = {}
if os.getenv("DFIO_API_KEY"):
    API_HEADERS["Authorization"] = f"Bearer {os.environ['DFIO_API_KEY']}"
if os.getenv("DFIO_USER_ROLE"):
    API_HEADERS["X-User-Role"] = os.environ["DFIO_USER_ROLE"]

@st.cache_data(ttl=30)
def fetch_model_status():
    try:
        response = requests.get(STATUS_ENDPOINT, headers=API_HEADERS, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def fetch_forecast(payload):
    return requests.post(
        AGENT_FORECAST_ENDPOINT,
        headers=API_HEADERS,
        json=payload,
        timeout=30,
    )

def submit_approval(payload):
    return requests.post(
        APPROVAL_ENDPOINT,
        headers=API_HEADERS,
        json=payload,
        timeout=15,
    )
