import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main_mod
from backend import data_preprocessing as backend_preprocessing
from backend import forecasting as backend_forecasting
from backend.main import app
from src.data_preprocessing import preprocess_input as src_preprocess_input
from src.forecasting import ForecastEngine as SrcForecastEngine
from src.forecasting import resolve_forecast_horizon

client = TestClient(app)

VALID_PAYLOAD = {
    "region": "North",
    "category": "Electronics",
    "forecast_period": "1 month",
    "store_id": "Store 1",
    "product_id": "P001",
    "weather_condition": "Sunny",
    "inventory_level": 250,
    "discount_rate": 10.0,
    "units_sold": 120,
    "units_ordered": 100,
    "promotion_active": True,
    "epidemic": 0,
    "competitor_pricing": 1499.0,
}


def test_v1_forecast_fails_visibly_when_model_is_missing():
    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)

    assert response.status_code == 500, response.text
    payload = response.json()

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "FORECAST_ERROR"
    assert "Trained model not found" in payload["error"]["message"]
    assert "request_id" in payload


def test_v1_forecast_rejects_invalid_promotion_type():
    invalid_payload = dict(VALID_PAYLOAD)
    invalid_payload["promotion_active"] = "yes"

    response = client.post("/api/v1/forecasts", json=invalid_payload)

    assert response.status_code == 422, response.text
    payload = response.json()

    assert payload["status"] == "failed"
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert "promotion_active" in payload["error"]["fields"]


def test_v1_forecast_rejects_unknown_store():
    invalid_payload = dict(VALID_PAYLOAD)
    invalid_payload["store_id"] = "Store 99"

    response = client.post("/api/v1/forecasts", json=invalid_payload)

    assert response.status_code == 404, response.text
    payload = response.json()
    assert payload["status"] == "failed"
    assert "Store not found" in payload["error"]["message"] or payload["error"]["code"] == "STORE_NOT_FOUND"


def test_v1_forecast_requires_api_key_when_auth_enabled(monkeypatch):
    monkeypatch.setattr(main_mod, "ENABLE_AUTH", True, raising=False)
    monkeypatch.setattr(main_mod, "API_KEY", "test-secret-key", raising=False)

    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)
    assert response.status_code == 401, response.text


def test_v1_forecast_does_not_hide_missing_model_errors(monkeypatch):
    audit_path = Path("logs/forecast_audit.json")
    if audit_path.exists():
        audit_path.unlink()

    monkeypatch.setattr(main_mod, "ENABLE_AUTH", False, raising=False)
    monkeypatch.setattr(main_mod, "API_KEY", "", raising=False)

    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)
    assert response.status_code == 500, response.text
    assert response.json()["status"] == "failed"
    assert "Trained model not found" in response.json()["error"]["message"]
    assert not audit_path.exists() or len(json.loads(audit_path.read_text(encoding="utf-8"))) == 0


def test_approval_requires_admin_role(monkeypatch):
    monkeypatch.setattr(main_mod, "ENABLE_AUTH", True, raising=False)
    monkeypatch.setattr(main_mod, "API_KEY", "test-secret-key", raising=False)
    monkeypatch.setattr(main_mod, "REQUIRE_ROLE_AUTH", True, raising=False)

    payload = {
        "forecast_id": "fcst_test_001",
        "store_id": "S001",
        "product_id": "P001",
        "predicted_demand": 120,
        "inventory_level": 50,
        "suggested_reorder_quantity": 40,
        "action": "approved",
        "modified_quantity": None,
        "manager_notes": "Looks good"
    }

    response = client.post(
        "/api/approve",
        json=payload,
        headers={"Authorization": "Bearer test-secret-key", "X-User-Role": "analyst"},
    )

    assert response.status_code == 403, response.text
    assert response.json()["error"]["code"] == "FORBIDDEN"


def test_backend_uses_authoritative_src_pipeline():
    assert backend_preprocessing.preprocess_input is src_preprocess_input
    assert backend_forecasting.ForecastEngine is SrcForecastEngine


def test_forecast_period_mapping_uses_requested_horizon():
    assert resolve_forecast_horizon("1 week") == 7
    assert resolve_forecast_horizon("1 month") == 30
    assert resolve_forecast_horizon("3 months") == 90
    assert resolve_forecast_horizon("6 months") == 180
    assert resolve_forecast_horizon("1 year") == 365
