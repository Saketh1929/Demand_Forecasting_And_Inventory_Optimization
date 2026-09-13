import json
from pathlib import Path

from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app

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


def test_v1_forecast_returns_spec_shape():
    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "completed"
    assert "request_id" in payload
    assert "prediction" in payload
    assert payload["prediction"]["forecast_period"] == "1 month"
    assert "predicted_demand_units" in payload["prediction"]
    assert "recommended_order_quantity" in payload["prediction"]
    assert "expected_stockout_risk" in payload["prediction"]
    assert "confidence_score" in payload["prediction"]
    assert "model" in payload
    assert payload["model"]["name"] == "dfio-demand-forecast"
    assert "input_echo" in payload
    assert payload["input_echo"]["region"] == "North"


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


def test_v1_forecast_writes_audit_record(monkeypatch):
    audit_path = Path("logs/forecast_audit.json")
    if audit_path.exists():
        audit_path.unlink()

    monkeypatch.setattr(main_mod, "ENABLE_AUTH", False, raising=False)
    monkeypatch.setattr(main_mod, "API_KEY", "", raising=False)

    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text
    assert audit_path.exists(), "audit log should be created"

    data = json.loads(audit_path.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) >= 1
    assert data[-1]["request_id"].startswith("fcst_")
    assert data[-1]["model_version"] == "1.0.0"


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
