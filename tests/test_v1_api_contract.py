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


def test_v1_forecast_succeeds_with_loaded_model_artifacts():
    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)

    assert response.status_code == 200, response.text
    payload = response.json()

    assert payload["status"] == "completed"
    assert payload["request_id"]
    assert payload["prediction"]["forecast_period"] == "1 month"
    assert payload["prediction"]["predicted_demand_units"] > 0
    assert payload["model"]["name"] == "dfio-demand-forecast"
    assert payload["input_echo"]["store_id"] == "Store 1"
    assert payload["input_echo"]["product_id"] == "P001"


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


def test_v1_forecast_records_completed_audit_entry_when_model_is_loaded(monkeypatch):
    audit_path = Path("logs/forecast_audit.json")
    if audit_path.exists():
        audit_path.unlink()

    monkeypatch.setattr(main_mod, "ENABLE_AUTH", False, raising=False)
    monkeypatch.setattr(main_mod, "API_KEY", "", raising=False)

    response = client.post("/api/v1/forecasts", json=VALID_PAYLOAD)
    assert response.status_code == 200, response.text

    payload = response.json()
    assert payload["status"] == "completed"
    assert payload["request_id"]
    assert audit_path.exists()

    records = json.loads(audit_path.read_text(encoding="utf-8"))
    assert len(records) > 0
    assert records[-1]["request_id"] == payload["request_id"]
    assert records[-1]["status"] == "completed"
    assert records[-1]["prediction"]["forecast_period"] == "1 month"


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


def test_legacy_forecast_endpoint_uses_requested_horizon():
    periods = ["1 week", "1 month", "3 months", "6 months", "1 year"]
    results = {}

    for period in periods:
        response = client.post(
            "/api/forecast",
            json={
                "date": "2024-02-15",
                "store_id": "S001",
                "product_id": "P0001",
                "category": "Groceries",
                "region": "North",
                "inventory_level": 120,
                "units_sold": 80,
                "units_ordered": 50,
                "price": 25.0,
                "discount": 10.0,
                "competitor_pricing": 25.0,
                "weather_condition": "Sunny",
                "promotion": 0,
                "seasonality": "Spring",
                "epidemic": 0,
                "forecast_period": period,
            },
        )

        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["forecast_period"] == period
        assert payload["horizon_days"] == resolve_forecast_horizon(period)
        assert payload["predicted_demand"] > 0
        results[period] = payload

    demand_values = [results[period]["predicted_demand"] for period in periods]
    gap_values = [results[period]["gap"] for period in periods]
    reorder_values = [results[period]["reorder_quantity"] for period in periods]

    assert demand_values[1:] > [demand_values[i] for i in range(len(demand_values) - 1)]
    assert all(current < previous for previous, current in zip(gap_values, gap_values[1:]))
    assert reorder_values[1:] > [reorder_values[i] for i in range(len(reorder_values) - 1)]

    assert results["1 week"]["forecast_period"] == "1 week"
    assert results["1 month"]["forecast_period"] == "1 month"
    assert results["3 months"]["forecast_period"] == "3 months"
    assert results["6 months"]["forecast_period"] == "6 months"
    assert results["1 year"]["forecast_period"] == "1 year"


def test_legacy_forecast_response_includes_backend_metrics_fields():
    response = client.post(
        "/api/forecast",
        json={
            "date": "2024-02-15",
            "store_id": "S001",
            "product_id": "P0001",
            "category": "Groceries",
            "region": "North",
            "inventory_level": 120,
            "units_sold": 80,
            "units_ordered": 50,
            "price": 25.0,
            "discount": 10.0,
            "competitor_pricing": 25.0,
            "weather_condition": "Sunny",
            "promotion": 0,
            "seasonality": "Spring",
            "epidemic": 0,
            "forecast_period": "1 month",
        },
    )

    assert response.status_code == 200, response.text
    payload = response.json()

    required_fields = {
        "forecast_id",
        "predicted_demand",
        "forecast_period",
        "horizon_days",
        "gap",
        "status",
        "urgency",
        "reorder_quantity",
        "requires_approval",
        "recommendation",
        "reasoning_source",
        "expected_stockout_risk",
        "confidence_score",
    }

    missing = required_fields - payload.keys()
    assert not missing, f"Missing required backend fields: {missing}"
    assert payload["forecast_period"] == "1 month"
    assert payload["horizon_days"] == 30
    assert isinstance(payload["expected_stockout_risk"], (int, float, type(None)))
    assert payload["confidence_score"] is None or isinstance(payload["confidence_score"], (int, float))
