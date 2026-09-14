import math
import pytest
from backend.agent import run_agent_pipeline, generate_fallback_recommendation

import src.data_preprocessing as dp_module

@pytest.fixture(autouse=True)
def reset_cached_sales_data():
    yield
    dp_module._CACHED_SALES_DF = None

SAMPLE_RAW_INPUT = {
    "store_id": "S001",
    "product_id": "P0001",
    "category": "Groceries",
    "region": "North",
    "date": "2023-01-01",
    "inventory_level": 250,
    "units_sold": 100,
    "units_ordered": 120,
    "price": 25.0,
    "discount": 0.05,
    "competitor_pricing": 25.0,
    "promotion": 0,
    "weather_condition": "Sunny",
    "holiday": 0,
    "seasonality": "Winter",
    "epidemic": 0,
}

EXPECTED_M4_KEYS = {
    "forecast_id",
    "timestamp",
    "date",
    "store_id",
    "product_id",
    "category",
    "predicted_demand",
    "inventory_level",
    "gap",
    "status",
    "reorder_quantity",
    "urgency",
    "requires_approval",
    "recommendation",
    "reasoning_source",
    "tool_trace",
}

def test_pipeline_preserves_all_m4_contract_keys():
    """Verify that all 15 M4 response keys are preserved in the pipeline output."""
    res = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period="7 days")
    missing_keys = EXPECTED_M4_KEYS - set(res.keys())
    assert not missing_keys, f"Missing required M4 keys: {missing_keys}"
    assert len(res["tool_trace"]) == 4
    step_tools = [step["tool"] for step in res["tool_trace"]]
    assert step_tools == [
        "data_preprocessing",
        "forecasting_model",
        "inventory_optimizer",
        "agent_llm_reasoning",
    ]

def test_pipeline_multi_horizon_behavioral_scaling():
    """Verify behavioral property: cumulative demand scales monotonically with horizon length."""
    res_7 = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period="7 days")
    res_30 = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period="30 days")
    res_90 = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period="90 days")

    # Horizon days verification
    assert res_7["horizon_days"] == 7
    assert res_30["horizon_days"] == 30
    assert res_90["horizon_days"] == 90

    # Behavioral scaling property: longer horizon produces strictly larger cumulative demand
    assert res_30["predicted_demand"] > res_7["predicted_demand"], (
        f"Expected 30-day demand ({res_30['predicted_demand']}) > 7-day demand ({res_7['predicted_demand']})"
    )
    assert res_90["predicted_demand"] > res_30["predicted_demand"], (
        f"Expected 90-day demand ({res_90['predicted_demand']}) > 30-day demand ({res_30['predicted_demand']})"
    )

def test_pipeline_daily_breakdown_properties():
    """Verify daily_forecasts length and that cumulative demand equals sum of daily forecasts."""
    horizon = "7 days"
    res = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period=horizon)

    assert len(res["daily_forecasts"]) == 7
    # Cumulative demand equals the sum of daily predictions within rounding tolerance
    sum_dailies = round(sum(res["daily_forecasts"]), 1)
    assert abs(res["predicted_demand"] - sum_dailies) <= 0.5, (
        f"predicted_demand ({res['predicted_demand']}) differs from sum of dailies ({sum_dailies})"
    )

def test_pipeline_inventory_uses_cumulative_horizon_demand():
    """Verify inventory gap calculation uses the multi-day cumulative demand."""
    res = run_agent_pipeline(SAMPLE_RAW_INPUT, forecast_period="30 days")
    
    expected_gap = round(SAMPLE_RAW_INPUT["inventory_level"] - res["predicted_demand"], 2)
    assert res["gap"] == expected_gap
    # With 250 inventory against 30-day demand (> 1000 units), stockout is Critical
    assert res["status"] == "Shortage"
    assert res["urgency"] == "Critical"
    assert res["requires_approval"] is True
    shortage_amount = abs(expected_gap)
    assert res["reorder_quantity"] == int(math.ceil(shortage_amount * 1.20))

def test_fallback_recommendation_structure():
    """Verify fallback recommendation handles various statuses and horizon context."""
    rec_short = generate_fallback_recommendation(
        store_id="S001",
        product_id="P0001",
        category="Groceries",
        predicted_demand=500.0,
        inventory_level=100,
        status="Shortage",
        reorder_qty=480,
        urgency="Critical",
        forecast_period="30 days",
        horizon_days=30,
    )
    assert "ALERT [Critical URGENCY]" in rec_short
    assert "Reorder 480 units" in rec_short
    assert "30 days" in rec_short

    rec_over = generate_fallback_recommendation(
        store_id="S001",
        product_id="P0001",
        category="Groceries",
        predicted_demand=100.0,
        inventory_level=500,
        status="Overstock",
        reorder_qty=0,
        urgency="None",
        forecast_period="7 days",
        horizon_days=7,
    )
    assert "NOTICE: Overstock position" in rec_over
    assert "Pause stock replenishment" in rec_over
