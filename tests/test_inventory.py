import math
import pytest
from backend.inventory import evaluate_inventory

def test_inventory_shortage_critical():
    """Verify shortage calculation when gap < -50 (Critical urgency)."""
    demand = 500.0
    stock = 100
    res = evaluate_inventory(predicted_demand=demand, inventory_level=stock)
    
    expected_gap = stock - demand
    shortage_amount = abs(expected_gap)
    expected_reorder = int(math.ceil(shortage_amount * 1.20))
    
    assert res["status"] == "Shortage"
    assert res["gap"] == expected_gap
    assert res["reorder_quantity"] == expected_reorder
    assert res["urgency"] == "Critical"
    assert res["requires_approval"] is True
    assert "shortage" in res["description"].lower()

def test_inventory_shortage_moderate():
    """Verify shortage calculation when -50 <= gap < 0 (Moderate urgency)."""
    demand = 120.0
    stock = 100
    res = evaluate_inventory(predicted_demand=demand, inventory_level=stock)
    
    expected_gap = stock - demand
    shortage_amount = abs(expected_gap)
    expected_reorder = int(math.ceil(shortage_amount * 1.20))
    
    assert res["status"] == "Shortage"
    assert res["gap"] == expected_gap
    assert res["reorder_quantity"] == expected_reorder
    assert res["urgency"] == "Moderate"
    assert res["requires_approval"] is True

def test_inventory_overstock():
    """Verify overstock when current stock exceeds predicted demand by > 20%."""
    demand = 100.0
    stock = 130
    res = evaluate_inventory(predicted_demand=demand, inventory_level=stock)
    
    assert res["status"] == "Overstock"
    assert res["reorder_quantity"] == 0
    assert res["urgency"] == "None"
    assert res["requires_approval"] is False
    assert "excess" in res["description"].lower() or "overstock" in res["description"].lower()

def test_inventory_sufficient():
    """Verify optimal stock when stock is between demand and demand + 20%."""
    demand = 100.0
    stock = 110
    res = evaluate_inventory(predicted_demand=demand, inventory_level=stock)
    
    assert res["status"] == "Sufficient"
    assert res["reorder_quantity"] == 0
    assert res["urgency"] == "None"
    assert res["requires_approval"] is False

def test_inventory_horizon_days_passed():
    """Verify optional horizon_days is passed and reflected in response dictionary."""
    demand = 1000.0
    stock = 200
    res = evaluate_inventory(predicted_demand=demand, inventory_level=stock, horizon_days=30)
    
    assert res["horizon_days"] == 30
    assert "30-day horizon" in res["description"]
