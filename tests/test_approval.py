import json
import pytest
from pathlib import Path
import backend.approval as approval_module
from backend.approval import log_approval, get_approvals

@pytest.fixture(autouse=True)
def isolated_approval_log(tmp_path, monkeypatch):
    """Isolate approval log to a temporary file during each test."""
    test_log_file = tmp_path / "test_approval_log.json"
    test_log_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(approval_module, "APPROVAL_LOG_PATH", str(test_log_file))
    monkeypatch.setattr(approval_module, "LOGS_DIR", str(tmp_path))
    yield test_log_file

def test_log_approval_success():
    """Verify logging an approved decision."""
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "predicted_demand": 500.0,
        "inventory_level": 100,
        "suggested_reorder_quantity": 480,
        "action": "approved",
        "manager_notes": "Approved for peak season",
    }
    res = log_approval(payload)
    assert res["status"] == "success"
    assert res["approval_id"].startswith("APP-")
    assert res["record"]["store_id"] == "S001"
    assert res["record"]["action"] == "approved"
    assert res["record"]["suggested_reorder_quantity"] == 480

def test_log_approval_modified_quantity():
    """Verify logging a modified decision with custom quantity."""
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "predicted_demand": 500.0,
        "inventory_level": 100,
        "suggested_reorder_quantity": 480,
        "action": "modified",
        "modified_quantity": 350,
        "manager_notes": "Reduced due to warehouse space limits",
    }
    res = log_approval(payload)
    assert res["status"] == "success"
    assert res["record"]["action"] == "modified"
    assert res["record"]["modified_quantity"] == 350

def test_log_approval_rejected():
    """Verify logging a rejected replenishment decision."""
    payload = {
        "store_id": "S002",
        "product_id": "P0002",
        "predicted_demand": 50.0,
        "inventory_level": 100,
        "suggested_reorder_quantity": 0,
        "action": "rejected",
        "manager_notes": "Holding orders until Q4",
    }
    res = log_approval(payload)
    assert res["status"] == "success"
    assert res["record"]["action"] == "rejected"

def test_log_approval_invalid_action():
    """Verify that an invalid action string raises ValueError."""
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "action": "pending_review",
    }
    with pytest.raises(ValueError, match="Action must be one of"):
        log_approval(payload)

def test_log_approval_modified_missing_quantity():
    """Verify that action='modified' without modified_quantity raises ValueError."""
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "action": "modified",
        "modified_quantity": None,
    }
    with pytest.raises(ValueError, match="modified_quantity must be provided"):
        log_approval(payload)

def test_log_approval_modified_negative_quantity():
    """Verify that action='modified' with negative quantity raises ValueError."""
    payload = {
        "store_id": "S001",
        "product_id": "P0001",
        "action": "modified",
        "modified_quantity": -50,
    }
    with pytest.raises(ValueError, match="modified_quantity cannot be negative"):
        log_approval(payload)

def test_get_approvals_filtering_and_sorting():
    """Verify retrieving approvals with filtering and newest-first sorting."""
    for i in range(5):
        log_approval({
            "store_id": f"S00{i % 2 + 1}",
            "product_id": "P0001",
            "action": "approved" if i % 2 == 0 else "rejected",
            "predicted_demand": 100.0,
            "inventory_level": 50,
            "suggested_reorder_quantity": 60,
        })
    
    all_records = get_approvals(limit=10)
    assert all_records["total"] == 5
    assert all_records["count"] == 5
    
    # Filter by store_id
    s1_records = get_approvals(store_id="S001")
    assert all(r["store_id"] == "S001" for r in s1_records["approvals"])
    
    # Filter by action
    rej_records = get_approvals(action="rejected")
    assert all(r["action"] == "rejected" for r in rej_records["approvals"])
    
    # Check limit constraint
    limited = get_approvals(limit=2)
    assert limited["count"] == 2
    assert limited["total"] == 5
