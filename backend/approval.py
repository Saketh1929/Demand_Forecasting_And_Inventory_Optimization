import os
import json
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from backend.config import LOGS_DIR, APPROVAL_LOG_PATH

def _ensure_log_file():
    """
    Ensures logs directory and approval_log.json exist on disk.
    """
    if not os.path.exists(LOGS_DIR):
        os.makedirs(LOGS_DIR, exist_ok=True)

    if not os.path.exists(APPROVAL_LOG_PATH):
        with open(APPROVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

def log_approval(approval_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Records a human manager replenishment decision into logs/approval_log.json.
    """
    _ensure_log_file()

    action = str(approval_data.get("action", "approved")).lower().strip()
    valid_actions = {"approved", "modified", "rejected"}
    if action not in valid_actions:
        raise ValueError(f"Action must be one of: 'approved', 'modified', or 'rejected', got '{action}'")

    raw_modified_qty = approval_data.get("modified_quantity")
    if action == "modified":
        if raw_modified_qty is None:
            raise ValueError("modified_quantity must be provided when action is 'modified'")
        try:
            modified_quantity = int(raw_modified_qty)
            if modified_quantity < 0:
                raise ValueError("modified_quantity cannot be negative")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid modified_quantity: {e}")
    else:
        modified_quantity = int(raw_modified_qty) if raw_modified_qty is not None else None

    approval_id = f"APP-{uuid.uuid4().hex[:8].upper()}"
    logged_at = datetime.now().isoformat()

    record = {
        "approval_id": approval_id,
        "logged_at": logged_at,
        "forecast_id": approval_data.get("forecast_id", ""),
        "store_id": str(approval_data.get("store_id", "")),
        "product_id": str(approval_data.get("product_id", "")),
        "predicted_demand": float(approval_data.get("predicted_demand", 0.0)),
        "inventory_level": int(approval_data.get("inventory_level", 0)),
        "suggested_reorder_quantity": int(approval_data.get("suggested_reorder_quantity", 0)),
        "action": action,
        "modified_quantity": modified_quantity,
        "manager_notes": str(approval_data.get("manager_notes", "") or ""),
    }

    try:
        with open(APPROVAL_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    logs.append(record)

    with open(APPROVAL_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    return {
        "status": "success",
        "approval_id": approval_id,
        "logged_at": logged_at,
        "record": record
    }

def get_approvals(
    store_id: Optional[str] = None,
    product_id: Optional[str] = None,
    action: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:
    """
    Retrieves decision history logs from approval_log.json with optional filters.
    """
    _ensure_log_file()

    try:
        with open(APPROVAL_LOG_PATH, "r", encoding="utf-8") as f:
            logs = json.load(f)
    except Exception:
        logs = []

    filtered = logs

    if store_id:
        filtered = [item for item in filtered if item.get("store_id") == store_id]

    if product_id:
        filtered = [item for item in filtered if item.get("product_id") == product_id]

    if action:
        filtered = [item for item in filtered if item.get("action") == action.lower()]

    # Sort newest first
    filtered = sorted(filtered, key=lambda x: x.get("logged_at", ""), reverse=True)

    limited = filtered[:limit] if limit > 0 else filtered

    return {
        "total": len(filtered),
        "count": len(limited),
        "approvals": limited
    }
