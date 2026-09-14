import math
from typing import Optional, Dict, Any

def evaluate_inventory(
    predicted_demand: float,
    inventory_level: int,
    horizon_days: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluates inventory gap, status classification, 20% safety stock reorder quantity,
    urgency level, and approval requirements over the forecast horizon.
    
    Formula: Gap = Inventory Level - Predicted Demand
    """
    inventory_level = int(inventory_level)
    predicted_demand = float(predicted_demand)
    
    gap = round(inventory_level - predicted_demand, 2)
    horizon_suffix = f" over {horizon_days}-day horizon" if horizon_days else ""
    
    if gap < 0:
        status = "Shortage"
        shortage_amount = abs(gap)
        # 20% safety stock buffer on shortage quantity
        reorder_quantity = int(math.ceil(shortage_amount * 1.20))
        urgency = "Critical" if gap < -50 else "Moderate"
        requires_approval = True
        description = (
            f"Inventory shortage detected! Current stock ({inventory_level}) is below predicted demand "
            f"({predicted_demand}){horizon_suffix}. Suggested reorder includes a 20% safety buffer."
        )
    elif predicted_demand > 0 and gap > (predicted_demand * 0.20):
        status = "Overstock"
        reorder_quantity = 0
        urgency = "None"
        requires_approval = False
        description = (
            f"Excess inventory detected! Stock level ({inventory_level}) exceeds predicted demand "
            f"({predicted_demand}){horizon_suffix} by over 20%. Consider promotions or holding further replenishment."
        )
    else:
        status = "Sufficient"
        reorder_quantity = 0
        urgency = "None"
        requires_approval = False
        description = (
            f"Inventory level ({inventory_level}) is optimal for forecasted demand "
            f"({predicted_demand}){horizon_suffix}. No reorder required."
        )

    res = {
        "predicted_demand": predicted_demand,
        "inventory_level": inventory_level,
        "gap": gap,
        "status": status,
        "reorder_quantity": reorder_quantity,
        "urgency": urgency,
        "requires_approval": requires_approval,
        "description": description,
    }
    if horizon_days is not None:
        res["horizon_days"] = horizon_days
    return res
