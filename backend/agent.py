import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
from backend.config import GEMINI_API_KEY
from backend.data_preprocessing import preprocess_input
from backend.forecasting import ForecastEngine, forecast_tool, resolve_forecast_horizon
from backend.inventory import evaluate_inventory

def generate_fallback_recommendation(
    store_id: str,
    product_id: str,
    category: str,
    predicted_demand: float,
    inventory_level: int,
    status: str,
    reorder_qty: int,
    urgency: str,
    forecast_period: str = "1 month",
    horizon_days: Optional[int] = None,
) -> str:
    """
    Structured executive recommendation generator used when Gemini API key is absent or unavailable.
    """
    period_desc = f" ({forecast_period}, {horizon_days} days)" if horizon_days else f" ({forecast_period})"
    if status == "Shortage":
        return (
            f"ALERT [{urgency} URGENCY]: Stockout risk identified for Product {product_id} ({category}) at Store {store_id}. "
            f"Forecasted demand{period_desc} is {predicted_demand} units against a current stock level of {inventory_level} units. "
            f"Recommended Action: Reorder {reorder_qty} units immediately (includes 20% safety stock buffer)."
        )
    elif status == "Overstock":
        return (
            f"NOTICE: Overstock position for Product {product_id} ({category}) at Store {store_id}. "
            f"Current stock ({inventory_level} units) exceeds forecasted demand{period_desc} ({predicted_demand} units) by >20%. "
            f"Recommended Action: Pause stock replenishment. Consider localized targeted promotions to clear excess inventory."
        )
    else:
        return (
            f"STATUS OPTIMAL: Inventory level ({inventory_level} units) for Product {product_id} ({category}) at Store {store_id} "
            f"is well-aligned with expected demand{period_desc} ({predicted_demand} units). "
            f"Recommended Action: No immediate reorder required. Maintain current stock monitoring."
        )

def query_gemini_llm(prompt: str, return_model: bool = False) -> Any:
    """
    Queries Google Gemini LLM via google-genai SDK if API key is provided.
    Returns None (or (None, None)) if unavailable or if query fails.
    """
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_google_gemini_api_key_here":
        return (None, None) if return_model else None

    try:
        from google import genai
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Try active models in priority order
        for model_name in [
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "gemini-flash-latest",
            "gemini-2.5-flash",
            "gemini-1.5-flash",
        ]:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                if response and hasattr(response, "text") and response.text:
                    text = response.text.strip()
                    return (text, model_name) if return_model else text
            except Exception:
                continue
    except Exception:
        pass
    
    return (None, None) if return_model else None

def run_agent_pipeline(raw_input: dict, forecast_period: str = "1 month") -> dict:
    """
    Complete agent pipeline orchestrator:
    1. Preprocesses input features
    2. Executes the ML forecast for the requested period via ForecastEngine
    3. Evaluates inventory gap & status rules using cumulative horizon demand
    4. Generates executive LLM recommendation & tool execution trace
    """
    forecast_id = f"FC-{uuid.uuid4().hex[:8].upper()}"
    timestamp = datetime.now().isoformat()
    tool_trace = []

    # Step 1: Preprocessing
    prep_res = preprocess_input(raw_input)
    tool_trace.append({
        "step": 1,
        "tool": "data_preprocessing",
        "status": "success",
        "details": f"Extracted temporal & encoded categorical features. Feature count: {len(prep_res.get('feature_names', []))}"
    })

    # Step 2: Forecasting via ForecastEngine (recursive multi-day rollout)
    horizon_days = resolve_forecast_horizon(forecast_period)
    engine = ForecastEngine()
    forecast_res = engine.forecast(current_row=raw_input, horizon=horizon_days)
    predicted_demand = round(float(forecast_res.total_demand), 1)
    daily_forecasts = [round(float(p), 1) for p in forecast_res.predictions]
    mean_daily_demand = round(float(forecast_res.mean_daily_demand), 1)

    tool_trace.append({
        "step": 2,
        "tool": "forecasting_model",
        "status": "success",
        "details": (
            f"Predicted Cumulative Demand: {predicted_demand} units across {horizon_days} days "
            f"(Mean: {mean_daily_demand} units/day; Source: M3 XGBoost ForecastEngine)"
        )
    })

    # Step 3: Inventory Evaluation using cumulative horizon demand
    inventory_level = raw_input.get("inventory_level", 100)
    inventory_res = evaluate_inventory(predicted_demand, inventory_level, horizon_days=horizon_days)
    tool_trace.append({
        "step": 3,
        "tool": "inventory_optimizer",
        "status": "success",
        "details": f"Status: {inventory_res['status']} | Gap: {inventory_res['gap']} | Reorder Qty: {inventory_res['reorder_quantity']} (Urgency: {inventory_res['urgency']})"
    })

    # Step 4: Agent Rationale & Recommendation
    store_id = str(raw_input.get("store_id", "S001"))
    product_id = str(raw_input.get("product_id", "P0001"))
    category = str(raw_input.get("category", "Groceries"))

    llm_prompt = f"""
You are an expert Supply Chain & Inventory Optimization AI Agent.
Analyze the following store inventory & demand forecast result and provide a concise, professional 2-sentence executive summary and recommendation.

Data Context:
- Store ID: {store_id}
- Product ID: {product_id}
- Category: {category}
- Forecast Period: {forecast_period} ({horizon_days} days)
- Current Inventory Level: {inventory_level} units
- Predicted Cumulative Demand: {predicted_demand} units (Mean: {mean_daily_demand} units/day)
- Inventory Gap (Inventory - Demand): {inventory_res['gap']} units
- Stock Status: {inventory_res['status']}
- Suggested Reorder Quantity (includes 20% safety stock buffer): {inventory_res['reorder_quantity']} units
- Urgency Level: {inventory_res['urgency']}

Provide a clear executive recommendation explaining why the action is suggested.
"""
    llm_recommendation, llm_model = query_gemini_llm(llm_prompt, return_model=True)
    if llm_recommendation:
        recommendation = llm_recommendation
        reasoning_source = "Gemini LLM (google-genai)"
        llm_details = f"Generated recommendation via {reasoning_source} (model: {llm_model})"
    else:
        recommendation = generate_fallback_recommendation(
            store_id, product_id, category, predicted_demand, inventory_level,
            inventory_res["status"], inventory_res["reorder_quantity"], inventory_res["urgency"],
            forecast_period=forecast_period, horizon_days=horizon_days,
        )
        reasoning_source = "Executive Fallback Rules"
        llm_details = f"Generated recommendation via {reasoning_source}"

    tool_trace.append({
        "step": 4,
        "tool": "agent_llm_reasoning",
        "status": "success",
        "details": llm_details
    })

    return {
        "forecast_id": forecast_id,
        "timestamp": timestamp,
        "date": raw_input.get("date", datetime.now().strftime("%Y-%m-%d")),
        "store_id": store_id,
        "product_id": product_id,
        "category": category,
        "predicted_demand": predicted_demand,
        "inventory_level": inventory_level,
        "gap": inventory_res["gap"],
        "status": inventory_res["status"],
        "reorder_quantity": inventory_res["reorder_quantity"],
        "urgency": inventory_res["urgency"],
        "requires_approval": inventory_res["requires_approval"],
        "recommendation": recommendation,
        "reasoning_source": reasoning_source,
        "tool_trace": tool_trace,
        "horizon_days": horizon_days,
        "forecast_period": forecast_period,
        "daily_forecasts": daily_forecasts,
        "mean_daily_demand": mean_daily_demand,
    }

