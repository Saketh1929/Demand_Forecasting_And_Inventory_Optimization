import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import JSONResponse

from backend.config import (
    STORES, CATEGORIES, REGIONS, WEATHER_CONDITIONS, SEASONS,
    PROMOTION_OPTIONS, EPIDEMIC_OPTIONS, MODEL_PATH, GEMINI_API_KEY,
    VALID_STORE_LABELS, VALID_PRODUCT_IDS, MODEL_INFO, APPROVED_MODEL_REGISTRY
)
from backend.forecasting import load_model
from backend.agent import run_agent_pipeline
from backend.approval import log_approval, get_approvals
from backend.api.schemas import (
    ForecastRequest, ForecastResponse, ForecastRequestV1, ForecastResponseV1,
    PredictionResponse, ModelMetadata, InputEcho, ApprovalRequest,
    OptionsResponse, HealthResponse, ModelStatusResponse
)
from backend.api.security import generate_request_id, build_error_response

router = APIRouter()
AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent.parent / "logs" / "forecast_audit.json"

def ensure_audit_log() -> None:
    AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.write_text("[]", encoding="utf-8")

def append_forecast_audit(request_id: str, payload: Dict[str, Any], model_version: str, status: str, prediction: Dict[str, Any]) -> None:
    ensure_audit_log()
    records = json.loads(AUDIT_LOG_PATH.read_text(encoding="utf-8") or "[]")
    records.append({
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "payload": payload,
        "model_version": model_version,
        "status": status,
        "prediction": prediction,
    })
    AUDIT_LOG_PATH.write_text(json.dumps(records, indent=2), encoding="utf-8")

def normalize_store_id(raw_store_id: str) -> str:
    raw_store_id = str(raw_store_id).strip()
    if raw_store_id.lower().startswith("store "):
        try:
            store_number = int(raw_store_id.split()[-1])
            return f"S{store_number:03d}"
        except ValueError:
            return raw_store_id
    return raw_store_id

def normalize_product_id(raw_product_id: str) -> str:
    raw_product_id = str(raw_product_id).strip().upper()
    if raw_product_id.lower().startswith("product "):
        try:
            product_number = int(raw_product_id.split()[-1])
            return f"P{product_number:04d}"
        except ValueError:
            return raw_product_id
    if raw_product_id.startswith("P") and raw_product_id[1:].isdigit():
        try:
            return f"P{int(raw_product_id[1:]):04d}"
        except ValueError:
            return raw_product_id
    return raw_product_id

def normalize_legacy_forecast_input(payload: ForecastRequestV1) -> dict:
    legacy_payload = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "store_id": normalize_store_id(payload.store_id),
        "product_id": normalize_product_id(payload.product_id),
        "category": payload.category,
        "region": payload.region,
        "inventory_level": int(payload.inventory_level),
        "units_sold": int(payload.units_sold),
        "units_ordered": int(payload.units_ordered),
        "price": float(payload.competitor_pricing),
        "discount": float(payload.discount_rate),
        "competitor_pricing": float(payload.competitor_pricing),
        "weather_condition": payload.weather_condition,
        "promotion": 1 if payload.promotion_active else 0,
        "seasonality": "Spring",
        "epidemic": int(payload.epidemic),
        "forecast_period": payload.forecast_period,
    }
    return legacy_payload

def compute_stockout_risk(inventory_level: int, predicted_demand: float) -> float:
    if predicted_demand <= 0:
        return 0.0
    if inventory_level <= 0:
        return 1.0
    risk = max(0.0, 1.0 - (inventory_level / max(predicted_demand, 1.0)))
    return round(min(1.0, max(0.0, risk)), 4)

def validate_master_data(payload: ForecastRequestV1) -> None:
    if payload.store_id not in VALID_STORE_LABELS and payload.store_id not in STORES:
        raise HTTPException(status_code=404, detail=f"Store not found: {payload.store_id}")
    if payload.product_id not in VALID_PRODUCT_IDS:
        raise HTTPException(status_code=404, detail=f"Product not found: {payload.product_id}")
    if payload.region not in REGIONS:
        raise HTTPException(status_code=404, detail=f"Region not found: {payload.region}")
    if payload.category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Category not found: {payload.category}")

def validate_model_registry(model_name: str, model_version: str) -> None:
    registry = APPROVED_MODEL_REGISTRY.get(model_name)
    if registry is None:
        raise HTTPException(status_code=503, detail=f"Model not approved: {model_name}")
    if registry.get("version") != model_version:
        raise HTTPException(status_code=503, detail=f"Model version not approved: {model_version}")

@router.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    model = load_model()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        gemini_key_configured=bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_google_gemini_api_key_here")
    )

@router.get("/api/v1/health", tags=["Health"])
def health_check_v1():
    model = load_model()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "gemini_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_google_gemini_api_key_here"),
    }

@router.get("/api/options", response_model=OptionsResponse, tags=["Metadata"])
def get_options():
    return OptionsResponse(
        stores=STORES,
        categories=CATEGORIES,
        regions=REGIONS,
        weather_conditions=WEATHER_CONDITIONS,
        seasons=SEASONS,
        promotion_options=PROMOTION_OPTIONS,
        epidemic_options=EPIDEMIC_OPTIONS
    )

@router.get("/api/v1/regions", tags=["Metadata"])
def get_regions():
    return {"regions": REGIONS}

@router.get("/api/v1/categories", tags=["Metadata"])
def get_categories():
    return {"categories": CATEGORIES}

@router.get("/api/v1/stores", tags=["Metadata"])
def get_stores(region: Optional[str] = Query(default=None, description="Optional region filter")):
    display_stores = [f"Store {i}" for i in range(1, 6)]
    return {"stores": display_stores}

@router.get("/api/v1/products", tags=["Metadata"])
def get_products(category: Optional[str] = Query(default=None, description="Optional category filter")):
    products = [f"P{i:04d}" for i in range(1, 21)]
    return {"products": products}

@router.get("/api/v1/model/status", response_model=ModelStatusResponse, tags=["Metadata"])
def get_model_status():
    model = load_model()
    return ModelStatusResponse(
        status="online" if model is not None else "offline",
        model_name=MODEL_INFO["name"],
        model_version=MODEL_INFO["version"],
        accuracy=MODEL_INFO["accuracy"],
        last_loaded_at=datetime.now().isoformat(),
    )

@router.get("/api/v1/ready", tags=["Health"])
def readiness_check():
    model = load_model()
    return {
        "status": "ready" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": MODEL_INFO["name"],
        "model_version": MODEL_INFO["version"],
        "timestamp": datetime.now().isoformat(),
    }

@router.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def create_forecast(payload: ForecastRequest):
    try:
        raw_input = payload.model_dump()
        result = run_agent_pipeline(raw_input, forecast_period=payload.forecast_period)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast pipeline error: {str(e)}"
        )

@router.post("/api/v1/forecasts", response_model=ForecastResponseV1, tags=["Forecasting"])
def create_forecast_v1(payload: ForecastRequestV1):
    request_id = generate_request_id()
    try:
        validate_master_data(payload)
        validate_model_registry(MODEL_INFO["name"], MODEL_INFO["version"])
        legacy_input = normalize_legacy_forecast_input(payload)
        result = run_agent_pipeline(legacy_input, forecast_period=payload.forecast_period)

        predicted_demand = float(result.get("predicted_demand", 0.0))
        inventory_level = int(legacy_input.get("inventory_level", 0))
        reorder_quantity = int(max(0, result.get("reorder_quantity", 0)))
        risk = compute_stockout_risk(inventory_level, predicted_demand)
        confidence = 0.91 if result.get("reasoning_source") == "Gemini LLM (google-genai)" else 0.78

        prediction_payload = {
            "forecast_period": payload.forecast_period,
            "predicted_demand_units": int(round(predicted_demand)),
            "recommended_order_quantity": reorder_quantity,
            "expected_stockout_risk": risk,
            "confidence_score": round(confidence, 4),
        }

        response = ForecastResponseV1(
            request_id=request_id,
            status="completed",
            prediction=PredictionResponse(**prediction_payload),
            model=ModelMetadata(
                name=MODEL_INFO["name"],
                version=MODEL_INFO["version"],
                accuracy=MODEL_INFO["accuracy"],
            ),
            input_echo=InputEcho(
                region=payload.region,
                category=payload.category,
                store_id=payload.store_id,
                product_id=payload.product_id,
            ),
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )

        append_forecast_audit(
            request_id=request_id,
            payload=payload.model_dump(),
            model_version="1.0.0",
            status="completed",
            prediction=prediction_payload,
        )
        return response
    except Exception as e:
        status_code = 500
        error_code = "FORECAST_ERROR"
        message = f"Forecast pipeline error: {str(e)}"
        if isinstance(e, HTTPException):
            status_code = e.status_code
            error_code = "STORE_NOT_FOUND" if "Store not found" in str(e.detail) else "PRODUCT_NOT_FOUND" if "Product not found" in str(e.detail) else "MODEL_NOT_APPROVED" if "Model not approved" in str(e.detail) else "VALIDATION_ERROR"
            message = str(e.detail)
        error_body = build_error_response(error_code, message, {}, request_id=request_id)
        return JSONResponse(status_code=status_code, content=error_body)

@router.post("/api/approve", tags=["Approvals"])
def submit_approval(payload: ApprovalRequest):
    if payload.action.lower() not in ["approved", "modified", "rejected"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Action must be one of: 'approved', 'modified', or 'rejected'."
        )

    if payload.action.lower() == "modified" and payload.modified_quantity is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="modified_quantity must be provided when action is 'modified'."
        )

    try:
        res = log_approval(payload.model_dump())
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Approval logging error: {str(e)}"
        )

@router.get("/api/approvals", tags=["Approvals"])
def list_approvals(
    store_id: Optional[str] = Query(default=None, description="Filter by Store ID"),
    product_id: Optional[str] = Query(default=None, description="Filter by Product ID"),
    action: Optional[str] = Query(default=None, description="Filter by action (approved, modified, rejected)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return")
):
    try:
        return get_approvals(store_id=store_id, product_id=product_id, action=action, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch approval history: {str(e)}"
        )
