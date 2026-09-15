import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from time import time
from typing import Optional, List, Dict, Any, Literal

from fastapi import FastAPI, Query, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt

from backend.config import (
    STORES, CATEGORIES, REGIONS, WEATHER_CONDITIONS, SEASONS,
    PROMOTION_OPTIONS, EPIDEMIC_OPTIONS, MODEL_PATH, GEMINI_API_KEY,
    API_KEY, ENABLE_AUTH, VALID_STORE_LABELS, VALID_PRODUCT_IDS, MODEL_INFO,
    APPROVED_MODEL_REGISTRY, ALLOWED_ROLES, PROTECTED_ENDPOINT_POLICIES
)
from backend.forecasting import load_model
from backend.agent import run_agent_pipeline
from backend.approval import log_approval, get_approvals

ENABLE_AUTH = ENABLE_AUTH
API_KEY = API_KEY
AUDIT_LOG_PATH = Path(__file__).resolve().parent.parent / "logs" / "forecast_audit.json"
REQUEST_LIMIT_PER_MINUTE = 60
REQUIRE_ROLE_AUTH = True
request_timestamps = defaultdict(list)

app = FastAPI(
    title="Demand Forecasting & Inventory Optimization Agent API",
    description="FastAPI production backend providing ML demand prediction, real-time inventory gap analysis, Gemini LLM rationale, and human-in-the-loop audit approvals.",
    version="1.0.0"
)

# Enable CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def generate_request_id() -> str:
    current_time = datetime.now(timezone.utc)
    return f"fcst_{current_time.strftime('%Y%m%d%H%M%S')}_{abs(hash(current_time.microsecond)) % 10000:04d}"


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


def build_error_response(code: str, message: str, fields: Dict[str, str], request_id: Optional[str] = None):
    return {
        "status": "failed",
        "error": {
            "code": code,
            "message": message,
            "fields": fields,
        },
        "request_id": request_id,
    }


def get_required_roles_for_request(request: Request) -> Optional[set]:
    for path_prefix, policy in PROTECTED_ENDPOINT_POLICIES.items():
        if request.url.path.startswith(path_prefix):
            if request.method not in policy.get("methods", set()):
                return None
            return policy.get("roles", set())
    return None


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    field_errors: Dict[str, str] = {}
    for error in exc.errors():
        loc = error.get("loc", [])
        field_name = loc[-1] if len(loc) > 1 else "request"
        field_errors[field_name] = error.get("msg", "Invalid value.")

    return JSONResponse(
        status_code=422,
        content=build_error_response(
            "VALIDATION_ERROR",
            "One or more input values are invalid.",
            field_errors,
            request_id=generate_request_id(),
        ),
    )


@app.middleware("http")
async def enforce_security_and_rate_limits(request: Request, call_next):
    now = time()
    client_ip = request.client.host if request.client else "unknown"
    request_timestamps[client_ip] = [ts for ts in request_timestamps[client_ip] if now - ts < 60]
    if len(request_timestamps[client_ip]) >= REQUEST_LIMIT_PER_MINUTE:
        return JSONResponse(
            status_code=429,
            content={
                "status": "failed",
                "error": {"code": "RATE_LIMIT_EXCEEDED", "message": "Too many requests. Please retry later.", "fields": {}},
                "request_id": generate_request_id(),
            },
        )
    request_timestamps[client_ip].append(now)

    if not ENABLE_AUTH:
        return await call_next(request)

    if request.url.path.startswith("/docs") or request.url.path.startswith("/openapi") or request.url.path.startswith("/redoc"):
        return await call_next(request)

    auth_header = request.headers.get("Authorization", "")
    if auth_header != f"Bearer {API_KEY}":
        return JSONResponse(
            status_code=401,
            content={
                "status": "failed",
                "error": {"code": "UNAUTHORIZED", "message": "Missing or invalid API key.", "fields": {}},
                "request_id": generate_request_id(),
            },
        )

    required_roles = get_required_roles_for_request(request)
    if REQUIRE_ROLE_AUTH and required_roles is not None:
        user_role = request.headers.get("X-User-Role", "").lower()
        if user_role not in ALLOWED_ROLES or user_role not in required_roles:
            return JSONResponse(
                status_code=403,
                content={
                    "status": "failed",
                    "error": {"code": "FORBIDDEN", "message": "You do not have permission to perform this action.", "fields": {}},
                    "request_id": generate_request_id(),
                },
            )
    return await call_next(request)


# Legacy Pydantic models for current agent pipeline
class ForecastRequest(BaseModel):
    date: str = Field(default="2024-02-15", description="Transaction date (YYYY-MM-DD)")
    store_id: str = Field(default="S001", description="Store location ID")
    product_id: str = Field(default="P0001", description="Product SKU ID")
    category: str = Field(default="Groceries", description="Product category")
    region: str = Field(default="North", description="Geographic region")
    inventory_level: int = Field(default=100, ge=0, description="Current stock on hand")
    units_sold: Optional[int] = Field(default=0, ge=0, description="Units sold")
    units_ordered: Optional[int] = Field(default=0, ge=0, description="Units ordered")
    price: float = Field(default=25.0, gt=0, description="Unit price")
    discount: float = Field(default=0.0, ge=0, le=100, description="Discount percentage")
    competitor_pricing: float = Field(default=25.0, gt=0, description="Competitor price benchmark")
    weather_condition: str = Field(default="Sunny", description="Weather state")
    promotion: int = Field(default=0, ge=0, le=1, description="Promotion active (0 or 1)")
    seasonality: str = Field(default="Spring", description="Current season")
    epidemic: int = Field(default=0, ge=0, le=1, description="Epidemic flag (0 or 1)")
    forecast_period: str = Field(default="1 month", description="Requested forecast horizon: 1 week, 1 month, 3 months, 6 months, or 1 year")


class ToolTraceItem(BaseModel):
    step: int
    tool: str
    status: str
    details: str


class ForecastResponse(BaseModel):
    forecast_id: str
    timestamp: str
    date: str
    store_id: str
    product_id: str
    category: str
    forecast_period: str = Field(default="1 month", description="Effective forecast horizon")
    horizon_days: int = Field(default=30, description="Effective forecast horizon in days")
    predicted_demand: float
    inventory_level: int
    gap: float
    status: str
    reorder_quantity: int
    urgency: str
    requires_approval: bool
    recommendation: str
    reasoning_source: str
    expected_stockout_risk: Optional[float] = Field(default=None, description="Stockout risk computed by backend logic")
    confidence_score: Optional[float] = Field(default=None, description="Backend confidence metric, if available")
    tool_trace: List[ToolTraceItem]


class ApprovalRequest(BaseModel):
    forecast_id: Optional[str] = Field(default="", description="Associated forecast ID")
    store_id: str = Field(..., description="Store location ID")
    product_id: str = Field(..., description="Product SKU ID")
    predicted_demand: float = Field(..., description="Predicted demand value")
    inventory_level: int = Field(..., ge=0, description="Current inventory level")
    suggested_reorder_quantity: int = Field(..., ge=0, description="Suggested order quantity")
    action: str = Field(..., description="Action decision: approved, modified, or rejected")
    modified_quantity: Optional[int] = Field(default=None, ge=0, description="Adjusted quantity if modified")
    manager_notes: Optional[str] = Field(default="", description="Manager review feedback/comments")


class OptionsResponse(BaseModel):
    stores: List[str]
    categories: List[str]
    regions: List[str]
    weather_conditions: List[str]
    seasons: List[str]
    promotion_options: List[int]
    epidemic_options: List[int]


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool
    model_path: str
    gemini_key_configured: bool


# V1 API contract models
class ForecastRequestV1(BaseModel):
    region: Literal["North", "South", "East", "West"]
    category: Literal["Electronics", "Clothing", "Groceries", "Toys", "Furniture"]
    forecast_period: Literal["1 week", "1 month", "3 months", "6 months", "1 year"]
    store_id: str = Field(..., description="Store name or identifier, e.g. 'Store 1'")
    product_id: str = Field(..., description="Product ID, e.g. 'P001'")
    weather_condition: Literal["Sunny", "Rainy", "Snowy", "Cloudy"]
    inventory_level: StrictInt = Field(..., ge=0, le=1000)
    discount_rate: StrictFloat = Field(..., ge=0, le=100)
    units_sold: StrictInt = Field(..., ge=0, le=1000)
    units_ordered: StrictInt = Field(..., ge=0, le=1000)
    promotion_active: StrictBool = Field(...)
    epidemic: StrictInt = Field(..., ge=0, le=1)
    competitor_pricing: StrictFloat = Field(..., ge=0, le=10000)


class PredictionResponse(BaseModel):
    forecast_period: str
    predicted_demand_units: int
    recommended_order_quantity: int
    expected_stockout_risk: float
    confidence_score: float


class ModelMetadata(BaseModel):
    name: str = "dfio-demand-forecast"
    version: str = "1.0.0"
    accuracy: float = 0.942


class InputEcho(BaseModel):
    region: str
    category: str
    store_id: str
    product_id: str


class ForecastResponseV1(BaseModel):
    request_id: str
    status: Literal["completed", "queued", "failed"] = "completed"
    prediction: PredictionResponse
    model: ModelMetadata
    input_echo: InputEcho
    generated_at: str


class ModelStatusResponse(BaseModel):
    status: Literal["online", "offline"] = "online"
    model_name: str = "dfio-demand-forecast"
    model_version: str = "1.0.0"
    accuracy: float = 0.942
    last_loaded_at: str


class GenericErrorResponse(BaseModel):
    status: str = "failed"
    error: Dict[str, Any]
    request_id: Optional[str] = None


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


# API Endpoints
@app.get("/api/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Service health check endpoint verifying model state and API availability.
    """
    model = load_model()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        model_path=str(MODEL_PATH),
        gemini_key_configured=bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_google_gemini_api_key_here")
    )


@app.get("/api/v1/health", tags=["Health"])
def health_check_v1():
    model = load_model()
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
        "gemini_key_configured": bool(GEMINI_API_KEY and GEMINI_API_KEY != "your_google_gemini_api_key_here"),
    }


@app.get("/api/options", response_model=OptionsResponse, tags=["Metadata"])
def get_options():
    """
    Returns valid categorical dropdown options for frontend UI population.
    """
    return OptionsResponse(
        stores=STORES,
        categories=CATEGORIES,
        regions=REGIONS,
        weather_conditions=WEATHER_CONDITIONS,
        seasons=SEASONS,
        promotion_options=PROMOTION_OPTIONS,
        epidemic_options=EPIDEMIC_OPTIONS
    )


@app.get("/api/v1/regions", tags=["Metadata"])
def get_regions():
    return {"regions": REGIONS}


@app.get("/api/v1/categories", tags=["Metadata"])
def get_categories():
    return {"categories": CATEGORIES}


@app.get("/api/v1/stores", tags=["Metadata"])
def get_stores(region: Optional[str] = Query(default=None, description="Optional region filter")):
    display_stores = [f"Store {i}" for i in range(1, 6)]
    return {"stores": display_stores}


@app.get("/api/v1/products", tags=["Metadata"])
def get_products(category: Optional[str] = Query(default=None, description="Optional category filter")):
    products = [f"P{i:04d}" for i in range(1, 21)]
    return {"products": products}


@app.get("/api/v1/model/status", response_model=ModelStatusResponse, tags=["Metadata"])
def get_model_status():
    model = load_model()
    return ModelStatusResponse(
        status="online" if model is not None else "offline",
        model_name=MODEL_INFO["name"],
        model_version=MODEL_INFO["version"],
        accuracy=MODEL_INFO["accuracy"],
        last_loaded_at=datetime.now().isoformat(),
    )


@app.get("/api/v1/ready", tags=["Health"])
def readiness_check():
    model = load_model()
    return {
        "status": "ready" if model is not None else "degraded",
        "model_loaded": model is not None,
        "model_name": MODEL_INFO["name"],
        "model_version": MODEL_INFO["version"],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def create_forecast(payload: ForecastRequest):
    """
    Generates demand forecast, evaluates inventory gap/status, and returns AI agent recommendation.
    """
    try:
        raw_input = payload.model_dump()
        result = run_agent_pipeline(raw_input, forecast_period=payload.forecast_period)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast pipeline error: {str(e)}"
        )


@app.post("/api/v1/forecasts", response_model=ForecastResponseV1, tags=["Forecasting"])
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


@app.post("/api/approve", tags=["Approvals"])
def submit_approval(payload: ApprovalRequest):
    """
    Records human manager approval decision into persistent audit log.
    """
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


@app.get("/api/approvals", tags=["Approvals"])
def list_approvals(
    store_id: Optional[str] = Query(default=None, description="Filter by Store ID"),
    product_id: Optional[str] = Query(default=None, description="Filter by Product ID"),
    action: Optional[str] = Query(default=None, description="Filter by action (approved, modified, rejected)"),
    limit: int = Query(default=50, ge=1, le=500, description="Max records to return")
):
    """
    Retrieves human manager approval decision history log.
    """
    try:
        return get_approvals(store_id=store_id, product_id=product_id, action=action, limit=limit)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch approval history: {str(e)}"
        )
