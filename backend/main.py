from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, Query, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.config import (
    STORES, CATEGORIES, REGIONS, WEATHER_CONDITIONS, SEASONS,
    PROMOTION_OPTIONS, EPIDEMIC_OPTIONS, MODEL_PATH, GEMINI_API_KEY
)
from backend.forecasting import load_model
from backend.agent import run_agent_pipeline
from backend.approval import log_approval, get_approvals

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

# Pydantic Request & Response Schemas
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
    predicted_demand: float
    inventory_level: int
    gap: float
    status: str
    reorder_quantity: int
    urgency: str
    requires_approval: bool
    recommendation: str
    reasoning_source: str
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

@app.post("/api/forecast", response_model=ForecastResponse, tags=["Forecasting"])
def create_forecast(payload: ForecastRequest):
    """
    Generates demand forecast, evaluates inventory gap/status, and returns AI agent recommendation.
    """
    try:
        raw_input = payload.model_dump()
        result = run_agent_pipeline(raw_input)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forecast pipeline error: {str(e)}"
        )

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
