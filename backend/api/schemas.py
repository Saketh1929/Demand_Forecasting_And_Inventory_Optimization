from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt

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
