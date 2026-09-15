# DFIO AI Inventory Assistant
## Backend and API Handoff Specification

This document describes the current Streamlit interface in `app.py` and the backend contract required to connect it to a demand forecasting service.

## 1. Product Overview

DFIO (Demand Forecasting and Inventory Optimization) helps local retailers estimate future product demand from store, product, inventory, sales, pricing, promotion, weather, and epidemic-related signals.

The current interface provides:

- Region, category, and forecast-period filters in the sidebar.
- Store and product selection.
- Market and environmental inputs.
- Sales and promotion inputs.
- Product attributes and competitor pricing.
- A live summary of key numeric inputs.
- Forecast engine status, model accuracy, and prediction mode indicators.
- A `RUN AI DEMAND FORECAST` action.

The current forecast action is UI-only. It displays a success message and an informational message, but it does not yet call an API or display a prediction.

## 2. Current UI Inputs

### 2.1 Sidebar context filters

| UI label | Suggested API field | Type | Allowed values | Required |
|---|---|---|---|---|
| Select Region | `region` | string | `North`, `South`, `East`, `West` | Yes for forecast |
| Select Category | `category` | string | `Electronics`, `Clothing`, `Groceries`, `Toys`, `Furniture` | Yes for forecast |
| Forecast Period | `forecast_period` | string | `1 week`, `1 month`, `3 months`, `6 months`, `1 year` | Yes for forecast |

These values are currently displayed as the current selection. They should also be included in the forecast request.

### 2.2 Product and location intelligence

| UI label | Suggested API field | Type | Allowed values | Required |
|---|---|---|---|---|
| Store Name | `store_id` | string | `Store 1`, `Store 2`, `Store 3`, `Store 4` | Yes |
| Product ID | `product_id` | string | `P001`, `P002`, `P003`, `P004` | Yes |

The UI currently uses store names as identifiers. The backend may retain these values initially or introduce a separate display name and stable identifier later.

### 2.3 Market and environmental conditions

| UI label | Suggested API field | Type | Range / values | Required |
|---|---|---|---|---|
| Weather Condition | `weather_condition` | string | `Sunny`, `Rainy`, `Snowy`, `Cloudy` | Yes |
| Inventory Level | `inventory_level` | integer | `0` to `1000` | Yes |
| Discount Rate (%) | `discount_rate` | number | `0` to `100` | Yes |

### 2.4 Sales and promotion signals

| UI label | Suggested API field | Type | Range / values | Required |
|---|---|---|---|---|
| Units Sold | `units_sold` | integer | `0` to `1000`, UI step `5` | Yes |
| Units Ordered | `units_ordered` | integer | `0` to `1000`, UI step `5` | Yes |
| Promotion Active | `promotion_active` | boolean | `true` or `false` | Yes |

### 2.5 Product attributes

| UI label | Suggested API field | Type | Allowed values | Required |
|---|---|---|---|---|
| Epidemic | `epidemic` | integer | `0` or `1` | Yes |

`0` means no relevant outbreak and `1` means an outbreak may affect demand.

### 2.6 Competitor pricing

| UI label | Suggested API field | Type | Range / format | Required |
|---|---|---|---|---|
| Competitor Pricing | `competitor_pricing` | number | `0` to `10000`, displayed with two decimals | Yes |

The UI describes this value as the competitor product price in Indian rupees.

## 3. Proposed Forecast API

### Endpoint

```http
POST /api/v1/forecasts
Content-Type: application/json
```

### Request body

```json
{
  "region": "North",
  "category": "Electronics",
  "forecast_period": "1 month",
  "store_id": "Store 1",
  "product_id": "P001",
  "weather_condition": "Sunny",
  "inventory_level": 250,
  "discount_rate": 10.0,
  "units_sold": 120,
  "units_ordered": 100,
  "promotion_active": true,
  "epidemic": 0,
  "competitor_pricing": 1499.00
}
```

### Validation rules

The API should reject the request with HTTP `422` when:

- A required field is missing or null.
- An enum value is outside the values listed above.
- A numeric value is outside its permitted range.
- `inventory_level`, `units_sold`, or `units_ordered` is not an integer.
- `epidemic` is not `0` or `1`.
- `promotion_active` is not a boolean.
- `discount_rate` or `competitor_pricing` is not numeric.

The backend should validate the same rules as the UI. Backend validation remains authoritative because requests may come from clients other than Streamlit.

## 4. Proposed Forecast Response

```json
{
  "request_id": "fcst_01JABC123",
  "status": "completed",
  "prediction": {
    "forecast_period": "1 month",
    "predicted_demand_units": 184,
    "recommended_order_quantity": 134,
    "expected_stockout_risk": 0.12,
    "confidence_score": 0.91
  },
  "model": {
    "name": "dfio-demand-forecast",
    "version": "1.0.0",
    "accuracy": 0.942
  },
  "input_echo": {
    "region": "North",
    "category": "Electronics",
    "store_id": "Store 1",
    "product_id": "P001"
  },
  "generated_at": "2026-09-13T12:00:00Z"
}
```

### Response field definitions

| Field | Type | Description |
|---|---|---|
| `request_id` | string | Traceable identifier for the forecast request. |
| `status` | string | `completed`, `queued`, or `failed`. |
| `predicted_demand_units` | integer | Expected units demanded during the selected period. |
| `recommended_order_quantity` | integer | Suggested replenishment quantity based on demand and current inventory. |
| `expected_stockout_risk` | number | Probability from `0` to `1`. |
| `confidence_score` | number | Model confidence from `0` to `1`. |
| `model.accuracy` | number | Model accuracy from `0` to `1`; display as a percentage in the UI. |
| `generated_at` | string | ISO 8601 UTC timestamp. |

The exact prediction fields may change with the selected forecast horizon, but the response should always provide a demand estimate, a replenishment recommendation, confidence, and model metadata.

## 5. Error Contract

All API errors should use a consistent structure:

```json
{
  "status": "failed",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "One or more input values are invalid.",
    "fields": {
      "discount_rate": "Must be between 0 and 100."
    }
  },
  "request_id": "fcst_01JABC123"
}
```

Recommended status codes:

| HTTP status | Use |
|---|---|
| `400` | Malformed JSON or invalid request shape |
| `422` | Valid JSON with invalid field values |
| `404` | Store or product does not exist |
| `409` | Conflicting or stale inventory data |
| `429` | Rate limit exceeded |
| `500` | Unexpected server error |
| `503` | Forecast model unavailable |

## 6. Backend Responsibilities

1. Expose the forecast endpoint and validate every request.
2. Resolve the store, product, region, and category against master data.
3. Convert the API payload into the feature order expected by the trained model.
4. Load the approved model version and preprocessing pipeline.
5. Generate demand, replenishment, risk, and confidence outputs.
6. Record request metadata and model version for auditability.
7. Return consistent error responses without exposing stack traces or sensitive internals.
8. Apply authentication, authorization, rate limiting, and HTTPS before production use.
9. Support health and readiness checks for the UI status indicators.

## 7. Supporting Endpoints

The UI currently hard-codes its select-box options. For production, these can be served by metadata endpoints:

```http
GET /api/v1/regions
GET /api/v1/categories
GET /api/v1/stores?region=North
GET /api/v1/products?category=Electronics
GET /api/v1/model/status
GET /api/v1/health
```

Suggested model status response:

```json
{
  "status": "online",
  "model_name": "dfio-demand-forecast",
  "model_version": "1.0.0",
  "accuracy": 0.942,
  "last_loaded_at": "2026-09-13T11:45:00Z"
}
```

## 8. Streamlit Integration Flow

When the user clicks the forecast button, the frontend should:

1. Confirm all required select-box inputs have values.
2. Build the request object using the field names in Section 3.
3. Send the object to `POST /api/v1/forecasts`.
4. Show a spinner while the request is in progress.
5. Render the returned forecast, recommendation, risk, confidence, and model version.
6. Show field-level validation errors when the API returns `422`.
7. Show a retry-friendly message for model or network failures.
8. Keep the submitted request and response associated with `request_id`.

The current success message can remain as the loading/completion state, but it should be driven by the actual API response rather than by the button click alone.

## 9. Suggested UI Result Area

After a successful response, the page should display:

- Predicted demand in units.
- Recommended order quantity.
- Stockout risk as a percentage.
- Confidence score as a percentage.
- Model accuracy and model version.
- Forecast generation timestamp.
- The selected store, product, region, category, and period.

If the API returns `queued`, the UI should show the request ID and poll a job-status endpoint such as `GET /api/v1/forecasts/{request_id}`. For an initial synchronous model, `completed` is sufficient.

## 10. Non-Functional Requirements

- Use HTTPS in all non-local environments.
- Add authentication before exposing the API outside the local machine.
- Log request IDs, latency, model version, and outcome.
- Do not log sensitive user or business data unnecessarily.
- Set request and model inference timeouts.
- Version the API under `/api/v1` so future model changes do not silently break the UI.
- Add automated tests for validation, successful predictions, model failures, and unknown store/product combinations.
- Add OpenAPI documentation so the endpoint can be consumed by frontend and integration teams.

## 11. Current Implementation Status

| Area | Status |
|---|---|
| Streamlit layout and styling | Implemented |
| Sidebar filters and current-selection display | Implemented |
| Forecast input controls | Implemented |
| Live numeric input summary | Implemented |
| Forecast button | Implemented as UI action only |
| API request integration | Pending |
| Input validation before submission | Pending |
| Model inference service | Pending |
| Forecast result visualization | Pending |
| Persistent forecast history | Pending |
| Authentication and production security | Pending |
