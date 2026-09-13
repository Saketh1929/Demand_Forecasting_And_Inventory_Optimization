# API Alignment Implementation Plan

## 1. Purpose

This document captures the current backend flow, the contract mismatch with the approved API specification, and the exact work required to bring the implementation into alignment.

It is intended to be the working tracker for updating the backend in a structured, reviewable way.

---

## 2. Current Backend Flow

### Existing request flow

The current API is implemented in [backend/main.py](backend/main.py), and the main forecast pipeline is orchestrated in [backend/agent.py](backend/agent.py).

Current flow:

1. Client sends request to `POST /api/forecast`
2. FastAPI validates input using `ForecastRequest`
3. Payload is converted to Python dict through `payload.model_dump()`
4. `run_agent_pipeline(raw_input)` is called
5. `preprocess_input(raw_input)` prepares features
6. `forecast_tool(prep_res)` predicts demand
7. `evaluate_inventory(predicted_demand, inventory_level)` determines status and order quantity
8. Recommendation is generated via Gemini if available, otherwise fallback rules
9. Final response is returned as a flat JSON object with fields such as:
   - forecast_id
   - timestamp
   - date
   - store_id
   - product_id
   - category
   - predicted_demand
   - gap
   - status
   - reorder_quantity
   - urgency
   - recommendation
   - tool_trace

### Current route set

- `GET /api/health`
- `GET /api/options`
- `POST /api/forecast`
- `POST /api/approve`
- `GET /api/approvals`

This is a working internal backend, but it does not match the structured API contract described in the handoff specification.

---

## 3. Spec vs Current Implementation

### 3.1 Route mismatch

| Item | Spec requirement | Current implementation |
|---|---|---|
| Endpoint versioning | `/api/v1/...` | `/api/...` |
| Forecast endpoint | `POST /api/v1/forecasts` | `POST /api/forecast` |
| Health endpoint | `GET /api/v1/health` | `GET /api/health` |
| Metadata endpoints | region/category/store/product/model status | combined `/api/options` |

### 3.2 Payload mismatch

| Item | Spec requirement | Current implementation |
|---|---|---|
| Fields | `region`, `category`, `forecast_period`, `store_id`, `product_id`, `weather_condition`, `inventory_level`, `discount_rate`, `units_sold`, `units_ordered`, `promotion_active`, `epidemic`, `competitor_pricing` | `date`, `store_id`, `product_id`, `category`, `region`, `inventory_level`, `units_sold`, `units_ordered`, `price`, `discount`, `competitor_pricing`, `weather_condition`, `promotion`, `seasonality`, `epidemic` |
| Boolean handling | `promotion_active` must be boolean | `promotion` is integer 0/1 |
| Numeric field names | `discount_rate` and `competitor_pricing` | `discount` and `competitor_pricing` |
| Forecast horizon | explicit `forecast_period` field | no `forecast_period` field |

### 3.3 Response mismatch

| Item | Spec requirement | Current implementation |
|---|---|---|
| Top-level object | `request_id`, `status`, `prediction`, `model`, `input_echo`, `generated_at` | `forecast_id`, `timestamp`, `date`, `store_id`, `product_id`, `category`, `predicted_demand`, `gap`, `status`, `reorder_quantity`, etc. |
| Result shape | nested prediction object | flat object |
| Model metadata | `model.name`, `model.version`, `accuracy` | not returned in required structure |
| Error contract | standard `status + error + request_id` | generic `HTTPException` detail text |

### 3.4 Validation mismatch

Current validation is not yet aligned with the specification’s authoritative backend contract. The API should reject invalid requests with `422` for:

- required missing values
- enum values outside allowed sets
- numeric range violations
- non-integer `inventory_level`, `units_sold`, `units_ordered`
- `epidemic` not equal to `0` or `1`
- `promotion_active` not boolean
- `discount_rate` or `competitor_pricing` not numeric

The current code does not fully enforce these rules in the forma required by the spec.

---

## 4. Present State Summary

### What is already working

- FastAPI app with CORS enabled
- Forecast pipeline exists and runs
- Model preprocessing exists
- Inventory evaluation exists
- Agent recommendation generation exists
- Approval logging exists
- Health metadata retrieval exists

### What is not yet aligned

- API versioning
- route contract
- request schema contract
- response schema contract
- validation behavior
- supporting metadata endpoints
- canonical error response structure
- model status endpoint
- explicit forecast horizon handling
- clean OpenAPI contract for frontend integration

---

## 5. Implementation Requirements

### Phase 1: Contract alignment

Update the backend contract to match the specification.

Required changes:

- add `/api/v1` namespace for all primary endpoints
- create `POST /api/v1/forecasts`
- create `GET /api/v1/health`
- create `GET /api/v1/model/status`
- create metadata endpoints for regions/categories/stores/products
- use request models matching the spec exactly
- return response models matching the expected nested structure
- return consistent error envelopes

### Phase 2: Validation enforcement

Add strict validation that mirrors the UI contract and rejects invalid payloads with HTTP 422.

Required validation rules:

- region must be one of allowed values
- category must be one of allowed values
- `forecast_period` must be one of supported values
- `promotion_active` must be boolean
- `epidemic` must be 0 or 1
- inventory/sales/order values must be integers
- discount and competitor price must be numeric and within allowed ranges
- return field-level validation responses

### Phase 3: Forecast result model

Define a canonical forecast response capable of returning:

- `request_id`
- `status`
- `prediction.forecast_period`
- `prediction.predicted_demand_units`
- `prediction.recommended_order_quantity`
- `prediction.expected_stockout_risk`
- `prediction.confidence_score`
- `model.name`
- `model.version`
- `model.accuracy`
- `input_echo`
- `generated_at`

This should replace the current internal flat response structure.

### Phase 4: Inventory and business logic adaptation

Adapt the current logic in [backend/agent.py](backend/agent.py) to produce output consistent with the contract while preserving the current forecasting logic.

Map current values to spec fields:

- `predicted_demand` -> `prediction.predicted_demand_units`
- `reorder_quantity` -> `prediction.recommended_order_quantity`
- `gap` -> internal business delta, not necessarily exposed directly
- `status` from inventory evaluation -> should be translated to API `status` semantics if needed
- `requires_approval` -> internal/decision data, not part of the public response unless explicitly included

### Phase 5: Error handling

Implement a consistent error format with:

- `status: "failed"`
- `error.code`
- `error.message`
- `error.fields`
- `request_id`

This should replace ad hoc `HTTPException(detail=...)` responses in the main endpoint.

### Phase 6: Supporting endpoints and metadata

Add metadata endpoints so frontend select options are not hard-coded in UI.

Expected endpoints:

- `GET /api/v1/regions`
- `GET /api/v1/categories`
- `GET /api/v1/stores?region=North`
- `GET /api/v1/products?category=Electronics`
- `GET /api/v1/model/status`

### Phase 7: Testing and verification

Add automated tests for:

- valid forecast request succeeds
- invalid values return 422
- model failure path behaves correctly
- unknown store/product combinations are handled
- metadata endpoints return correct options
- output schema matches response model

---

## 6. Files Likely to Change

Primary files:

- [backend/main.py](backend/main.py)
- [backend/agent.py](backend/agent.py)
- [backend/config.py](backend/config.py)
- [backend/forecasting.py](backend/forecasting.py)
- [backend/inventory.py](backend/inventory.py)
- [backend/data_preprocessing.py](backend/data_preprocessing.py)

Potential supporting files:

- tests for validation and endpoint behavior
- OpenAPI-focused docs or schema helper files

---

## 7. Recommended Update Order

1. Define the v1 request/response schema models
2. Add the versioned routes and metadata endpoints
3. Update the agent pipeline to map results into the new schema
4. Add strict validation and standard error responses
5. Verify against sample payloads from the spec
6. Add regression tests
7. Validate with FastAPI OpenAPI docs and smoke testing

---

## 8. Acceptance Criteria

The work is considered complete when:

- all primary routes are under `/api/v1`
- request and response bodies match the spec contract
- invalid payloads return `422` with consistent error objects
- valid forecast response includes prediction/model/input metadata
- metadata and model status endpoints are available
- OpenAPI docs reflect the served contract
- tests pass for valid and invalid forecast scenarios

---

## 9. Notes

This current implementation is a solid internal prototype and is a good base for the formal contract update, but it is not yet ready to be treated as the production API contract described by the spec.

The work should be done in a versioned, incremental manner so the frontend can evolve without breaking against the older prototype structure.
