# Backend API Documentation & Usage Guide

This directory contains the production FastAPI service for the **Demand Forecasting & Inventory Optimization Agent**.

---

## 📌 Features & Architecture

- **Machine Learning & Fallback Forecasting**: Attempts to load `models/best_model.pkl` and `models/encoders.pkl`. If absent, seamlessly falls back to a deterministic, feature-based heuristic forecasting engine.
- **Inventory Optimization**: Computes Gap ($\text{Inventory Level} - \text{Predicted Demand}$), classifies status into `Shortage`, `Sufficient`, or `Overstock`, and calculates a 20% safety stock buffer reorder quantity.
- **Agent Orchestration**: Combines preprocessing, demand prediction, inventory evaluation, and Gemini LLM rationale (`google-genai` SDK with executive fallback).
- **Human-in-the-Loop Audit Logging**: Persists manager decisions (`approved`, `modified`, `rejected`) into `logs/approval_log.json`.

---

## 🚀 Running the Server

Start the FastAPI server using Uvicorn:

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Interactive OpenAPI Swagger documentation will be available at:
👉 **`http://localhost:8000/docs`**

---

## 🔌 Endpoints Reference

### 1. `GET /api/health`
Checks service health status, model load state, and Gemini configuration.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "timestamp": "2026-09-13T11:00:00.000000",
  "model_loaded": false,
  "model_path": "C:\\...\\models\\best_model.pkl",
  "gemini_key_configured": false
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/health"
```

---

### 2. `GET /api/options`
Returns valid dropdown options for Store ID, Category, Region, Weather, Season, etc.

**Response (200 OK):**
```json
{
  "stores": ["S001", "S002", "S003", "S004", "S005"],
  "categories": ["Groceries", "Clothing", "Electronics", "Furniture", "Toys"],
  "regions": ["North", "South", "East", "West"],
  "weather_conditions": ["Sunny", "Cloudy", "Rainy", "Snowy"],
  "seasons": ["Winter", "Spring", "Summer", "Autumn"],
  "promotion_options": [0, 1],
  "epidemic_options": [0, 1]
}
```

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/options"
```

---

### 3. `POST /api/forecast`
Submits store parameters, predicts customer demand, evaluates inventory gap, and returns AI agent recommendations.

**Request Payload Schema:**
```json
{
  "date": "2024-02-15",
  "store_id": "S001",
  "product_id": "P0001",
  "category": "Groceries",
  "region": "North",
  "inventory_level": 100,
  "units_sold": 0,
  "units_ordered": 0,
  "price": 25.0,
  "discount": 5.0,
  "competitor_pricing": 24.5,
  "weather_condition": "Sunny",
  "promotion": 1,
  "seasonality": "Spring",
  "epidemic": 0
}
```

**Response Payload Schema (200 OK):**
```json
{
  "forecast_id": "FC-A1B2C3D4",
  "timestamp": "2026-09-13T11:00:00.000000",
  "date": "2024-02-15",
  "store_id": "S001",
  "product_id": "P0001",
  "category": "Groceries",
  "predicted_demand": 182.0,
  "inventory_level": 100,
  "gap": -82.0,
  "status": "Shortage",
  "reorder_quantity": 99,
  "urgency": "Critical",
  "requires_approval": true,
  "recommendation": "ALERT [Critical URGENCY]: Stockout risk identified for Product P0001 (Groceries) at Store S001...",
  "reasoning_source": "Executive Fallback Rules",
  "tool_trace": [
    {
      "step": 1,
      "tool": "data_preprocessing",
      "status": "success",
      "details": "Extracted temporal & encoded categorical features."
    },
    {
      "step": 2,
      "tool": "forecasting_model",
      "status": "fallback_applied",
      "details": "Predicted Demand: 182.0 units (Source: heuristic_fallback)"
    },
    {
      "step": 3,
      "tool": "inventory_optimizer",
      "status": "success",
      "details": "Status: Shortage | Gap: -82.0 | Reorder Qty: 99 (Urgency: Critical)"
    },
    {
      "step": 4,
      "tool": "agent_llm_reasoning",
      "status": "success",
      "details": "Generated recommendation via Executive Fallback Rules"
    }
  ]
}
```

**Python Example:**
```python
import requests

payload = {
    "date": "2024-02-15",
    "store_id": "S001",
    "product_id": "P0001",
    "category": "Groceries",
    "inventory_level": 100,
    "promotion": 1
}

response = requests.post("http://localhost:8000/api/forecast", json=payload)
print(response.json())
```

---

### 4. `POST /api/approve`
Records a human manager replenishment decision into `logs/approval_log.json`.

**Request Payload Schema:**
```json
{
  "forecast_id": "FC-A1B2C3D4",
  "store_id": "S001",
  "product_id": "P0001",
  "predicted_demand": 182.0,
  "inventory_level": 100,
  "suggested_reorder_quantity": 99,
  "action": "approved",
  "modified_quantity": null,
  "manager_notes": "Order approved for immediate restock."
}
```

**Response Payload Schema (200 OK):**
```json
{
  "status": "success",
  "approval_id": "APP-E5F6G7H8",
  "logged_at": "2026-09-13T11:05:00.000000",
  "record": {
    "approval_id": "APP-E5F6G7H8",
    "logged_at": "2026-09-13T11:05:00.000000",
    "forecast_id": "FC-A1B2C3D4",
    "store_id": "S001",
    "product_id": "P0001",
    "predicted_demand": 182.0,
    "inventory_level": 100,
    "suggested_reorder_quantity": 99,
    "action": "approved",
    "modified_quantity": null,
    "manager_notes": "Order approved for immediate restock."
  }
}
```

---

### 5. `GET /api/approvals`
Retrieves human manager decision history logs with optional query filters.

**Query Parameters:**
- `store_id` (string, optional)
- `product_id` (string, optional)
- `action` (string: `approved`, `modified`, `rejected`, optional)
- `limit` (integer, default: 50)

**cURL Example:**
```bash
curl -X GET "http://localhost:8000/api/approvals?store_id=S001&action=approved&limit=10"
```

---

## 🛡️ Graceful Fallback Guarantees

1. **No Model PKL Required**: If `models/best_model.pkl` is missing, the system uses the feature-based heuristic engine derived from EDA insights.
2. **No Gemini Key Required**: If `GEMINI_API_KEY` is not provided in `.env`, the system generates structured executive rationale rules without erroring out.
3. **Automatic Log Initialization**: `logs/approval_log.json` is created automatically on first manager approval submission.
