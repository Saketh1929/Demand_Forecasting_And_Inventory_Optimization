# Architecture — Demand Forecasting & Inventory Optimization Agent

This document describes the full end-to-end architecture: data → analysis → model → agent → UI, mapped directly onto the repo's actual folder structure.

---

## 1. Repository Structure

```
Demand-Forecasting-Inventory-Agent/
├── data/
│   └── sales_data.csv              # Raw historical retail dataset (Jan 2022 – Jan 2024)
├── models/
│   └── best_model.pkl              # Trained, serialized forecasting model
├── notebooks/
│   ├── 01_data_understanding.ipynb # Data quality, shape, demand/inventory mismatch discovery
│   ├── 02_eda_problem_discovery.ipynb # Category/region/promo/weather/season/epidemic drivers
│   ├── 03_demand_analysis.ipynb    # Time trends, rolling demand, feature engineering
│   ├── 04_forecasting.ipynb        # Model training, comparison, selection → best_model.pkl
│   └── 05_inventory_analysis.ipynb # Inventory-vs-prediction logic, shortage/reorder rules
├── outputs/
│   ├── figures/                    # Saved charts from the notebooks
│   └── reports/                    # Generated analysis/summary reports
├── src/
│   ├── data_preprocessing.py       # preprocess_input(row: dict) -> np.ndarray
│   ├── forecasting.py              # forecast_tool() — loads best_model.pkl, predicts demand
│   ├── inventory.py                # Compares prediction vs inventory, computes gap/status
│   └── agent.py                    # Orchestrates preprocessing → forecast → inventory → recommendation
├── README.md
└── requirements.txt
```

Every `src/` module has a 1:1 origin in a notebook: the notebooks are where the logic was *discovered and validated*, `src/` is where it's *productionized*.

---

## 2. End-to-End Flow

```
┌────────────────────────────────────────────────────────────────────────┐
│                         OFFLINE / RESEARCH PHASE                         │
│                         (notebooks/ → models/)                           │
└────────────────────────────────────────────────────────────────────────┘

  data/sales_data.csv
        │
        ▼
  01_data_understanding.ipynb
   - shape, dtypes, missing values, duplicates
   - discovers Demand > Inventory Level mismatch (shortage signal)
   - discovers Demand > Units Sold (lost-sales signal)
        │
        ▼
  02_eda_problem_discovery.ipynb
   - Inventory_Shortage & Potential_Lost_Sales indicators defined
   - drivers analyzed: Category, Region, Promotion, Weather Condition,
     Seasonality, Epidemic, Discount, Competitor Pricing
        │
        ▼
  03_demand_analysis.ipynb
   - time trend + rolling/moving-average demand analysis
   - feature engineering: Year, Month, Day, DayOfWeek, WeekOfYear,
     Quarter, IsWeekend, categorical encodings
        │
        ▼
  04_forecasting.ipynb
   - train/test split, model candidates (e.g. regression / gradient boosting)
   - hyperparameter tuning, evaluation (RMSE/MAE/R²)
   - best model serialized → models/best_model.pkl
        │
        ▼
  05_inventory_analysis.ipynb
   - defines shortage / sufficient-stock / overstock rules from
     predicted demand vs inventory level
   - validates the gap logic that inventory.py later encodes
        │
        ▼
  src/ (productionized logic, imported by the agent — not re-derived)

┌────────────────────────────────────────────────────────────────────────┐
│                          ONLINE / SERVING PHASE                          │
│                    (form UI → src/agent.py → response)                   │
└────────────────────────────────────────────────────────────────────────┘

  User fills form:
   Store ID · Product ID · Category · Region · Weather Condition ·
   Seasonality · Promotion · Epidemic · Price · Discount · Inventory Level
        │
        ▼
  Backend API receives structured payload
        │
        ▼
  src/data_preprocessing.py
   preprocess_input(row: dict) -> np.ndarray
   - applies same encodings/feature engineering as notebook 03
        │
        ▼
  src/forecasting.py
   forecast_tool(features: np.ndarray) -> float
   - loads models/best_model.pkl once at startup
   - returns predicted demand
        │
        ▼
  src/inventory.py
   - Gap = Predicted Demand − Inventory Level
   - Status = Shortage / Sufficient / Overstock (rules validated in notebook 05)
        │
        ▼
  src/agent.py
   - orchestrates the three modules above
   - LLM reasoning layer (Gemini) turns the numeric result into a
     natural-language recommendation
   - returns a structured response + the tool-call trace shown in the UI
        │
        ▼
  Frontend result card:
   Predicted Demand · Gap · Status · Recommendation · Tool-call trace
```

---

## 3. Dataset Schema (confirmed from `01_data_understanding.ipynb`)

| Column | Role |
|---|---|
| Store ID | Categorical — store identifier |
| Product ID | Categorical — product identifier |
| Category | Categorical — product category (e.g. Groceries, Clothing, Furniture) |
| Region | Categorical — store region (e.g. North, South, East, West) |
| Date | Temporal — Jan 2022 to Jan 2024 |
| Demand | Target variable |
| Units Sold | Actual sales realized |
| Units Ordered | Replenishment quantity |
| Inventory Level | Stock on hand |
| Price | Selling price |
| Discount | Discount applied |
| Competitor Pricing | External pricing signal |
| Weather Condition | Sunny / Cloudy / Rainy / Snowy |
| Seasonality | Winter / Spring / Summer / Autumn |
| Promotion | Yes/No flag |
| Epidemic | Yes/No flag |

**Engineered features** (from `03_demand_analysis.ipynb`): `Year`, `Month`, `Day`, `DayOfWeek`, `WeekOfYear`, `Quarter`, `IsWeekend`.

**Derived business indicators** (from `01`/`02`): `Inventory_Shortage` (Demand > Inventory Level), `Potential_Lost_Sales` (Demand > Units Sold), `Demand_Sales_Difference`, `Demand_Inventory_Difference` — these are the signals `inventory.py` and the agent's reasoning are built on.

---

## 4. Why This Architecture (grounded in your own EDA findings)

The notebooks justify specific design choices rather than generic ones:

- **Forecasting must use more than raw historical demand** — Category, Region, Promotion, Weather, Seasonality, Epidemic, and Discount all show measurable relationships with Demand (notebook 02). This is why `preprocess_input()` needs the full feature set, not just a time series of past demand.
- **Inventory logic needs its own layer, separate from the model** — the model predicts demand; whether that demand constitutes a "problem" depends on comparing it to `Inventory Level`, which is exactly what notebooks 01/02 formalized as `Inventory_Shortage`. This is why `inventory.py` is a distinct module from `forecasting.py`.
- **The agent's job is reasoning, not prediction** — the ML model and the inventory-gap rules already produce the numbers; the LLM layer's only job is turning `{predicted_demand, gap, status}` into a clear recommendation sentence, plus (optionally) incorporating free-text context the structured data can't capture.

---

## 5. Tech Stack

| Layer | Technology |
|---|---|
| Data analysis / feature engineering | Python, pandas, numpy, matplotlib |
| Model training | scikit-learn / gradient boosting (finalized in `04_forecasting.ipynb`), serialized with `pickle`/`joblib` → `best_model.pkl` |
| Backend API | Python (FastAPI or Flask) |
| Preprocessing | `src/data_preprocessing.py` — mirrors notebook 03's encodings |
| Forecasting | `src/forecasting.py` — wraps `best_model.pkl` |
| Inventory logic | `src/inventory.py` — shortage/sufficient/overstock rules from notebook 05 |
| Agent reasoning | `src/agent.py` + Gemini API |
| Frontend | Streamlit (form + dropdowns + result card) |
| Reports/figures | `outputs/figures/`, `outputs/reports/` |

---

## 6. Module Responsibility Map

| File | Responsibility | Notebook it comes from |
|---|---|---|
| `src/data_preprocessing.py` | `preprocess_input(row: dict) -> np.ndarray` | 03_demand_analysis.ipynb (feature engineering) |
| `src/forecasting.py` | `forecast_tool(features) -> float`, loads `best_model.pkl` | 04_forecasting.ipynb |
| `src/inventory.py` | Gap + status calculation | 01, 02, 05 (shortage/lost-sales definitions) |
| `src/agent.py` | Orchestration + LLM recommendation | New — ties the above together |

---

## 7. Notes for the Team

- Whoever owns `04_forecasting.ipynb` and `05_inventory_analysis.ipynb` should keep the feature list and encoding order **identical** to what `03_demand_analysis.ipynb` produces — `preprocess_input()` must match exactly what the model was trained on, or predictions will be silently wrong rather than erroring out.
- `best_model.pkl` should be re-saved (and its expected feature order documented in `forecasting.py`) any time the feature engineering notebook changes.
- The categorical values (Category, Region, Weather Condition, Seasonality) should be encoded the same way in both training and `preprocess_input()` — e.g. a shared encoder/mapping saved alongside `best_model.pkl`, not hand-duplicated in two places.
