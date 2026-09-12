# Demand Forecasting and Inventory Optimization

A retail demand forecasting foundation for predicting demand across store-product pairs and preparing the results for inventory planning. The current implementation focuses on reliable data preparation, historical model evaluation, and recursive future forecasting with XGBoost.

## What the Project Does

The workflow:

1. Loads historical sales data.
2. Removes fields that would leak future inventory or sales information.
3. Creates calendar, lag, and rolling-demand features.
4. Compares a seven-day moving-average baseline with plain and optimized XGBoost models.
5. Trains the selected XGBoost model on all available history.
6. Forecasts future demand for any positive horizon, such as 1, 7, or 30 days.
7. Produces uncertainty bounds and an in-memory demand handoff for a future inventory optimizer.

The forecasting output is currently printed to the console and held in memory. A production API, dashboard, and inventory ordering workflow are planned but are not implemented yet.

## Repository Structure

```text
Demand_Forecasting_&_Inventory_Optimization_Agent/
├── data/
│   ├── raw/sales_data.csv                 # Input sales history
│   └── processed/sales_data_processed.csv # Generated model-ready data
├── forecasting/
│   ├── demand_forecast_pipeline.py       # Standalone preprocessing script
│   ├── forecast_core.py                  # Shared feature and model functions
│   ├── forecast_engine.py                # Future recursive forecasting
│   └── simple_model.py                   # Historical model comparison
├── inventory/                            # Reserved for inventory optimization
├── api/                                  # Reserved for a future service layer
├── dashboard/src/                        # Reserved for a future dashboard
├── outputs/
│   └── model_residual_std.txt            # Forecast uncertainty calibration
├── docs/                                 # Architecture and implementation guides
└── .gitignore
```

The raw and processed CSV datasets are excluded from version control by `.gitignore`. Provide the expected input data locally before running the pipeline.

## Requirements

- Python 3.10 or newer
- pip

Install the Python dependencies from the project root:

```powershell
python -m pip install pandas numpy scikit-learn xgboost tabulate
```

For an isolated environment on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install pandas numpy scikit-learn xgboost tabulate
```

## Run the Workflow

Run these commands from the project root:

### 1. Build the processed dataset

```powershell
python forecasting/demand_forecast_pipeline.py
```

This reads `data/raw/sales_data.csv`, engineers model features, and writes `data/processed/sales_data_processed.csv`.

### 2. Evaluate the models

```powershell
python forecasting/simple_model.py
```

This performs a chronological train/test evaluation and reports MAE, RMSE, and WAPE for the moving-average baseline, plain XGBoost, and optimized XGBoost models. It also writes the calibrated residual standard deviation used by the forecast engine to `outputs/model_residual_std.txt`.

### 3. Generate future demand forecasts

```powershell
python forecasting/forecast_engine.py --horizon 1
python forecasting/forecast_engine.py --horizon 7
python forecasting/forecast_engine.py --horizon 30
```

The `--horizon` value can be any positive integer. If it is omitted, the script prompts for a horizon and defaults to seven days when no valid input is provided.

## Forecast Results

The engine generates records containing:

- `Forecast_Date`
- `Store_ID` and `Product_ID`
- `Predicted_Demand`
- `Low` and `High` uncertainty estimates
- `Horizon_Expected_Demand`
- Original store and product labels

It also prints daily aggregate statistics and an `inventory_demand` table with:

- `Store_ID`
- `Product_ID`
- `Horizon_Expected_Demand`
- `Expected_Daily_Demand`

This inventory handoff is currently in memory only; no forecast CSV or order recommendation file is generated.

## Modeling Approach

`forecast_core.py` is the shared production feature contract. It prepares:

- Calendar features such as weekday, month, week, quarter, year, and weekend flag.
- Demand lags for 1, 7, and 14 days.
- Seven-day and fourteen-day rolling means and standard deviations.
- Encoded store and product identifiers.
- One-hot encoded categorical business features.

The production model is an optimized `xgboost.XGBRegressor`. Multi-day forecasts are recursive: each predicted day is added to the history used to calculate features for the next day.

## Current Limitations and Roadmap

- Inventory coverage, safety stock, reorder points, target inventory, risk classification, and order quantities are not implemented yet.
- The API and dashboard directories are placeholders.
- Forecast results are not persisted to files or a database.
- Static inputs such as price, discount, weather, and seasonality use their latest known values throughout a forecast horizon.
- Recursive forecast errors can compound over longer horizons.
- Model comparison currently uses a chronological train/test split; walk-forward validation and additional models are future improvements.

## Documentation

- [Forecasting Architecture](docs/Forecasting_Architecture.md)
- [Forecasting Code Guide](docs/Forecasting_Code_Guide.md)
- [Finalized Architecture Gap Analysis](docs/Finalized_Architecture_Gap_Analysis.md)
- [Project Documentation](docs/Documentation.md)

## Git Workflow

From the project directory:

```powershell
git status
git add .
git commit -m "Describe your change"
git push origin main
```
