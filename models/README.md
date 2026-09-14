# M3 Model Card

## Training Source

`data/preprocessed_sales_data.csv` is dynamically generated from `data/sales_data.csv`. The target variable is `Demand`. Horizons 7, 30, 90, 180, and 365 are generated recursively using the single-step model via `ForecastEngine`.

## Model

The authoritative model artifact is located at `models/best_model.pkl`.

## Features

The model uses exactly **38 features** based on the `FEATURE_ORDER` contract defined in `src.data_preprocessing`. 
`Date`, raw Store/Product IDs, `Inventory Level`, `Units Sold`, and `Units Ordered` are explicitly excluded to prevent data leakage.

## Recursive Serving Contract

The serving engine strictly enforces the 38-feature `FEATURE_ORDER` contract. 
After every prediction step, the prediction is appended to the internal working history. Lag and rolling statistics are recalculated purely based on past demand and prior predictions before the next day is predicted.

## Evaluation

The generated metrics are recorded in `outputs/reports/model_evaluation.md`.
The latest daily-horizon evaluation on the canonical 38-feature pipeline produced:
- **R²**: `0.7361`
- **MAE**: `16.8299`
- **RMSE**: `23.0801`
- **WAPE**: `17.0537%`
