# M3 Model Card

## Training source

`data/preprocessed_sales_data.csv` is kept unchanged. Training filters it to
`forecast_horizon == 1`, which defines the one-day-ahead target
`Target_Demand`. Horizons 7, 30, 90, 180, and 365 are generated recursively
with the same one-day model.

## Model

artifact is `src/models/best_model.pkl`.

## Excluded columns

`Date`, raw Store/Product IDs, `forecast_period`, `forecast_horizon`,
`Inventory Level`, `Units Sold`, `Units Ordered`, and `Target_Demand` are not
model inputs. The inventory and sales columns are excluded to avoid leakage.

## Recursive serving contract

The serving engine must send the exact `feature_order_` stored on the model.
After every prediction, the prediction is appended to history and the lag and
rolling columns are updated before the next day is predicted.

## Evaluation

The generated metrics are recorded in `outputs/reports/model_evaluation.md`.
The current daily-horizon run produced test MAE `34.4715`, RMSE `43.1469`,
WAPE `34.9528%`, and R2 `0.0791`.
