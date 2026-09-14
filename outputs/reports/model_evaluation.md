# M3 Model Evaluation Report

## Model
Optimized XGBoost regressor trained from `data/preprocessed_sales_data.csv`.
All preprocessed rows (74,600 total) are used. Multi-day forecast horizons are generated recursively at inference time by ForecastEngine.

## Chronological Split
| Set | Date range | Rows |
|---|---|---:|
| Train | Through 2023-06-30 | 53,200 |
| Validation | 2023-07-01 to 2023-09-30 | 9,200 |
| Test | 2023-10-01 onward | 12,200 |

## Metrics
| Set | MAE | RMSE | WAPE (%) | R² |
|---|---:|---:|---:|---:|
| Validation | 18.7541 | 26.3455 | 16.2371 | 0.6964 |
| Test | 16.8299 | 23.0801 | 17.0537 | 0.7361 |

Residual standard deviation on the held-out test set: `23.0798`.

## Feature Contract
Target column: `Demand` (Demand at date t).
Endogenous variables (Inventory Level, Units Sold, Units Ordered) are excluded from model features.
Total features: **38** (see `FEATURE_ORDER` in `src/data_preprocessing.py`).

```text
Store ID, Product ID, Price, Discount, Promotion, Competitor Pricing, Epidemic, day_of_week, month, day_of_month, week, quarter, year, is_weekend, lag_1, lag_7, lag_14, rolling_mean_7, rolling_std_7, rolling_mean_14, rolling_std_14, Category_Clothing, Category_Electronics, Category_Furniture, Category_Groceries, Category_Toys, Region_East, Region_North, Region_South, Region_West, Weather Condition_Cloudy, Weather Condition_Rainy, Weather Condition_Snowy, Weather Condition_Sunny, Seasonality_Autumn, Seasonality_Spring, Seasonality_Summer, Seasonality_Winter
```

## Deployment Artifact
`models/best_model.pkl` — retrained on Train + Validation after evaluation.

> **Note**: The deployment model does NOT have a `feature_order_` attribute.
> `ForecastEngine` therefore uses the modern `preprocess_input()` path
> (38-feature contract) for all inference.
