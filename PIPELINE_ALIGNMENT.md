# Pipeline Schema Alignment

## Purpose

This document describes the corrected demand forecasting pipeline implemented in:

- `src/data_preprocessing.py`
- `src/model_traning/train_model.py`
- `src/forecasting.py`
- `tests/`

The pipeline now uses one canonical 38-feature contract from data preparation through model training and recursive inference.

## Ground Truth

The raw dataset contains these categorical values:

| Column | Values |
|---|---|
| Category | Clothing, Electronics, Furniture, Groceries, Toys |
| Region | East, North, South, West |
| Weather Condition | Cloudy, Rainy, Snowy, Sunny |
| Seasonality | Autumn, Spring, Summer, Winter |

The preprocessing schema includes every value present in the raw data. This prevents valid production records from being rejected during preparation or inference.

## Canonical Feature Contract

`FEATURE_ORDER` in `src/data_preprocessing.py` is the single source of truth. It contains exactly 38 features in this order:

1. Store ID
2. Product ID
3. Price
4. Discount
5. Promotion
6. Competitor Pricing
7. Epidemic
8. day_of_week
9. month
10. day_of_month
11. week
12. quarter
13. year
14. is_weekend
15. lag_1
16. lag_7
17. lag_14
18. rolling_mean_7
19. rolling_std_7
20. rolling_mean_14
21. rolling_std_14
22. Category_Clothing
23. Category_Electronics
24. Category_Furniture
25. Category_Groceries
26. Category_Toys
27. Region_East
28. Region_North
29. Region_South
30. Region_West
31. Weather Condition_Cloudy
32. Weather Condition_Rainy
33. Weather Condition_Snowy
34. Weather Condition_Sunny
35. Seasonality_Autumn
36. Seasonality_Spring
37. Seasonality_Summer
38. Seasonality_Winter

Inventory and realized-sales fields are intentionally excluded from the model features:

- Inventory Level
- Units Sold
- Units Ordered

This prevents target leakage and keeps inventory decisions downstream from demand prediction.

## Implementation Plan

### 1. Align categorical mappings

`ONE_HOT_MAPPING` was expanded to include all real category values:

- `Clothing`
- `East`
- `Cloudy`
- `Autumn`

The same mapping is used during training and inference. Unknown values still raise a clear `ValueError`.

### 2. Align feature order

The feature list was expanded from the incomplete contract to 38 features by adding the four missing one-hot indicators:

- `Category_Clothing`
- `Region_East`
- `Weather Condition_Cloudy`
- `Seasonality_Autumn`

### 3. Correct seasonality inference

When Seasonality is omitted, the fallback now maps:

- December, January, February -> Winter
- March, April, May -> Spring
- June, July, August -> Summer
- September, October, November -> Autumn

### 4. Align training with preprocessing

`train_model.py` imports `FEATURE_ORDER` and the preprocessing paths directly from `data_preprocessing.py`.

The training target is `Demand`. The old `Target_Demand` and `forecast_horizon` requirements are not part of the canonical pipeline.

Before training, `load_training_data()` validates that the preprocessed CSV contains:

- The `Demand` target column
- Every feature in `FEATURE_ORDER`
- A parseable `Date` column

If the preprocessed file is missing, the script runs `prepare_data()` to generate it. If the file exists but has an incompatible schema, training stops with an actionable error instructing the user to regenerate it.

### 5. Keep deployment inference on the canonical path

The deployment model must not receive a `feature_order_` attribute. `ForecastEngine` uses that attribute as the legacy-model detection flag.

Without the flag, inference calls `preprocess_input()` and sends the same 38-feature order used during training.

## Pipeline Walkthrough

### Step 1: Raw data validation

`prepare_data()` loads `data/sales_data.csv`, parses `Date`, sorts each Store ID/Product ID series chronologically, and validates required columns, duplicates, missing values, ordering, and minimum history.

Each store-product series must contain at least 14 observations.

### Step 2: Feature engineering

For each store-product series:

- Calendar features are derived from the observation date.
- `lag_1`, `lag_7`, and `lag_14` are calculated from historical `Demand`.
- Seven-day and 14-day rolling means and standard deviations exclude the current target row.
- Store ID and Product ID are deterministically label encoded.
- The four categorical columns are one-hot encoded with the complete mapping.

Rows without enough prior history are removed. The resulting training target remains `Demand` at date `t`.

### Step 3: Artifact generation

The preprocessing stage writes:

- `data/preprocessed_sales_data.csv`
- `models/encoders.pkl`
- `models/feature_columns.pkl`

The feature-column artifact contains the authoritative 38-feature order.

### Step 4: Model training

`src/model_traning/train_model.py`:

1. Loads or generates the canonical preprocessed dataset.
2. Verifies `Demand` and all 38 features.
3. Sorts records by date.
4. Creates chronological train, validation, and test partitions.
5. Trains the evaluation model on the training partition.
6. Calculates validation and test metrics.
7. Retrains the deployment model on train plus validation data.
8. Saves `models/best_model.pkl` without `feature_order_`.
9. Writes the residual standard deviation and evaluation report.

### Step 5: Recursive inference

`ForecastEngine` predicts one day at a time:

1. Normalize the current input row.
2. Build the 38-feature vector with `preprocess_input()`.
3. Predict `Demand(t)`.
4. Clamp negative predictions to zero.
5. Append the prediction to working history.
6. Advance the date by one day.
7. Repeat until the requested horizon is complete.

The newly predicted demand becomes the next step's lag and rolling-history input. Future actual demand is never used during the rollout.

## Execution Plan

Run these commands from the repository root after changing the source or raw data:

```powershell
python src/data_preprocessing.py
python src/model_traning/train_model.py
python -m pytest tests/ -v
```

The preprocessing and training commands regenerate the model artifacts. The final command verifies the complete preprocessing, training, and recursive forecasting behavior.

## Verification Status

The complete test suite was run after the implementation changes:

```text
43 passed in 12.78s
```

The tests cover:

- Exact 38-feature count and order
- Real categorical values, including Clothing, East, Cloudy, and Autumn
- Lag and rolling-window alignment
- Leakage prevention
- Input validation and unknown-category rejection
- Autumn seasonality inference
- Recursive forecast history updates
- Forecast horizon handling
- Negative prediction clamping
- Future business schedule overrides
- Deployment model absence of the legacy `feature_order_` flag
- Training-data schema validation

## Current Source of Truth

For implementation behavior, use this order of authority:

1. `src/data_preprocessing.py` for feature definitions and encodings
2. `src/model_traning/train_model.py` for training and artifact generation
3. `src/forecasting.py` for serving and recursive forecasting
4. `tests/` for executable behavioral verification

Older project documents may describe the former 34- or 41-feature pipeline. This document records the corrected implementation and its operating procedure.
