"""Train and evaluate the M3 model using the canonical 38-feature pipeline.

Pipeline contract:
    data_preprocessing.prepare_data()
        -> data/preprocessed_sales_data.csv  (Date + 38 FEATURE_ORDER cols + Demand)
        -> src/models/encoders.pkl
        -> src/models/feature_columns.pkl
    train_model.train()
        -> src/models/best_model.pkl         (XGBoost, NO feature_order_ attribute)
        -> outputs/model_residual_std.txt
        -> outputs/reports/model_evaluation.md

Key design rules:
1. TARGET_COLUMN = "Demand"  (matches data_preprocessing.prepare_data() output)
2. FEATURE_ORDER is imported directly from data_preprocessing — single source of truth.
3. The deployment model must NOT have a `feature_order_` attribute set.
   ForecastEngine uses that attribute as a flag to route to the legacy 41-feature path.
   Without it, ForecastEngine correctly uses preprocess_input() with the 38-feature contract.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# Ensure project root is on sys.path so src.data_preprocessing is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

from src.data_preprocessing import (
    DATA_PATH,
    ENCODERS_PATH,
    FEATURE_COLUMNS_PATH,
    FEATURE_ORDER,
    PROCESSED_PATH,
    prepare_data,
)

PROJECT_ROOT = _PROJECT_ROOT
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "preprocessed_sales_data.csv"
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "best_model.pkl"
RESIDUAL_PATH = PROJECT_ROOT / "outputs" / "model_residual_std.txt"
REPORT_PATH = PROJECT_ROOT / "outputs" / "reports" / "model_evaluation.md"

# The target column produced by data_preprocessing.prepare_data()
TARGET_COLUMN = "Demand"


def load_training_data() -> pd.DataFrame:
    """Load the canonical M3 training dataset, regenerating it from raw data if absent.

    The preprocessed CSV must have been produced by data_preprocessing.prepare_data().
    If it does not exist (first run), prepare_data() is called automatically.
    """
    if not PROCESSED_DATA_PATH.exists():
        print(
            f"Preprocessed dataset not found at {PROCESSED_DATA_PATH}. "
            "Running data_preprocessing.prepare_data() to generate it..."
        )
        prepare_data(
            data_path=DATA_PATH,
            processed_path=PROCESSED_PATH,
            encoders_path=ENCODERS_PATH,
            feature_columns_path=FEATURE_COLUMNS_PATH,
        )

    processed = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["Date"])

    if TARGET_COLUMN not in processed.columns:
        raise ValueError(
            f"Preprocessed CSV at {PROCESSED_DATA_PATH} is missing the target column "
            f"'{TARGET_COLUMN}'. Delete the file and re-run to regenerate:\n"
            "    python src/data_preprocessing.py"
        )

    missing_features = [c for c in FEATURE_ORDER if c not in processed.columns]
    if missing_features:
        raise ValueError(
            f"Preprocessed CSV is missing {len(missing_features)} required feature "
            f"columns: {missing_features}.\n"
            "Delete the file and re-run to regenerate:\n"
            "    python src/data_preprocessing.py"
        )

    return processed.sort_values("Date").reset_index(drop=True)


def get_feature_order() -> list[str]:
    """Return the authoritative feature list imported from data_preprocessing.

    This is the single source of truth — identical at training and inference time.
    Currently 38 features (2 identifiers + 5 business + 7 calendar + 3 lags +
    4 rolling + 5+4+4+4 one-hot dummies for Category/Region/Weather/Seasonality).
    """
    return list(FEATURE_ORDER)


def _metrics(y_true: pd.Series, predictions: np.ndarray, split: str) -> dict[str, Any]:
    actual = y_true.to_numpy(dtype=float)
    error = actual - predictions
    return {
        "split": split,
        "rows": int(len(actual)),
        "mae": round(float(mean_absolute_error(actual, predictions)), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "wape": round(float(np.sum(np.abs(error)) / np.sum(np.abs(actual)) * 100), 4),
        "r2": round(float(r2_score(actual, predictions)), 4),
    }


def build_model() -> xgb.XGBRegressor:
    """Return the fixed M3 XGBoost configuration used for reproducible training."""
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.1,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def write_report(
    metrics: list[dict[str, Any]],
    row_counts: dict[str, int],
    residual_std: float,
    feature_order: list[str],
) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# M3 Model Evaluation Report",
        "",
        "## Model",
        "Optimized XGBoost regressor trained from `data/preprocessed_sales_data.csv`.",
        f"All preprocessed rows ({sum(row_counts.values()):,} total) are used. "
        "Multi-day forecast horizons are generated recursively at inference time by ForecastEngine.",
        "",
        "## Chronological Split",
        "| Set | Date range | Rows |",
        "|---|---|---:|",
        f"| Train | Through 2023-06-30 | {row_counts['train']:,} |",
        f"| Validation | 2023-07-01 to 2023-09-30 | {row_counts['validation']:,} |",
        f"| Test | 2023-10-01 onward | {row_counts['test']:,} |",
        "",
        "## Metrics",
        "| Set | MAE | RMSE | WAPE (%) | R² |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in metrics:
        lines.append(
            f"| {result['split']} | {result['mae']} | {result['rmse']} | "
            f"{result['wape']} | {result['r2']} |"
        )
    lines.extend(
        [
            "",
            f"Residual standard deviation on the held-out test set: `{residual_std:.4f}`.",
            "",
            "## Feature Contract",
            f"Target column: `{TARGET_COLUMN}` (Demand at date t).",
            "Endogenous variables (Inventory Level, Units Sold, Units Ordered) are "
            "excluded from model features.",
            f"Total features: **{len(feature_order)}** (see `FEATURE_ORDER` in "
            "`src/data_preprocessing.py`).",
            "",
            "```text",
            ", ".join(feature_order),
            "```",
            "",
            "## Deployment Artifact",
            "`src/models/best_model.pkl` — retrained on Train + Validation after evaluation.",
            "",
            "> **Note**: The deployment model does NOT have a `feature_order_` attribute.",
            "> `ForecastEngine` therefore uses the modern `preprocess_input()` path",
            "> (38-feature contract) for all inference.",
        ]
    )
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train() -> dict[str, Any]:
    """Train, evaluate, and save the M3 XGBoost model.

    Returns a summary dict with paths, feature count, and evaluation metrics.
    """
    frame = load_training_data()
    feature_order = get_feature_order()

    # Chronological split (no shuffling — time-series ordering must be preserved)
    train_df = frame[frame["Date"] <= "2023-06-30"].copy()
    validation_df = frame[
        (frame["Date"] >= "2023-07-01") & (frame["Date"] <= "2023-09-30")
    ].copy()
    test_df = frame[frame["Date"] >= "2023-10-01"].copy()

    if min(len(train_df), len(validation_df), len(test_df)) == 0:
        raise ValueError(
            "Chronological split produced an empty partition. "
            f"Sizes — train: {len(train_df)}, val: {len(validation_df)}, test: {len(test_df)}."
        )

    print(
        f"Split sizes — train: {len(train_df):,}, "
        f"validation: {len(validation_df):,}, test: {len(test_df):,}"
    )

    # Evaluation model: train on train set only
    eval_model = build_model()
    eval_model.fit(train_df[feature_order], train_df[TARGET_COLUMN])
    validation_predictions = eval_model.predict(validation_df[feature_order])
    test_predictions = eval_model.predict(test_df[feature_order])

    metrics = [
        _metrics(validation_df[TARGET_COLUMN], validation_predictions, "Validation"),
        _metrics(test_df[TARGET_COLUMN], test_predictions, "Test"),
    ]
    residual_std = float(np.std(test_df[TARGET_COLUMN].to_numpy() - test_predictions))

    # Deployment model: retrain on Train + Validation for maximum coverage
    deployment_df = pd.concat([train_df, validation_df], ignore_index=True)
    deployment_model = build_model()
    deployment_model.fit(deployment_df[feature_order], deployment_df[TARGET_COLUMN])

    # IMPORTANT: Do NOT set feature_order_ on the deployment model.
    # ForecastEngine checks for that attribute to decide which inference path to use.
    # Without feature_order_, ForecastEngine uses the modern preprocess_input() path
    # with the 38-feature FEATURE_ORDER contract from data_preprocessing.py.

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(deployment_model, MODEL_PATH)
    print(f"Deployment model saved to {MODEL_PATH}")

    RESIDUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESIDUAL_PATH.write_text(f"{residual_std:.4f}\n", encoding="utf-8")

    write_report(
        metrics,
        {"train": len(train_df), "validation": len(validation_df), "test": len(test_df)},
        residual_std,
        feature_order,
    )

    summary = {
        "model_path": str(MODEL_PATH),
        "report_path": str(REPORT_PATH),
        "feature_count": len(feature_order),
        "feature_order": feature_order,
        "metrics": metrics,
    }
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    train()
