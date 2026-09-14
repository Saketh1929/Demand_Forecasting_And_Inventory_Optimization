"""Tests for the authoritative trained model and preprocessing artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

from src.data_preprocessing import FEATURE_ORDER


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


def test_authoritative_model_artifacts_are_loadable_and_aligned() -> None:
    model_path = MODEL_DIR / "best_model.pkl"
    encoders_path = MODEL_DIR / "encoders.pkl"
    feature_columns_path = MODEL_DIR / "feature_columns.pkl"

    for artifact_path in (model_path, encoders_path, feature_columns_path):
        assert artifact_path.is_file()
        assert artifact_path.stat().st_size > 0

    with model_path.open("rb") as file:
        model = pickle.load(file)
    with encoders_path.open("rb") as file:
        encoders = pickle.load(file)
    with feature_columns_path.open("rb") as file:
        feature_columns = pickle.load(file)

    assert model.n_features_in_ == len(FEATURE_ORDER) == 38
    assert feature_columns == FEATURE_ORDER
    assert encoders["feature_order"] == FEATURE_ORDER