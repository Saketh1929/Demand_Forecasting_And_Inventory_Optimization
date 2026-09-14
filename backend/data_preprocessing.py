"""Compatibility wrapper that points the backend to the authoritative ML preprocessing implementation."""
from src.data_preprocessing import (
    extract_date_features,
    load_encoders,
    normalize_product_id,
    normalize_store_id,
    preprocess_input,
)

__all__ = [
    "preprocess_input",
    "extract_date_features",
    "load_encoders",
    "normalize_store_id",
    "normalize_product_id",
]
