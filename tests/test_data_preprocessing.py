import numpy as np
import pandas as pd
import pytest

from src.data_preprocessing import fit_encoders, preprocess_input


@pytest.fixture
def encoder_path(tmp_path):
    training_data = pd.DataFrame(
        [
            {
                "Date": "2024-01-14",
                "Store ID": "S001",
                "Product ID": "P0001",
                "Category": "Electronics",
                "Region": "North",
                "Inventory Level": 100,
                "Units Sold": 20,
                "Units Ordered": 30,
                "Price": 500.0,
                "Discount": 10,
                "Weather Condition": "Sunny",
                "Promotion": 1,
                "Competitor Pricing": 520.0,
                "Seasonality": "Winter",
                "Epidemic": 0,
            },
            {
                "Date": "2024-01-15",
                "Store ID": "S002",
                "Product ID": "P0002",
                "Category": "Toys",
                "Region": "South",
                "Inventory Level": 80,
                "Units Sold": 15,
                "Units Ordered": 20,
                "Price": 100.0,
                "Discount": 0,
                "Weather Condition": "Rainy",
                "Promotion": 0,
                "Competitor Pricing": 110.0,
                "Seasonality": "Spring",
                "Epidemic": 1,
            },
        ]
    )
    path = tmp_path / "encoders.pkl"
    fit_encoders(training_data, path)
    return path


def test_preprocess_input_returns_stable_numeric_matrix(encoder_path):
    row = {
        "store_id": "S001",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "inventory_level": 100,
        "units_sold": 20,
        "units_ordered": 30,
        "price": 500,
        "discount_rate": 10,
        "weather_condition": "Sunny",
        "promotion_active": 1,
        "competitor_pricing": 520,
        "seasonality": "Winter",
        "epidemic": 0,
        "date": "2024-01-14",
    }

    features = preprocess_input(row, encoder_path)

    assert features.shape == (1, 25)
    assert features.dtype == np.float64
    assert np.isfinite(features).all()
    assert features[0, 8:15].tolist() == [2024, 1, 14, 6, 2, 1, 1]


def test_preprocess_input_rejects_unknown_categories(encoder_path):
    row = {
        "store_id": "S999",
        "product_id": "P0001",
        "category": "Electronics",
        "region": "North",
        "inventory_level": 100,
        "units_sold": 20,
        "units_ordered": 30,
        "price": 500,
        "discount_rate": 10,
        "weather_condition": "Sunny",
        "promotion_active": 1,
        "competitor_pricing": 520,
        "seasonality": "Winter",
        "epidemic": 0,
    }

    with pytest.raises(ValueError, match="Unknown Store ID"):
        preprocess_input(row, encoder_path)