import pandas as pd
from src.data_preprocessing.features import REQUIRED_COLUMNS, GROUP_COLUMNS, MIN_HISTORY

def validate_data(df: pd.DataFrame, min_history: int = MIN_HISTORY) -> bool:
    """Validate schema, completeness, chronology, duplicates, and minimum history."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    missing_values = df[REQUIRED_COLUMNS].isna().sum()
    missing_values = missing_values[missing_values > 0].to_dict()
    if missing_values:
        raise ValueError(f"Required columns contain missing values: {missing_values}")

    try:
        parsed_dates = pd.to_datetime(df["Date"], errors="raise")
    except (TypeError, ValueError) as error:
        raise ValueError("Date contains invalid values.") from error

    if df.duplicated(subset=[*GROUP_COLUMNS, "Date"]).any():
        raise ValueError("Duplicate Store ID/Product ID/Date rows found.")

    expected_order = df.assign(Date=parsed_dates).sort_values([*GROUP_COLUMNS, "Date"]).index
    if not expected_order.equals(df.index):
        raise ValueError("Data must be sorted by Store ID, Product ID, and Date.")

    group_sizes = df.groupby(GROUP_COLUMNS, sort=False).size()
    short_groups = group_sizes[group_sizes < min_history]
    if not short_groups.empty:
        examples = [f"{group}: {size}" for group, size in short_groups.head(5).items()]
        raise ValueError(
            f"Store-product groups need at least {min_history} rows for lag/rolling features; "
            f"short groups include {', '.join(examples)}."
        )
    return True
