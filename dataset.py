"""
This module provides functions to load and transform datasets using the Dataset class from the fairml_datasets library.
It includes:
- `load_and_transform_data`: Loads a dataset by ID, optionally transforms it, and returns relevant information.
- `train_test_split`: Splits a DataFrame into training and testing sets using the Dataset's built-in method.
"""
import pandas as pd
from fairml_datasets import Dataset

def load_and_transform_data(id: str, transform: bool = True) -> dict:
    """Example of loading a dataset using the Dataset class."""
    dataset = Dataset.from_id(id)

    # Load as pandas DataFrame
    df = dataset.load()

    sensitive_columns = dataset.sensitive_columns

    if transform:
        # Transform to e.g. impute missing data
        df, transformation_info = dataset.transform(df)

        # Sensitive columns may change due to transformation
        sensitive_columns = transformation_info.sensitive_columns


    return {
        "dataset": dataset,
        "dataframe": df,
        "target_column": dataset.get_target_column(),
        "feature_columns": dataset.get_feature_columns(),
        "sensitive_columns": sensitive_columns
    }

def train_test_val_split(dataset: Dataset, df: pd.DataFrame, test_size: float = 0.3) -> tuple:
    """Example of splitting the dataset into train and test sets."""
    return dataset.train_test_val_split(df, test_size=test_size)