"""
This module provides functions to load and transform datasets using the Dataset class from the fairml_datasets library.
It includes:
- `load_and_transform_data`: Loads a dataset by ID, optionally transforms it, and returns relevant information.
- `train_test_split`: Splits a DataFrame into training and testing sets using the Dataset's built-in method.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from fairml_datasets import Dataset

from torch.utils.data import Dataset as TorchDataset, DataLoader


class TabularDataset(TorchDataset):
    def __init__(self, X: np.ndarray, A: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.A = A.astype(np.int64)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.A[idx], self.y[idx]
    

def load_and_transform_data(id: str, transform: bool = True) -> Dict:
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

def load_and_transform_data(id: str, transform: bool = True) -> Dict:
    dataset = Dataset.from_id(id)
    df = dataset.load()
    sensitive_columns = dataset.sensitive_columns
    if transform:
        df, transformation_info = dataset.transform(df)
        sensitive_columns = transformation_info.sensitive_columns
    return {
        "dataset": dataset,
        "dataframe": df,
        "target_column": dataset.get_target_column(),
        "feature_columns": dataset.get_feature_columns(df),
        "sensitive_columns": sensitive_columns,
    }

def prepare_features(df: pd.DataFrame, feature_cols: List[str], sensitive_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[feature_cols].values
    A = df[sensitive_cols].values[:,0] if len(sensitive_cols)>0 else np.zeros(len(df), dtype=int)
    y = df[target_col].values
    return X, A, y

def create_dataloaders(train_df, val_df, test_df,
                       feature_columns,
                       sensitive_columns,
                       target_column,
                       batch_size):
    X_train, A_train, y_train = prepare_features(train_df, feature_columns, sensitive_columns, target_column)
    X_val, A_val, y_val = prepare_features(val_df, feature_columns, sensitive_columns, target_column)
    X_test, A_test, y_test = prepare_features(test_df, feature_columns, sensitive_columns, target_column)

    train_ds = TabularDataset(X_train, A_train, y_train)
    val_ds = TabularDataset(X_val, A_val, y_val)
    test_ds = TabularDataset(X_test, A_test, y_test)

    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            DataLoader(val_ds, batch_size=batch_size),
            DataLoader(test_ds, batch_size=batch_size))

