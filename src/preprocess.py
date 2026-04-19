from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import RAW_DATA_PATH, RANDOM_STATE, TARGET_COL, TEST_SIZE, VAL_SIZE
from .features import prepare_model_frame


def load_raw_data(path: str | None = None) -> pd.DataFrame:
    csv_path = RAW_DATA_PATH if path is None else path
    return pd.read_csv(csv_path)


def make_supervised_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' is missing.")

    y = df[TARGET_COL].astype(int).copy()
    X = prepare_model_frame(df.drop(columns=[TARGET_COL]))
    return X, y


def split_dataset(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_state: int = RANDOM_STATE,
) -> dict[str, pd.DataFrame]:
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[TARGET_COL],
    )
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=train_df[TARGET_COL],
    )
    return {"train": train_df, "val": val_df, "test": test_df}


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_columns = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in X.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_columns),
            ("cat", categorical_pipeline, categorical_columns),
        ],
        sparse_threshold=0.0,
    )


def load_raw() -> pd.DataFrame:
    return load_raw_data()


def split_data(df: pd.DataFrame) -> tuple[Any, Any, Any, Any, Any, Any]:
    splits = split_dataset(df)
    X_train, y_train = make_supervised_frame(splits["train"])
    X_val, y_val = make_supervised_frame(splits["val"])
    X_test, y_test = make_supervised_frame(splits["test"])
    return X_train, X_val, X_test, y_train, y_val, y_test


SENSOR_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]


def run_pipeline() -> tuple[Any, Any, Any, Any, Any, Any]:
    raw_df = load_raw_data()
    return split_data(raw_df)


if __name__ == "__main__":
    train_X, val_X, test_X, train_y, val_y, test_y = run_pipeline()
    print(f"Train rows: {len(train_X)}, positives: {int(train_y.sum())}")
    print(f"Val rows:   {len(val_X)}, positives: {int(val_y.sum())}")
    print(f"Test rows:  {len(test_X)}, positives: {int(test_y.sum())}")

