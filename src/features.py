from __future__ import annotations

import numpy as np
import pandas as pd

from .config import ID_COLUMNS, LEAKAGE_COLUMNS, RAW_INPUT_COLUMNS, SENSOR_COLS

ENGINEERED_FEATURES = [
    "temp_delta_k",
    "power_w",
    "wear_torque_interaction",
    "rpm_per_temp",
    "thermal_stress_index",
]


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, SENSOR_COLS)

    enriched = df.copy()
    safe_process_temperature = enriched["Process temperature [K]"].replace(0, np.nan)

    enriched["temp_delta_k"] = (
        enriched["Process temperature [K]"] - enriched["Air temperature [K]"]
    )
    enriched["power_w"] = (
        enriched["Torque [Nm]"]
        * enriched["Rotational speed [rpm]"]
        * (2 * np.pi / 60)
    )
    enriched["wear_torque_interaction"] = (
        enriched["Tool wear [min]"] * enriched["Torque [Nm]"]
    )
    enriched["rpm_per_temp"] = (
        enriched["Rotational speed [rpm]"] / safe_process_temperature
    )
    enriched["thermal_stress_index"] = (
        enriched["temp_delta_k"] * enriched["Torque [Nm]"]
    )

    return enriched.replace([np.inf, -np.inf], np.nan)


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    validate_required_columns(df, RAW_INPUT_COLUMNS)

    feature_frame = df.copy()
    columns_to_drop = [
        column
        for column in ID_COLUMNS + LEAKAGE_COLUMNS
        if column in feature_frame.columns
    ]
    if columns_to_drop:
        feature_frame = feature_frame.drop(columns=columns_to_drop)

    feature_frame = add_engineered_features(feature_frame)

    ordered_columns = [
        column
        for column in RAW_INPUT_COLUMNS + ENGINEERED_FEATURES
        if column in feature_frame.columns
    ]
    return feature_frame[ordered_columns]

