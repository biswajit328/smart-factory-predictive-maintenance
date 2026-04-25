from __future__ import annotations

import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import keras
import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras import layers

from .config import (
    FEATURE_GROUPS,
    RANDOM_STATE,
    TYPE_FEATURE_COLUMNS,
    V2_BATCH_SIZE,
    V2_EPOCHS,
    V2_THRESHOLD_BETA,
    V2_THRESHOLD_PRECISION_FLOOR,
    V2_WINDOW_SIZE,
)
from .model import choose_probability_threshold, evaluate_classifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class SequenceDataset:
    inputs: dict[str, np.ndarray]
    labels: np.ndarray
    metadata: pd.DataFrame


def add_type_indicator_columns(stream_df: pd.DataFrame) -> pd.DataFrame:
    prepared = stream_df.copy()
    for type_column in TYPE_FEATURE_COLUMNS:
        prepared[type_column] = 0.0

    for machine_type in ["H", "L", "M"]:
        column_name = f"type_{machine_type}"
        prepared[column_name] = (prepared["machine_type"] == machine_type).astype(float)

    return prepared


def build_sequence_dataset(
    stream_df: pd.DataFrame,
    window_size: int = V2_WINDOW_SIZE,
) -> SequenceDataset:
    prepared = add_type_indicator_columns(stream_df)

    grouped_inputs = {group_name: [] for group_name in FEATURE_GROUPS}
    labels = []
    metadata_rows: list[dict[str, object]] = []

    ordered = prepared.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    for machine_id, machine_frame in ordered.groupby("machine_id", sort=False):
        machine_frame = machine_frame.reset_index(drop=True)
        for end_index in range(window_size - 1, len(machine_frame)):
            window_frame = machine_frame.iloc[end_index - window_size + 1 : end_index + 1]
            for group_name, columns in FEATURE_GROUPS.items():
                grouped_inputs[group_name].append(window_frame[columns].to_numpy(dtype=np.float32))
            labels.append(float(machine_frame.loc[end_index, "failure_next_horizon"]))
            metadata_rows.append(
                {
                    "machine_id": machine_id,
                    "machine_type": machine_frame.loc[end_index, "machine_type"],
                    "timestamp": machine_frame.loc[end_index, "timestamp"],
                    "step": int(machine_frame.loc[end_index, "step"]),
                    "breakdown_event": int(machine_frame.loc[end_index, "breakdown_event"]),
                    "failure_next_horizon": int(machine_frame.loc[end_index, "failure_next_horizon"]),
                }
            )

    inputs = {
        group_name: np.asarray(values, dtype=np.float32) for group_name, values in grouped_inputs.items()
    }
    return SequenceDataset(
        inputs=inputs,
        labels=np.asarray(labels, dtype=np.float32),
        metadata=pd.DataFrame(metadata_rows),
    )


def split_machine_ids(
    machine_ids: list[str] | np.ndarray,
    seed: int = RANDOM_STATE,
) -> dict[str, list[str]]:
    unique_machine_ids = list(dict.fromkeys(machine_ids))
    if len(unique_machine_ids) < 3:
        raise ValueError("At least three machines are required for train/val/test splits.")

    rng = np.random.default_rng(seed)
    rng.shuffle(unique_machine_ids)

    test_count = max(1, round(len(unique_machine_ids) * 0.15))
    val_count = max(1, round(len(unique_machine_ids) * 0.15))
    if test_count + val_count >= len(unique_machine_ids):
        val_count = 1
        test_count = 1

    test_ids = unique_machine_ids[:test_count]
    val_ids = unique_machine_ids[test_count : test_count + val_count]
    train_ids = unique_machine_ids[test_count + val_count :]

    if not train_ids:
        raise ValueError("Split configuration left no machines for training.")

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }


def build_split_masks(metadata: pd.DataFrame, split_ids: dict[str, list[str]]) -> dict[str, np.ndarray]:
    machine_id_series = metadata["machine_id"].to_numpy()
    return {
        split_name: np.isin(machine_id_series, split_machine_ids)
        for split_name, split_machine_ids in split_ids.items()
    }


def subset_inputs(inputs: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    return {group_name: values[mask] for group_name, values in inputs.items()}


def fit_branch_scalers(train_inputs: dict[str, np.ndarray]) -> dict[str, StandardScaler]:
    scalers = {}
    for group_name, values in train_inputs.items():
        scaler = StandardScaler()
        scaler.fit(values.reshape(-1, values.shape[-1]))
        scalers[group_name] = scaler
    return scalers


def transform_branch_inputs(
    inputs: dict[str, np.ndarray],
    scalers: dict[str, StandardScaler],
) -> dict[str, np.ndarray]:
    transformed = {}
    for group_name, values in inputs.items():
        scaler = scalers[group_name]
        transformed[group_name] = (
            scaler.transform(values.reshape(-1, values.shape[-1])).reshape(values.shape).astype(np.float32)
        )
    return transformed


def transform_single_window(
    window_by_group: dict[str, np.ndarray],
    scalers: dict[str, StandardScaler],
) -> dict[str, np.ndarray]:
    transformed = {}
    for group_name, values in window_by_group.items():
        transformed[group_name] = scalers[group_name].transform(values).astype(np.float32)
    return transformed


def build_temporal_fusion_model(
    window_size: int = V2_WINDOW_SIZE,
    feature_groups: dict[str, list[str]] | None = None,
) -> keras.Model:
    groups = FEATURE_GROUPS if feature_groups is None else feature_groups

    def build_branch(name: str, feature_count: int, filters: int):
        inputs = keras.Input(shape=(window_size, feature_count), name=name)
        x = layers.Conv1D(filters, kernel_size=3, padding="same", activation="relu")(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv1D(filters, kernel_size=3, padding="same", activation="relu")(x)
        x = layers.GlobalAveragePooling1D()(x)
        return inputs, x

    thermal_input, thermal_branch = build_branch(
        "thermal",
        len(groups["thermal"]),
        filters=16,
    )
    mechanical_input, mechanical_branch = build_branch(
        "mechanical",
        len(groups["mechanical"]),
        filters=24,
    )
    electrical_input, electrical_branch = build_branch(
        "electrical",
        len(groups["electrical"]),
        filters=16,
    )

    x = layers.Concatenate()([thermal_branch, mechanical_branch, electrical_branch])
    x = layers.Dense(48, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Dense(24, activation="relu")(x)
    output = layers.Dense(1, activation="sigmoid", name="failure_probability")(x)

    model = keras.Model(
        inputs=[thermal_input, mechanical_input, electrical_input],
        outputs=output,
        name="TemporalSensorFusionCNN",
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            keras.metrics.AUC(name="roc_auc"),
            keras.metrics.AUC(name="pr_auc", curve="PR"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def compute_class_weight(y_train: np.ndarray) -> dict[int, float]:
    positive_count = float(y_train.sum())
    negative_count = float(len(y_train) - positive_count)
    if positive_count == 0:
        return {0: 1.0, 1: 1.0}
    return {0: 1.0, 1: negative_count / positive_count}


def train_temporal_fusion_model(
    train_inputs: dict[str, np.ndarray],
    train_labels: np.ndarray,
    val_inputs: dict[str, np.ndarray],
    val_labels: np.ndarray,
    epochs: int = V2_EPOCHS,
    batch_size: int = V2_BATCH_SIZE,
) -> tuple[keras.Model, keras.callbacks.History]:
    keras.utils.set_random_seed(RANDOM_STATE)
    model = build_temporal_fusion_model(
        window_size=train_inputs["thermal"].shape[1],
    )
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=3,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        class_weight=compute_class_weight(train_labels),
        callbacks=callbacks,
    )
    return model, history


def predict_probabilities(model: keras.Model, inputs: dict[str, np.ndarray]) -> np.ndarray:
    return model.predict(inputs, verbose=0).ravel()


def evaluate_temporal_fusion_model(
    model: keras.Model,
    val_inputs: dict[str, np.ndarray],
    val_labels: np.ndarray,
    test_inputs: dict[str, np.ndarray],
    test_labels: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    val_probabilities = predict_probabilities(model, val_inputs)
    threshold_selection = choose_probability_threshold(
        val_labels,
        val_probabilities,
        precision_floor=V2_THRESHOLD_PRECISION_FLOOR,
        beta=V2_THRESHOLD_BETA,
    )
    test_probabilities = predict_probabilities(model, test_inputs)
    classifier_metrics = evaluate_classifier(
        y_true=test_labels,
        probabilities=test_probabilities,
        threshold=threshold_selection.threshold,
        beta=V2_THRESHOLD_BETA,
    )
    metrics_payload = {
        "threshold_selection": threshold_selection.to_dict(),
        "classification": classifier_metrics,
    }
    return metrics_payload, test_probabilities


def save_training_history_plot(history: keras.callbacks.History, output_path) -> None:
    history_frame = pd.DataFrame(history.history)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history_frame["loss"], label="train_loss")
    axes[0].plot(history_frame["val_loss"], label="val_loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history_frame["pr_auc"], label="train_pr_auc")
    axes[1].plot(history_frame["val_pr_auc"], label="val_pr_auc")
    axes[1].set_title("Precision-Recall AUC")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_branch_scalers(scalers: dict[str, StandardScaler], output_path) -> None:
    joblib.dump(scalers, output_path)
