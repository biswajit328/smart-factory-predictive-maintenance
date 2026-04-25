from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import (
    RANDOM_STATE,
    V2_BATCH_SIZE,
    V2_BRANCH_IMPORTANCE_PATH,
    V2_BRANCH_IMPORTANCE_PLOT_PATH,
    V2_CALIBRATION_CURVE_PATH,
    V2_EPOCHS,
    V2_METRICS_PATH,
    V2_MODEL_PATH,
    V2_NUM_MACHINES,
    V2_NUM_STEPS,
    V2_OUTPUT_DIR,
    V2_PR_CURVE_PATH,
    V2_ROC_CURVE_PATH,
    V2_SCALERS_PATH,
    V2_SENSOR_EVENTS_PATH,
    V2_SIMULATED_STREAM_PATH,
    V2_TEST_PREDICTIONS_PATH,
    V2_THRESHOLD_ANALYSIS_PATH,
    V2_THRESHOLD_BETA,
    V2_TRAINING_HISTORY_PATH,
    V2_WINDOW_SIZE,
    V2_METADATA_PATH,
    repo_relative,
)
from .logging_utils import configure_logging, get_logger
from .v2_neural import (
    build_sequence_dataset,
    build_split_masks,
    evaluate_temporal_fusion_model,
    fit_branch_scalers,
    save_branch_scalers,
    save_training_history_plot,
    split_machine_ids,
    subset_inputs,
    train_temporal_fusion_model,
    transform_branch_inputs,
)
from .model import build_threshold_analysis
from .v2_streaming import simulate_factory_stream, to_sensor_events

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve


logger = get_logger(__name__)


def _save_probability_curves(
    y_true,
    probabilities,
    roc_auc: float,
    pr_auc: float,
) -> None:
    precision, recall, _ = precision_recall_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Neural PR Curve (AP = {pr_auc:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(V2_PR_CURVE_PATH, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_true, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#d95f02", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"Neural ROC Curve (AUC = {roc_auc:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(V2_ROC_CURVE_PATH, dpi=150)
    plt.close(fig)


def _save_calibration_plot(y_true, probabilities, brier_score: float) -> None:
    observed_rate, predicted_rate = calibration_curve(
        y_true,
        probabilities,
        n_bins=8,
        strategy="quantile",
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted_rate, observed_rate, marker="o", lw=2, label="model")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure rate")
    ax.set_title(f"Neural Calibration (Brier = {brier_score:.3f})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(V2_CALIBRATION_CURVE_PATH, dpi=150)
    plt.close(fig)


def _compute_branch_importance(
    model,
    test_inputs: dict[str, np.ndarray],
    test_labels: np.ndarray,
    baseline_pr_auc: float,
    baseline_roc_auc: float,
) -> pd.DataFrame:
    rows = []

    for group_name in test_inputs:
        ablated_inputs = {
            name: values.copy()
            for name, values in test_inputs.items()
        }
        ablated_inputs[group_name] = np.zeros_like(ablated_inputs[group_name])

        ablated_probabilities = model.predict(ablated_inputs, verbose=0).ravel()
        ablated_pr_auc = float(average_precision_score(test_labels, ablated_probabilities))
        ablated_roc_auc = float(roc_auc_score(test_labels, ablated_probabilities))
        rows.append(
            {
                "branch": group_name,
                "baseline_pr_auc": round(float(baseline_pr_auc), 4),
                "ablated_pr_auc": round(ablated_pr_auc, 4),
                "pr_auc_drop": round(float(baseline_pr_auc - ablated_pr_auc), 4),
                "baseline_roc_auc": round(float(baseline_roc_auc), 4),
                "ablated_roc_auc": round(ablated_roc_auc, 4),
                "roc_auc_drop": round(float(baseline_roc_auc - ablated_roc_auc), 4),
            }
        )

    return pd.DataFrame(rows).sort_values("pr_auc_drop", ascending=False).reset_index(drop=True)


def _save_branch_importance_plot(branch_importance: pd.DataFrame) -> None:
    plot_frame = branch_importance.sort_values("pr_auc_drop")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(plot_frame["branch"], plot_frame["pr_auc_drop"], color="#2a9d8f")
    ax.set_xlabel("PR-AUC drop after branch ablation")
    ax.set_title("Neural Sensor Branch Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(V2_BRANCH_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)


def train_smart_factory_v2(
    num_machines: int = V2_NUM_MACHINES,
    steps: int = V2_NUM_STEPS,
    window_size: int = V2_WINDOW_SIZE,
    epochs: int = V2_EPOCHS,
    batch_size: int = V2_BATCH_SIZE,
    seed: int = RANDOM_STATE,
    save_artifacts: bool = True,
):
    stream_df = simulate_factory_stream(
        num_machines=num_machines,
        steps=steps,
        seed=seed,
    )
    sensor_events_df = to_sensor_events(stream_df)

    sequence_dataset = build_sequence_dataset(stream_df, window_size=window_size)
    split_ids = split_machine_ids(
        sequence_dataset.metadata["machine_id"].tolist(),
        seed=seed,
    )
    split_masks = build_split_masks(sequence_dataset.metadata, split_ids)

    train_inputs_raw = subset_inputs(sequence_dataset.inputs, split_masks["train"])
    val_inputs_raw = subset_inputs(sequence_dataset.inputs, split_masks["val"])
    test_inputs_raw = subset_inputs(sequence_dataset.inputs, split_masks["test"])
    y_train = sequence_dataset.labels[split_masks["train"]]
    y_val = sequence_dataset.labels[split_masks["val"]]
    y_test = sequence_dataset.labels[split_masks["test"]]

    scalers = fit_branch_scalers(train_inputs_raw)
    train_inputs = transform_branch_inputs(train_inputs_raw, scalers)
    val_inputs = transform_branch_inputs(val_inputs_raw, scalers)
    test_inputs = transform_branch_inputs(test_inputs_raw, scalers)

    model, history = train_temporal_fusion_model(
        train_inputs=train_inputs,
        train_labels=y_train,
        val_inputs=val_inputs,
        val_labels=y_val,
        epochs=epochs,
        batch_size=batch_size,
    )

    evaluation, test_probabilities = evaluate_temporal_fusion_model(
        model=model,
        val_inputs=val_inputs,
        val_labels=y_val,
        test_inputs=test_inputs,
        test_labels=y_test,
    )
    threshold_analysis = build_threshold_analysis(
        y_test,
        test_probabilities,
        beta=V2_THRESHOLD_BETA,
    )
    branch_importance = _compute_branch_importance(
        model=model,
        test_inputs=test_inputs,
        test_labels=y_test,
        baseline_pr_auc=evaluation["classification"]["pr_auc"],
        baseline_roc_auc=evaluation["classification"]["roc_auc"],
    )

    test_metadata = sequence_dataset.metadata.loc[split_masks["test"]].copy().reset_index(drop=True)
    threshold = evaluation["threshold_selection"]["threshold"]
    test_metadata["failure_probability"] = test_probabilities
    test_metadata["classification_flag"] = test_metadata["failure_probability"] >= threshold
    test_metadata["maintenance_priority"] = (test_metadata["failure_probability"] * 100).round(2)

    metrics_payload = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "simulator": {
            "num_machines": num_machines,
            "steps_per_machine": steps,
            "window_size": window_size,
            "seed": seed,
        },
        "splits": {
            "train_machines": split_ids["train"],
            "val_machines": split_ids["val"],
            "test_machines": split_ids["test"],
            "train_rows": int(split_masks["train"].sum()),
            "val_rows": int(split_masks["val"].sum()),
            "test_rows": int(split_masks["test"].sum()),
            "train_positive_rate": round(float(y_train.mean()), 4),
            "val_positive_rate": round(float(y_val.mean()), 4),
            "test_positive_rate": round(float(y_test.mean()), 4),
        },
        "architecture": {
            "name": "TemporalSensorFusionCNN",
            "sensor_groups": {
                "thermal": ["air_temp_k", "process_temp_k", "humidity_pct"],
                "mechanical": [
                    "rotational_speed_rpm",
                    "torque_nm",
                    "tool_wear_min",
                    "vibration_mm_s",
                    "type_H",
                    "type_L",
                    "type_M",
                ],
                "electrical": ["pressure_bar", "current_a", "acoustic_db"],
            },
        },
        "calibration": {
            "brier_score": evaluation["classification"]["brier_score"],
            "plot_path": repo_relative(V2_CALIBRATION_CURVE_PATH),
        },
        "branch_importance": branch_importance.to_dict(orient="records"),
        **evaluation,
    }

    metadata_payload = {
        "window_size": window_size,
        "probability_threshold": threshold,
        "required_sensor_columns": [
            "air_temp_k",
            "process_temp_k",
            "rotational_speed_rpm",
            "torque_nm",
            "tool_wear_min",
            "vibration_mm_s",
            "pressure_bar",
            "current_a",
            "acoustic_db",
            "humidity_pct",
        ],
        "type_values": ["H", "L", "M"],
        "feature_groups": metrics_payload["architecture"]["sensor_groups"],
    }

    bundle = {
        "model": model,
        "scalers": scalers,
        "metadata": metadata_payload,
        "stream_df": stream_df,
        "sensor_events_df": sensor_events_df,
        "test_predictions_df": test_metadata,
        "branch_importance_df": branch_importance,
        "metrics": metrics_payload,
    }

    if save_artifacts:
        V2_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        model.save(V2_MODEL_PATH)
        save_branch_scalers(scalers, V2_SCALERS_PATH)
        V2_METADATA_PATH.write_text(json.dumps(metadata_payload, indent=2), encoding="utf-8")
        V2_METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        stream_df.to_csv(V2_SIMULATED_STREAM_PATH, index=False)
        sensor_events_df.to_csv(V2_SENSOR_EVENTS_PATH, index=False)
        test_metadata.to_csv(V2_TEST_PREDICTIONS_PATH, index=False)
        threshold_analysis.to_csv(V2_THRESHOLD_ANALYSIS_PATH, index=False)
        branch_importance.to_csv(V2_BRANCH_IMPORTANCE_PATH, index=False)
        save_training_history_plot(history, V2_TRAINING_HISTORY_PATH)
        _save_probability_curves(
            y_true=y_test,
            probabilities=test_probabilities,
            roc_auc=evaluation["classification"]["roc_auc"],
            pr_auc=evaluation["classification"]["pr_auc"],
        )
        _save_calibration_plot(
            y_true=y_test,
            probabilities=test_probabilities,
            brier_score=evaluation["classification"]["brier_score"],
        )
        _save_branch_importance_plot(branch_importance)

    return bundle, metrics_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the v2 smart factory neural fusion model.")
    parser.add_argument("--machines", type=int, default=V2_NUM_MACHINES, help="Number of simulated machines.")
    parser.add_argument("--steps", type=int, default=V2_NUM_STEPS, help="Number of timesteps per machine.")
    parser.add_argument("--epochs", type=int, default=V2_EPOCHS, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=V2_BATCH_SIZE, help="Training batch size.")
    return parser


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    _, metrics = train_smart_factory_v2(
        num_machines=args.machines,
        steps=args.steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    summary = {
        "roc_auc": metrics["classification"]["roc_auc"],
        "pr_auc": metrics["classification"]["pr_auc"],
        "precision": metrics["classification"]["precision"],
        "recall": metrics["classification"]["recall"],
        "threshold": metrics["threshold_selection"]["threshold"],
    }
    logger.info("v2_training_completed", extra=summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
