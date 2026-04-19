from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve

from .anomaly import evaluate_anomaly_detector, fit_anomaly_detector
from .config import (
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    FEATURE_IMPORTANCE_PATH,
    FEATURE_IMPORTANCE_PLOT_PATH,
    METRICS_PATH,
    MODEL_BUNDLE_PATH,
    OUTPUT_DIR,
    PRECISION_RECALL_CURVE_PATH,
    PROBABILITY_DISTRIBUTION_PATH,
    RANDOM_STATE,
    ROC_CURVE_PATH,
)
from .model import (
    build_feature_importance_table,
    build_training_pipeline,
    choose_probability_threshold,
    evaluate_classifier,
)
from .preprocess import build_preprocessor, load_raw_data, make_supervised_frame, split_dataset


def _save_confusion_matrix_plot(confusion_matrix_values) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        confusion_matrix_values,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Normal", "Failure"],
        yticklabels=["Normal", "Failure"],
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(CONFUSION_MATRIX_PATH, dpi=150)
    plt.close(fig)


def _save_probability_plot(y_test, probabilities) -> None:
    probability_frame = pd.DataFrame(
        {
            "failure_probability": probabilities,
            "actual_label": ["Failure" if value == 1 else "Normal" for value in y_test],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=probability_frame,
        x="failure_probability",
        hue="actual_label",
        bins=30,
        stat="density",
        common_norm=False,
        ax=ax,
    )
    ax.set_title("Predicted Failure Probability Distribution")
    fig.tight_layout()
    fig.savefig(PROBABILITY_DISTRIBUTION_PATH, dpi=150)
    plt.close(fig)


def _save_curve_plots(y_test, probabilities, pr_auc: float, roc_auc: float) -> None:
    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#1f77b4", lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {pr_auc:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PRECISION_RECALL_CURVE_PATH, dpi=150)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#ff7f0e", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(ROC_CURVE_PATH, dpi=150)
    plt.close(fig)


def _save_feature_importance_artifacts(feature_importance: pd.DataFrame) -> None:
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    top_features = feature_importance.head(12).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top_features["feature"], top_features["importance"], color="#2a9d8f")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(fig)


def train_project_model(
    raw_df: pd.DataFrame | None = None,
    save_artifacts: bool = True,
    model_params: dict | None = None,
):
    source_df = load_raw_data() if raw_df is None else raw_df.copy()
    splits = split_dataset(source_df, random_state=RANDOM_STATE)

    X_train, y_train = make_supervised_frame(splits["train"])
    X_val, y_val = make_supervised_frame(splits["val"])
    X_test, y_test = make_supervised_frame(splits["test"])

    preprocessor = build_preprocessor(X_train)
    pipeline = build_training_pipeline(preprocessor, **(model_params or {}))
    pipeline.fit(X_train, y_train)

    val_probabilities = pipeline.predict_proba(X_val)[:, 1]
    threshold_selection = choose_probability_threshold(y_val, val_probabilities)

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    classifier_metrics = evaluate_classifier(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=threshold_selection.threshold,
    )

    transformed_train = pipeline.named_steps["preprocess"].transform(X_train)
    transformed_test = pipeline.named_steps["preprocess"].transform(X_test)
    anomaly_model, anomaly_threshold = fit_anomaly_detector(transformed_train, y_train)
    anomaly_metrics = evaluate_anomaly_detector(
        anomaly_model,
        anomaly_threshold,
        transformed_test,
        y_test,
    )

    feature_importance = build_feature_importance_table(pipeline)

    metrics_payload = {
        "run_timestamp": datetime.utcnow().isoformat() + "Z",
        "dataset": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "train_failure_rate": round(float(y_train.mean()), 4),
            "val_failure_rate": round(float(y_val.mean()), 4),
            "test_failure_rate": round(float(y_test.mean()), 4),
        },
        "model": {
            "name": "RandomForestClassifier",
            "random_state": RANDOM_STATE,
            "notes": "Leakage-free tabular classifier trained on raw machine attributes plus engineered physics features.",
        },
        "threshold_selection": threshold_selection.to_dict(),
        "classification": classifier_metrics,
        "anomaly_detection": anomaly_metrics,
        "top_features": feature_importance.head(10).to_dict(orient="records"),
    }

    bundle = {
        "pipeline": pipeline,
        "anomaly_model": anomaly_model,
        "probability_threshold": threshold_selection.threshold,
        "anomaly_threshold": anomaly_threshold,
        "metrics": metrics_payload,
        "feature_names": feature_importance["feature"].tolist(),
        "model_params": model_params or {},
    }

    if save_artifacts:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, MODEL_BUNDLE_PATH)
        METRICS_PATH.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        CLASSIFICATION_REPORT_PATH.write_text(
            classifier_metrics["classification_report"],
            encoding="utf-8",
        )
        _save_confusion_matrix_plot(classifier_metrics["confusion_matrix"])
        _save_curve_plots(
            y_test=y_test,
            probabilities=test_probabilities,
            pr_auc=classifier_metrics["pr_auc"],
            roc_auc=classifier_metrics["roc_auc"],
        )
        _save_probability_plot(y_test=y_test, probabilities=test_probabilities)
        _save_feature_importance_artifacts(feature_importance)

    return bundle, metrics_payload


def main() -> None:
    _, metrics = train_project_model()
    summary = {
        "pr_auc": metrics["classification"]["pr_auc"],
        "roc_auc": metrics["classification"]["roc_auc"],
        "precision": metrics["classification"]["precision"],
        "recall": metrics["classification"]["recall"],
        "threshold": metrics["threshold_selection"]["threshold"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
