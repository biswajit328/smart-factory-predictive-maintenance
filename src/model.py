from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypedDict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from .config import RANDOM_STATE, THRESHOLD_BETA, THRESHOLD_PRECISION_FLOOR


@dataclass
class ThresholdSelection:
    threshold: float
    precision: float
    recall: float
    fbeta: float
    precision_floor_met: bool
    strategy: str

    def to_dict(self) -> "ThresholdSelectionDict":
        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "fbeta": round(self.fbeta, 4),
            "precision_floor_met": self.precision_floor_met,
            "strategy": self.strategy,
        }


class ThresholdSelectionDict(TypedDict):
    threshold: float
    precision: float
    recall: float
    fbeta: float
    precision_floor_met: bool
    strategy: str


class ClassificationMetricsDict(TypedDict):
    roc_auc: float
    pr_auc: float
    brier_score: float
    precision: float
    recall: float
    fbeta: float
    accuracy: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    confusion_matrix: list[list[int]]
    classification_report: str


def build_classifier(
    random_state: int = RANDOM_STATE,
    **overrides: Any,
) -> RandomForestClassifier:
    params: dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "max_features": "sqrt",
        "random_state": random_state,
        "n_jobs": 1,
    }
    params.update(overrides)
    return RandomForestClassifier(**params)


def build_training_pipeline(preprocessor, **model_overrides) -> Pipeline:
    classifier = build_classifier(**model_overrides)
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def choose_probability_threshold(
    y_true,
    probabilities,
    precision_floor: float = THRESHOLD_PRECISION_FLOOR,
    beta: float = THRESHOLD_BETA,
) -> ThresholdSelection:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)

    best_any = None
    best_with_floor = None

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        precision = precision_score(y_true, predictions, zero_division=0)
        recall = recall_score(y_true, predictions, zero_division=0)
        fbeta = fbeta_score(y_true, predictions, beta=beta, zero_division=0)
        candidate = (fbeta, recall, precision, threshold)

        if best_any is None or candidate > best_any:
            best_any = candidate
        if precision >= precision_floor and (best_with_floor is None or candidate > best_with_floor):
            best_with_floor = candidate

    if best_with_floor is not None:
        fbeta, recall, precision, threshold = best_with_floor
        strategy = f"Best F{beta:.1f} threshold while keeping precision >= {precision_floor:.2f}"
        return ThresholdSelection(
            threshold=float(threshold),
            precision=float(precision),
            recall=float(recall),
            fbeta=float(fbeta),
            precision_floor_met=True,
            strategy=strategy,
        )

    if best_any is None:
        raise ValueError("No threshold candidates were evaluated.")
    fbeta, recall, precision, threshold = best_any
    strategy = f"Fallback best F{beta:.1f} threshold because precision floor was not met"
    return ThresholdSelection(
        threshold=float(threshold),
        precision=float(precision),
        recall=float(recall),
        fbeta=float(fbeta),
        precision_floor_met=False,
        strategy=strategy,
    )


def build_threshold_analysis(
    y_true,
    probabilities,
    beta: float = THRESHOLD_BETA,
) -> pd.DataFrame:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    rows = []

    for threshold in np.linspace(0.05, 0.95, 181):
        predictions = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
                "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
                "fbeta": round(float(fbeta_score(y_true, predictions, beta=beta, zero_division=0)), 4),
                "true_positives": int(tp),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_negatives": int(tn),
                "alert_rate": round(float(predictions.mean()), 4),
            }
        )

    return pd.DataFrame(rows)


def evaluate_classifier(
    y_true,
    probabilities,
    threshold: float,
    beta: float = THRESHOLD_BETA,
) -> ClassificationMetricsDict:
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "roc_auc": round(float(roc_auc_score(y_true, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_true, probabilities)), 4),
        "brier_score": round(float(brier_score_loss(y_true, probabilities)), 4),
        "precision": round(float(precision_score(y_true, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, predictions, zero_division=0)), 4),
        "fbeta": round(float(fbeta_score(y_true, predictions, beta=beta, zero_division=0)), 4),
        "accuracy": round(float((predictions == y_true).mean()), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_true,
            predictions,
            target_names=["Normal", "Failure"],
            digits=4,
            zero_division=0,
        ),
    }


def build_feature_importance_table(pipeline: Pipeline) -> pd.DataFrame:
    preprocessor = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )
    return importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
