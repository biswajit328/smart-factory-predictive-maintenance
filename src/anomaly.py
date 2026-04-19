from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .config import ANOMALY_QUANTILE, RANDOM_STATE


def build_anomaly_detector(random_state: int = RANDOM_STATE) -> IsolationForest:
    return IsolationForest(
        n_estimators=300,
        contamination="auto",
        random_state=random_state,
    )


def score_anomalies(model: IsolationForest, X_matrix) -> np.ndarray:
    return -model.score_samples(X_matrix)


def fit_anomaly_detector(
    X_matrix,
    y_train,
    quantile: float = ANOMALY_QUANTILE,
    random_state: int = RANDOM_STATE,
) -> tuple[IsolationForest, float]:
    y_array = np.asarray(y_train).astype(int)
    normal_matrix = X_matrix[y_array == 0]

    detector = build_anomaly_detector(random_state=random_state)
    detector.fit(normal_matrix)

    normal_scores = score_anomalies(detector, normal_matrix)
    threshold = float(np.quantile(normal_scores, quantile))
    return detector, threshold


def evaluate_anomaly_detector(model: IsolationForest, threshold: float, X_matrix, y_true):
    y_array = np.asarray(y_true).astype(int)
    scores = score_anomalies(model, X_matrix)
    predictions = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_array,
        predictions,
        average="binary",
        zero_division=0,
    )

    return {
        "threshold": round(float(threshold), 4),
        "roc_auc": round(float(roc_auc_score(y_array, scores)), 4),
        "pr_auc": round(float(average_precision_score(y_array, scores)), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1": round(float(f1), 4),
    }

