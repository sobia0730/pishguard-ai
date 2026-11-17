"""Evaluate phishing as the positive class."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, auc, confusion_matrix, f1_score, precision_recall_curve,
    precision_score, recall_score, roc_curve,
)


SOURCE_TO_MODEL = {0: 1, 1: 0}
LABEL_NAMES = {0: "legitimate", 1: "phishing"}
MAX_VALIDATION_FPR = 0.10


def encode_labels(labels: pd.Series) -> np.ndarray:
    """Convert source 0=phishing into classifier 1=phishing."""
    if not labels.isin(SOURCE_TO_MODEL).all():
        raise ValueError("Expected original labels 0 (phishing) and 1 (legitimate).")
    return labels.map(SOURCE_TO_MODEL).to_numpy(dtype="int8")


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Maximize recall under the false-positive cap on validation data."""
    fpr, recall, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    eligible = np.flatnonzero((fpr <= MAX_VALIDATION_FPR) & np.isfinite(thresholds))
    if not len(eligible):
        # Every finite cut exceeded the cap. Reject all predictions for this trial.
        return float(np.nextafter(1.0, np.inf))
    best = min(eligible, key=lambda i: (-recall[i], fpr[i], -thresholds[i]))
    return float(thresholds[best])


def evaluate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Evaluation requires both classifier labels.")
    predicted = (scores >= threshold).astype("int8")
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    tn, fp, _, _ = matrix.ravel()
    pr_precision, pr_recall, _ = precision_recall_curve(labels, scores)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "pr_auc": float(auc(pr_recall, pr_precision)),
        "false_positive_rate": float(fp / (fp + tn)),
        "confusion_matrix": matrix.tolist(),
    }
