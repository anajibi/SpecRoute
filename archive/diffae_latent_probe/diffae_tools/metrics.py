from __future__ import annotations

from typing import Dict

import numpy as np

try:
    from scipy.stats import pearsonr, spearmanr
except Exception:  # pragma: no cover - optional in minimal envs
    pearsonr = spearmanr = None

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


def binary_classification_metrics(y_true, y_score, threshold: float = 0.5) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "positive_rate": float(y_pred.mean()),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["auroc"] = float("nan")
    return metrics


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(float)
    y_pred = np.asarray(y_pred).astype(float)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    if pearsonr is not None and len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        try:
            metrics["pearson"] = float(pearsonr(y_true, y_pred)[0])
        except Exception:
            metrics["pearson"] = float("nan")
    else:
        metrics["pearson"] = float("nan")
    if spearmanr is not None and len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        try:
            metrics["spearman"] = float(spearmanr(y_true, y_pred)[0])
        except Exception:
            metrics["spearman"] = float("nan")
    else:
        metrics["spearman"] = float("nan")
    return metrics


