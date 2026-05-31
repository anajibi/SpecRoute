from __future__ import annotations

import numpy as np


def target_edit_success(
    original_prob: float,
    edited_prob: float,
    direction: str,
) -> float:
    delta = edited_prob - original_prob
    if direction == "positive":
        return delta
    return -delta


def target_flip_rate(original_prob: float, edited_prob: float, threshold: float = 0.5) -> float:
    return float((original_prob >= threshold) != (edited_prob >= threshold))


def non_target_prob_delta(original_probs: np.ndarray, edited_probs: np.ndarray) -> float:
    return float(np.mean(np.abs(edited_probs - original_probs)))


def non_target_flip_rate(original_probs: np.ndarray, edited_probs: np.ndarray, threshold: float = 0.5) -> float:
    original_labels = original_probs >= threshold
    edited_labels = edited_probs >= threshold
    return float(np.mean(original_labels != edited_labels))

