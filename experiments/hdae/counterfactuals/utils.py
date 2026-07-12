from typing import Sequence

import numpy as np


def summarize_attribute_changes(before: np.ndarray, after: np.ndarray, target_index: int,
                                severe_threshold: float = 0.25,
                                preservation_indices: Sequence[int] = None) -> dict:
    """Summarize target edit strength and non-target preservation.

    ``before`` and ``after`` are probabilities/logits already converted to
    probabilities with shape ``(N, 40)``.
    """
    delta = after - before
    if preservation_indices is None:
        non_target = [i for i in range(delta.shape[1]) if i != target_index]
    else:
        non_target = [int(i) for i in preservation_indices]
    abs_non_target = np.abs(delta[:, non_target]) if non_target else np.zeros((delta.shape[0], 0))
    before_binary = before >= 0.5
    after_binary = after >= 0.5
    non_target_flips = before_binary[:, non_target] != after_binary[:, non_target] if non_target else np.zeros((delta.shape[0], 0), dtype=bool)
    return {
        "target_delta_mean": float(delta[:, target_index].mean()),
        "target_delta_abs_mean": float(np.abs(delta[:, target_index]).mean()),
        "target_flip_rate": float((before_binary[:, target_index] != after_binary[:, target_index]).mean()),
        "non_target_abs_delta_mean": float(abs_non_target.mean()) if non_target else float("nan"),
        "non_target_abs_delta_max_mean": float(abs_non_target.max(axis=1).mean()) if non_target else float("nan"),
        "non_target_severe_fraction": float((abs_non_target > severe_threshold).mean()) if non_target else float("nan"),
        "non_target_flip_fraction": float(non_target_flips.mean()) if non_target else float("nan"),
        "non_target_any_flip_rate": float(non_target_flips.any(axis=1).mean()) if non_target else float("nan"),
        "num_images": int(delta.shape[0]),
        "num_non_target_attributes": int(len(non_target)),
    }
def summarize_attribute_changes(before: np.ndarray, after: np.ndarray, target_index: int,
                                severe_threshold: float = 0.25,
                                preservation_indices: Sequence[int] = None) -> dict:
    """Summarize target edit strength and non-target preservation.

    ``before`` and ``after`` are probabilities/logits already converted to
    probabilities with shape ``(N, 40)``.
    """
    delta = after - before
    if preservation_indices is None:
        non_target = [i for i in range(delta.shape[1]) if i != target_index]
    else:
        non_target = [int(i) for i in preservation_indices]
    abs_non_target = np.abs(delta[:, non_target]) if non_target else np.zeros((delta.shape[0], 0))
    before_binary = before >= 0.5
    after_binary = after >= 0.5
    non_target_flips = before_binary[:, non_target] != after_binary[:, non_target] if non_target else np.zeros((delta.shape[0], 0), dtype=bool)
    return {
        "target_delta_mean": float(delta[:, target_index].mean()),
        "target_delta_abs_mean": float(np.abs(delta[:, target_index]).mean()),
        "target_flip_rate": float((before_binary[:, target_index] != after_binary[:, target_index]).mean()),
        "non_target_abs_delta_mean": float(abs_non_target.mean()) if non_target else float("nan"),
        "non_target_abs_delta_max_mean": float(abs_non_target.max(axis=1).mean()) if non_target else float("nan"),
        "non_target_severe_fraction": float((abs_non_target > severe_threshold).mean()) if non_target else float("nan"),
        "non_target_flip_fraction": float(non_target_flips.mean()) if non_target else float("nan"),
        "non_target_any_flip_rate": float(non_target_flips.any(axis=1).mean()) if non_target else float("nan"),
        "num_images": int(delta.shape[0]),
        "num_non_target_attributes": int(len(non_target)),
    }
