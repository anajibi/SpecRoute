from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


def train_attribute_directions(
    z_sem: torch.Tensor,
    attributes: pd.DataFrame,
    target_attributes: list[str],
    output_dir: str | Path,
    class_weight: str = "balanced",
    normalize_direction: bool = True,
) -> dict[str, dict[str, float]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_np = z_sem.detach().cpu().numpy()
    metrics_rows = []
    directions = {}

    for attr in target_attributes:
        y = attributes[attr].values.astype(int)
        x_train, x_val, y_train, y_val = train_test_split(z_np, y, test_size=0.2, random_state=0, stratify=y)
        clf = LogisticRegression(max_iter=200, class_weight=class_weight)
        clf.fit(x_train, y_train)
        logits = clf.decision_function(x_val)
        pred = clf.predict(x_val)
        accuracy = float((pred == y_val).mean())
        balanced = float(balanced_accuracy_score(y_val, pred))
        auroc = float("nan")
        try:
            auroc = float(roc_auc_score(y_val, logits))
        except ValueError:
            auroc = float("nan")

        w = clf.coef_.reshape(-1)
        if normalize_direction:
            norm = np.linalg.norm(w) + 1e-8
            w = w / norm

        directions[attr] = {"w": w, "b": float(clf.intercept_[0])}
        metrics_rows.append(
            {
                "attribute": attr,
                "accuracy": accuracy,
                "balanced_accuracy": balanced,
                "auroc": auroc,
                "positive_prevalence": float(y.mean()),
                "negative_prevalence": float(1.0 - y.mean()),
            }
        )

    direction_payload = {
        attr: {
            "w": torch.tensor(payload["w"]),
            "b": torch.tensor(payload["b"]),
        }
        for attr, payload in directions.items()
    }
    torch.save(direction_payload, output_dir / "directions.pt")
    pd.DataFrame(metrics_rows).to_csv(output_dir / "classifier_metrics.csv", index=False)
    return {row["attribute"]: row for row in metrics_rows}
