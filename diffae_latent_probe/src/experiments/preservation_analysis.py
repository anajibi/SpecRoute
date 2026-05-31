from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.celeba_attributes import get_semantic_exclusion_group
from src.metrics.attribute_preservation import (
    non_target_flip_rate,
    non_target_prob_delta,
    target_edit_success,
    target_flip_rate,
)


def run_preservation_analysis(
    original_probs: pd.DataFrame,
    edited_probs: pd.DataFrame,
    attributes_df: pd.DataFrame,
    target_attributes: list[str],
    attr_names: list[str],
    alpha_values: list[float],
    output_dir: str | Path,
    prevalence_min: float,
    prevalence_max: float,
    use_semantic_exclusion_groups: bool,
    stats_output_path: str | Path | None = None,
    summary_output_path: str | Path | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    for attr in attr_names:
        values = attributes_df[attr].astype(float)
        prevalence = float(values.mean())
        stats_rows.append(
            {
                "attribute": attr,
                "prevalence": prevalence,
                "num_positive": int(values.sum()),
                "num_negative": int(len(values) - values.sum()),
                "keep_for_preservation_eval": prevalence_min <= prevalence <= prevalence_max,
            }
        )
    stats_df = pd.DataFrame(stats_rows)
    if stats_output_path is None:
        stats_df.to_csv(output_dir / "attribute_stats.csv", index=False)
    else:
        Path(stats_output_path).parent.mkdir(parents=True, exist_ok=True)
        stats_df.to_csv(stats_output_path, index=False)

    kept_attrs = stats_df[stats_df["keep_for_preservation_eval"]]["attribute"].tolist()

    metrics_rows = []
    for attr in target_attributes:
        exclusion = [attr]
        if use_semantic_exclusion_groups:
            exclusion = get_semantic_exclusion_group(attr)
        preserved = [a for a in kept_attrs if a not in exclusion]

        for alpha in alpha_values:
            subset = edited_probs[(edited_probs["attribute"] == attr) & (edited_probs["alpha"] == alpha)]
            for _, row in subset.iterrows():
                image_id = row["image_id"]
                original_row = original_probs.loc[original_probs["image_id"] == image_id]
                if original_row.empty:
                    continue
                original_row = original_row.iloc[0]
                original_target = float(original_row[attr])
                edited_target = float(row[attr])
                direction = "positive" if alpha >= 0 else "negative"
                success = target_edit_success(original_target, edited_target, direction)
                flip = target_flip_rate(original_target, edited_target)

                if preserved:
                    original_preserved = original_row[preserved].values.astype(float)
                    edited_preserved = row[preserved].values.astype(float)
                    prob_delta = non_target_prob_delta(original_preserved, edited_preserved)
                    flip_rate = non_target_flip_rate(original_preserved, edited_preserved)
                else:
                    prob_delta = float("nan")
                    flip_rate = float("nan")

                metrics_rows.append(
                    {
                        "image_id": image_id,
                        "edited_attribute": attr,
                        "alpha": alpha,
                        "direction": direction,
                        "success_score": success,
                        "target_flip": flip,
                        "non_target_mean_abs_change": prob_delta,
                        "non_target_flip_rate": flip_rate,
                        "preservation_accuracy": 1.0 - flip_rate,
                        "semantic_exclusion": use_semantic_exclusion_groups,
                    }
                )

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(output_dir / "preservation_metrics.csv", index=False)

    summary_rows = []
    for (attr, alpha), group in metrics_df.groupby(["edited_attribute", "alpha"]):
        summary_rows.append(
            {
                "edited_attribute": attr,
                "alpha": float(alpha),
                "success_score_mean": float(group["success_score"].mean()),
                "success_score_std": float(group["success_score"].std(ddof=0)),
                "target_flip_rate": float(group["target_flip"].mean()),
                "non_target_mean_abs_change": float(group["non_target_mean_abs_change"].mean()),
                "non_target_flip_rate": float(group["non_target_flip_rate"].mean()),
                "preservation_accuracy": float(group["preservation_accuracy"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    if summary_output_path is None:
        summary_df.to_csv(output_dir / "preservation_summary_all_filtered.csv", index=False)
    else:
        Path(summary_output_path).parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_output_path, index=False)

    if use_semantic_exclusion_groups:
        summary_df.to_csv(output_dir / "preservation_summary_semantic_excluded.csv", index=False)
