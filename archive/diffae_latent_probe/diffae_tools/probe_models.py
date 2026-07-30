from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple
import json
import logging

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .latent_codec import flatten_latent_for_probe
from .metrics import binary_classification_metrics, regression_metrics

LOGGER = logging.getLogger(__name__)


@dataclass
class ProbeSuiteConfig:
    seed: int = 0
    test_size: float = 0.15
    val_size: float = 0.15
    semantic_mode: str = "auto"
    stochastic_mode: str = "summary"
    feature_target_dim: int = 512
    use_shuffled_control: bool = True
    use_random_baseline: bool = True


def _primary_metric_name(label_type: str) -> str:
    return "auroc" if label_type == "binary" else "r2"


def _safe_stratify(y: np.ndarray) -> Optional[np.ndarray]:
    y = np.asarray(y)
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        return None
    if counts.min() < 2:
        return None
    return y


def _split_indices(y: np.ndarray, seed: int, test_size: float, val_size: float):
    indices = np.arange(len(y))
    stratify = _safe_stratify(y)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=test_size + val_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    temp_y = y[temp_idx]
    stratify_temp = _safe_stratify(temp_y)
    relative_val = val_size / (test_size + val_size)
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1.0 - relative_val,
        random_state=seed,
        shuffle=True,
        stratify=stratify_temp,
    )
    return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)


def _make_features(semantic, stochastic, semantic_mode, stochastic_mode, target_dim, seed):
    sem_feats, sem_projector = flatten_latent_for_probe(
        semantic,
        mode=semantic_mode,
        target_dim=target_dim,
        random_state=seed,
    )
    sto_feats, sto_projector = flatten_latent_for_probe(
        stochastic,
        mode=stochastic_mode,
        target_dim=target_dim,
        random_state=seed,
    )
    concat_feats = np.concatenate([sem_feats, sto_feats], axis=1)
    return {
        "semantic": (sem_feats, sem_projector),
        "stochastic": (sto_feats, sto_projector),
        "concat": (concat_feats, None),
    }


def _normalize_binary_labels(series: pd.Series) -> np.ndarray:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    unique = set(np.unique(values[~np.isnan(values)]).tolist())
    if unique.issubset({-1.0, 1.0}):
        return (values > 0).astype(int)
    if unique.issubset({0.0, 1.0}):
        return values.astype(int)
    return (values > np.nanmedian(values)).astype(int)


def _fit_and_score(X, y, label_type: str, seed: int):
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    train_idx, val_idx, test_idx = _split_indices(y, seed=seed, test_size=0.15, val_size=0.15)

    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

    if label_type == "binary":
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=seed,
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)
        val_score = pipeline.predict_proba(X_val)[:, 1]
        test_score = pipeline.predict_proba(X_test)[:, 1]
        train_score = pipeline.predict_proba(X_train)[:, 1]
        val_metrics = binary_classification_metrics(y_val, val_score)
        test_metrics = binary_classification_metrics(y_test, test_score)
        train_metrics = binary_classification_metrics(y_train, train_score)
        coef = pipeline.named_steps["model"].coef_
        intercept = pipeline.named_steps["model"].intercept_
        scaler = pipeline.named_steps["scaler"]
    else:
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0, random_state=seed)),
            ]
        )
        pipeline.fit(X_train, y_train)
        val_score = pipeline.predict(X_val)
        test_score = pipeline.predict(X_test)
        train_score = pipeline.predict(X_train)
        val_metrics = regression_metrics(y_val, val_score)
        test_metrics = regression_metrics(y_test, test_score)
        train_metrics = regression_metrics(y_train, train_score)
        coef = pipeline.named_steps["model"].coef_
        intercept = np.asarray([pipeline.named_steps["model"].intercept_])
        scaler = pipeline.named_steps["scaler"]

    return {
        "train_idx": train_idx,
        "val_idx": val_idx,
        "test_idx": test_idx,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "train_score": train_score,
        "val_score": val_score,
        "test_score": test_score,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "coef": coef,
        "intercept": intercept,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
    }


def _score_row(
    attribute: str,
    label_type: str,
    latent_source: str,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
):
    primary = _primary_metric_name(label_type)
    return {
        "attribute": attribute,
        "label_type": label_type,
        "latent_source": latent_source,
        "primary_metric": primary,
        "train_primary_score": float(train_metrics.get(primary, np.nan)),
        "val_primary_score": float(val_metrics.get(primary, np.nan)),
        "test_primary_score": float(test_metrics.get(primary, np.nan)),
        "train_metrics": json.dumps(train_metrics),
        "val_metrics": json.dumps(val_metrics),
        "test_metrics": json.dumps(test_metrics),
    }


def train_probe_suite(
    semantic_latents,
    stochastic_latents,
    labels_frame: pd.DataFrame,
    image_id_column: str,
    output_dir: str | Path,
    cfg: Optional[ProbeSuiteConfig] = None,
    label_columns: Optional[list[str]] = None,
):
    """Train probe models for each label against each latent source.

    Returns a DataFrame with one row per attribute/source combination.
    """
    cfg = cfg or ProbeSuiteConfig()
    output_dir = Path(output_dir)
    coeff_dir = output_dir / "probe_coefficients"
    pred_dir = output_dir / "predictions"
    coeff_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    if image_id_column not in labels_frame.columns:
        raise KeyError(f"image_id column '{image_id_column}' not found in labels dataframe")

    labels_frame = labels_frame.copy()
    labels_frame[image_id_column] = labels_frame[image_id_column].astype(str)

    latent_features = _make_features(
        semantic_latents,
        stochastic_latents,
        cfg.semantic_mode,
        cfg.stochastic_mode,
        cfg.feature_target_dim,
        cfg.seed,
    )

    results = []
    label_columns = label_columns or [c for c in labels_frame.columns if c != image_id_column]

    for attribute in label_columns:
        series = labels_frame[attribute]
        numeric = pd.to_numeric(series, errors="coerce")
        valid_mask = ~numeric.isna()
        if valid_mask.sum() < 4:
            LOGGER.warning("Skipping %s because too few valid labels", attribute)
            continue
        y_raw = numeric[valid_mask].to_numpy()
        if set(np.unique(y_raw).tolist()).issubset({-1.0, 0.0, 1.0}):
            if len(np.unique(y_raw)) <= 2:
                label_type = "binary"
                y = _normalize_binary_labels(series[valid_mask])
            else:
                label_type = "continuous"
                y = y_raw.astype(float)
        else:
            # Heuristic: integer/boolean-looking columns with <=2 unique values are binary.
            unique = np.unique(y_raw)
            if len(unique) <= 2:
                label_type = "binary"
                y = _normalize_binary_labels(series[valid_mask])
            else:
                label_type = "continuous"
                y = y_raw.astype(float)

        image_ids = labels_frame.loc[valid_mask, image_id_column].astype(str).to_numpy()

        for latent_source, (X_full, projector) in latent_features.items():
            X = np.asarray(X_full)[valid_mask.to_numpy()]
            fit_out = _fit_and_score(X, y, label_type=label_type, seed=cfg.seed)
            row = _score_row(attribute, label_type, latent_source, fit_out["train_metrics"], fit_out["val_metrics"], fit_out["test_metrics"])
            row["num_examples"] = int(len(y))
            row["feature_dim"] = int(X.shape[1])
            row["projector"] = "none" if projector is None else projector.__class__.__name__
            results.append(row)

            coef_path = coeff_dir / f"{attribute}__{latent_source}.npz"
            np.savez(
                coef_path,
                coef=fit_out["coef"],
                intercept=fit_out["intercept"],
                scaler_mean=fit_out["scaler_mean"],
                scaler_scale=fit_out["scaler_scale"],
                label_type=label_type,
                latent_source=latent_source,
                attribute=attribute,
            )

            pred_df = pd.DataFrame(
                {
                    image_id_column: image_ids[fit_out["train_idx"]].tolist()
                    + image_ids[fit_out["val_idx"]].tolist()
                    + image_ids[fit_out["test_idx"]].tolist(),
                    "split": ["train"] * len(fit_out["train_idx"]) + ["val"] * len(fit_out["val_idx"]) + ["test"] * len(fit_out["test_idx"]),
                    "y_true": np.concatenate([fit_out["y_train"], fit_out["y_val"], fit_out["y_test"]]),
                    "y_pred": np.concatenate([
                        fit_out["train_score"],
                        fit_out["val_score"],
                        fit_out["test_score"],
                    ]),
                }
            )
            pred_df["latent_source"] = latent_source
            pred_df["attribute"] = attribute
            pred_df.to_csv(pred_dir / f"{attribute}__{latent_source}.csv", index=False)

        if cfg.use_shuffled_control:
            perm = np.random.default_rng(cfg.seed).permutation(len(y))
            for latent_source, (X_full, projector) in latent_features.items():
                X = np.asarray(X_full)[valid_mask.to_numpy()][perm]
                fit_out = _fit_and_score(X, y, label_type=label_type, seed=cfg.seed)
                row = _score_row(attribute, label_type, f"{latent_source}_shuffled", fit_out["train_metrics"], fit_out["val_metrics"], fit_out["test_metrics"])
                row["num_examples"] = int(len(y))
                row["feature_dim"] = int(X.shape[1])
                row["projector"] = "none" if projector is None else projector.__class__.__name__
                results.append(row)

        if cfg.use_random_baseline:
            train_idx, val_idx, test_idx = _split_indices(y, seed=cfg.seed, test_size=cfg.test_size, val_size=cfg.val_size)
            if label_type == "binary":
                train_prob = float(np.mean(y[train_idx]))
                val_score = np.full(len(val_idx), train_prob, dtype=np.float32)
                test_score = np.full(len(test_idx), train_prob, dtype=np.float32)
                train_score = np.full(len(train_idx), train_prob, dtype=np.float32)
                train_metrics = binary_classification_metrics(y[train_idx], train_score)
                val_metrics = binary_classification_metrics(y[val_idx], val_score)
                test_metrics = binary_classification_metrics(y[test_idx], test_score)
            else:
                train_mean = float(np.mean(y[train_idx]))
                val_score = np.full(len(val_idx), train_mean, dtype=np.float32)
                test_score = np.full(len(test_idx), train_mean, dtype=np.float32)
                train_score = np.full(len(train_idx), train_mean, dtype=np.float32)
                train_metrics = regression_metrics(y[train_idx], train_score)
                val_metrics = regression_metrics(y[val_idx], val_score)
                test_metrics = regression_metrics(y[test_idx], test_score)
            row = _score_row(attribute, label_type, "random_baseline", train_metrics, val_metrics, test_metrics)
            row["num_examples"] = int(len(y))
            row["feature_dim"] = 0
            row["projector"] = "none"
            results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "probe_results.csv", index=False)
    return results_df

