from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


KNOWN_METADATA_COLUMNS = {
    "image_id",
    "original_path",
    "aligned_path",
    "success",
    "failure_reason",
    "split",
}


@dataclass
class LabelSchema:
    image_id_column: str
    label_columns: list[str]
    binary_columns: list[str]
    continuous_columns: list[str]


class FlexibleLabelDataset:
    """CSV-backed labels for latent probing experiments.

    Supports standard custom CSVs with an `image_id` column and any number of
    binary or continuous attribute columns. Columns outside the metadata set are
    treated as labels.
    """

    def __init__(self, csv_path: str | Path, image_id_column: str = "image_id", label_columns: Optional[list[str]] = None):
        self.csv_path = Path(csv_path)
        self.image_id_column = image_id_column
        self.frame = pd.read_csv(self.csv_path)
        if self.image_id_column not in self.frame.columns:
            # Fall back to the index/name if the CSV already uses it as a first column.
            if self.frame.columns[0] != self.image_id_column:
                self.frame = self.frame.rename(columns={self.frame.columns[0]: self.image_id_column})
        self.frame[self.image_id_column] = self.frame[self.image_id_column].astype(str)

        if label_columns is None:
            label_columns = [c for c in self.frame.columns if c not in KNOWN_METADATA_COLUMNS]
        self.missing_label_columns = [c for c in label_columns if c not in self.frame.columns]
        self.label_columns = [c for c in label_columns if c in self.frame.columns]
        self.schema = self._infer_schema()

    def _infer_schema(self) -> LabelSchema:
        binary_columns = []
        continuous_columns = []
        for col in self.label_columns:
            series = pd.to_numeric(self.frame[col], errors="coerce")
            values = set(series.dropna().unique().tolist())
            if values.issubset({0, 1, -1}):
                binary_columns.append(col)
            else:
                continuous_columns.append(col)
        return LabelSchema(
            image_id_column=self.image_id_column,
            label_columns=self.label_columns,
            binary_columns=binary_columns,
            continuous_columns=continuous_columns,
        )

    @property
    def image_ids(self) -> list[str]:
        return self.frame[self.image_id_column].astype(str).tolist()

    def subset_for_image_ids(self, image_ids: Iterable[str]) -> pd.DataFrame:
        ids = pd.Index([str(each) for each in image_ids])
        out = self.frame[self.frame[self.image_id_column].astype(str).isin(ids)].copy()
        out[self.image_id_column] = out[self.image_id_column].astype(str)
        return out

    def get_label_matrix(self, columns: Optional[list[str]] = None) -> np.ndarray:
        columns = columns or self.label_columns
        return self.frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy()

    def get_column(self, column: str) -> np.ndarray:
        return pd.to_numeric(self.frame[column], errors="coerce").to_numpy()

    def train_val_test_split(self, seed: int = 0, train_size: float = 0.7, val_size: float = 0.15):
        """Return row indices for a 70/15/15 split."""
        from sklearn.model_selection import train_test_split

        indices = np.arange(len(self.frame))
        train_idx, temp_idx = train_test_split(indices, train_size=train_size, random_state=seed, shuffle=True)
        relative_val = val_size / (1.0 - train_size)
        val_idx, test_idx = train_test_split(temp_idx, train_size=relative_val, random_state=seed, shuffle=True)
        return np.sort(train_idx), np.sort(val_idx), np.sort(test_idx)

    def merge_with_latents(self, latent_image_ids: Iterable[str]) -> pd.DataFrame:
        latent_image_ids = [str(each) for each in latent_image_ids]
        ordered = pd.DataFrame({self.image_id_column: latent_image_ids})
        merged = ordered.merge(self.frame, on=self.image_id_column, how="left")
        return merged

    def __len__(self):
        return len(self.frame)


