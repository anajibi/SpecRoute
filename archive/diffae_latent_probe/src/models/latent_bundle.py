from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import json

from src.utils.io import ensure_dir, write_json


@dataclass
class LatentBundle:
    z_sem: torch.Tensor
    x_t: torch.Tensor
    image_ids: list[str]
    attributes: pd.DataFrame
    metadata: dict[str, Any]

    def save(self, output_dir: str | Path) -> None:
        output_dir = ensure_dir(output_dir)
        torch.save(self.z_sem.cpu(), output_dir / "z_sem.pt")
        torch.save(self.x_t.cpu(), output_dir / "x_T.pt")
        pd.DataFrame({"image_id": self.image_ids}).to_csv(output_dir / "image_ids.csv", index=False)
        self.attributes.to_csv(output_dir / "attributes.csv", index=False)
        write_json(output_dir / "latent_metadata.json", self.metadata)

    @classmethod
    def load(cls, output_dir: str | Path) -> "LatentBundle":
        output_dir = Path(output_dir)
        z_sem = torch.load(output_dir / "z_sem.pt", map_location="cpu")
        x_t = torch.load(output_dir / "x_T.pt", map_location="cpu")
        image_ids = pd.read_csv(output_dir / "image_ids.csv")["image_id"].tolist()
        attributes = pd.read_csv(output_dir / "attributes.csv")
        metadata_path = output_dir / "latent_metadata.json"
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text())
        return cls(z_sem=z_sem, x_t=x_t, image_ids=image_ids, attributes=attributes, metadata=metadata)

    def subset(self, indices: list[int]) -> "LatentBundle":
        return LatentBundle(
            z_sem=self.z_sem[indices],
            x_t=self.x_t[indices],
            image_ids=[self.image_ids[i] for i in indices],
            attributes=self.attributes.iloc[indices].reset_index(drop=True),
            metadata=self.metadata,
        )
