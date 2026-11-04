"""One result shape, whichever dataset produced it.

The two pipelines wrote different JSON, so a table could not be assembled without a human knowing
which file meant what. Every field a reader needs to interpret the numbers is recorded here
alongside them: the model, the guidance scale, the null set, which attributes were preserved-checked
and which FC term each CF1 used.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json


@dataclass
class EvalResult:
    model: str
    dataset: str
    guidance: float
    null_set: str                 # "target-only" or "all"
    n_samples: int
    metrics: dict = field(default_factory=dict)
    conditioned: tuple = ()
    preserved_unobs: tuple = ()   # the attributes FC_unobs was taken over
    floor: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)

    def to_json(self, path=None) -> str:
        s = json.dumps(asdict(self), indent=1, sort_keys=False)
        if path:
            with open(path, "w") as f:
                f.write(s)
        return s

    @staticmethod
    def from_json(path) -> "EvalResult":
        with open(path) as f:
            return EvalResult(**json.load(f))
