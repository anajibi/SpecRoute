"""The 64px reader bank: per attribute, the architecture that won on held-out F1.

No single architecture is best. Across the 40 CelebA attributes at 64px, convnext_small wins
17, convnext_base 16 and convnext_tiny 7, with under a point between their means -- so the
admitted set is the per-attribute best, not any one run. This loads all three and dispatches
each attribute to its winner, which is what `best_of_three.json` records.

Everything is scored at 64px: the protocol's measurements are defined there, so a 256px output
is downsampled before it is read.
"""
from __future__ import annotations
import json, os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

RUN_FILES = {
    "convnext_small": "_readers64.pt",
    "convnext_tiny": "_readers64_convnext_tiny.pt",
    "convnext_base": "_readers64_convnext_base.pt",
}
RUN_KEY = {"convnext_small@20": "convnext_small",
           "convnext_tiny@30": "convnext_tiny",
           "convnext_base@30": "convnext_base"}


class ReaderBank:
    def __init__(self, d="experiments/hdae/outputs/attr_predictors_celebahq64", device="cuda"):
        self.device = device
        self.models, self.attrs_of = {}, {}
        for arch, fn in RUN_FILES.items():
            ck = torch.load(os.path.join(d, fn), map_location="cpu", weights_only=False)
            m = getattr(models, arch)(weights=None)
            m.classifier[2] = nn.Linear(m.classifier[2].in_features, len(ck["attrs"]))
            m.load_state_dict({k: (v.float() if v.is_floating_point() else v)
                               for k, v in ck["state_dict"].items()})
            self.models[arch] = m.to(device).eval()
            self.attrs_of[arch] = list(ck["attrs"])
            self.size = ck["img_size"]
        best = json.load(open(os.path.join(d, "best_of_three.json")))["per_attribute"]
        # attribute -> (arch, head index within that arch's output)
        self.route = {a: (RUN_KEY[v["run"]], self.attrs_of[RUN_KEY[v["run"]]].index(a))
                      for a, v in best.items()}
        self.attrs = list(best)
        self.admitted = {a: bool(v["admitted"]) for a, v in best.items()}

    @torch.no_grad()
    def read(self, imgs01: torch.Tensor) -> dict:
        """imgs01: (N,3,H,W) in [0,1], any H. Returns {attribute: int array of 0/1}."""
        x = imgs01
        if x.shape[-1] != self.size:
            x = F.interpolate(x, size=(self.size,) * 2, mode="bicubic", align_corners=False).clamp(0, 1)
        x = x.to(self.device)
        logits = {}
        for arch, m in self.models.items():
            logits[arch] = m(x).float().cpu().numpy()
        return {a: (logits[arch][:, j] > 0).astype(np.int8)
                for a, (arch, j) in self.route.items()}
