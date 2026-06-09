"""Metrics used by preservation and counterfactual probes."""
import torch
import torch.nn.functional as F

def cosine_similarity(a,b): return F.cosine_similarity(a.flatten(1),b.flatten(1))
def mse_distance(a,b): return (a-b).square().flatten(1).mean(-1)
def attribute_agreement(a,b,threshold=0.): return ((a>threshold)==(b>threshold)).float().mean(-1)
