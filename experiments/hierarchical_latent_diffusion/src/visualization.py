"""Visualization helpers."""
from pathlib import Path
import torch

def save_image_grid(images,path,nrow=8):
    from torchvision.utils import save_image
    Path(path).parent.mkdir(parents=True,exist_ok=True); save_image(images.clamp(-1,1).add(1).div(2),path,nrow=nrow)
