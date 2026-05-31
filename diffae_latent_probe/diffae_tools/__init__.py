"""Utilities for probing DiffAE latent components."""

from .image_io import (
    DiffAEAlignmentResult,
    center_crop_resize,
    load_image_tensor,
    save_tensor_image,
)
from .model_loader import DiffAEModelWrapper
from .latent_codec import (
    LatentBundle,
    flatten_latent_for_probe,
    load_latent_bundle,
    save_latent_bundle,
)
from .latent_dataset import FlexibleLabelDataset
from .probe_models import train_probe_suite
from .metrics import (
    binary_classification_metrics,
    regression_metrics,
)
from .plotting import (
    save_attribute_bars,
    save_attribute_heatmap,
    save_reconstruction_panel,
    save_swap_panel,
)

