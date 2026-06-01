from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


DEFAULT_CELEBA_GAN_XAI_ATTRIBUTES = ["Smiling", "Eyeglasses", "Young", "Male", "Blond_Hair"]


@dataclass(frozen=True)
class PretrainedAttributeClassifierInfo:
    model_id: str
    checkpoint_filename: str
    attributes: list[str]
    architecture: str
    input_size: int


class SelectedAttributeClassifier(nn.Module):
    """Differentiable wrapper that exposes one pretrained CelebA attribute logit.

    The wrapped checkpoint is trained once outside this repo and is downloaded as
    pretrained weights.  The diffusion sampler still expects a module whose
    output contains the target logit at index 0, so this wrapper selects exactly
    one column from the shared multi-attribute classifier.
    """

    def __init__(self, backbone: nn.Module, attribute_index: int, input_size: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.attribute_index = int(attribute_index)
        self.input_size = int(input_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        classifier_input = images
        if classifier_input.shape[-2:] != (self.input_size, self.input_size):
            classifier_input = F.interpolate(
                classifier_input,
                size=(self.input_size, self.input_size),
                mode="bilinear",
                align_corners=False,
            )
        logits = self.backbone(classifier_input)
        if hasattr(logits, "logits"):
            logits = logits.logits
        return logits[:, self.attribute_index : self.attribute_index + 1]


def _download_checkpoint(model_id: str, checkpoint_filename: str, cache_dir: str | Path | None) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=model_id, filename=checkpoint_filename, cache_dir=cache_dir))


def _load_torch_payload(checkpoint_path: str | Path) -> object:
    checkpoint_path = Path(checkpoint_path)
    try:
        return torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(checkpoint_path, map_location="cpu")


def _state_dict_from_payload(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "classifier_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(isinstance(key, str) for key in payload):
            tensor_values = [value for value in payload.values() if torch.is_tensor(value)]
            if tensor_values:
                return payload
    raise ValueError("Checkpoint does not contain a recognizable PyTorch state dict")


def _strip_prefix(state_dict: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(key.startswith(prefix) for key in state_dict):
        return state_dict
    return {key.removeprefix(prefix): value for key, value in state_dict.items()}


def _build_torchvision_classifier(architecture: str, num_attributes: int) -> nn.Module:
    if architecture != "resnet18":
        raise ValueError(f"Unsupported pretrained classifier architecture: {architecture}")
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_attributes)
    return model


def _load_torchvision_hf_classifier(
    classifier_cfg: dict,
    device: torch.device,
) -> tuple[nn.Module, PretrainedAttributeClassifierInfo]:
    model_id = str(classifier_cfg.get("model_id", "pymlex/celeba-gan-xai"))
    checkpoint_filename = str(classifier_cfg.get("checkpoint_filename", "attribute_classifier.pt"))
    architecture = str(classifier_cfg.get("architecture", "resnet18"))
    attributes = list(classifier_cfg.get("attributes") or DEFAULT_CELEBA_GAN_XAI_ATTRIBUTES)
    input_size = int(classifier_cfg.get("input_size", 64))
    cache_dir = classifier_cfg.get("cache_dir")

    checkpoint_path = _download_checkpoint(model_id, checkpoint_filename, cache_dir)
    payload = _load_torch_payload(checkpoint_path)
    state_dict = _state_dict_from_payload(payload)
    state_dict = _strip_prefix(state_dict, "module.")
    state_dict = _strip_prefix(state_dict, "model.")
    state_dict = _strip_prefix(state_dict, "backbone.")

    model = _build_torchvision_classifier(architecture, len(attributes))
    model.load_state_dict(state_dict)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    info = PretrainedAttributeClassifierInfo(
        model_id=model_id,
        checkpoint_filename=checkpoint_filename,
        attributes=attributes,
        architecture=architecture,
        input_size=input_size,
    )
    return model, info


def load_pretrained_attribute_classifiers(
    classifier_cfg: dict,
    target_attributes: Sequence[str],
    device: torch.device,
) -> tuple[dict[str, nn.Module], PretrainedAttributeClassifierInfo]:
    """Load off-the-shelf pretrained classifiers for the requested attributes.

    The default provider downloads a CelebA attribute classifier from Hugging
    Face Hub and wraps the shared multi-output model as one single-logit module
    per target attribute.  No project-local classifier training or checkpoint
    cache is used by the editing script.
    """

    provider = str(classifier_cfg.get("provider", "hf_torchvision_state_dict"))
    if provider != "hf_torchvision_state_dict":
        raise ValueError(f"Unsupported pretrained classifier provider: {provider}")

    shared_model, info = _load_torchvision_hf_classifier(classifier_cfg, device)
    attribute_to_index = {attribute: index for index, attribute in enumerate(info.attributes)}
    missing = [attribute for attribute in target_attributes if attribute not in attribute_to_index]
    if missing:
        raise ValueError(
            "Requested attributes are not available in the pretrained classifier: "
            f"{missing}. Available attributes: {info.attributes}"
        )

    classifiers = {
        attribute: SelectedAttributeClassifier(
            shared_model,
            attribute_to_index[attribute],
            info.input_size,
        ).to(device).eval()
        for attribute in target_attributes
    }
    return classifiers, info
