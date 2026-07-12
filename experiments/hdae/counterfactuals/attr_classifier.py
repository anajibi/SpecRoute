import torch
import torch.nn as nn
import torch.nn.functional as F
import  logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

class HuggingFaceResNetWrapper(nn.Module):
    """
    Optimized for 64x64 inputs. Uses a ResNet backbone which natively handles
    variable spatial resolutions without requiring aggressive, blurry upscaling.
    """

    def __init__(self, model_name="microsoft/resnet-50", num_attributes=40):
        super().__init__()
        logging.info(f"Initializing CNN backbone: {model_name}")

        from transformers import AutoImageProcessor, AutoModelForImageClassification

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.register_buffer("mean", torch.tensor(processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(processor.image_std).view(1, 3, 1, 1))

        self.cnn = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_attributes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

    def forward(self, x):
        # 1. Convert [-1, 1] diffusion range to [0, 1]
        if x.min() < 0:
            x = (x + 1.0) / 2.0

        # 2. ResNets can technically process 64x64 natively, but the pretrained ImageNet
        # weights expect features at a slightly larger scale. Interpolating to 128x128
        # is a highly calculated middle-ground: it prevents the CNN's pooling layers
        # from crushing the 64x64 feature map down to a 2x2 grid before the final layer,
        # without introducing the severe blurring of a 224x224 upscale.
        x_resized = F.interpolate(x, size=(128, 128), mode='bilinear', align_corners=False)

        # 3. Apply ImageNet Normalization
        x_norm = (x_resized - self.mean) / self.std

        outputs = self.cnn(pixel_values=x_norm)
        return outputs.logits


def load_classifier(device):
    """
    Load a HuggingFace ResNet-based attribute classifier from a local checkpoint.
    """
    import torch

    checkpoint_path = "/home/anajibi/HDM/experiments/hdae/outputs/finetuned_attr_classifier.pt"
    logging.info(f"Loading attribute classifier from {checkpoint_path}")

    # Load the state dict and attribute names
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    attribute_names = checkpoint["attribute_names"]

    # Initialize the model with the correct number of attributes
    model = HuggingFaceResNetWrapper()
    model.load_state_dict(state_dict)
    model.to(device).eval()

    return model, checkpoint