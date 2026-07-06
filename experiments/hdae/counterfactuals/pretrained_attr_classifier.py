import torch
import torch.nn as nn
from transformers import AutoModelForImageClassification, AutoImageProcessor


class HuggingFaceCelebAClassifier(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224-in21k", num_classes=40):
        super().__init__()

        # 1. Load the Image Processor (Handles Resizing & ImageNet Normalization)
        self.processor = AutoImageProcessor.from_pretrained(model_name)

        # 2. Load the ViT and explicitly set it for 40-label Multi-Class
        # ignore_mismatched_sizes=True strips the original ImageNet classification head
        # and randomly initializes a new linear layer with 40 outputs.
        self.vit = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            problem_type="multi_label_classification",
            ignore_mismatched_sizes=True
        )

        # 3. Register the ImageNet normalization parameters so they live on the GPU
        self.register_buffer("mean", torch.tensor(self.processor.image_mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.processor.image_std).view(1, 3, 1, 1))

    def preprocess(self, x):
        """
        Converts your diffusion [-1, 1] tensors into the exact format the ViT expects.
        x: Tensor of shape [B, 3, H, W] in range [-1, 1]
        """
        # Step A: Convert [-1, 1] to [0, 1]
        x_01 = (x + 1.0) / 2.0

        # Step B: Resize to 224x224 (Standard ViT resolution)
        import torch.nn.functional as F
        x_resized = F.interpolate(x_01, size=(224, 224), mode='bilinear', align_corners=False)

        # Step C: Apply ImageNet Normalization
        x_norm = (x_resized - self.mean) / self.std
        return x_norm

    def forward(self, x):
        # Preprocess the raw diffusion tensor
        x_processed = self.preprocess(x)

        # Pass through the Vision Transformer
        # We return logits (raw scores) directly. Do NOT apply Sigmoid here if you are training.
        outputs = self.vit(pixel_values=x_processed)
        return outputs.logits


# ==========================================
# Example Fine-Tuning Setup
# ==========================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Initialize the architecture
    model = HuggingFaceCelebAClassifier().to(device)

    # REQUIRED: Multi-label classification requires BCEWithLogitsLoss.
    # CrossEntropyLoss will fail because it assumes only ONE attribute can be true.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

    # Dummy diffusion image batch [-1, 1] and Multi-label Ground Truth [0, 1]
    dummy_images = torch.rand(8, 3, 256, 256, device=device) * 2 - 1
    dummy_labels = torch.randint(0, 2, (8, 40), dtype=torch.float32, device=device)

    # Forward pass
    logits = model(dummy_images)

    # Calculate Loss
    loss = criterion(logits, dummy_labels)

    print(f"Model successfully loaded and initialized.")
    print(f"Logits Shape: {logits.shape}")  # Expected: [8, 40]
    print(f"Calculated Loss: {loss.item():.4f}")