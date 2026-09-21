"""ResNet, ViT, and CLIP wrappers."""

import argparse

import open_clip
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import (
    ResNet50_Weights,
    ViT_B_16_Weights,
    resnet50,
    vit_b_16,
)

# The openai weights were trained with QuickGELU; the plain ViT-B-32 config
# silently substitutes standard GELU and degrades every CLIP result.
CLIP_ARCH = "ViT-B-32-quickgelu"
CLIP_PRETRAINED = "openai"
PROMPT = "a photo of a {}."
MODEL_NAMES = ("resnet50", "vit_b_16", "clip")


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_resnet50():
    """Frozen ResNet-50 (IMAGENET1K_V2); fc dropped so forward gives GAP features."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Identity()
    return model


def load_vit_b16():
    """Frozen ViT-B/16 (IMAGENET1K_V1); heads dropped so forward gives the class token."""
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads = nn.Identity()
    return model


def load_clip():
    """Frozen OpenCLIP ViT-B-32 with the openai weights, plus its tokenizer."""
    model, _, _ = open_clip.create_model_and_transforms(
        CLIP_ARCH, pretrained=CLIP_PRETRAINED
    )
    return model, open_clip.get_tokenizer(CLIP_ARCH)


def load_backbone(name):
    """Return a frozen, eval-mode backbone by name."""
    if name == "resnet50":
        model = load_resnet50()
    elif name == "vit_b_16":
        model = load_vit_b16()
    elif name == "clip":
        model, _ = load_clip()
    else:
        raise ValueError(f"unknown backbone: {name}")

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model.to(device())


@torch.no_grad()
def encode(model, name, batch):
    """Final representation for one batch of already-normalized images."""
    if name == "clip":
        return F.normalize(model.encode_image(batch), dim=-1)
    return model(batch)


@torch.no_grad()
def extract_features(model, name, loader, normalize, intervention=None):
    """Run a frozen backbone over a loader, returning features and labels.

    The intervention is applied to the common [0, 1] image, before normalization,
    so every backbone sees pixel-identical inputs.
    """
    features, labels = [], []
    for images, targets in loader:
        images = images.to(device(), non_blocking=True)
        if intervention is not None:
            images = intervention(images)
        features.append(encode(model, name, normalize(images)).float().cpu())
        labels.append(targets)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def zero_shot_classifier(model, tokenizer, classes, prompt=PROMPT):
    """Normalized text embeddings for the fixed prompt, one column per class."""
    tokens = tokenizer([prompt.format(name) for name in classes]).to(device())
    return F.normalize(model.encode_text(tokens), dim=-1)


def download_weights():
    """Populate the local checkpoint cache for all three backbones."""
    for name in MODEL_NAMES:
        model = load_backbone(name)
        print(f"{name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")
    print(f"cuda available: {torch.cuda.is_available()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download", action="store_true", help="prefetch pretrained weights only"
    )
    args = parser.parse_args()

    if args.download:
        download_weights()


if __name__ == "__main__":
    main()
