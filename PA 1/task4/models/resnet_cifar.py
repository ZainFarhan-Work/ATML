"""CIFAR-appropriate ResNet-18: 3x3 stem, no max-pool, 32x32 input."""

import torch
from torch import nn
from torchvision.models import resnet18

FEATURE_DIM = 512


class ResNetCifar(nn.Module):
    """ResNet-18 adapted to 32x32, exposing the penultimate feature.

    The manual's changes: the 7x7 stride-2 stem becomes 3x3 stride-1 and the
    initial max-pool is removed, so spatial resolution is not thrown away on
    small images.
    """

    def __init__(self, num_classes=10, num_dummies=0):
        super().__init__()
        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)
        # PROSER's dummy classifiers; absent (0) for Vanilla and GCSC.
        self.dummies = nn.Linear(FEATURE_DIM, num_dummies) if num_dummies else None

    def features(self, x):
        """Penultimate 512-d feature f(x)."""
        return self.backbone(x)

    def forward(self, x):
        feature = self.features(x)
        return self.classifier(feature), feature

    def dummy_logits(self, feature):
        """Strongest dummy response, or None when there are no placeholders."""
        if self.dummies is None:
            return None
        return self.dummies(feature).max(dim=1, keepdim=True).values

    def forward_pre(self, x):
        """Everything up to and including layer2 (the manifold-mixup point)."""
        b = self.backbone
        h = b.relu(b.bn1(b.conv1(x)))
        h = b.maxpool(h)
        h = b.layer1(h)
        return b.layer2(h)

    def forward_post(self, h):
        """layer3 onwards, returning logits and the feature."""
        b = self.backbone
        h = b.layer3(h)
        h = b.layer4(h)
        feature = torch.flatten(b.avgpool(h), 1)
        return self.classifier(feature), feature
