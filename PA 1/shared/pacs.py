"""PACS dataset access, backed by the parquet release."""

import io
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

DEFAULT_ROOT = Path(__file__).resolve().parents[2] / "datasets" / "pacs"
PARQUET = DEFAULT_ROOT / "pacs_train.parquet"

SOURCE_DOMAINS = ("photo", "art_painting", "cartoon")
TARGET_DOMAIN = "sketch"
DOMAINS = (*SOURCE_DOMAINS, TARGET_DOMAIN)
CLASSES = ("dog", "elephant", "giraffe", "guitar", "horse", "house", "person")

# The weights' own preprocessing constants (ResNet18_Weights.IMAGENET1K_V1).
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def train_transform():
    """Resize 256, random 224 crop, horizontal flip, as the manual specifies."""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def eval_transform():
    """Resize 256 then centre crop 224, for validation and evaluation."""
    return transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )


def load_table(parquet=PARQUET):
    """Image bytes plus domain and label for all 9,991 PACS images.

    The parquet stores an `image` struct; the bytes are lifted into their own
    column so the Dataset does not re-enter the struct per item.
    """
    frame = pd.read_parquet(parquet)
    frame["bytes"] = frame["image"].apply(lambda cell: cell["bytes"])
    return frame.drop(columns=["image"])


class PacsSubset(Dataset):
    """A fixed list of PACS rows under one transform."""

    def __init__(self, table, indices, transform):
        self.rows = table.loc[list(indices)].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.loc[index]
        image = Image.open(io.BytesIO(row["bytes"])).convert("RGB")
        return self.transform(image), int(row["label"])


class DomainTagged(PacsSubset):
    """Same images, but yielding the domain id instead of the class label.

    Used by the separability probe, where the question is whether the feature
    still says which domain an image came from.
    """

    def __init__(self, table, indices, transform, domain_id):
        super().__init__(table, indices, transform)
        self.domain_id = domain_id

    def __getitem__(self, index):
        image, _ = super().__getitem__(index)
        return image, torch.tensor(self.domain_id)
