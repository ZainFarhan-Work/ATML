"""The PACS protocol shared by Tasks 2 and 3: splits, model, batches, training."""

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from shared.pacs import (
    CLASSES,
    SOURCE_DOMAINS,
    TARGET_DOMAIN,
    DomainTagged,
    PacsSubset,
    eval_transform,
    load_table,
    train_transform,
)

SEED = 6304
VAL_FRACTION = 0.2
SPLIT_PATH = Path(__file__).resolve().parent / "splits" / f"pacs_sketch_seed{SEED}.json"

PER_SOURCE_BATCH = 8   # 8 from each of the three source domains ...
TARGET_BATCH = 24      # ... so source and target batch sizes match
EVAL_BATCH = 128
WORKERS = 4

MAX_EPOCHS = 30
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_splits(table, path=SPLIT_PATH, seed=SEED):
    """Stratified 80/20 train/val per source domain; the whole target domain.

    Written once and reused by both tasks, so Task 3 trains on exactly the same
    source data Task 2 used.
    """
    if path.exists():
        return json.loads(path.read_text())

    rng = np.random.RandomState(seed)
    splits = {"seed": seed, "classes": list(CLASSES), "sources": {}}
    for domain in SOURCE_DOMAINS:
        rows = table[table["domain"] == domain]
        train_idx, val_idx = [], []
        for label in sorted(rows["label"].unique()):
            idx = rows.index[rows["label"] == label].to_numpy().copy()
            rng.shuffle(idx)
            cut = int(round(len(idx) * VAL_FRACTION))
            val_idx += idx[:cut].tolist()
            train_idx += idx[cut:].tolist()
        splits["sources"][domain] = {
            "train": sorted(train_idx),
            "val": sorted(val_idx),
        }
    splits["target"] = sorted(table.index[table["domain"] == TARGET_DOMAIN].tolist())

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))
    return splits


def build_model(seed=SEED):
    """ResNet-18 (IMAGENET1K_V1) with a fresh 7-class head, split into F and C."""
    from torchvision.models import ResNet18_Weights, resnet18

    torch.manual_seed(seed)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    features = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
    classifier = nn.Linear(model.fc.in_features, len(CLASSES))
    return features.to(device()), classifier.to(device())


def train_mode(module):
    """Train mode, but BatchNorm kept in eval so running stats never update.

    Required by the manual: updating BN statistics on mixed source/target
    batches would itself be a form of adaptation. Affine gamma/beta stay trainable.
    """
    module.train()
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    return module


def loaders(table, splits, with_target=True):
    """One training loader per source domain, one for target, plus eval loaders.

    Task 3 passes with_target=False so that no Sketch image is ever loaded by the
    training procedure, which is that task's hard constraint.
    """
    train_loaders = {
        domain: DataLoader(
            PacsSubset(table, splits["sources"][domain]["train"], train_transform()),
            batch_size=PER_SOURCE_BATCH, shuffle=True, drop_last=True,
            num_workers=WORKERS, persistent_workers=True,
        )
        for domain in SOURCE_DOMAINS
    }
    target_loader = (
        DataLoader(
            PacsSubset(table, splits["target"], train_transform()),
            batch_size=TARGET_BATCH, shuffle=True, drop_last=True,
            num_workers=WORKERS, persistent_workers=True,
        )
        if with_target
        else None
    )
    val_loaders = {
        domain: DataLoader(
            PacsSubset(table, splits["sources"][domain]["val"], eval_transform()),
            batch_size=EVAL_BATCH, num_workers=WORKERS,
        )
        for domain in SOURCE_DOMAINS
    }
    target_eval = (
        DataLoader(
            PacsSubset(table, splits["target"], eval_transform()),
            batch_size=EVAL_BATCH, num_workers=WORKERS,
        )
        if with_target
        else None
    )
    return train_loaders, target_loader, val_loaders, target_eval


def infinite(loader):
    """Cycle a loader forever, as the manual allows for unequal domain sizes."""
    while True:
        yield from loader


@torch.no_grad()
def evaluate(features, classifier, loader):
    """Top-1 accuracy and macro-F1 on one loader."""
    features.eval()
    classifier.eval()
    predicted, actual = [], []
    for images, labels in loader:
        logits = classifier(features(images.to(device(), non_blocking=True)))
        predicted.append(logits.argmax(dim=1).cpu())
        actual.append(labels)
    predicted, actual = torch.cat(predicted), torch.cat(actual)
    return {
        "accuracy": (predicted == actual).float().mean().item() * 100,
        "macro_f1": f1_score(actual.numpy(), predicted.numpy(), average="macro") * 100,
        "predicted": predicted,
        "actual": actual,
    }


def source_validation(features, classifier, val_loaders):
    """Per-domain validation plus the mean macro-F1 used for checkpoint selection."""
    per_domain = {
        domain: evaluate(features, classifier, loader)
        for domain, loader in val_loaders.items()
    }
    mean_f1 = float(np.mean([result["macro_f1"] for result in per_domain.values()]))
    mean_accuracy = float(np.mean([r["accuracy"] for r in per_domain.values()]))
    return per_domain, mean_f1, mean_accuracy


def steps_per_epoch(train_loaders):
    """One epoch = a pass over the largest source domain's training loader."""
    return max(len(loader) for loader in train_loaders.values())


def domain_loaders(table, splits):
    """Equal-sized source-validation and target sets for the separability probe."""
    source_val = [
        index
        for domain in SOURCE_DOMAINS
        for index in splits["sources"][domain]["val"]
    ]
    rng = np.random.RandomState(SEED)
    size = min(len(source_val), len(splits["target"]))
    source_pick = rng.choice(source_val, size, replace=False).tolist()
    target_pick = rng.choice(splits["target"], size, replace=False).tolist()
    return (
        DataLoader(
            DomainTagged(table, source_pick, eval_transform(), 0),
            batch_size=EVAL_BATCH, num_workers=WORKERS,
        ),
        DataLoader(
            DomainTagged(table, target_pick, eval_transform(), 1),
            batch_size=EVAL_BATCH, num_workers=WORKERS,
        ),
    )
