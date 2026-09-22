"""Freeze each model and save features and logits once.

Every score is then computed from exactly the same saved tensors, as the manual
requires. CIFAR-100 is read here for the first time - never during training.
"""

import argparse
import sys
from pathlib import Path

TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.cifar10 import CifarSubset, decode_all, eval_transform, load_table  # noqa: E402
from data.cifar100_unknowns import load_unknowns  # noqa: E402
from data.make_splits import build_splits  # noqa: E402
from models.resnet_cifar import ResNetCifar  # noqa: E402

CACHE_DIR = TASK4 / "cache"
RUNS = ("vanilla", "gcsc", "proser")
BATCH_SIZE = 512
WORKERS = 4
NUM_DUMMIES = 5


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_run(name):
    """Restore a trained model; PROSER keeps its five dummy classifiers."""
    state = torch.load(CACHE_DIR / f"{name}.pt", map_location=device())
    model = ResNetCifar(num_dummies=NUM_DUMMIES if name == "proser" else 0)
    model.load_state_dict(state["model"])
    return model.eval().to(device())


@torch.no_grad()
def run_split(model, loader, with_dummy):
    """Penultimate features, the ten known logits, and PROSER's dummy column."""
    features, logits, dummies, labels = [], [], [], []
    for images, targets in loader:
        batch_logits, batch_features = model(images.to(device(), non_blocking=True))
        features.append(batch_features.cpu())
        logits.append(batch_logits.cpu())
        if with_dummy:
            dummies.append(model.dummy_logits(batch_features).cpu())
        labels.append(targets)
    packed = {
        "features": torch.cat(features),
        "logits": torch.cat(logits),
        "labels": torch.cat(labels),
    }
    if with_dummy:
        packed["dummy"] = torch.cat(dummies)
    return packed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args()

    train_table = load_table("train")
    test_table = load_table("test")
    splits = build_splits(train_table)
    train_images = decode_all(train_table, CACHE_DIR / "cifar10_train_u8.npy")
    test_images = decode_all(test_table, CACHE_DIR / "cifar10_test_u8.npy")
    near, far = load_unknowns()

    def loader_for(table, indices, images):
        return DataLoader(
            CifarSubset(table, indices, eval_transform(), images=images),
            batch_size=BATCH_SIZE, num_workers=WORKERS,
        )

    # Unknowns keep their own decode cache; labels here are CIFAR-100 fine labels.
    near_images = decode_all(near, CACHE_DIR / "cifar100_near_u8.npy")
    far_images = decode_all(far, CACHE_DIR / "cifar100_far_u8.npy")

    splits_to_run = {
        "train": loader_for(train_table, splits["train"], train_images),
        "val": loader_for(train_table, splits["val"], train_images),
        "test": loader_for(test_table, range(len(test_table)), test_images),
        "near": loader_for(near.reset_index(drop=True), range(len(near)), near_images),
        "far": loader_for(far.reset_index(drop=True), range(len(far)), far_images),
    }
    print({name: len(loader.dataset) for name, loader in splits_to_run.items()}, flush=True)

    for name in args.runs:
        print(f"[{name}]", flush=True)
        model = load_run(name)
        packed = {
            split: run_split(model, loader, with_dummy=(name == "proser"))
            for split, loader in splits_to_run.items()
        }
        path = CACHE_DIR / f"outputs_{name}.pt"
        torch.save(packed, path)
        print(f"  wrote {path.name}", flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
