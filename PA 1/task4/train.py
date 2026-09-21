"""Train Vanilla, GCSC and PROSER on CIFAR-10. No CIFAR-100 image is loaded here."""

import argparse
import json
import sys
import time
from pathlib import Path

TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.cifar10 import (  # noqa: E402
    CifarSubset,
    decode_all,
    eval_transform,
    load_table,
    train_transform,
)  # noqa: E402
from data.make_splits import SEED, build_splits  # noqa: E402
from methods import proser as proser_losses  # noqa: E402
from methods.manifold_mixup import different_class_pairs, mix  # noqa: E402
from methods.vanilla import loss as cross_entropy  # noqa: E402
from models.resnet_cifar import ResNetCifar  # noqa: E402

CACHE_DIR = TASK4 / "cache"
RESULTS_DIR = TASK4 / "results"
BATCH_SIZE = 128
WORKERS = 4
NUM_DUMMIES = 5


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def validate(model, loader):
    """CIFAR-10 validation accuracy; the only checkpoint-selection signal."""
    model.eval()
    correct = total = 0
    for images, labels in loader:
        logits, _ = model(images.to(device(), non_blocking=True))
        correct += (logits.argmax(dim=1).cpu() == labels).sum().item()
        total += len(labels)
    return 100.0 * correct / total


def train(config):
    """One model under the fixed CIFAR recipe."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    table = load_table("train")
    splits = build_splits(table)
    images = decode_all(table, CACHE_DIR / "cifar10_train_u8.npy")
    train_loader = DataLoader(
        CifarSubset(table, splits["train"],
                    train_transform(config.get("randaugment", False)), images=images),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
        num_workers=WORKERS, persistent_workers=True,
    )
    val_loader = DataLoader(
        CifarSubset(table, splits["val"], eval_transform(), images=images),
        batch_size=512, num_workers=WORKERS,
    )

    is_proser = config["method"] == "proser"
    model = ResNetCifar(num_dummies=NUM_DUMMIES if is_proser else 0).to(device())
    if is_proser:
        # Placeholders are fine-tuned from the selected Vanilla checkpoint.
        state = torch.load(CACHE_DIR / "vanilla.pt", map_location=device())
        model.load_state_dict(state["model"], strict=False)

    epochs = config["epochs"]
    optimizer = torch.optim.SGD(
        model.parameters(), lr=config["learning_rate"],
        momentum=0.9, weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    rng = np.random.RandomState(SEED)

    history, best = [], {"accuracy": -1.0, "epoch": -1, "state": None}
    for epoch in range(epochs):
        model.train()
        running = 0.0
        started = time.time()
        for images, labels in train_loader:
            images = images.to(device(), non_blocking=True)
            labels = labels.to(device(), non_blocking=True)

            if not is_proser:
                logits, _ = model(images)
                loss = cross_entropy(logits, labels)
            else:
                # Half the batch trains classifier placeholders, half trains
                # data placeholders, as the manual requires.
                half = len(images) // 2
                known_logits, feature = model(images[:half])
                loss = proser_losses.classifier_placeholder_loss(
                    known_logits, model.dummy_logits(feature), labels[:half]
                )

                hidden = model.forward_pre(images[half:])
                order, valid = different_class_pairs(labels[half:].cpu())
                if valid.any():
                    mixed, _ = mix(hidden, order.to(device()), rng)
                    mixed_logits, mixed_feature = model.forward_post(mixed[valid.to(device())])
                    loss = loss + proser_losses.data_placeholder_loss(
                        mixed_logits, model.dummy_logits(mixed_feature)
                    )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item()

        scheduler.step()
        accuracy = validate(model, val_loader)
        row = {
            "epoch": epoch + 1,
            "loss": round(running / len(train_loader), 4),
            "val_accuracy": round(accuracy, 2),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  {config['name']} epoch {row['epoch']:>3}: loss {row['loss']:.4f} "
                  f"val {row['val_accuracy']:.2f}% ({row['seconds']}s)", flush=True)

        if accuracy > best["accuracy"]:
            best = {
                "accuracy": accuracy,
                "epoch": epoch + 1,
                "state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
            }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best["state"], "config": config}, CACHE_DIR / f"{config['name']}.pt")
    (RESULTS_DIR / f"history_{config['name']}.json").write_text(json.dumps(history))
    print(f"  {config['name']}: best epoch {best['epoch']} "
          f"(CIFAR-10 val accuracy {best['accuracy']:.2f}%)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    args = parser.parse_args()

    for path in args.configs:
        config = yaml.safe_load(Path(path).read_text())
        print(f"[{config['name']}]", flush=True)
        train(config)


if __name__ == "__main__":
    main()
