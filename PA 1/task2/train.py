"""One training loop; the method only changes the alignment term."""

import argparse
import json
import sys
import time
from pathlib import Path

TASK2 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK2))
sys.path.insert(0, str(TASK2.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from methods import cdan, dann  # noqa: E402
from methods.dan import mmd_loss  # noqa: E402
from models.domain_discriminator import build_discriminator, schedule  # noqa: E402
from shared.pacs import CLASSES, SOURCE_DOMAINS  # noqa: E402
from shared.pacs_protocol import (  # noqa: E402
    LEARNING_RATE,
    MAX_EPOCHS,
    PATIENCE,
    SEED,
    WEIGHT_DECAY,
    build_model,
    build_splits,
    device,
    infinite,
    loaders,
    load_table,
    source_validation,
    steps_per_epoch,
    train_mode,
)

CACHE_DIR = TASK2 / "cache"
RESULTS_DIR = TASK2 / "results"
FEATURE_DIM = 512
DISCRIMINATOR_LR = 1e-3  # 10x the backbone rate; see the note in train()


def train(config, table, splits):
    """Train one method under the fixed protocol; returns the best checkpoint."""
    method = config["method"]
    torch.manual_seed(SEED)

    train_loaders, target_loader, val_loaders, _ = loaders(table, splits)
    features, classifier = build_model(SEED)

    groups = [
        {
            "params": list(features.parameters()) + list(classifier.parameters()),
            "lr": LEARNING_RATE,
        }
    ]
    discriminator = None
    if method in ("dann", "cdan"):
        width = FEATURE_DIM * len(CLASSES) if method == "cdan" else FEATURE_DIM
        discriminator = build_discriminator(width).to(device())
        # The discriminator gets 10x the backbone learning rate, as in Ganin et
        # al.'s original DANN (newly initialised layers train faster). With the
        # manual's frozen BatchNorm statistics nothing renormalises activations,
        # so a discriminator that cannot keep up lets the backbone maximise the
        # domain loss by inflating feature norm without bound; at equal rates
        # this diverged within one epoch. Backbone and classifier keep 1e-4 for
        # every method, so the shared pipeline is unchanged.
        groups.append(
            {"params": list(discriminator.parameters()), "lr": DISCRIMINATOR_LR}
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)

    source_iters = {d: iter(infinite(loader)) for d, loader in train_loaders.items()}
    target_iter = iter(infinite(target_loader))
    per_epoch = steps_per_epoch(train_loaders)
    total_steps = per_epoch * MAX_EPOCHS

    history, best = [], {"mean_f1": -1.0, "epoch": -1, "state": None}
    epochs_without_gain = 0
    step = 0

    for epoch in range(MAX_EPOCHS):
        train_mode(features)
        train_mode(classifier)
        if discriminator is not None:
            discriminator.train()

        sums = {"cls": 0.0, "align": 0.0, "domain_acc": 0.0}
        started = time.time()
        for _ in range(per_epoch):
            images, labels = [], []
            for domain in SOURCE_DOMAINS:
                batch_images, batch_labels = next(source_iters[domain])
                images.append(batch_images)
                labels.append(batch_labels)
            source_images = torch.cat(images).to(device(), non_blocking=True)
            source_labels = torch.cat(labels).to(device(), non_blocking=True)
            target_images = next(target_iter)[0].to(device(), non_blocking=True)

            source_features = features(source_images)
            logits = classifier(source_features)
            classification = F.cross_entropy(logits, source_labels)

            alignment = torch.zeros((), device=device())
            domain_accuracy = float("nan")
            if method != "source_only":
                target_features = features(target_images)
                if method == "dan":
                    alignment = config["lambda_mmd"] * mmd_loss(
                        source_features, target_features
                    )
                else:
                    alpha = schedule(step / total_steps, config.get("max_alpha", 1.0))
                    if method == "dann":
                        alignment, domain_accuracy = dann.domain_loss(
                            discriminator, source_features, target_features, alpha
                        )
                    else:
                        target_logits = classifier(target_features)
                        alignment, domain_accuracy = cdan.domain_loss(
                            discriminator,
                            (source_features, F.softmax(logits, dim=1)),
                            (target_features, F.softmax(target_logits, dim=1)),
                            alpha,
                        )

            loss = classification + alignment
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["cls"] += classification.item()
            sums["align"] += float(alignment)
            if domain_accuracy == domain_accuracy:  # not nan
                sums["domain_acc"] += domain_accuracy
            step += 1

        per_domain, mean_f1, mean_accuracy = source_validation(
            features, classifier, val_loaders
        )
        row = {
            "epoch": epoch + 1,
            "cls_loss": round(sums["cls"] / per_epoch, 4),
            "align_loss": round(sums["align"] / per_epoch, 4),
            "domain_acc": round(sums["domain_acc"] / per_epoch, 4),
            "mean_source_val_f1": round(mean_f1, 2),
            "mean_source_val_acc": round(mean_accuracy, 2),
            **{f"{d}_val_f1": round(r["macro_f1"], 2) for d, r in per_domain.items()},
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        print(f"  {method} epoch {row['epoch']:>2}: "
              f"cls {row['cls_loss']:.3f} align {row['align_loss']:.3f} "
              f"val_f1 {row['mean_source_val_f1']:.2f} ({row['seconds']}s)", flush=True)

        if mean_f1 > best["mean_f1"]:
            best = {
                "mean_f1": mean_f1,
                "epoch": epoch + 1,
                "state": {
                    "features": {k: v.cpu().clone() for k, v in features.state_dict().items()},
                    "classifier": {k: v.cpu().clone() for k, v in classifier.state_dict().items()},
                },
            }
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= PATIENCE:
                print(f"  {method}: early stop after epoch {epoch + 1}", flush=True)
                break

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], CACHE_DIR / f"{config['name']}.pt")
    (RESULTS_DIR / f"history_{config['name']}.json").write_text(json.dumps(history))
    print(f"  {method}: best epoch {best['epoch']} "
          f"(mean source val macro-F1 {best['mean_f1']:.2f})", flush=True)
    return best


def load_config(path):
    """A method config layered on base.yaml."""
    base = yaml.safe_load((TASK2 / "configs" / "base.yaml").read_text())
    base.update(yaml.safe_load(Path(path).read_text()))
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+", help="config files to run in order")
    args = parser.parse_args()

    table = load_table()
    splits = build_splits(table)
    print(f"sources: {[len(splits['sources'][d]['train']) for d in SOURCE_DOMAINS]} train, "
          f"target: {len(splits['target'])} images", flush=True)

    for path in args.configs:
        config = load_config(path)
        print(f"[{config['name']}]", flush=True)
        train(config, table, splits)


if __name__ == "__main__":
    main()
