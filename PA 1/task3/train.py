"""Train DAN-DG and SAM on the three source domains. Sketch is never loaded.

ERM is not trained here: the manual requires reusing Task 2's source-only
checkpoint unchanged.
"""

import argparse
import json
import sys
import time
from pathlib import Path

TASK3 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK3))
sys.path.insert(0, str(TASK3.parent))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import yaml  # noqa: E402

from methods.dan_dg import lambda_at, pairwise_mmd  # noqa: E402
from methods.sam import ascend, descend  # noqa: E402
from shared.pacs import SOURCE_DOMAINS  # noqa: E402
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
    load_table,
    loaders,
    source_validation,
    steps_per_epoch,
    train_mode,
)

CACHE_DIR = TASK3 / "cache"
RESULTS_DIR = TASK3 / "results"


def source_batch(source_iters):
    """Eight images from each source domain, kept separable by domain."""
    per_domain = []
    for domain in SOURCE_DOMAINS:
        images, labels = next(source_iters[domain])
        per_domain.append(
            (images.to(device(), non_blocking=True), labels.to(device(), non_blocking=True))
        )
    return per_domain


def train(config, table, splits, probe_steps=0):
    """Train one Task 3 method under the Task 2 protocol, minus any target access.

    probe_steps > 0 runs only that many updates, prints collapse diagnostics, and
    writes nothing - a cheap check before committing to a full run.
    """
    method = config["method"]
    torch.manual_seed(SEED)

    train_loaders, target_loader, val_loaders, target_eval = loaders(
        table, splits, with_target=False
    )
    assert target_loader is None and target_eval is None, "Sketch must not be loaded"

    features, classifier = build_model(SEED)
    parameters = list(features.parameters()) + list(classifier.parameters())
    optimizer = torch.optim.AdamW(
        parameters, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    source_iters = {d: iter(infinite(loader)) for d, loader in train_loaders.items()}
    per_epoch = steps_per_epoch(train_loaders)
    total_steps = per_epoch * MAX_EPOCHS
    step = 0
    detach_bandwidth = config.get("detach_bandwidth", True)

    history, best = [], {"mean_f1": -1.0, "epoch": -1, "state": None}
    epochs_without_gain = 0

    for epoch in range(MAX_EPOCHS):
        train_mode(features)
        train_mode(classifier)
        sums = {"cls": 0.0, "align": 0.0, "grad_norm": 0.0, "feat_std": 0.0, "lambda": 0.0}
        started = time.time()

        for _ in range(per_epoch):
            batch = source_batch(source_iters)
            weight = lambda_at(config, step / total_steps) if method == "dan_dg" else 0.0
            spread = {}

            def forward():
                """Classification over pooled sources, plus DAN-DG's pairwise MMD."""
                domain_features = [features(images) for images, _ in batch]
                pooled = torch.cat(domain_features)
                # Mean per-dimension std across the batch: 0 means every image
                # maps to the same vector, i.e. the collapse seen at lambda = 1.
                spread["std"] = pooled.detach().std(dim=0).mean().item()
                spread["norm"] = pooled.detach().norm(dim=1).mean().item()
                logits = classifier(pooled)
                labels = torch.cat([label for _, label in batch])
                classification = F.cross_entropy(logits, labels)
                alignment = torch.zeros((), device=device())
                if method == "dan_dg":
                    alignment = weight * pairwise_mmd(
                        domain_features, detach_bandwidth=detach_bandwidth
                    )
                return classification, alignment

            classification, alignment = forward()
            (classification + alignment).backward()

            grad_norm = 0.0
            if method == "sam":
                # First pass gave the ascent direction; re-evaluate at theta + eps
                # and update theta with that gradient. BatchNorm stays frozen in
                # both passes via train_mode().
                perturbations, norm = ascend(parameters, config["rho"])
                grad_norm = float(norm)
                optimizer.zero_grad()
                perturbed_classification, perturbed_alignment = forward()
                (perturbed_classification + perturbed_alignment).backward()
                descend(parameters, perturbations)

            optimizer.step()
            optimizer.zero_grad()

            sums["cls"] += classification.item()
            sums["align"] += float(alignment)
            sums["grad_norm"] += grad_norm
            sums["feat_std"] += spread["std"]
            sums["lambda"] += weight
            step += 1

            if probe_steps:
                if step % 25 == 0 or step == 1:
                    print(f"  probe step {step:>4}: lambda {weight:.3f} "
                          f"cls {classification.item():.3f} align {float(alignment):.4f} "
                          f"|f| {spread['norm']:.2f} feat_std {spread['std']:.4f}", flush=True)
                if step >= probe_steps:
                    healthy = spread["std"] > 0.1 and classification.item() < 1.5
                    print(f"  probe verdict: {'HEALTHY' if healthy else 'COLLAPSING'} "
                          f"(feat_std {spread['std']:.4f}, cls {classification.item():.3f}; "
                          f"collapse looks like feat_std -> 0 and cls -> 1.946 = ln 7)",
                          flush=True)
                    return None

        per_domain, mean_f1, mean_accuracy = source_validation(
            features, classifier, val_loaders
        )
        worst_f1 = min(result["macro_f1"] for result in per_domain.values())
        row = {
            "epoch": epoch + 1,
            "cls_loss": round(sums["cls"] / per_epoch, 4),
            "align_loss": round(sums["align"] / per_epoch, 4),
            "grad_norm": round(sums["grad_norm"] / per_epoch, 4),
            "feat_std": round(sums["feat_std"] / per_epoch, 4),
            "lambda": round(sums["lambda"] / per_epoch, 4),
            "mean_source_val_f1": round(mean_f1, 2),
            "mean_source_val_acc": round(mean_accuracy, 2),
            "worst_source_val_f1": round(worst_f1, 2),
            **{f"{d}_val_f1": round(r["macro_f1"], 2) for d, r in per_domain.items()},
            "seconds": round(time.time() - started, 1),
        }
        history.append(row)
        print(f"  {config['name']} epoch {row['epoch']:>2}: cls {row['cls_loss']:.3f} "
              f"align {row['align_loss']:.3f} lambda {row['lambda']:.3f} "
              f"feat_std {row['feat_std']:.4f} val_f1 {row['mean_source_val_f1']:.2f} "
              f"worst {row['worst_source_val_f1']:.2f} ({row['seconds']}s)", flush=True)

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
                print(f"  {config['name']}: early stop after epoch {epoch + 1}", flush=True)
                break

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], CACHE_DIR / f"{config['name']}.pt")
    (RESULTS_DIR / f"history_{config['name']}.json").write_text(json.dumps(history))
    print(f"  {config['name']}: best epoch {best['epoch']} "
          f"(mean source val macro-F1 {best['mean_f1']:.2f})", flush=True)
    return best


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configs", nargs="+")
    parser.add_argument(
        "--probe", type=int, default=0, metavar="STEPS",
        help="run only STEPS updates, print collapse diagnostics, write nothing",
    )
    args = parser.parse_args()

    table = load_table()
    splits = build_splits(table)
    for path in args.configs:
        config = load_config(path)
        print(f"[{config['name']}]", flush=True)
        train(config, table, splits, probe_steps=args.probe)


if __name__ == "__main__":
    main()
