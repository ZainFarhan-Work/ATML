"""Final Task 3 evaluation. This is the only place Sketch is loaded.

Run after every Task 3 decision is fixed: source metrics, Sketch metrics, the
source-domain separability probe and the common sharpness proxy.
"""

import argparse
import json
import sys
from pathlib import Path

TASK3 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK3))
sys.path.insert(0, str(TASK3.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from evaluation.domain_metrics import summarise  # noqa: E402
from evaluation.sharpness import PER_DOMAIN, SEED, sharpness  # noqa: E402
from evaluation.source_domain_separability import collect, separability  # noqa: E402
from shared.pacs import CLASSES, SOURCE_DOMAINS, PacsSubset, eval_transform  # noqa: E402
from shared.pacs_protocol import (  # noqa: E402
    EVAL_BATCH,
    build_model,
    build_splits,
    device,
    evaluate,
    load_table,
    loaders,
    source_validation,
)

CACHE_DIR = TASK3 / "cache"
TASK2_CACHE = TASK3.parent / "task2" / "cache"
RESULTS_DIR = TASK3 / "results"

# ERM is Task 2's source-only checkpoint, reused unchanged.
RUNS = {
    "erm": TASK2_CACHE / "source_only.pt",
    "dan_dg": CACHE_DIR / "dan_dg.pt",
    "sam": CACHE_DIR / "sam.pt",
    "sam_rho0.01": CACHE_DIR / "sam_rho0.01.pt",
    "sam_rho0.1": CACHE_DIR / "sam_rho0.1.pt",
    "dan_dg_lambda0.1": CACHE_DIR / "dan_dg_lambda0.1.pt",
    "dan_dg_lambda10": CACHE_DIR / "dan_dg_lambda10.pt",
    # Optional stabilised variants; skipped automatically if not trained.
    "dan_dg_ramp": CACHE_DIR / "dan_dg_ramp.pt",
    "dan_dg_bwgrad": CACHE_DIR / "dan_dg_bwgrad.pt",
}
MAIN = ("erm", "dan_dg", "sam")

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"


def load_run(path):
    """Restore one run's frozen backbone and classifier."""
    state = torch.load(path, map_location=device())
    features, classifier = build_model()
    features.load_state_dict(state["features"])
    classifier.load_state_dict(state["classifier"])
    return features.eval(), classifier.eval()


def sharpness_batch(table, splits, seed=SEED):
    """A fixed validation batch: 32 images from each source domain."""
    rng = np.random.RandomState(seed)
    picked = []
    for domain in SOURCE_DOMAINS:
        pool = splits["sources"][domain]["val"]
        picked += rng.choice(pool, min(PER_DOMAIN, len(pool)), replace=False).tolist()
    loader = DataLoader(
        PacsSubset(table, picked, eval_transform()), batch_size=len(picked)
    )
    images, labels = next(iter(loader))
    return images.to(device()), labels.to(device())


def per_class_accuracy(predicted, actual):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    return {
        name: float((predicted[actual == index] == index).mean() * 100)
        for index, name in enumerate(CLASSES)
    }


def confusions(predicted, actual, class_name, top=3):
    predicted, actual = np.asarray(predicted), np.asarray(actual)
    index = list(CLASSES).index(class_name)
    wrong = predicted[(actual == index) & (predicted != index)]
    return pd.Series([CLASSES[p] for p in wrong]).value_counts().head(top).to_dict()


def plot_curves(out_dir=RESULTS_DIR / "figures"):
    """Classification loss and the MMD penalty, to show how each method trained."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = ("dan_dg", "dan_dg_lambda0.1", "sam", "sam_rho0.1")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for color, name in zip(SERIES_COLORS, names):
        path = RESULTS_DIR / f"history_{name}.json"
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        epochs = [r["epoch"] for r in history]
        axes[0].plot(epochs, [r["cls_loss"] for r in history],
                     color=color, linewidth=2, marker="o", markersize=4, label=name)
        axes[1].plot(epochs, [r["align_loss"] for r in history],
                     color=color, linewidth=2, marker="o", markersize=4, label=name)

    axes[0].set_title("Classification loss (source)", fontsize=10, color=SECONDARY_INK)
    axes[1].set_title("MMD penalty (DAN-DG only)", fontsize=10, color=SECONDARY_INK)
    for axis in axes:
        axis.set_xlabel("epoch", fontsize=9, color=MUTED_INK)
        axis.grid(True, color=MUTED_INK, alpha=0.25, linewidth=0.6)
        axis.tick_params(colors=MUTED_INK, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED_INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)

    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task3_training_curves.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args()

    table = load_table()
    splits = build_splits(table)
    _, _, val_loaders, sketch_loader = loaders(table, splits, with_target=True)
    probe_loaders = {
        domain: DataLoader(
            PacsSubset(table, splits["sources"][domain]["val"], eval_transform()),
            batch_size=EVAL_BATCH,
        )
        for domain in SOURCE_DOMAINS
    }
    images, labels = sharpness_batch(table, splits)

    rows, per_class, confusion_rows, baseline = [], {}, {}, None
    for name in args.runs:
        path = RUNS[name]
        if not path.exists():
            print(f"[{name}] checkpoint missing, skipped", flush=True)
            continue
        print(f"[{name}]", flush=True)
        features, classifier = load_run(path)

        per_domain, mean_f1, mean_accuracy = source_validation(
            features, classifier, val_loaders
        )
        sketch = evaluate(features, classifier, sketch_loader)
        score = separability(
            [collect(features, probe_loaders[d], device(), i)
             for i, d in enumerate(SOURCE_DOMAINS)]
        )
        delta, base_loss, grad_norm = sharpness(features, classifier, images, labels)

        row = {
            "run": name,
            **{f"{d}_val_acc": round(per_domain[d]["accuracy"], 2) for d in SOURCE_DOMAINS},
            **{f"{d}_val_f1": round(per_domain[d]["macro_f1"], 2) for d in SOURCE_DOMAINS},
            **{k: round(v, 2) for k, v in summarise(per_domain).items()},
            "sketch_acc": round(sketch["accuracy"], 2),
            "sketch_f1": round(sketch["macro_f1"], 2),
            "source_domain_separability": round(score, 2),
            "sharpness_delta": round(delta, 4),
            "val_loss": round(base_loss, 4),
            "grad_norm": round(grad_norm, 4),
        }
        if name == "erm":
            baseline = sketch["accuracy"]
        rows.append(row)

        per_class[name] = per_class_accuracy(sketch["predicted"], sketch["actual"])
        confusion_rows[name] = {
            klass: confusions(sketch["predicted"], sketch["actual"], klass)
            for klass in CLASSES
        }
        print(f"  sketch acc {row['sketch_acc']}  separability {row['source_domain_separability']}"
              f"  sharpness {row['sharpness_delta']}", flush=True)

        del features, classifier
        torch.cuda.empty_cache()

    table_out = pd.DataFrame(rows)
    if baseline is not None:
        table_out["sketch_acc_change"] = (table_out["sketch_acc"] - baseline).round(2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_out.to_csv(RESULTS_DIR / "task3_final.csv", index=False)

    class_frame = pd.DataFrame(per_class).round(2)
    for name in class_frame.columns:
        if name != "erm":
            class_frame[f"{name}_delta"] = (class_frame[name] - class_frame["erm"]).round(2)
    class_frame.to_csv(RESULTS_DIR / "task3_per_class_sketch.csv")
    (RESULTS_DIR / "task3_confusions.json").write_text(json.dumps(confusion_rows))

    print(f"\n{table_out.to_string(index=False)}")
    print(f"\nper-class Sketch accuracy:\n{class_frame.to_string()}")
    print(f"\nwrote {RESULTS_DIR / 'task3_final.csv'}")
    plot_curves()


if __name__ == "__main__":
    main()
