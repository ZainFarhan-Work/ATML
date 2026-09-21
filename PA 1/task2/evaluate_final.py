"""Final Task 2 evaluation: target metrics, separability, per-class transfer.

Run only after every checkpoint is fixed. This is the first place target labels
are used.
"""

import argparse
import json
import sys
from pathlib import Path

TASK2 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK2))
sys.path.insert(0, str(TASK2.parent))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from evaluation.class_analysis import confusions, per_class_accuracy  # noqa: E402
from evaluation.domain_separability import collect, separability  # noqa: E402
from shared.pacs import CLASSES, SOURCE_DOMAINS  # noqa: E402
from shared.pacs_protocol import (  # noqa: E402
    build_model,
    build_splits,
    device,
    domain_loaders,
    evaluate,
    load_table,
    loaders,
    source_validation,
)

CACHE_DIR = TASK2 / "cache"
RESULTS_DIR = TASK2 / "results"
RUNS = ("source_only", "dan", "dann", "cdan", "dan_lambda0.1", "dan_lambda10")
MAIN = ("source_only", "dan", "dann", "cdan")

# Categorical slots 1-4 of the validated default palette, fixed order.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"


def load_run(name):
    """Restore one run's frozen backbone and classifier."""
    state = torch.load(CACHE_DIR / f"{name}.pt", map_location=device())
    features, classifier = build_model()
    features.load_state_dict(state["features"])
    classifier.load_state_dict(state["classifier"])
    return features.eval(), classifier.eval()


def plot_curves(out_dir=RESULTS_DIR / "figures"):
    """Classification and alignment curves, to show each method trained as intended."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for color, name in zip(SERIES_COLORS, MAIN):
        history = json.loads((RESULTS_DIR / f"history_{name}.json").read_text())
        epochs = [row["epoch"] for row in history]
        axes[0].plot(epochs, [r["cls_loss"] for r in history],
                     color=color, linewidth=2, marker="o", markersize=4, label=name)
        axes[1].plot(epochs, [r["align_loss"] for r in history],
                     color=color, linewidth=2, marker="o", markersize=4, label=name)

    axes[0].set_title("Classification loss (source)", fontsize=10, color=SECONDARY_INK)
    axes[1].set_title("Alignment / domain loss", fontsize=10, color=SECONDARY_INK)
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
    path = out_dir / "task2_training_curves.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=list(RUNS))
    args = parser.parse_args()

    table = load_table()
    splits = build_splits(table)
    _, _, val_loaders, target_eval = loaders(table, splits)
    source_domain_loader, target_domain_loader = domain_loaders(table, splits)

    rows, per_class, baseline_target = [], {}, None
    for name in args.runs:
        print(f"[{name}]", flush=True)
        features, classifier = load_run(name)

        per_domain, mean_f1, mean_accuracy = source_validation(
            features, classifier, val_loaders
        )
        target = evaluate(features, classifier, target_eval)
        score = separability(
            collect(features, source_domain_loader, device()),
            collect(features, target_domain_loader, device()),
        )

        row = {
            "run": name,
            **{f"{d}_val_acc": round(per_domain[d]["accuracy"], 2) for d in SOURCE_DOMAINS},
            "mean_source_acc": round(mean_accuracy, 2),
            "mean_source_f1": round(mean_f1, 2),
            "target_acc": round(target["accuracy"], 2),
            "target_f1": round(target["macro_f1"], 2),
            "domain_separability": round(score, 2),
        }
        if name == "source_only":
            baseline_target = target["accuracy"]
        rows.append(row)

        per_class[name] = per_class_accuracy(
            target["predicted"], target["actual"], CLASSES
        )
        per_class[f"{name}__confusions"] = {
            klass: confusions(target["predicted"], target["actual"], list(CLASSES), klass)
            for klass in CLASSES
        }
        print(f"  target acc {row['target_acc']}  separability {row['domain_separability']}",
              flush=True)

        del features, classifier
        torch.cuda.empty_cache()

    table_out = pd.DataFrame(rows)
    table_out["target_acc_change"] = (table_out["target_acc"] - baseline_target).round(2)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "task2_final.csv"
    table_out.to_csv(path, index=False)

    class_frame = pd.DataFrame(
        {name: per_class[name] for name in args.runs}
    ).round(2)
    for name in args.runs:
        if name != "source_only":
            class_frame[f"{name}_delta"] = (
                class_frame[name] - class_frame["source_only"]
            ).round(2)
    class_frame.to_csv(RESULTS_DIR / "task2_per_class_target.csv")
    (RESULTS_DIR / "task2_confusions.json").write_text(
        json.dumps({k: v for k, v in per_class.items() if k.endswith("__confusions")})
    )

    print(f"\n{table_out.to_string(index=False)}")
    print(f"\nper-class target accuracy:\n{class_frame.to_string()}")
    print(f"\nwrote {path}")
    plot_curves()


if __name__ == "__main__":
    main()
