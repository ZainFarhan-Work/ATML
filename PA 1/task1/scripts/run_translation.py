"""Accuracy and prediction consistency as the object is displaced."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from analysis.evaluate_bias import (  # noqa: E402
    head_probabilities,
    metrics,
    prediction_consistency,
    zero_shot_probabilities,
)
from data.make_subset import CLASSES, DEFAULT_ROOT, load_splits, load_stl10  # noqa: E402
from data.transforms import DIRECTIONS, DISPLACEMENTS, normalizer, translate  # noqa: E402
from models.backbones import (  # noqa: E402
    MODEL_NAMES,
    device,
    extract_features,
    load_backbone,
    load_clip,
    zero_shot_classifier,
)

TASK1 = Path(__file__).resolve().parents[1]
CACHE_DIR = TASK1 / "cache"
RESULTS_DIR = TASK1 / "results"
BATCH_SIZE = 128
WORKERS = 4

# Categorical slots 1-4, fixed order, from the validated default palette.
SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"


def conditions():
    """(label, displacement, direction, transform) for every required condition."""
    yield "clean", 0, "none", lambda images: images
    for delta in DISPLACEMENTS:
        if delta == 0:
            continue
        for name, (x_step, y_step) in DIRECTIONS.items():
            shift = (x_step * delta, y_step * delta)
            yield (
                f"{name}{delta}",
                delta,
                name,
                lambda images, shift=shift: translate(images, *shift),
            )


def probabilities_for(name, model, loader, zero_shot=None):
    """Class probabilities under every displacement condition."""
    normalize = normalizer(name)
    head = None
    if zero_shot is None:
        state = torch.load(CACHE_DIR / f"{name}_head.pt")
        head = nn.Linear(state["weight"].shape[1], len(CLASSES))
        head.load_state_dict(state)
        head.eval()

    outputs = {}
    for label, delta, direction, transform in conditions():
        features, targets = extract_features(model, name, loader, normalize, transform)
        if zero_shot is None:
            probabilities = head_probabilities(head, features)
        else:
            text_features, logit_scale = zero_shot
            probabilities = zero_shot_probabilities(
                features.to(device()), text_features, logit_scale
            ).cpu()
        outputs[label] = (delta, direction, probabilities, targets)
    print(f"  {len(outputs)} conditions done")
    return outputs


def rows_for(system, outputs):
    """Per-direction rows plus the across-direction mean the manual asks for."""
    _, _, clean_probabilities, clean_targets = outputs["clean"]
    rows = []
    for delta, direction, probabilities, targets in outputs.values():
        rows.append(
            {
                "system": system,
                "displacement": delta,
                "direction": direction,
                "top1": metrics(probabilities, targets)["top1"],
                "consistency": prediction_consistency(
                    clean_probabilities, probabilities
                ),
            }
        )
    frame = pd.DataFrame(rows)
    means = (
        frame[frame["direction"] != "none"]
        .groupby("displacement", as_index=False)[["top1", "consistency"]]
        .mean()
        .assign(system=system, direction="mean")
    )
    clean = frame[frame["direction"] == "none"].assign(direction="mean")
    return pd.concat([frame, clean, means], ignore_index=True)


def plot(table, out_dir=RESULTS_DIR / "figures"):
    """Accuracy and consistency against displacement, averaged over directions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    means = table[table["direction"] == "mean"].sort_values("displacement")
    figure, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharex=True)

    for axis, column, title in (
        (axes[0], "top1", "Top-1 accuracy (%)"),
        (axes[1], "consistency", "Prediction consistency (%)"),
    ):
        for color, (system, group) in zip(SERIES_COLORS, means.groupby("system")):
            axis.plot(
                group["displacement"], group[column],
                color=color, linewidth=2, marker="o", markersize=5, label=system,
            )
        axis.set_title(title, fontsize=10, color=SECONDARY_INK)
        axis.set_xlabel("displacement (pixels)", fontsize=9, color=MUTED_INK)
        axis.set_xticks(list(DISPLACEMENTS))
        axis.grid(True, color=MUTED_INK, alpha=0.25, linewidth=0.6)
        axis.tick_params(colors=MUTED_INK, labelsize=8)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(MUTED_INK)

    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "translation_curves.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    dataset = Subset(load_stl10(args.root, "test"), load_splits()["eval_subset"])
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)

    tables = []
    for name in MODEL_NAMES:
        print(f"[{name}]")
        model = load_backbone(name)
        tables.append(rows_for(f"{name} + linear head",
                               probabilities_for(name, model, loader)))

        if name == "clip":
            print("[clip zero-shot]")
            clip_model, tokenizer = load_clip()
            clip_model = clip_model.eval().to(device())
            zero_shot = (
                zero_shot_classifier(clip_model, tokenizer, CLASSES),
                clip_model.logit_scale.exp().item(),
            )
            tables.append(
                rows_for("clip zero-shot",
                         probabilities_for(name, model, loader, zero_shot))
            )
            del clip_model

        del model
        torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.concat(tables, ignore_index=True).round(2)
    path = RESULTS_DIR / "translation.csv"
    table.to_csv(path, index=False)
    print(f"\n{table[table['direction'] == 'mean'].to_string(index=False)}")
    print(f"wrote {path}")
    plot(table)


if __name__ == "__main__":
    main()
