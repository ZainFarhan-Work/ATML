"""Compact report figures for Task 1: cue-conflict examples with predictions, one-panel translation curve.

Reads saved outputs and the saved linear heads; nothing is trained. Examples are picked by a fixed rule
(first stem in sorted order per category), never by inspecting which look best.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from data.make_cue_conflicts import CONFLICT_DIR  # noqa: E402
from data.make_subset import CLASSES  # noqa: E402
from models.backbones import load_clip, device, zero_shot_classifier  # noqa: E402
from run_shape_texture import ConflictSet, balanced, predictions_for  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGURES = RESULTS / "figures"
OUT = Path(__file__).resolve().parents[3] / "Local Reports" / "PA1" / "figures"
SYSTEMS = ("ResNet-50", "ViT-B/16", "CLIP head", "CLIP zero-shot")
COLORS = {"clip + linear head": "#2a78d6", "clip zero-shot": "#eb6834",
          "resnet50 + linear head": "#1baf7a", "vit_b_16 + linear head": "#e8a000"}
LABELS = {"clip + linear head": "CLIP + head", "clip zero-shot": "CLIP zero-shot",
          "resnet50 + linear head": "ResNet-50", "vit_b_16 + linear head": "ViT-B/16"}


def all_predictions(frame):
    loader = DataLoader(ConflictSet(frame, CONFLICT_DIR / "images"), batch_size=64)
    out = {}
    out["ResNet-50"] = predictions_for("resnet50", loader)[0]
    out["ViT-B/16"] = predictions_for("vit_b_16", loader)[0]
    out["CLIP head"] = predictions_for("clip", loader)[0]
    model, tokenizer = load_clip()
    model = model.eval().to(device())
    zero_shot = (zero_shot_classifier(model, tokenizer, CLASSES), model.logit_scale.exp().item())
    out["CLIP zero-shot"] = predictions_for("clip", loader, zero_shot)[0]
    return out


def cue_conflict_examples():
    frame = balanced(pd.read_csv(CONFLICT_DIR / "accepted.csv")).sort_values("stem").reset_index(drop=True)
    preds = all_predictions(frame)
    content = frame.content_class.map(CLASSES.index)
    style = frame.style_class.map(CLASSES.index)

    def kind(i, p):
        return "shape" if p[i] == content[i] else "texture" if p[i] == style[i] else "other"

    kinds = pd.DataFrame({s: [kind(i, preds[s]) for i in range(len(frame))] for s in SYSTEMS})
    rules = [
        ("all systems: shape", (kinds == "shape").all(axis=1)),
        ("ResNet-50 texture, others shape", (kinds["ResNet-50"] == "texture") & (kinds[list(SYSTEMS[1:])] == "shape").all(axis=1)),
        ("all systems: texture", (kinds == "texture").all(axis=1)),
        ("all systems: neither", (kinds == "other").all(axis=1)),
    ]
    rows = []
    for title, mask in rules:
        hits = frame[mask]
        print(f"{title}: {len(hits)} candidates")
        if len(hits):
            i = hits.index[0]
            rows.append((title, i))

    fig, axes = plt.subplots(1, len(rows), figsize=(5.4, 2.2))
    axes = [axes] if len(rows) == 1 else axes
    records = []
    for axis, (title, i) in zip(axes, rows):
        row = frame.loc[i]
        axis.imshow(Image.open(CONFLICT_DIR / "images" / f"{row.stem}.png").convert("RGB"))
        axis.set_xticks([])
        axis.set_yticks([])
        for side in axis.spines.values():
            side.set_visible(False)
        axis.set_title(f"shape: {row.content_class}, texture: {row.style_class}", fontsize=6.8, color="#333333")
        short = {"ResNet-50": "ResNet", "ViT-B/16": "ViT", "CLIP head": "CLIP head", "CLIP zero-shot": "zero-shot"}
        pred = [f"{short[s]}: {CLASSES[int(preds[s][i])]}" for s in SYSTEMS]
        axis.set_xlabel(f"{pred[0]}, {pred[1]}\n{pred[2]}, {pred[3]}", fontsize=6.3, color="#333333", labelpad=2)
        records.append({"category": title, "stem": row.stem, "content": row.content_class, "style": row.style_class,
                        **{s: CLASSES[int(preds[s][i])] for s in SYSTEMS}})
    fig.tight_layout(pad=0.6, w_pad=0.6, rect=(0, 0.03, 1, 0.97))
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "cue_conflict_examples.png", dpi=250, facecolor="white")
    pd.DataFrame(records).to_csv(RESULTS / "cue_conflict_examples.csv", index=False)
    print(pd.DataFrame(records).to_string(index=False))


def translation_panel():
    table = pd.read_csv(RESULTS / "translation.csv")
    table = table.groupby(["system", "displacement"], as_index=False)[["top1", "consistency"]].mean()
    fig, axis = plt.subplots(figsize=(2.6, 2.1))
    for system, group in table.groupby("system"):
        group = group.sort_values("displacement")
        axis.plot(group["displacement"], group["consistency"], marker="o", markersize=3.5, linewidth=1.6,
                  color=COLORS[system], label=LABELS[system])
    axis.set_xticks([0, 8, 16, 32])
    axis.set_xlabel("displacement (px)", fontsize=7)
    axis.set_ylabel("prediction consistency (%)", fontsize=7)
    axis.tick_params(labelsize=6.5)
    axis.grid(alpha=0.25)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.set_ylim(96.3, 102.2)
    axis.legend(fontsize=5.6, frameon=False, loc="upper right", ncol=2, columnspacing=0.8, handlelength=1.2)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "translation_consistency.png", dpi=250, facecolor="white")


if __name__ == "__main__":
    translation_panel()
    cue_conflict_examples()
