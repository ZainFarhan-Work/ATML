"""Score the colour interventions against each model's own clean baseline."""

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
from data.make_subset import (  # noqa: E402
    CLASSES,
    DEFAULT_ROOT,
    load_splits,
    load_stl10,
)
from data.transforms import INTERVENTIONS, normalizer  # noqa: E402
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


def eval_loader(root, indices):
    """The one fixed 500-image evaluation subset, in a fixed order."""
    dataset = Subset(load_stl10(root, "test"), indices)
    return DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=WORKERS)


def load_head(name, dimension, classes):
    """Reload the linear head trained during the clean baseline."""
    head = nn.Linear(dimension, classes)
    head.load_state_dict(torch.load(CACHE_DIR / f"{name}_head.pt"))
    return head.eval()


def probabilities_per_intervention(name, model, loader, zero_shot=None):
    """Class probabilities for every intervention, keyed by intervention name."""
    normalize = normalizer(name)
    outputs = {}
    for label, intervention in INTERVENTIONS.items():
        features, targets = extract_features(
            model, name, loader, normalize, intervention
        )
        if zero_shot is None:
            head = load_head(name, features.shape[1], len(CLASSES))
            outputs[label] = (head_probabilities(head, features), targets)
        else:
            text_features, logit_scale = zero_shot
            outputs[label] = (
                zero_shot_probabilities(
                    features.to(device()), text_features, logit_scale
                ).cpu(),
                targets,
            )
        print(f"  {label}: done")
    return outputs


def rows_for(system, outputs):
    """One table row per intervention, relative to that system's clean row."""
    clean_probabilities, clean_targets = outputs["clean"]
    clean = metrics(clean_probabilities, clean_targets)
    rows = []
    for label, (probabilities, targets) in outputs.items():
        row = metrics(probabilities, targets)
        rows.append(
            {
                "system": system,
                "intervention": label,
                **row,
                "delta_top1": row["top1"] - clean["top1"],
                "consistency": prediction_consistency(
                    clean_probabilities, probabilities
                ),
            }
        )
    return rows


def color_bias(root=DEFAULT_ROOT):
    """Grayscale and hue rotation for the three heads plus zero-shot CLIP."""
    loader = eval_loader(root, load_splits()["eval_subset"])
    rows = []

    for name in MODEL_NAMES:
        print(f"[{name}]")
        model = load_backbone(name)
        rows += rows_for(
            f"{name} + linear head",
            probabilities_per_intervention(name, model, loader),
        )

        if name == "clip":
            print("[clip zero-shot]")
            clip_model, tokenizer = load_clip()
            clip_model = clip_model.eval().to(device())
            zero_shot = (
                zero_shot_classifier(clip_model, tokenizer, CLASSES),
                clip_model.logit_scale.exp().item(),
            )
            rows += rows_for(
                "clip zero-shot",
                probabilities_per_intervention(name, model, loader, zero_shot),
            )
            del clip_model

        del model
        torch.cuda.empty_cache()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).round(2)
    path = RESULTS_DIR / "color_bias.csv"
    table.to_csv(path, index=False)
    print(f"\n{table.to_string(index=False)}\nwrote {path}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    color_bias(args.root)


if __name__ == "__main__":
    main()
