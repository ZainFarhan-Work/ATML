"""Patch shuffle: destroy global layout, keep local evidence."""

import argparse
import json
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
from data.transforms import (  # noqa: E402
    PATCH_GRID,
    normalizer,
    patch_permutations,
    patch_shuffle,
)
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
SEED = 6304


class Shuffler:
    """Applies each image's own permutation, in dataset order.

    The loader is unshuffled, so resetting before each pass gives every model
    pixel-identical shuffled images.
    """

    def __init__(self, permutations):
        self.permutations = permutations
        self.cursor = 0

    def reset(self):
        self.cursor = 0
        return self

    def __call__(self, images):
        take = self.permutations[self.cursor : self.cursor + len(images)]
        self.cursor += len(images)
        return patch_shuffle(images, take)


def probabilities_for(name, model, loader, shuffler, zero_shot=None):
    """Clean and patch-shuffled probabilities for one system."""
    normalize = normalizer(name)
    head = None
    if zero_shot is None:
        state = torch.load(CACHE_DIR / f"{name}_head.pt")
        head = nn.Linear(state["weight"].shape[1], len(CLASSES))
        head.load_state_dict(state)
        head.eval()

    outputs = {}
    for label, intervention in (("clean", None), ("patch_shuffle", shuffler.reset())):
        features, targets = extract_features(
            model, name, loader, normalize, intervention
        )
        if zero_shot is None:
            probabilities = head_probabilities(head, features)
        else:
            text_features, logit_scale = zero_shot
            probabilities = zero_shot_probabilities(
                features.to(device()), text_features, logit_scale
            ).cpu()
        outputs[label] = (probabilities, targets)
    return outputs


def rows_for(system, outputs):
    """Accuracy, drop and consistency relative to the clean condition."""
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
                "delta_top1": round(row["top1"] - clean["top1"], 2),
                "consistency": prediction_consistency(
                    clean_probabilities, probabilities
                ),
            }
        )
    return rows


def save_example(dataset, permutations, out_dir=RESULTS_DIR / "figures"):
    """Clean/shuffled pairs for the report figure."""
    from torchvision.utils import save_image

    images = torch.stack([dataset[i][0] for i in range(6)])
    shuffled = patch_shuffle(images, permutations[:6])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "patch_shuffle_examples.png"
    save_image(torch.cat([images, shuffled]), path, nrow=6)
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    indices = load_splits()["eval_subset"]
    dataset = Subset(load_stl10(args.root, "test"), indices)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE)

    permutations = patch_permutations(len(indices), PATCH_GRID, SEED)
    perm_path = RESULTS_DIR / "splits" / f"patch_permutations_seed{SEED}.json"
    perm_path.parent.mkdir(parents=True, exist_ok=True)
    perm_path.write_text(json.dumps(permutations.tolist()))
    print(f"{len(permutations)} permutations ({PATCH_GRID}x{PATCH_GRID}) -> {perm_path}")

    shuffler = Shuffler(permutations)
    rows = []
    for name in MODEL_NAMES:
        print(f"[{name}]")
        model = load_backbone(name)
        rows += rows_for(
            f"{name} + linear head", probabilities_for(name, model, loader, shuffler)
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
                probabilities_for(name, model, loader, shuffler, zero_shot),
            )
            del clip_model

        del model
        torch.cuda.empty_cache()

    table = pd.DataFrame(rows).round(2)
    path = RESULTS_DIR / "patch_shuffle.csv"
    table.to_csv(path, index=False)
    print(f"\n{table.to_string(index=False)}\nwrote {path}")
    save_example(dataset, permutations)


if __name__ == "__main__":
    main()
