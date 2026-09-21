"""Representation stability (I_T) and t-SNE of clean vs transformed features."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Dataset, Subset  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from analysis.feature_similarity import cosine_stability, random_pair_floor  # noqa: E402
from analysis.representation import fit_projection, plot_panels  # noqa: E402
from data.make_cue_conflicts import CONFLICT_DIR, source_indices  # noqa: E402
from data.make_subset import CLASSES, DEFAULT_ROOT, load_splits, load_stl10  # noqa: E402
from data.transforms import (  # noqa: E402
    DIRECTIONS,
    PATCH_GRID,
    grayscale,
    normalizer,
    patch_permutations,
    patch_shuffle,
    translate,
)
from models.backbones import MODEL_NAMES, extract_features, load_backbone  # noqa: E402

TASK1 = Path(__file__).resolve().parents[1]
RESULTS_DIR = TASK1 / "results"
BATCH_SIZE = 128
SEED = 6304
TRANSLATION = 32


class ImagePaths(Dataset):
    """Conflict PNGs, already 224x224, in a fixed order."""

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = Image.open(self.paths[index]).convert("RGB")
        return TF.to_tensor(image), 0


class Shuffler:
    """Per-image patch permutations, consumed in dataset order."""

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


def balanced_conflicts(seed=SEED):
    """The same balanced conflict set used for the shape-bias headline."""
    accepted = pd.read_csv(CONFLICT_DIR / "accepted.csv")
    smallest = accepted.groupby(["pair", "direction"]).size().min()
    chosen = (
        accepted.groupby(["pair", "direction"], group_keys=False)
        .apply(lambda group: group.sample(smallest, random_state=seed))
        .reset_index(drop=True)
    )
    return chosen.merge(source_indices(), on="stem")


def features_for(name, model, loader, intervention=None):
    return extract_features(model, name, loader, normalizer(name), intervention)[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()

    indices = load_splits()["eval_subset"]
    test_set = load_stl10(args.root, "test")
    eval_loader = DataLoader(Subset(test_set, indices), batch_size=BATCH_SIZE)
    eval_labels = np.array([test_set.labels[i] for i in indices])

    conflicts = balanced_conflicts()
    conflict_loader = DataLoader(
        ImagePaths(CONFLICT_DIR / "images" / f"{stem}.png" for stem in conflicts["stem"]),
        batch_size=BATCH_SIZE,
    )
    content_loader = DataLoader(
        Subset(test_set, conflicts["content_index"].tolist()), batch_size=BATCH_SIZE
    )
    style_loader = DataLoader(
        Subset(test_set, conflicts["style_index"].tolist()), batch_size=BATCH_SIZE
    )
    conflict_labels = np.array(
        [CLASSES.index(name) for name in conflicts["content_class"]]
    )
    print(f"{len(conflicts)} balanced conflicts, {len(indices)} eval images")

    shuffler = Shuffler(patch_permutations(len(indices), PATCH_GRID, SEED))
    rows = []

    for name in MODEL_NAMES:
        print(f"[{name}]")
        model = load_backbone(name)

        clean = features_for(name, model, eval_loader)
        gray = features_for(name, model, eval_loader, grayscale)
        shuffled = features_for(name, model, eval_loader, shuffler.reset())

        translated = {
            direction: features_for(
                name, model, eval_loader,
                lambda images, step=step: translate(
                    images, step[0] * TRANSLATION, step[1] * TRANSLATION
                ),
            )
            for direction, step in DIRECTIONS.items()
        }

        conflict = features_for(name, model, conflict_loader)
        content = features_for(name, model, content_loader)
        style = features_for(name, model, style_loader)
        del model
        torch.cuda.empty_cache()

        pairs = {
            "grayscale": (clean, gray),
            f"translation{TRANSLATION}": None,  # averaged over directions below
            "patch_shuffle": (clean, shuffled),
            "cue_conflict_vs_content": (content, conflict),
            "cue_conflict_vs_style": (style, conflict),
        }
        for condition, pair in pairs.items():
            if pair is None:
                value = float(
                    np.mean([cosine_stability(clean, t) for t in translated.values()])
                )
            else:
                value = cosine_stability(*pair)
            rows.append(
                {
                    "model": name,
                    "condition": condition,
                    "cosine_stability": round(value, 4),
                    "n": len(conflict) if "conflict" in condition else len(clean),
                }
            )
        rows.append(
            {
                "model": name,
                "condition": "random_pair_floor",
                "cosine_stability": round(random_pair_floor(clean, SEED), 4),
                "n": len(clean),
            }
        )

        stacked = np.concatenate(
            [
                clean.numpy(), gray.numpy(),
                translated["right"].numpy(), shuffled.numpy(), conflict.numpy(),
            ]
        )
        condition = np.array(
            ["clean"] * len(clean)
            + ["grayscale"] * len(gray)
            + [f"translation{TRANSLATION}"] * len(clean)
            + ["patch_shuffle"] * len(clean)
            + ["cue_conflict"] * len(conflict)
        )
        labels = np.concatenate(
            [eval_labels, eval_labels, eval_labels, eval_labels, conflict_labels]
        )
        print(f"  t-SNE on {len(stacked)} points ...")
        points = fit_projection(stacked, SEED)
        plot_panels(
            points, condition, labels, CLASSES,
            f"{name}: t-SNE (perplexity 30, cosine metric, seed {SEED})",
            RESULTS_DIR / "figures" / f"tsne_{name}.png",
        )

    table = pd.DataFrame(rows)
    path = RESULTS_DIR / "representation_stability.csv"
    table.to_csv(path, index=False)
    print(f"\n{table.to_string(index=False)}\nwrote {path}")


if __name__ == "__main__":
    main()
