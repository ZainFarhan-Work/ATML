"""Run Task 1 experiments."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from analysis.evaluate_bias import (  # noqa: E402
    head_probabilities,
    metrics,
    train_linear_head,
    zero_shot_probabilities,
)
from data.make_subset import (  # noqa: E402
    CLASSES,
    DEFAULT_ROOT,
    build_splits,
    download_stl10,
    load_splits,
    load_stl10,
)
from data.transforms import normalizer  # noqa: E402
from models.backbones import (  # noqa: E402
    MODEL_NAMES,
    device,
    download_weights,
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


def features_for(name, root, splits, refresh=False):
    """Frozen features for the train/val split and the fixed evaluation subset."""
    cache = CACHE_DIR / f"{name}_clean.pt"
    if cache.exists() and not refresh:
        print(f"  cached features: {cache.name}")
        return torch.load(cache)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model = load_backbone(name)
    normalize = normalizer(name)
    cached = {}
    for part, dataset_split, indices in (
        ("train", "train", splits["train"]),
        ("val", "train", splits["val"]),
        ("eval", "test", splits["eval_subset"]),
    ):
        dataset = Subset(load_stl10(root, dataset_split), indices)
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE, num_workers=WORKERS, pin_memory=True
        )
        features, labels = extract_features(model, name, loader, normalize)
        cached[part] = (features, labels)
        print(f"  {part}: {tuple(features.shape)}")

    torch.save(cached, cache)
    del model
    torch.cuda.empty_cache()
    return cached


def clean_baseline(root=DEFAULT_ROOT, refresh=False):
    """Three linear heads plus zero-shot CLIP on the clean evaluation subset."""
    splits = load_splits()
    rows = []

    for name in MODEL_NAMES:
        print(f"[{name}]")
        cached = features_for(name, root, splits, refresh)
        head, _ = train_linear_head(*cached["train"], *cached["val"])
        eval_features, eval_labels = cached["eval"]
        row = metrics(head_probabilities(head, eval_features), eval_labels)
        rows.append({"model": f"{name} + linear head", **row})
        torch.save(head.state_dict(), CACHE_DIR / f"{name}_head.pt")

        if name == "clip":
            print("[clip zero-shot]")
            model, tokenizer = load_clip()
            model = model.eval().to(device())
            text_features = zero_shot_classifier(model, tokenizer, CLASSES)
            probabilities = zero_shot_probabilities(
                eval_features.to(device()),
                text_features,
                model.logit_scale.exp().item(),
            )
            row = metrics(probabilities.cpu(), eval_labels)
            rows.append({"model": "clip zero-shot", **row})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows).round(2)
    path = RESULTS_DIR / "clean_baseline.csv"
    table.to_csv(path, index=False)
    print(f"\n{table.to_string(index=False)}\nwrote {path}")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--download", action="store_true", help="fetch STL-10 and the backbone weights"
    )
    parser.add_argument("--splits", action="store_true", help="build the split JSON")
    parser.add_argument("--clean", action="store_true", help="run the clean baseline")
    parser.add_argument(
        "--refresh", action="store_true", help="re-extract cached features"
    )
    args = parser.parse_args()

    if args.download:
        download_stl10(args.root)
        download_weights()
    if args.splits:
        build_splits(args.root)
    if args.clean:
        clean_baseline(args.root, args.refresh)


if __name__ == "__main__":
    main()
