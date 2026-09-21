"""Score the accepted cue conflicts for shape bias and coverage."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision.transforms import functional as TF  # noqa: E402

from analysis.evaluate_bias import head_probabilities, zero_shot_probabilities  # noqa: E402
from data.make_cue_conflicts import CONFLICT_DIR  # noqa: E402
from data.make_subset import CLASSES  # noqa: E402
from data.transforms import normalizer  # noqa: E402
from models.backbones import (  # noqa: E402
    MODEL_NAMES,
    device,
    encode,
    load_backbone,
    load_clip,
    zero_shot_classifier,
)

TASK1 = Path(__file__).resolve().parents[1]
CACHE_DIR = TASK1 / "cache"
RESULTS_DIR = TASK1 / "results"
BATCH_SIZE = 64
SEED = 6304


class ConflictSet(Dataset):
    """The accepted conflict images, with their content and style class indices."""

    def __init__(self, frame, image_dir):
        self.frame = frame.reset_index(drop=True)
        self.image_dir = Path(image_dir)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.loc[index]
        image = Image.open(self.image_dir / f"{row.stem}.png").convert("RGB")
        return (
            TF.to_tensor(image),
            CLASSES.index(row.content_class),
            CLASSES.index(row.style_class),
        )


def balanced(frame, seed=SEED):
    """Equal numbers per class pair and direction, as the manual asks."""
    smallest = frame.groupby(["pair", "direction"]).size().min()
    return (
        frame.groupby(["pair", "direction"], group_keys=False)
        .apply(lambda group: group.sample(smallest, random_state=seed))
        .reset_index(drop=True)
    )


@torch.no_grad()
def predictions_for(name, loader, zero_shot=None):
    """Predicted class per conflict image, plus its content and style labels."""
    model = load_backbone(name)
    normalize = normalizer(name)
    head = None
    if zero_shot is None:
        state = torch.load(CACHE_DIR / f"{name}_head.pt")
        head = nn.Linear(state["weight"].shape[1], len(CLASSES))
        head.load_state_dict(state)
        head.eval()

    predicted, content, style = [], [], []
    for images, content_labels, style_labels in loader:
        features = encode(model, name, normalize(images.to(device()))).float()
        if zero_shot is None:
            probabilities = head_probabilities(head, features.cpu())
        else:
            text_features, logit_scale = zero_shot
            probabilities = zero_shot_probabilities(
                features, text_features, logit_scale
            ).cpu()
        predicted.append(probabilities.argmax(dim=1))
        content.append(content_labels)
        style.append(style_labels)

    del model
    torch.cuda.empty_cache()
    return torch.cat(predicted), torch.cat(content), torch.cat(style)


def bias_metrics(predicted, content, style):
    """Shape bias over the two intended classes, and how often either was chosen."""
    shape = (predicted == content).sum().item()
    texture = (predicted == style).sum().item()
    other = len(predicted) - shape - texture
    decided = shape + texture
    return {
        "n": len(predicted),
        "shape": shape,
        "texture": texture,
        "other": other,
        "shape_bias": round(100 * shape / decided, 2) if decided else float("nan"),
        "coverage": round(100 * decided / len(predicted), 2),
    }


def score(frame, label, image_dir):
    """Every system's shape bias on one set of conflicts."""
    loader = DataLoader(ConflictSet(frame, image_dir), batch_size=BATCH_SIZE)
    rows = []
    for name in MODEL_NAMES:
        rows.append(
            {"set": label, "system": f"{name} + linear head",
             **bias_metrics(*predictions_for(name, loader))}
        )
        print(f"  {name}: done")

        if name == "clip":
            clip_model, tokenizer = load_clip()
            clip_model = clip_model.eval().to(device())
            zero_shot = (
                zero_shot_classifier(clip_model, tokenizer, CLASSES),
                clip_model.logit_scale.exp().item(),
            )
            rows.append(
                {"set": label, "system": "clip zero-shot",
                 **bias_metrics(*predictions_for(name, loader, zero_shot))}
            )
            del clip_model
            torch.cuda.empty_cache()
            print("  clip zero-shot: done")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conflicts", type=Path, default=CONFLICT_DIR)
    args = parser.parse_args()

    accepted = pd.read_csv(args.conflicts / "accepted.csv")
    image_dir = args.conflicts / "images"

    print(f"[balanced subsample]")
    rows = score(balanced(accepted), "balanced", image_dir)
    print(f"[all accepted]")
    rows += score(accepted, "all_accepted", image_dir)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    path = RESULTS_DIR / "shape_texture.csv"
    table.to_csv(path, index=False)
    print(f"\n{table.to_string(index=False)}\nwrote {path}")


if __name__ == "__main__":
    main()
