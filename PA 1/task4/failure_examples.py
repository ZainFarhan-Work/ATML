"""Individual accepted-unknown failures under the vanilla MLS threshold, plus score agreement.

Reads only saved outputs. Run after every model, score and threshold is fixed:
this is the final failure analysis, which may not feed back into any of them.
"""

import json
import sys
from pathlib import Path

TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from data.cifar10 import CLASSES  # noqa: E402
from data.cifar100_unknowns import FAR_INDICES, NEAR_INDICES  # noqa: E402
from evaluation.thresholds import threshold  # noqa: E402
from scores import energy, mahalanobis, mls, msp  # noqa: E402

CACHE_DIR = TASK4 / "cache"
RESULTS_DIR = TASK4 / "results"
NAMES = {v: k for k, v in {**NEAR_INDICES, **FAR_INDICES}.items()}
POSTHOC = {"MSP": msp, "MLS": mls, "Energy": energy, "Mahalanobis": mahalanobis}
PER_GROUP = 6


def plot_examples(table, out_dir=RESULTS_DIR / "figures"):
    """Grid of the failures: image, true unknown class, assigned known class, score."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    images = {g: np.load(CACHE_DIR / f"cifar100_{g}_u8.npy") for g in ("near", "far")}
    figure, axes = plt.subplots(2, PER_GROUP, figsize=(1.9 * PER_GROUP, 5.6))
    for row, group in enumerate(("near", "far")):
        members = table[table["group"] == group].reset_index(drop=True)
        for column in range(PER_GROUP):
            axis = axes[row, column]
            axis.axis("off")
            if column >= len(members):
                continue
            record = members.loc[column]
            axis.imshow(images[group][record["index"]].transpose(1, 2, 0), interpolation="nearest")
            axis.set_title(
                f"{record['unknown_class']}\n-> {record['predicted_class']}\nMLS {record['mls_score']:.1f}",
                fontsize=8, color="#52514e",
            )
        axes[row, 0].text(-0.15, 0.5, group, transform=axes[row, 0].transAxes,
                          rotation=90, va="center", ha="right", fontsize=10, color="#0b0b0b")
    figure.suptitle("Accepted unknowns under the vanilla MLS threshold", fontsize=10)
    figure.tight_layout(rect=(0.03, 0, 1, 0.94), h_pad=2.2)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task4_failure_examples.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    packed = torch.load(CACHE_DIR / "outputs_vanilla.pt")
    stats = mahalanobis.fit(packed["train"]["features"], packed["train"]["labels"])
    scores = {
        name: {s: mod.score(packed[s]["logits"], packed[s]["features"], stats).numpy()
               for s in ("val", "test", "near", "far")}
        for name, mod in POSTHOC.items()
    }
    tau = threshold(scores["MLS"]["val"])

    rows = []
    for group in ("near", "far"):
        s = scores["MLS"][group]
        labels = packed[group]["labels"].numpy()
        predicted = packed[group]["logits"].argmax(dim=1).numpy()
        accepted = np.where(s <= tau)[0]

        # Deterministic, not cherry-picked: the most confidently accepted example
        # (lowest unknownness) from each of the unknown classes that slip through most.
        per_class = pd.Series([NAMES[int(labels[i])] for i in accepted]).value_counts()
        for unknown_class in per_class.index[:PER_GROUP]:
            members = [i for i in accepted if NAMES[int(labels[i])] == unknown_class]
            i = min(members, key=lambda j: s[j])
            rows.append({
                "group": group,
                "index": int(i),
                "unknown_class": unknown_class,
                "predicted_class": CLASSES[int(predicted[i])],
                "mls_score": round(float(s[i]), 3),
                "tau": round(tau, 3),
                "margin_below_tau": round(float(tau - s[i]), 3),
                "class_accepted": int(per_class[unknown_class]),
                "class_total": 100,
            })

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "task4_failure_examples.csv", index=False)

    # Which known class absorbs accepted unknowns, per group.
    absorbed = {}
    for group in ("near", "far"):
        s = scores["MLS"][group]
        predicted = packed[group]["logits"].argmax(dim=1).numpy()
        accepted = s <= tau
        absorbed[group] = {
            "accepted": int(accepted.sum()), "of": len(s),
            "by_known_class": pd.Series([CLASSES[p] for p in predicted[accepted]])
            .value_counts().to_dict(),
        }
        # Per unknown class acceptance rate.
        labels = packed[group]["labels"].numpy()
        absorbed[group]["acceptance_by_unknown_class"] = {
            NAMES[int(c)]: round(float(accepted[labels == c].mean() * 100), 1)
            for c in np.unique(labels)
        }
    (RESULTS_DIR / "task4_absorption.json").write_text(json.dumps(absorbed, indent=1))

    # Do the four scores agree? Rank correlation over the pooled unknowns, and overlap of the
    # unknowns each one accepts at its own validation-calibrated threshold.
    names = list(POSTHOC)
    pooled = {n: np.concatenate([scores[n]["near"], scores[n]["far"]]) for n in names}
    rho = pd.DataFrame(
        [[spearmanr(pooled[a], pooled[b]).correlation for b in names] for a in names],
        index=names, columns=names,
    ).round(3)
    rho.to_csv(RESULTS_DIR / "task4_score_rank_correlation.csv")

    accepted_sets = {
        n: set(np.where(pooled[n] <= threshold(scores[n]["val"]))[0]) for n in names
    }
    overlap = pd.DataFrame(
        [[len(accepted_sets[a] & accepted_sets[b]) / max(1, len(accepted_sets[a] | accepted_sets[b]))
          for b in names] for a in names],
        index=names, columns=names,
    ).round(3)
    overlap.to_csv(RESULTS_DIR / "task4_score_accept_overlap.csv")
    sizes = {n: len(accepted_sets[n]) for n in names}
    everyone = set.intersection(*accepted_sets.values())

    print(f"vanilla MLS tau = {tau:.4f}\n")
    print(table.to_string(index=False))
    print(f"\nabsorption:\n{json.dumps(absorbed, indent=1)}")
    print(f"\nSpearman rank correlation of scores over the 1,600 unknowns:\n{rho.to_string()}")
    print(f"\nJaccard overlap of accepted unknowns:\n{overlap.to_string()}")
    print(f"\naccepted unknowns per score: {sizes}; accepted by all four: {len(everyone)}")
    plot_examples(table)


if __name__ == "__main__":
    main()
