"""Open-set evaluation: AUROC, validation-calibrated rejection, failure analysis.

Reads only the saved outputs, so every score sees identical examples.
"""

import argparse
import json
import sys
from pathlib import Path

TASK4 = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK4))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from data.cifar10 import CLASSES  # noqa: E402
from data.cifar100_unknowns import FAR_INDICES, NEAR_INDICES  # noqa: E402
from evaluation.failure_analysis import accepted_unknown_classes  # noqa: E402
from evaluation.metrics import auroc, closed_set_accuracy  # noqa: E402
from evaluation.thresholds import acceptance_rate, threshold  # noqa: E402
from methods.proser import detection_score  # noqa: E402
from scores import energy, mahalanobis, mls, msp  # noqa: E402

CACHE_DIR = TASK4 / "cache"
RESULTS_DIR = TASK4 / "results"
POSTHOC = {"MSP": msp, "MLS": mls, "Energy": energy, "Mahalanobis": mahalanobis}
UNKNOWN_NAMES = {v: k for k, v in {**NEAR_INDICES, **FAR_INDICES}.items()}

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
MUTED_INK = "#898781"
SECONDARY_INK = "#52514e"


def scores_for(module, packed, stats):
    """One score across val / test / near / far, from the saved tensors."""
    return {
        split: module.score(packed[split]["logits"], packed[split]["features"], stats)
        for split in ("val", "test", "near", "far")
    }


def proser_scores(packed):
    """PROSER's placeholder-based detection score: dummy evidence minus best known."""
    return {
        split: detection_score(packed[split]["logits"], packed[split]["dummy"])
        for split in ("val", "test", "near", "far")
    }


def summarise(name, score_name, packed, s):
    """AUROC on the three comparisons, plus the 95%-TPR operating point."""
    both = np.concatenate([s["near"].numpy(), s["far"].numpy()])
    tau = threshold(s["val"].numpy())
    accept_test = acceptance_rate(s["test"].numpy(), tau)
    accept_near = acceptance_rate(s["near"].numpy(), tau)
    accept_far = acceptance_rate(s["far"].numpy(), tau)
    return {
        "model": name,
        "score": score_name,
        "csa": round(closed_set_accuracy(packed["test"]["logits"], packed["test"]["labels"]), 2),
        "auroc_near": round(auroc(s["test"].numpy(), s["near"].numpy()), 2),
        "auroc_far": round(auroc(s["test"].numpy(), s["far"].numpy()), 2),
        "auroc_all": round(auroc(s["test"].numpy(), both), 2),
        "tau": round(tau, 4),
        "test_acceptance": round(accept_test, 2),
        "near_rejection": round(100 - accept_near, 2),
        "far_rejection": round(100 - accept_far, 2),
        "fpr95_near": round(accept_near, 2),
        "fpr95_far": round(accept_far, 2),
    }


def plot_distributions(packed, stats, out_dir=RESULTS_DIR / "figures"):
    """Known vs near vs far score distributions for MSP, MLS and Mahalanobis."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shown = ("MSP", "MLS", "Mahalanobis")
    figure, axes = plt.subplots(1, len(shown), figsize=(10.5, 3.2))
    for axis, score_name in zip(axes, shown):
        s = scores_for(POSTHOC[score_name], packed, stats)
        for color, (label, values) in zip(
            SERIES_COLORS, (("known (test)", s["test"]), ("near", s["near"]), ("far", s["far"]))
        ):
            axis.hist(values.numpy(), bins=60, density=True, histtype="step",
                      linewidth=2, color=color, label=label)
        axis.set_title(score_name, fontsize=10, color=SECONDARY_INK)
        axis.set_xlabel("unknownness", fontsize=9, color=MUTED_INK)
        axis.tick_params(colors=MUTED_INK, labelsize=8)
        axis.set_yticks([])
        for side in ("top", "right", "left"):
            axis.spines[side].set_visible(False)
        axis.spines["bottom"].set_color(MUTED_INK)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)

    figure.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "task4_score_distributions.png"
    figure.savefig(path, dpi=200, facecolor="#fcfcfb")
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=["vanilla", "gcsc", "proser"])
    args = parser.parse_args()

    rows, failures = [], {}
    vanilla_packed = vanilla_stats = None

    for name in args.runs:
        packed = torch.load(CACHE_DIR / f"outputs_{name}.pt")
        # Mahalanobis is fitted on unaugmented CIFAR-10 *training* features only.
        stats = mahalanobis.fit(packed["train"]["features"], packed["train"]["labels"])
        if name == "vanilla":
            vanilla_packed, vanilla_stats = packed, stats

        # Step 2 of the manual compares all four post-hoc scores on vanilla;
        # step 6 compares models using MLS.
        wanted = POSTHOC.items() if name == "vanilla" else [("MLS", mls)]
        for score_name, module in wanted:
            s = scores_for(module, packed, stats)
            rows.append(summarise(name, score_name, packed, s))
            if name == "vanilla" and score_name == "MSP":
                tau = threshold(s["val"].numpy())
                failures["vanilla_MSP_near"] = accepted_unknown_classes(
                    s["near"].numpy(), tau, packed["near"]["labels"],
                    UNKNOWN_NAMES, packed["near"]["logits"].argmax(dim=1), CLASSES,
                )
                failures["vanilla_MSP_far"] = accepted_unknown_classes(
                    s["far"].numpy(), tau, packed["far"]["labels"],
                    UNKNOWN_NAMES, packed["far"]["logits"].argmax(dim=1), CLASSES,
                )

        if name == "proser":
            s = proser_scores(packed)
            rows.append(summarise(name, "Placeholder", packed, s))
            tau = threshold(s["val"].numpy())
            failures["proser_placeholder_near"] = accepted_unknown_classes(
                s["near"].numpy(), tau, packed["near"]["labels"],
                UNKNOWN_NAMES, packed["near"]["logits"].argmax(dim=1), CLASSES,
            )
        print(f"[{name}] done", flush=True)

    table = pd.DataFrame(rows)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "task4_osr.csv", index=False)
    (RESULTS_DIR / "task4_failures.json").write_text(json.dumps(failures, indent=1))

    print(f"\n{table.to_string(index=False)}")
    print(f"\naccepted unknowns (vanilla, MSP): {json.dumps(failures, indent=1)}")
    print(f"wrote {RESULTS_DIR / 'task4_osr.csv'}")
    if vanilla_packed is not None:
        plot_distributions(vanilla_packed, vanilla_stats)


if __name__ == "__main__":
    main()
