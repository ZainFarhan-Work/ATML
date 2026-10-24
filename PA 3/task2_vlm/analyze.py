"""Task 2 figures, tables, and qualitative complementary-pair selection.

    python -m task2_vlm.analyze --config configs/task2_vlm.yaml --run-name main
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from common.utils import load_json, load_yaml, read_jsonl, repo_path, save_json
from task2_vlm.evaluation import CONDITIONS
from task2_vlm.vlm_data import image_file

BOUNDS = [2250, 3750]


def learning_curves(cfg, run):
    res = repo_path(f"{cfg['paths']['results']}/{run}")
    log = pd.DataFrame(read_jsonl(res / "train_log.jsonl"))
    dev = pd.DataFrame(read_jsonl(res / "dev_metrics.jsonl"))
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.3))
    ax[0].plot(log["update"], log["loss"].rolling(25, min_periods=1).mean())
    ax[0].set(ylabel="train loss (25-update mean)")
    ax[1].plot(dev["update"], dev["caption_dev_loss"], marker="o")
    ax[1].set(ylabel="dev caption loss")
    for suite, ls in [("grounding", "-"), ("vqa", "--")]:
        for c in ["correct", "swap", "no_image"]:
            ax[2].plot(dev["update"], dev[f"{suite}_nll_{c}"], ls, marker=".", label=f"{suite}/{c}")
        ax[3].plot(dev["update"], dev[f"{suite}_delta_nll_swap"], ls, marker="o", label=suite)
    ax[2].set(ylabel="dev answer NLL")
    ax[3].set(ylabel="image-swap penalty ΔNLL")
    ax[2].legend(fontsize=6)
    ax[3].legend(fontsize=7)
    for a in ax:
        a.set_xlabel("main-run update")
        for b in BOUNDS:
            a.axvline(b, color="gray", ls=":", lw=0.8)
    fig.tight_layout()
    fig.savefig(repo_path(f"{cfg['paths']['figures']}/{run}_learning_curves.pdf"))


def final_tables(cfg, run):
    res = repo_path(f"{cfg['paths']['results']}/{run}")
    s = load_json(res / "final_summary.json")
    rows = []
    for suite in ("grounding", "vqa"):
        for c in CONDITIONS:
            row = {"suite": suite, "condition": c}
            for m in ["accuracy", "pair_accuracy", "identical_answer_rate", "empty_answer_rate", "follows_partner_rate",
                      "reference_nll", "gen_seconds_per_answer"]:
                d = s[suite][m][c]
                row[m] = f"{d['mean']:.3f} [{d['lo']:.3f},{d['hi']:.3f}]"
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(res / "final_table.csv", index=False)
    print(df.to_string())


def select_examples(cfg, run, per_bucket: int = 2):
    """Rule: within each suite take the first `per_bucket` pairs (file order) in each bucket:
    (a) grounded: both right under correct, both wrong under swap; (b) follows partner: both swap predictions equal the
    partner's answer; (c) image-insensitive: identical predictions under correct and swap, at least one wrong."""
    res = repo_path(f"{cfg['paths']['results']}/{run}")
    chosen = {}
    for suite in ("grounding", "vqa"):
        df = pd.DataFrame(read_jsonl(res / f"final_{suite}_examples.jsonl"))
        pairs = list(dict.fromkeys(df.pair_id))
        piv = lambda cond: df[df.condition == cond].set_index(["pair_id", "slot"])
        c, s = piv("correct"), piv("swap")
        buckets = {"grounded": [], "follows_partner": [], "image_insensitive": []}
        for pid in pairs:
            cc = [c.loc[(pid, k)] for k in (0, 1)]
            ss = [s.loc[(pid, k)] for k in (0, 1)]
            if all(x.correct for x in cc) and not any(x.correct for x in ss):
                buckets["grounded"].append(pid)
            if all(x.follows_partner for x in ss):
                buckets["follows_partner"].append(pid)
            if all(cc[k].prediction_norm == ss[k].prediction_norm for k in (0, 1)) and not all(x.correct for x in cc):
                buckets["image_insensitive"].append(pid)
        chosen[suite] = {k: v[:per_bucket] for k, v in buckets.items()}
    save_json(res / "qualitative_selection.json", chosen)
    return chosen


def plot_pair(cfg, run, suite, pid, out):
    res = repo_path(f"{cfg['paths']['results']}/{run}")
    df = pd.DataFrame(read_jsonl(res / f"final_{suite}_examples.jsonl"))
    d = df[df.pair_id == pid]
    fig, ax = plt.subplots(1, 2, figsize=(6, 3.2))
    for slot in (0, 1):
        r = d[(d.condition == "correct") & (d.slot == slot)].iloc[0]
        sw = d[(d.condition == "swap") & (d.slot == slot)].iloc[0]
        split, iid = r.image.split("/")
        ax[slot].imshow(Image.open(image_file(cfg, r.image)).convert("RGB"))
        ax[slot].axis("off")
        ax[slot].set_title(f"ref: {r.reference}\ncorrect→{r.prediction}\nswapped→{sw.prediction}", fontsize=8)
    fig.suptitle(d.iloc[0].question.split("\n")[0], fontsize=8)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--run-name", default="main")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    learning_curves(cfg, args.run_name)
    if (repo_path(cfg["paths"]["results"]) / args.run_name / "final_summary.json").exists():
        final_tables(cfg, args.run_name)
        chosen = select_examples(cfg, args.run_name)
        for suite, b in chosen.items():
            for bucket, pids in b.items():
                for pid in pids:
                    plot_pair(cfg, args.run_name, suite, pid,
                              repo_path(f"{cfg['paths']['figures']}/{args.run_name}_{suite}_{bucket}_{pid}.pdf"))


if __name__ == "__main__":
    main()
