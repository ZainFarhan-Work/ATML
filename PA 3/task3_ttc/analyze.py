"""Step 6: compute-allocation figures/tables and the stated-rule example selection for Task 3.

    python -m task3_ttc.analyze --config configs/task3_ttc.yaml
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json, load_yaml, repo_path, save_json

MAIN_METHODS = {"math": ["first_ans", "self_consistency", "mean_logp", "self_cert", "self_cert_rwc"],
                "mathvista": ["first_ans", "self_consistency", "mean_logp", "self_cert", "self_cert_rwc",
                              "vision_only", "visconf", "visconf_rwc"]}


def accuracy_curves(cfg, ds):
    res = repo_path(f"{cfg['paths']['results']}/{ds}")
    comp = pd.read_csv(res / "method_comparison_eval.csv")
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for m in MAIN_METHODS[ds]:
        d = comp[comp.method == m].sort_values("N")
        ax[0].plot(d.N, d.acc, marker="o", label=m)
        ax[1].plot(d.tokens, d.acc, marker="o", label=m)
    orc = comp[comp.method == "self_consistency"].sort_values("N")
    ax[0].plot(orc.N, orc.oracle, "k--", label="oracle coverage")
    ax[1].plot(orc.tokens, orc.oracle, "k--", label="oracle coverage")
    ad = res / "adaptive" / "heldout_results.json"
    if ad.exists():
        for fam, r in load_json(ad).items():
            ax[1].scatter([r["tokens"]], [r["acc"]], marker="*", s=120, label=f"adaptive:{fam}", zorder=5)
    dec = pd.read_csv(res / "decomposition_eval.csv")
    ax[2].stackplot(dec.N, dec.correct, dec.selection_fail, dec.sampling_fail,
                    labels=["selected correct (SC)", "selection failure", "sampling failure"], alpha=0.8)
    ax[0].set(xscale="log", xlabel="candidates N", ylabel="accuracy / coverage (%)", title=f"{ds}: accuracy vs N")
    ax[1].set(xscale="log", xlabel="mean generated tokens", ylabel="accuracy (%)", title="accuracy vs token cost")
    ax[2].set(xscale="log", xlabel="candidates N", ylabel="% questions", title="SC outcome decomposition")
    ax[0].legend(fontsize=6)
    ax[1].legend(fontsize=6)
    ax[2].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(repo_path(f"{cfg['paths']['figures']}/{ds}_compute.pdf"))


def diversity_table(cfg, ds):
    res = repo_path(f"{cfg['paths']['results']}/{ds}")
    df = pd.read_parquet(res / "selection.parquet")
    d = df[(df.split == "eval") & (df.method == "self_consistency")]
    t = d.groupby("N").agg(distinct_groups=("n_groups", "mean"), n_correct=("n_correct", "mean"), oracle=("oracle", "mean"),
                           no_answer_selected=("no_answer", "mean"))
    t.round(3).to_csv(res / "diversity_eval.csv")
    return t


def select_examples(cfg, ds):
    """Stated rules (held-out split, order 0, N=32, first three questions in sorted-ID order):
    disagreement = SC and self-certainty selections differ and exactly one is correct."""
    res = repo_path(f"{cfg['paths']['results']}/{ds}")
    df = pd.read_parquet(res / "selection.parquet")
    d = df[(df.split == "eval") & (df.N == 32) & (df.order == 0)]
    a = d[d.method == "self_consistency"].set_index("qid")
    b = d[d.method == "self_cert"].set_index("qid")
    dis = [q for q in sorted(a.index) if a.loc[q].selected_group != b.loc[q].selected_group and a.loc[q].correct != b.loc[q].correct]
    out = {"selection_disagreement": [{"qid": q, "sc_group": a.loc[q].selected_group, "sc_correct": bool(a.loc[q].correct),
                                       "selfcert_group": b.loc[q].selected_group, "selfcert_correct": bool(b.loc[q].correct)}
                                      for q in dis[:3]]}
    pre = res / "adaptive" / "premature_stop_cases.csv"
    if pre.exists() and pre.stat().st_size > 5:
        out["premature_stop"] = pd.read_csv(pre).head(3).to_dict("records")
    if ds == "math":
        rows = res.parent / "live" / "strategy_rows.json"
        if rows.exists():
            r = pd.DataFrame(load_json(rows))
            ref = r[r.strategy == "sequential_refine"]
            out["revision_wrong_to_correct"] = ref[~ref.initial_correct & ref.correct].head(3)[["qid", "initial_answer", "answer"]].to_dict("records")
            out["revision_correct_to_wrong"] = ref[ref.initial_correct & ~ref.correct].head(3)[["qid", "initial_answer", "answer"]].to_dict("records")
    save_json(res / "example_selection.json", out)


def plot_trajectories(cfg, ds):
    p = repo_path(f"{cfg['paths']['results']}/{ds}/trajectories.json")
    if not p.exists():
        return
    traj = load_json(p)
    fig, ax = plt.subplots(len(traj), 1, figsize=(8, 1.6 * len(traj)), squeeze=False)
    for a, t in zip(ax[:, 0], traj):
        a.plot(pd.Series(t["kl"]).rolling(8, min_periods=1).mean(), label="KL to uniform (8-tok mean)")
        if "vis" in t:
            a2 = a.twinx()
            a2.plot(pd.Series(t["vis"]).rolling(8, min_periods=1).mean(), color="C1", label="visual engagement")
        a.set_title(f"{t['qid']} rollout {t['rid']} ({t['kind']})", fontsize=7)
    fig.tight_layout()
    fig.savefig(repo_path(f"{cfg['paths']['figures']}/{ds}_trajectories.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    for ds in ("math", "mathvista"):
        if not (repo_path(cfg["paths"]["results"]) / ds / "selection.parquet").exists():
            continue
        accuracy_curves(cfg, ds)
        print(diversity_table(cfg, ds).round(3))
        select_examples(cfg, ds)
        plot_trajectories(cfg, ds)


if __name__ == "__main__":
    main()
