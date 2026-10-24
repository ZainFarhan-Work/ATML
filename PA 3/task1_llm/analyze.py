"""Task 1 tables and figures from saved results.

    python -m task1_llm.analyze --config configs/task1_llm.yaml
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.utils import load_json, load_yaml, read_jsonl, repo_path

METRICS = [("gsm8k_cot", "exact_match,flexible-extract", "GSM8K EM"),
           ("hellaswag", "acc_norm,none", "HellaSwag acc_norm"),
           ("boolq", "acc,none", "BoolQ acc"),
           ("arc_easy", "acc_norm,none", "ARC-E acc_norm")]
METHODS = ["full_ft", "full_ft_l2sp", "lora4", "lora16", "oplora16"]


def benchmark_table(cfg) -> pd.DataFrame:
    rows, base = [], None
    for cond in cfg["eval"]["conditions"]:
        r = load_json(f"{cfg['paths']['results']}/eval/{cond['name']}.json")["scores"]
        row = {"condition": cond["name"]}
        for task, key, label in METRICS:
            row[label] = 100 * r[task][key]
        if "hellaswag" in r:
            row["HellaSwag acc"] = 100 * r["hellaswag"]["acc,none"]
            row["ARC-E acc"] = 100 * r["arc_easy"]["acc,none"]
        rows.append(row)
    df = pd.DataFrame(rows).set_index("condition")
    for _, _, label in METRICS:
        df[f"Δ {label} (pp)"] = df[label] - df.loc["baseline", label]
    return df


def plot_losses(cfg):
    repo_path(cfg['paths']['figures']).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
    for m in METHODS:
        log = pd.DataFrame(read_jsonl(f"{cfg['paths']['results']}/{m}/train_log.jsonl"))
        if log.empty:
            continue
        sm = log["sft_loss"].rolling(25, min_periods=1).mean()
        ax[0].plot(log["update"], sm, label=m)
        ax[1].plot(log["update"], log["lr"], label=m)
    ax[0].set(xlabel="optimizer update", ylabel="SFT loss (25-update mean)")
    ax[1].set(xlabel="optimizer update", ylabel="learning rate")
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(repo_path(f"{cfg['paths']['figures']}/learning_curves.pdf"))


def l2sp_summary(cfg) -> dict:
    out = {}
    for m in ["full_ft", "full_ft_l2sp"]:
        snap = repo_path(f"{cfg['paths']['results']}/{m}/snapshot_1250.json")
        out[m] = load_json(snap).get("displacement_l2") if snap.exists() else None
    log = pd.DataFrame(read_jsonl(f"{cfg['paths']['results']}/full_ft_l2sp/train_log.jsonl"))
    if not log.empty:
        out["l2sp_final_omega"], out["l2sp_final_lam_omega"] = float(log.omega.iloc[-1]), float(log.lam_omega.iloc[-1])
    return out


def resource_table(cfg) -> pd.DataFrame:
    rows = []
    for m in METHODS:
        p = repo_path(f"{cfg['paths']['results']}/{m}/resources.json")
        if p.exists():
            r = load_json(p)
            rows.append({"method": m, "trainable_params": r["trainable_params"], "train_min": r["train_seconds"] / 60,
                         "peak_gpu_gb": r["peak_gpu_mb"] / 1024, "ckpt_mb": sum(r["checkpoint_mb"].values()),
                         "basis_mb": r.get("basis_storage_mb"), "svd_s": r.get("svd_seconds"),
                         "ref_mb": r.get("ref_bytes", 0) / 2**20})
    return pd.DataFrame(rows).set_index("method")


def plot_diagnostics(cfg):
    repo_path(cfg['paths']['figures']).mkdir(parents=True, exist_ok=True)
    d = load_json(f"{cfg['paths']['results']}/diagnostics.json")
    conds = list(d)
    layers = list(next(iter(d.values())))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))
    x = np.arange(len(conds))
    w = 0.8 / len(layers)
    for j, key in enumerate(["rho_k", "eta"]):
        for i, n in enumerate(layers):
            ax[j].bar(x + i * w, [d[c][n][key] for c in conds], w, label=f"layer {n.split('.')[2]}")
        ax[j].set_xticks(x + 0.4 - w / 2, conds, rotation=40, ha="right", fontsize=7)
        ax[j].set_ylabel(key)
        ax[j].set_yscale("log")
    for g, hatch in [("math_gsm8k_chat", ""), ("nonmath_hellaswag_raw", "//")]:
        ax[2].bar(x + (0 if not hatch else 0.4), [np.mean([d[c][n]["activation"][g]["mean"] for n in layers]) for c in conds],
                  0.4, hatch=hatch, label=g)
    ax[2].set_xticks(x + 0.2, conds, rotation=40, ha="right", fontsize=7)
    ax[2].set_ylabel("mean R(z), averaged over layers")
    ax[2].set_yscale("log")
    ax[0].legend(fontsize=7)
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(repo_path(f"{cfg['paths']['figures']}/diagnostics.pdf"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    out = repo_path(cfg["paths"]["results"])
    bt = benchmark_table(cfg)
    bt.round(2).to_csv(out / "benchmark_table.csv")
    print(bt.round(2).to_string())
    plot_losses(cfg)
    print(l2sp_summary(cfg))
    rt = resource_table(cfg)
    rt.to_csv(out / "resource_table.csv")
    print(rt.round(2).to_string())
    if (out / "diagnostics.json").exists():
        plot_diagnostics(cfg)


if __name__ == "__main__":
    main()
