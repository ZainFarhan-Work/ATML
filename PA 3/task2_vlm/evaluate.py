"""Final test evaluation of the Task 2 VLM under four image conditions.

    python -m task2_vlm.evaluate --config configs/task2_vlm.yaml --run-name main [--compress]
Writes per-example records and a summary with 95% pair-bootstrap intervals (4,000 resamples, shared indices).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import torch

from common.stats import bootstrap_matrix_ci, diff_ci_from_boots
from common.utils import (load_yaml, peak_memory_mb, repo_path, reset_peak_memory, save_json, set_seed, write_jsonl)
from task2_vlm.evaluation import CONDITIONS, final_eval_suite, make_mismatch_map
from task2_vlm.model import VLM
from task2_vlm.vlm_data import FeatureStore, load_splits

SUITES = [("grounding_test", "grounding"), ("vqa_test", "vqa")]


def load_final(cfg: dict, run_name: str, compress: bool) -> VLM:
    ck = repo_path(f"{cfg['paths']['checkpoints']}/{run_name}")
    model = VLM(cfg, compress=compress)
    model.connector.load_state_dict(torch.load(ck / "connector_stage3.pt", map_location="cuda"))
    model.enable_lora(cfg["seed"])
    model.load_lora_state(torch.load(ck / "lora_stage3.pt", map_location="cpu"))
    for m in model.adapters.values():
        m.drop = torch.nn.Identity()
    return model.eval()


def per_pair_matrix(df: pd.DataFrame, cond: str, col: str, pair_ids: list[str]) -> np.ndarray:
    """Per-pair mean of `col` (two examples per pair) for one condition, in `pair_ids` order."""
    g = df[df.condition == cond].groupby("pair_id")[col].mean()
    return g.loc[pair_ids].to_numpy(dtype=float)


def pair_accuracy(df: pd.DataFrame, cond: str, pair_ids: list[str]) -> np.ndarray:
    g = df[df.condition == cond].groupby("pair_id")["correct"].all().astype(float)
    return g.loc[pair_ids].to_numpy()


def identical_rate(df: pd.DataFrame, cond: str, pair_ids: list[str]) -> np.ndarray:
    d = df[df.condition == cond].groupby("pair_id")["prediction_norm"].nunique()
    return (d.loc[pair_ids] == 1).to_numpy(dtype=float)


def summarize(df: pd.DataFrame, suite_pairs: list[dict], n_boot: int, seed: int) -> dict:
    pair_ids = [p["pair_id"] for p in suite_pairs]
    metrics = {"accuracy": lambda c: per_pair_matrix(df, c, "soft_acc", pair_ids),
               "exact_acc": lambda c: per_pair_matrix(df, c, "correct", pair_ids),
               "pair_accuracy": lambda c: pair_accuracy(df, c, pair_ids),
               "identical_answer_rate": lambda c: identical_rate(df, c, pair_ids),
               "empty_answer_rate": lambda c: per_pair_matrix(df, c, "empty", pair_ids),
               "follows_partner_rate": lambda c: per_pair_matrix(df, c, "follows_partner", pair_ids),
               "reference_nll": lambda c: per_pair_matrix(df, c, "nll", pair_ids),
               "gen_seconds_per_answer": lambda c: per_pair_matrix(df, c, "gen_seconds", pair_ids)}
    out = {}
    for name, fn in metrics.items():
        arms = {c: fn(c) for c in CONDITIONS}
        ci, boots = bootstrap_matrix_ci(arms, n_boot=n_boot, seed=seed)  # same resampled pairs across conditions
        out[name] = {c: {"mean": ci[c][0], "lo": ci[c][1], "hi": ci[c][2]} for c in CONDITIONS}
        if name in {"accuracy", "pair_accuracy", "reference_nll"}:
            sign = 1 if name != "reference_nll" else -1  # advantage = correct better than control
            out[name]["advantage"] = {}
            for c in ["swap", "mismatch", "no_image"]:
                a, b = (arms["correct"], arms[c]) if sign == 1 else (arms[c], arms["correct"])
                ba, bb = (boots["correct"], boots[c]) if sign == 1 else (boots[c], boots["correct"])
                m, lo, hi = diff_ci_from_boots(ba, bb, a.mean(), b.mean())
                out[name]["advantage"][f"correct_minus_{c}" if sign == 1 else f"{c}_minus_correct"] = {"mean": m, "lo": lo, "hi": hi}
    # question-type breakdown (accuracy; types are fixed per pair)
    by_type = df.groupby(["condition", "kind" if df.suite.iloc[0] == "grounding_test" else "qtype"])["soft_acc"].mean()
    out["by_type"] = {f"{c}|{t}": float(v) for (c, t), v in by_type.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--compress", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    res = repo_path(f"{cfg['paths']['results']}/{args.run_name}")
    splits, store = load_splits(cfg), FeatureStore(cfg)
    model = load_final(cfg, args.run_name, args.compress)
    reset_peak_memory()
    summary = {}
    for suite, short in SUITES:
        pairs = splits[suite]
        mismatch = make_mismatch_map(pairs, cfg["seed"])
        save_json(res / f"mismatch_map_{short}.json", mismatch)  # fixed before any generation
        t0 = time.perf_counter()
        recs = final_eval_suite(model, store, pairs, suite, mismatch, cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"])
        write_jsonl(res / f"final_{short}_examples.jsonl", recs)
        summary[short] = summarize(pd.DataFrame(recs), pairs, cfg["eval"]["bootstrap"], cfg["seed"])
        summary[short]["eval_seconds"] = time.perf_counter() - t0
    summary["peak_gpu_mb"] = peak_memory_mb()
    save_json(res / "final_summary.json", summary)
    for short in ("grounding", "vqa"):
        print(short, {c: round(summary[short]["accuracy"][c]["mean"], 3) for c in CONDITIONS})


if __name__ == "__main__":
    main()
