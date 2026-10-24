"""Steps 1-2: repeated-sampling opportunity, quality scoring and aggregation on the supplied rollouts.

    python -m task3_ttc.offline --config configs/task3_ttc.yaml --dataset math|mathvista [--workers 8]
Writes results/task3/{dataset}/selection.parquet (every question x order x budget x method), the answer-group
table, and compact summary tables with paired question-level bootstrap intervals (held-out split).
"""
from __future__ import annotations

import argparse
from multiprocessing import Pool as MPPool

import numpy as np
import pandas as pd

from common.stats import bootstrap_matrix_ci, diff_ci_from_boots
from common.utils import load_yaml, repo_path, save_json
from task3_ttc.data import all_orders, load_pool
from task3_ttc.scoring import evaluate_question

_STATE = {}


def _work(args):
    qid, oi = args
    q, order = _STATE["pool"].questions[qid], _STATE["orders"][qid][oi]
    rows = evaluate_question(q, order, _STATE["budgets"], _STATE["vlm"])
    for r in rows:
        r["order"] = oi
    return rows


def run_selection(cfg: dict, pool, is_vlm: bool, workers: int) -> pd.DataFrame:
    _STATE.update(pool=pool, orders=all_orders(cfg, pool), budgets=cfg["budgets"], vlm=is_vlm)
    jobs = [(qid, oi) for qid in sorted(pool.questions) for oi in range(cfg["n_orders"])]
    if workers > 1:
        with MPPool(workers) as p:
            chunks = p.map(_work, jobs, chunksize=50)
    else:
        chunks = [_work(j) for j in jobs]
    df = pd.DataFrame([r for c in chunks for r in c])
    split_of = {q: s for s in ("dev", "eval") for q in pool.split[s]}
    df["split"] = df.qid.map(split_of)
    return df


def per_question(df: pd.DataFrame, split: str, value: str) -> pd.DataFrame:
    """Mean over the ten orders -> [question x (method, N)] table of `value`."""
    d = df[df.split == split]
    return d.groupby(["method", "N", "qid"])[value].mean().unstack("qid")


def summarize(df: pd.DataFrame, split: str, n_boot: int, seed: int) -> pd.DataFrame:
    acc, tok, orc = (per_question(df, split, v) for v in ("correct", "tokens", "oracle"))
    qids = acc.columns
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qids), size=(n_boot, len(qids)))  # paired: same resampled questions for every arm
    rows = []
    sc = {n: acc.loc[("self_consistency", n)].to_numpy() for n in acc.index.get_level_values("N").unique()}
    for (m, n), a in acc.iterrows():
        a = a.to_numpy()
        boots = a[idx].mean(1)
        base = sc[n]
        dboots = (a - base)[idx].mean(1)
        rows.append({"method": m, "N": n, "acc": 100 * a.mean(), "acc_lo": 100 * np.quantile(boots, .025),
                     "acc_hi": 100 * np.quantile(boots, .975), "tokens": tok.loc[(m, n)].mean(),
                     "oracle": 100 * orc.loc[(m, n)].mean(),
                     "diff_vs_sc": 100 * (a - base).mean(), "diff_lo": 100 * np.quantile(dboots, .025),
                     "diff_hi": 100 * np.quantile(dboots, .975)})
    return pd.DataFrame(rows)


def decomposition(df: pd.DataFrame, split: str, method: str = "self_consistency") -> pd.DataFrame:
    """Sampling failures (no correct candidate) vs selection failures (correct available, wrong selected)."""
    d = df[(df.split == split) & (df.method == method)].copy()
    d["sampling_fail"] = ~d.oracle
    d["selection_fail"] = d.oracle & ~d.correct
    return d.groupby("N")[["correct", "oracle", "sampling_fail", "selection_fail"]].mean().mul(100)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--dataset", choices=["math", "mathvista"], required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    is_vlm = args.dataset == "mathvista"
    out = repo_path(f"{cfg['paths']['results']}/{args.dataset}")
    out.mkdir(parents=True, exist_ok=True)

    pool = load_pool(cfg, args.dataset, use_cache=not args.no_cache)
    groups = {q.qid: {"gold": str(q.gold), "reps": q.reps,
                      "rollouts": [{"rid": r.rid, "group": r.group, "correct": r.correct, "tokens": r.n_tokens}
                                   for r in q.rollouts]} for q in pool.questions.values()}
    save_json(out / "answer_groups.json", groups)
    df = run_selection(cfg, pool, is_vlm, args.workers)
    df.to_parquet(out / "selection.parquet")

    for split in ("dev", "eval"):
        s = summarize(df, split, cfg["bootstrap"], cfg["order_seed"])
        s.round(3).to_csv(out / f"method_comparison_{split}.csv", index=False)
        decomposition(df, split).round(3).to_csv(out / f"decomposition_{split}.csv")
    print(pd.read_csv(out / "method_comparison_eval.csv").query("N == 32").to_string())


if __name__ == "__main__":
    main()
