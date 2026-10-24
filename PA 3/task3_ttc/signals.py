"""Step 3: what do length, confidence, and visual-engagement signals capture? (N = 32, held-out split)

    python -m task3_ttc.signals --config configs/task3_ttc.yaml --dataset math|mathvista
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common.stats import bootstrap_ci
from common.utils import load_yaml, repo_path, save_json
from task3_ttc.data import load_pool
from task3_ttc.scoring import candidate_scores


def quartile_gap(correct: np.ndarray, score: np.ndarray) -> float:
    """Accuracy in the top score quartile minus the bottom quartile within one question (higher score = better)."""
    k = len(score) // 4
    order = np.argsort(-score, kind="stable")  # ties -> earliest rollout first
    top, bot = order[:k], order[::-1][:k]
    return float(correct[top].mean() - correct[bot].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--dataset", choices=["math", "mathvista"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    is_vlm = args.dataset == "mathvista"
    pool = load_pool(cfg, args.dataset)
    qids = pool.split["eval"]
    out = repo_path(f"{cfg['paths']['results']}/{args.dataset}")

    gaps: dict[str, list[float]] = {}
    len_corr, cov_rows, tok_corr, roll_corr = [], [], [], []
    for qid in qids:
        q = pool.questions[qid]
        rolls = q.rollouts
        correct = np.array([r.correct for r in rolls], dtype=float)
        scores = candidate_scores(rolls, is_vlm)
        names = ["mean_logp", "self_cert", "neg_length"] + (["vision", "conf", "visconf", "visconf_tok"] if is_vlm else [])
        for s in names:
            gaps.setdefault(s, []).append(quartile_gap(correct, scores[s]))
        lens = np.array([r.n_tokens for r in rolls], dtype=float)
        if correct.std() > 0 and lens.std() > 0:
            len_corr.append(spearmanr(lens, correct).statistic)
        if is_vlm:
            cs = [np.clip((r.kl - np.quantile(np.concatenate([x.kl for x in rolls]), .05)) /
                          max(np.diff(np.quantile(np.concatenate([x.kl for x in rolls]), [.05, .95]))[0], 1e-8), 0, 1)
                  for r in rolls]
            per_roll = [pearsonr(c, r.vis).statistic for c, r in zip(cs, rolls) if c.std() > 0 and r.vis.std() > 0]
            tok_corr.append(float(np.mean(per_roll)) if per_roll else np.nan)
            cov = [float(np.mean(c * r.vis) - c.mean() * r.vis.mean()) for c, r in zip(cs, rolls)]
            # identity check: Z_tok - Z == Cov_t(c, v)
            z = scores["visconf_tok"] - scores["visconf"]
            assert np.allclose(z, cov, atol=1e-6), "Z_tok - Z must equal the within-rollout covariance"
            cov_rows.append({"qid": qid, "mean_cov": float(np.mean(cov)), "sd_cov": float(np.std(cov))})
            # rollout-level association between confidence and vision within the question
            if scores["conf"].std() > 0 and scores["vision"].std() > 0:
                roll_corr.append(spearmanr(scores["conf"], scores["vision"]).statistic)
    result = {"n_questions": len(qids),
              "quartile_gap_pp": {s: dict(zip(("mean", "lo", "hi"), (100 * x for x in bootstrap_ci(v)))) for s, v in gaps.items()},
              "length_correctness_spearman": {"mean": float(np.mean(len_corr)), "n_questions": len(len_corr)}}
    if is_vlm:
        result["confidence_vision"] = {"tokenwise_pearson_mean": float(np.nanmean(tok_corr)),
                                       "rollout_level_spearman_mean": float(np.nanmean(roll_corr)),
                                       "mean_token_cov": float(np.mean([r["mean_cov"] for r in cov_rows]))}
    save_json(out / "signals.json", result)

    # fixed illustrative trajectories: first 3 questions (permutation order) with mixed correctness
    traj = []
    for qid in pool.split["eval"]:
        q = pool.questions[qid]
        good = [r for r in q.rollouts if r.correct]
        bad = [r for r in q.rollouts if not r.correct]
        if good and bad:
            for tag, r in (("correct", good[0]), ("incorrect", bad[0])):
                item = {"qid": qid, "kind": tag, "rid": r.rid, "n_tokens": r.n_tokens, "kl": r.kl.tolist(), "logp": r.logp.tolist()}
                if is_vlm:
                    item["vis"] = r.vis.tolist()
                traj.append(item)
        if len(traj) >= 6:
            break
    save_json(out / "trajectories.json", traj)
    print(pd.DataFrame(result["quartile_gap_pp"]).T.round(2))


if __name__ == "__main__":
    main()
