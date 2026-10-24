"""Step 5: adaptive sampling (agreement-based and quality-aware stopping), simulated on the supplied rollouts.

    python -m task3_ttc.adaptive --config configs/task3_ttc.yaml --dataset math|mathvista
Thresholds are selected on the development split only, frozen, then evaluated on the held-out split.
Both controllers return the current self-consistency answer; only the stopping rule differs.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.utils import load_yaml, repo_path, save_json
from task3_ttc.answers import NO_ANSWER
from task3_ttc.data import all_orders, load_pool
from task3_ttc.scoring import (candidate_scores, leading_agreement, rwc_margin, rwc_weights, sel_vote)


def checkpoint_features(cfg: dict, pool, is_vlm: bool) -> pd.DataFrame:
    """For every question x order x check-point n: everything a stopping rule may look at (revealed candidates only)."""
    ns = sorted(set(cfg["adaptive"]["check_at"]) | {cfg["adaptive"]["max_n"]})
    signal = cfg["adaptive"]["quality_signal"]["mathvista" if is_vlm else "math"]
    orders = all_orders(cfg, pool)
    split_of = {q: s for s in ("dev", "eval") for q in pool.split[s]}
    rows = []
    for qid, q in pool.questions.items():
        pool_correct = np.mean([r.correct for r in q.rollouts])
        for oi, order in enumerate(orders[qid]):
            for n in ns:
                rolls = [q.rollouts[i] for i in order[:n]]
                groups = [r.group for r in rolls]
                lead, frac, unique = leading_agreement(groups)
                sc_pos = sel_vote(groups)
                w = rwc_weights(candidate_scores(rolls, is_vlm)[signal])
                rwc_pos = sel_vote(groups, w)
                rows.append({"qid": qid, "split": split_of[qid], "order": oi, "n": n, "frac": frac,
                             "unique_lead": unique and lead != NO_ANSWER, "sc_correct": rolls[sc_pos].correct,
                             "agree": groups[sc_pos] == groups[rwc_pos], "margin": rwc_margin(groups, w),
                             "tokens": sum(r.n_tokens for r in rolls), "pool_correct_frac": pool_correct,
                             "any_correct_revealed": any(r.correct for r in rolls)})
    return pd.DataFrame(rows)


def simulate(feat: pd.DataFrame, family: str, tau: float, gamma: float, check_at: list[int], max_n: int) -> pd.DataFrame:
    """Per (qid, order): stopping count, correctness and token cost of the policy."""
    f = feat.set_index(["qid", "order", "n"]).sort_index()
    base = f.xs(max_n, level="n")
    stop = pd.Series(max_n, index=base.index)
    for n in sorted(check_at, reverse=True):  # smallest satisfied check-point wins
        s = f.xs(n, level="n")
        ok = s.unique_lead & (s.frac >= tau - 1e-12)
        if family == "quality":
            ok = ok & s.agree & (s.margin >= gamma - 1e-12)
        stop[ok[ok].index] = n
    sel = f.loc[list(zip(stop.index.get_level_values(0), stop.index.get_level_values(1), stop.to_numpy()))]
    out = pd.DataFrame({"qid": stop.index.get_level_values(0), "order": stop.index.get_level_values(1), "n": stop.to_numpy(),
                        "correct": sel.sc_correct.to_numpy().astype(bool), "tokens": sel.tokens.to_numpy().astype(float),
                        "fixed32_correct": base.sc_correct.to_numpy().astype(bool),
                        "pool_correct_frac": base.pool_correct_frac.to_numpy().astype(float)})
    return out


def policy_summary(sim: pd.DataFrame) -> dict:
    return {"acc": 100 * sim.correct.mean(), "tokens": sim.tokens.mean(), "rollouts": sim.n.mean(),
            "stop_dist": {int(n): float(c) for n, c in sim.n.value_counts(normalize=True).sort_index().items()}}


def boot_by_question(sim: pd.DataFrame, n_boot: int, seed: int):
    g = sim.groupby("qid").agg(correct=("correct", "mean"), tokens=("tokens", "mean"), n=("n", "mean"),
                               fixed=("fixed32_correct", "mean"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), size=(n_boot, len(g)))
    out = {}
    for col in ["correct", "tokens", "n"]:
        b = g[col].to_numpy()[idx].mean(1)
        out[col] = (g[col].mean(), np.quantile(b, .025), np.quantile(b, .975))
    d = (g.correct - g.fixed).to_numpy()[idx].mean(1)
    out["acc_diff_vs_fixed32"] = ((g.correct - g.fixed).mean(), np.quantile(d, .025), np.quantile(d, .975))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--dataset", choices=["math", "mathvista"], required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    acfg, is_vlm = cfg["adaptive"], args.dataset == "mathvista"
    out = repo_path(f"{cfg['paths']['results']}/{args.dataset}/adaptive")
    out.mkdir(parents=True, exist_ok=True)

    pool = load_pool(cfg, args.dataset)
    feat = checkpoint_features(cfg, pool, is_vlm)
    feat.to_parquet(out / "checkpoint_features.parquet")
    dev, ev = feat[feat.split == "dev"], feat[feat.split == "eval"]

    grid = [("agreement", t, 0.0) for t in acfg["taus"]] + \
           [("quality", t, g) for t in acfg["taus"] for g in acfg["gammas"]]
    fixed32 = simulate(dev, "agreement", 2.0, 0.0, acfg["check_at"], acfg["max_n"])  # tau > 1 never stops
    fixed_acc = 100 * fixed32.correct.mean()

    dev_rows = []
    for fam, tau, gam in grid:
        s = policy_summary(simulate(dev, fam, tau, gam, acfg["check_at"], acfg["max_n"]))
        dev_rows.append({"family": fam, "tau": tau, "gamma": gam, **{k: v for k, v in s.items() if k != "stop_dist"}})
    dev_df = pd.DataFrame(dev_rows)
    dev_df.to_csv(out / "dev_grid.csv", index=False)

    chosen = {}
    for fam in ("agreement", "quality"):
        d = dev_df[(dev_df.family == fam) & (dev_df.acc >= fixed_acc - acfg["max_acc_drop_pp"])]
        if d.empty:  # fall back to the always-32 policy
            chosen[fam] = {"family": fam, "tau": 2.0, "gamma": 0.0, "fallback": True}
            continue
        best = d.sort_values(["tokens", "acc", "tau", "gamma"], ascending=[True, False, True, True]).iloc[0]
        chosen[fam] = {"family": fam, "tau": float(best.tau), "gamma": float(best.gamma), "fallback": False,
                       "dev_acc": float(best.acc), "dev_tokens": float(best.tokens), "fixed32_dev_acc": fixed_acc}
    save_json(out / "selected_thresholds.json", chosen)  # frozen before held-out evaluation

    results, cases = {}, []
    sim32 = simulate(ev, "agreement", 2.0, 0.0, acfg["check_at"], acfg["max_n"])
    results["fixed32_sc"] = {**policy_summary(sim32), "boot": boot_by_question(sim32, cfg["bootstrap"], cfg["order_seed"])}
    for fam, c in chosen.items():
        sim = simulate(ev, fam, c["tau"], c["gamma"], acfg["check_at"], acfg["max_n"])
        results[fam] = {**policy_summary(sim), "boot": boot_by_question(sim, cfg["bootstrap"], cfg["order_seed"]), "setting": c}
        bad = sim[(sim.n < acfg["max_n"]) & ~sim.correct]
        recov = bad[bad.fixed32_correct]
        results[fam]["incorrect_early_stops"] = {"n": int(len(bad)), "n_fixed32_would_recover": int(len(recov)),
                                                 "n_no_correct_candidate_anywhere": int((bad.pool_correct_frac == 0).sum())}
        cases += [{"family": fam, **r} for r in recov.sort_values(["qid", "order"]).head(10).to_dict("records")]
    save_json(out / "heldout_results.json", results)
    pd.DataFrame(cases).to_csv(out / "premature_stop_cases.csv", index=False)
    for k, v in results.items():
        print(k, round(v["acc"], 2), round(v["tokens"], 1), round(v["rollouts"], 2), v["stop_dist"])


if __name__ == "__main__":
    main()
