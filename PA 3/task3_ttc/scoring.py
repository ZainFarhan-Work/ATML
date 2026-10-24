"""Quality scores and selection rules (Task 3, Steps 2-3).

All functions take only information about the *revealed* candidates (in the current order);
gold answers and unseen rollouts are never touched here.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from task3_ttc.answers import NO_ANSWER
from task3_ttc.data import Question


def visconf_components(kl_list: list[np.ndarray], vis_list: list[np.ndarray]) -> dict[str, np.ndarray]:
    """VisConf for a candidate set G: calibrate token KL with G's pooled 5th/95th percentiles (linear interpolation).

    c_t = clip((k_t - Q05(G)) / max(Q95(G) - Q05(G), 1e-8), 0, 1); C_i = mean_t c_t; Z_i = V_i C_i.
    Also returns the token-wise fusion Z_tok_i = mean_t c_t v_t (Step 3 ablation).
    """
    pooled = np.concatenate(kl_list)
    q05, q95 = np.quantile(pooled, [0.05, 0.95])
    denom = max(float(q95 - q05), 1e-8)
    C, V, Z, Zt = [], [], [], []
    for kl, v in zip(kl_list, vis_list):
        c = np.clip((kl - q05) / denom, 0.0, 1.0)
        C.append(c.mean())
        V.append(v.mean())
        Z.append(v.mean() * c.mean())
        Zt.append((c * v).mean())
    return {"conf": np.array(C), "vision": np.array(V), "visconf": np.array(Z), "visconf_tok": np.array(Zt)}


def candidate_scores(rolls, is_vlm: bool) -> dict[str, np.ndarray]:
    s = {"mean_logp": np.array([r.mean_logp for r in rolls]), "self_cert": np.array([r.self_cert for r in rolls])}
    if is_vlm:
        s.update(visconf_components([r.kl for r in rolls], [r.vis for r in rolls]))
    s["neg_length"] = -np.array([r.n_tokens for r in rolls], dtype=float)
    return s


def rwc_weights(scores: np.ndarray) -> np.ndarray:
    """w_i = (N - r_i)/(N - 1) with descending average rank r_i (rank 1 = best). N=1 -> weight 1."""
    n = len(scores)
    if n == 1:
        return np.ones(1)
    r = rankdata(-scores, method="average")
    return (n - r) / (n - 1)


def _earliest(groups: list[str]) -> dict[str, int]:
    first: dict[str, int] = {}
    for pos, g in enumerate(groups):
        first.setdefault(g, pos)
    return first


def sel_first(groups, scores=None) -> int:
    return 0


def sel_highest(scores: np.ndarray) -> int:
    return int(np.argmax(scores))  # argmax returns the earliest maximum -> earliest-in-order tie break


def vote_totals(groups: list[str], weights: np.ndarray | None = None) -> dict[str, float]:
    tot: dict[str, float] = {}
    for i, g in enumerate(groups):
        tot[g] = tot.get(g, 0.0) + (1.0 if weights is None else float(weights[i]))
    return tot


def sel_vote(groups: list[str], weights: np.ndarray | None = None) -> int:
    """Self-consistency (weights=None) or RWC. Ties by earliest supporting rollout in the current order."""
    tot = vote_totals(groups, weights)
    first = _earliest(groups)
    best = max(tot, key=lambda g: (round(tot[g], 10), -first[g]))
    return first[best]


def rwc_margin(groups: list[str], weights: np.ndarray) -> float:
    """(largest - second-largest group vote) / total vote weight; runner-up counts as 0 if only one group."""
    tot = sorted(vote_totals(groups, weights).values(), reverse=True)
    total = float(np.sum(weights))
    second = tot[1] if len(tot) > 1 else 0.0
    return (tot[0] - second) / total if total > 0 else 0.0


def leading_agreement(groups: list[str]) -> tuple[str, float, bool]:
    """(leading group, support fraction, unique-lead flag) for self-consistency among revealed candidates."""
    tot = vote_totals(groups)
    top = max(tot.values())
    leaders = [g for g, v in tot.items() if v == top]
    return (min(leaders, key=lambda g: groups.index(g)), top / len(groups), len(leaders) == 1)


def method_names(is_vlm: bool) -> list[str]:
    base = ["first_ans", "self_consistency", "mean_logp", "self_cert", "self_cert_rwc", "shortest"]
    if is_vlm:
        base += ["vision_only", "vision_only_rwc", "visconf", "visconf_rwc", "visconf_tok", "visconf_tok_rwc"]
    return base


def select_all(groups: list[str], scores: dict[str, np.ndarray], is_vlm: bool) -> dict[str, int]:
    """Position (within the candidate order) selected by every method."""
    sel = {
        "first_ans": 0,
        "self_consistency": sel_vote(groups),
        "mean_logp": sel_highest(scores["mean_logp"]),
        "self_cert": sel_highest(scores["self_cert"]),
        "self_cert_rwc": sel_vote(groups, rwc_weights(scores["self_cert"])),
        "shortest": sel_highest(scores["neg_length"]),
    }
    if is_vlm:
        sel.update({
            "vision_only": sel_highest(scores["vision"]),
            "vision_only_rwc": sel_vote(groups, rwc_weights(scores["vision"])),
            "visconf": sel_highest(scores["visconf"]),
            "visconf_rwc": sel_vote(groups, rwc_weights(scores["visconf"])),
            "visconf_tok": sel_highest(scores["visconf_tok"]),
            "visconf_tok_rwc": sel_vote(groups, rwc_weights(scores["visconf_tok"])),
        })
    return sel


def evaluate_question(q: Question, order: np.ndarray, budgets: list[int], is_vlm: bool) -> list[dict]:
    """All methods at all budgets for one question and one candidate order (nested prefixes)."""
    rows = []
    for n in budgets:
        ids = order[:n]
        rolls = [q.rollouts[i] for i in ids]
        groups = [r.group for r in rolls]
        scores = candidate_scores(rolls, is_vlm)
        sel = select_all(groups, scores, is_vlm)
        oracle = any(r.correct for r in rolls)
        total_tokens = sum(r.n_tokens for r in rolls)
        for m, pos in sel.items():
            r = rolls[pos]
            rows.append({"qid": q.qid, "N": n, "method": m, "selected_rollout": r.rid, "selected_group": r.group,
                         "correct": bool(r.correct), "tokens": rolls[0].n_tokens if m == "first_ans" else total_tokens,
                         "oracle": oracle, "n_groups": len(set(groups)), "n_correct": sum(x.correct for x in rolls),
                         "no_answer": r.group == NO_ANSWER})
    return rows
