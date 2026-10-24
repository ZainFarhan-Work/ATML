"""Load and join the supplied rollout tables; split questions; build answer groups and correctness labels.

The supplied README.MD defines the exact schema. Column names below are the expected ones; if the export
differs, edit COLUMN_CANDIDATES (first match wins). `scripts/inspect_task3_data.py` prints the real schema.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common.utils import load_json, repo_path, save_json
from task3_ttc.answers import (NO_ANSWER, extract_last_boxed, group_math, group_mathvista, math_grade, mathvista_grade)

KEYS = ["dataset", "sample_id", "rollout_id"]

COLUMN_CANDIDATES = {
    "generations": {
        "text": ["text", "generated_text", "response", "output_text"],
        "answer": ["extracted_answer", "answer_extracted", "pred_answer", "prediction", "final_answer"],
        "gold": ["gold_answer", "gold", "reference_answer", "ground_truth", "target", "answer_gold"],
        "n_tokens": ["num_tokens", "n_tokens", "length", "gen_len", "generated_tokens", "output_tokens", "token_count"],
        "question": ["question", "problem", "question_text", "prompt_text"],
    },
    "probability": {
        "logp": ["logp", "token_logp", "logprobs", "sampled_logp"],
        "kl": ["kl_u_p", "kl", "kl_uniform"],
    },
    "attention": {
        "image": ["image", "attn_image", "image_attention", "attention_image", "A_image"],
        "prompt": ["prompt", "attn_prompt", "prompt_attention", "attention_prompt", "A_prompt", "prompt_text"],
        "generated": ["generated", "attn_generated", "generated_attention", "attention_generated", "A_generated", "generated_text"],
    },
}


def _pick(df: pd.DataFrame, cands: list[str], what: str, required: bool = True):
    for c in cands:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"no column for '{what}' among {cands}; available: {list(df.columns)}")
    return None


def _as_arrays(df: pd.DataFrame, col: str) -> list[np.ndarray]:
    return [np.asarray(v, dtype=np.float32) for v in df[col].tolist()]


@dataclass
class Rollout:
    rid: int
    n_tokens: int
    raw_answer: object
    group: str = NO_ANSWER
    correct: bool = False
    logp: np.ndarray | None = None
    kl: np.ndarray | None = None
    vis: np.ndarray | None = None        # per-token visual engagement v_t (VLM only)
    mean_logp: float = 0.0               # L_i
    self_cert: float = 0.0               # K_i
    vis_mean: float = 0.0                # V_i


@dataclass
class Question:
    qid: str
    gold: object
    rollouts: list[Rollout] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
    text: str | None = None
    reps: dict = field(default_factory=dict)  # group id -> representative answer string


@dataclass
class Pool:
    name: str
    questions: dict[str, Question]
    split: dict[str, list[str]]           # {"dev": [...], "eval": [...], "permutation": [...]}


def make_split(qids: list[str], seed: int, n_dev: int) -> dict[str, list[str]]:
    ordered = sorted(qids)
    perm = [str(x) for x in np.random.default_rng(seed).permutation(ordered)]
    return {"permutation": perm, "dev": perm[:n_dev], "eval": perm[n_dev:]}


def load_mathvista_meta(cfg: dict) -> dict[str, dict]:
    raw = load_json(cfg["data"]["mathvista_metadata"])
    items = raw.values() if isinstance(raw, dict) else raw
    out = {}
    for m in items:
        pid = str(m.get("pid", m.get("sample_id", m.get("id"))))
        out[pid] = m
    return out


def load_pool(cfg: dict, which: str, use_cache: bool = True) -> Pool:
    """which in {'math', 'mathvista'}. Joins generations + probability (+ attention) on (dataset, sample_id, rollout_id)."""
    cache = repo_path(f"cached/task3/pool_{which}.pkl")
    if use_cache and cache.exists():
        return pickle.loads(cache.read_bytes())

    d = cfg["data"]
    folder = d["llm_dir"] if which == "math" else d["vlm_dir"]
    root = repo_path(f"{d['root']}/{folder}")
    gen = pd.read_parquet(root / "generations.parquet")
    prob = pd.read_parquet(root / "probability_metrics.parquet")
    gc, pc = COLUMN_CANDIDATES["generations"], COLUMN_CANDIDATES["probability"]
    c_ans, c_gold, c_n = _pick(gen, gc["answer"], "answer"), _pick(gen, gc["gold"], "gold"), _pick(gen, gc["n_tokens"], "n_tokens")
    c_q = _pick(gen, gc["question"], "question", required=False)
    c_lp, c_kl = _pick(prob, pc["logp"], "logp"), _pick(prob, pc["kl"], "kl_u_p")

    df = gen.merge(prob[KEYS + [c_lp, c_kl]].rename(columns={c_lp: "lp_", c_kl: "kl_"}), on=KEYS, how="inner")
    if which == "mathvista":
        att = pd.read_parquet(root / "attention_metrics.parquet")
        ac = COLUMN_CANDIDATES["attention"]
        ci, cp, cg = _pick(att, ac["image"], "image"), _pick(att, ac["prompt"], "prompt"), _pick(att, ac["generated"], "generated")
        df = df.merge(att[KEYS + [ci, cp, cg]].rename(columns={ci: "ai_", cp: "ap_", cg: "ag_"}), on=KEYS, how="inner")
    if len(df) != len(gen):
        raise RuntimeError(f"join dropped rows: {len(gen)} -> {len(df)}")
    lp, kl = _as_arrays(df, "lp_"), _as_arrays(df, "kl_")
    vis = None
    if which == "mathvista":
        vis = []
        for ai, ap, ag in zip(df["ai_"], df["ap_"], df["ag_"]):
            ai, ap, ag = (np.asarray(x, dtype=np.float32) for x in (ai, ap, ag))
            vis.append(ai / np.maximum(ai + ap + ag, 1e-12))  # v_{i,t} = A_img / (A_img + A_prompt + A_gen)
    meta_all = load_mathvista_meta(cfg) if which == "mathvista" else {}

    questions: dict[str, Question] = {}
    for i, r in enumerate(df.to_dict("records")):
        qid = str(r["sample_id"])
        q = questions.setdefault(qid, Question(qid=qid, gold=r[c_gold], meta=meta_all.get(qid, {}),
                                               text=r[c_q] if c_q else None))
        n_tok = int(r[c_n])
        if len(lp[i]) != n_tok or len(kl[i]) != n_tok:
            raise RuntimeError(f"token-list misalignment for {qid}/{r['rollout_id']}: n={n_tok} logp={len(lp[i])} kl={len(kl[i])}")
        ro = Rollout(rid=int(r["rollout_id"]), n_tokens=n_tok, raw_answer=r[c_ans], logp=lp[i], kl=kl[i],
                     vis=vis[i] if vis is not None else None)
        ro.mean_logp, ro.self_cert = float(lp[i].mean()), float(kl[i].mean())
        ro.vis_mean = float(vis[i].mean()) if vis is not None else 0.0
        q.rollouts.append(ro)

    for q in questions.values():
        q.rollouts.sort(key=lambda x: x.rid)
        assert [r.rid for r in q.rollouts] == list(range(cfg["n_rollouts"])), f"{q.qid}: rollout ids {len(q.rollouts)}"
        answers = [r.raw_answer for r in q.rollouts]
        if which == "math":
            gids, reps = group_math(answers)
            gold_correct = {g: math_grade(rep, str(q.gold)) for g, rep in reps.items()}
        else:
            gids, reps = group_mathvista(answers, q.meta)
            gold_correct = {g: mathvista_grade(rep, q.gold, q.meta) for g, rep in reps.items()}
        q.reps = reps
        for ro, g in zip(q.rollouts, gids):
            ro.group, ro.correct = g, bool(gold_correct.get(g, False))

    pool = Pool(which, questions, make_split(list(questions), cfg["order_seed"], cfg["data"]["n_dev"]))
    save_json(cfg["data"]["split_path"].replace(".json", f"_{which}.json"), pool.split)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(pool))
    return pool


def candidate_orders(cfg: dict, qid: str, qpos: int) -> np.ndarray:
    """Ten seeded permutations of the 32 rollout IDs for one question. Shape [n_orders, 32]."""
    rng = np.random.default_rng([cfg["order_seed"], qpos])
    return np.stack([rng.permutation(cfg["n_rollouts"]) for _ in range(cfg["n_orders"])])


def all_orders(cfg: dict, pool: Pool) -> dict[str, np.ndarray]:
    """Orders keyed by question, seeded by the question's position in the sorted ID list (stable across splits)."""
    return {qid: candidate_orders(cfg, qid, i) for i, qid in enumerate(sorted(pool.questions))}
