"""Image conditions, teacher-forced NLL, development evaluation, and final generation metrics."""
from __future__ import annotations

import time

import numpy as np
import torch

from task2_vlm.vlm_data import FeatureStore, caption_example, pair_examples
from task2_vlm.vqa_norm import normalize_answer, soft_accuracy

CONDITIONS = ["correct", "swap", "mismatch", "no_image"]


def suite_examples(pairs: list[dict]) -> list[dict]:
    return [e for p in pairs for e in pair_examples(p)]


def make_mismatch_map(pairs: list[dict], seed: int) -> dict[str, str]:
    """Fixed mapping example-id -> unrelated image key from another pair in the same suite.

    Excludes both images of the original pair. Built before any generation and reused for all comparisons.
    """
    rng = np.random.default_rng(seed)
    all_pairs = [(p["pair_id"], [f"{p['split']}/{it['image_id']}" for it in p["items"]]) for p in pairs]
    mapping = {}
    for p in pairs:
        own = {f"{p['split']}/{it['image_id']}" for it in p["items"]}
        for slot in range(2):
            while True:
                pid, keys = all_pairs[int(rng.integers(len(all_pairs)))]
                if pid == p["pair_id"]:
                    continue
                k = keys[int(rng.integers(2))]
                if k not in own:
                    mapping[f"{p['pair_id']}|{slot}"] = k
                    break
    return mapping


def condition_keys(exs: list[dict], cond: str, mismatch: dict[str, str] | None) -> list[str | None]:
    if cond == "correct":
        return [e["key"] for e in exs]
    if cond == "swap":
        return [e["partner_key"] for e in exs]
    if cond == "mismatch":
        return [mismatch[f"{e['pair_id']}|{e['slot']}"] for e in exs]
    if cond == "no_image":
        return [None] * len(exs)
    raise ValueError(cond)


def _batches(n: int, bs: int):
    for s in range(0, n, bs):
        yield slice(s, min(n, s + bs))


@torch.no_grad()
def teacher_forced_nll(model, store: FeatureStore, exs: list[dict], keys: list[str | None], bs: int,
                       labels_key: str = "labels_noeos") -> list[float]:
    out = []
    for sl in _batches(len(exs), bs):
        k = keys[sl]
        feats = None if k[0] is None else store.get(k)
        batch = model.build_batch(exs[sl], feats, side="right")
        with torch.autocast("cuda", dtype=torch.float16):
            out += model.per_example_nll(batch, labels_key).tolist()
    return out


@torch.no_grad()
def dev_eval(model, store: FeatureStore, splits: dict, bs: int = 8) -> dict:
    """Caption reference loss + answer NLL (correct / complementary / no-image) on both dev suites."""
    was_training = model.training
    model.eval()
    res = {}
    caps = [caption_example(r) for r in splits["caption_dev"]]
    nll = teacher_forced_nll(model, store, caps, [e["key"] for e in caps], bs, labels_key="labels")
    res["caption_dev_loss"] = float(np.mean(nll))
    for suite, name in [("grounding_dev", "grounding"), ("vqa_dev", "vqa")]:
        exs = suite_examples(splits[suite])
        per = {c: np.array(teacher_forced_nll(model, store, exs, condition_keys(exs, c, None), bs))
               for c in ["correct", "swap", "no_image"]}
        for c, v in per.items():
            res[f"{name}_nll_{c}"] = float(v.mean())
        res[f"{name}_delta_nll_swap"] = float((per["swap"] - per["correct"]).mean())
        res[f"{name}_delta_nll_noimg"] = float((per["no_image"] - per["correct"]).mean())
    if was_training:
        model.train()
    return res


# ------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def final_eval_suite(model, store: FeatureStore, pairs: list[dict], suite: str, mismatch: dict, bs: int, max_new: int):
    """Greedy generation + teacher-forced NLL for all four conditions. Returns per-example records."""
    model.eval()
    exs = suite_examples(pairs)
    recs = []
    for cond in CONDITIONS:
        keys = condition_keys(exs, cond, mismatch)
        nll = teacher_forced_nll(model, store, exs, keys, bs)
        preds, t_gen = [], 0.0
        for sl in _batches(len(exs), bs):
            k = keys[sl]
            feats = None if k[0] is None else store.get(k)
            batch = model.build_batch(exs[sl], feats, side="left", with_target=False)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.float16):
                preds += model.generate(batch, max_new)
            torch.cuda.synchronize()
            t_gen += time.perf_counter() - t0
        for e, k, p, n in zip(exs, keys, preds, nll):
            canon = normalize_answer(e["target"])
            partner = pairs_partner_answer(pairs, e)
            norm_p = normalize_answer(p)
            rec = {"suite": suite, "condition": cond, "pair_id": e["pair_id"], "slot": e["slot"], "kind": e["kind"],
                   "qtype": e["qtype"], "question": e["question"], "image": k, "reference": e["target"],
                   "canonical": canon, "partner_canonical": normalize_answer(partner), "prediction": p,
                   "prediction_norm": norm_p, "correct": norm_p == canon, "follows_partner": norm_p == normalize_answer(partner),
                   "empty": norm_p == "", "nll": n, "gen_seconds": t_gen / len(exs)}
            rec["soft_acc"] = soft_accuracy(p, e["refs"]) if e.get("refs") else float(norm_p == canon)
            recs.append(rec)
    return recs


def pairs_partner_answer(pairs: list[dict], e: dict) -> str:
    for p in pairs:
        if p["pair_id"] == e["pair_id"]:
            return p["items"][1 - e["slot"]]["answer"]
    raise KeyError(e["pair_id"])
