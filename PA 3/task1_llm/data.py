"""MetaMathQA selection, formatting, and the fixed probe set for Task 1."""
from __future__ import annotations

import re
import unicodedata

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from common.utils import load_json, repo_path, save_json

IM_END = "<|im_end|>"


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_tokenizer(model_id: str, revision: str | None = None):
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def chat_prompt(tok, question: str) -> str:
    """User turn + assistant header with thinking disabled."""
    return tok.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def encode_example(tok, question: str, response: str) -> tuple[list[int], list[int]]:
    """Return (prompt_ids, response_ids); response_ids include the end-of-turn token."""
    prompt_ids = tok(chat_prompt(tok, question), add_special_tokens=False)["input_ids"]
    resp_ids = tok(response, add_special_tokens=False)["input_ids"]
    resp_ids = resp_ids + [tok.convert_tokens_to_ids(IM_END)]
    return prompt_ids, resp_ids


def gsm8k_test_questions(cfg) -> set[str]:
    ds = load_dataset(cfg["data"]["gsm8k_dataset"], "main", split="test")
    return {normalize_text(q) for q in ds["question"]}


def select_training_examples(cfg, tok) -> dict:
    """First n_train eligible MetaMathQA examples in source order. Cached on disk.

    Excluded: empty query/response, formatted length > max_len tokens, or query /
    original_question that matches a GSM8K test question after normalization.
    Nothing is shuffled or truncated.
    """
    d = cfg["data"]
    cache = repo_path("cached/task1/train_tokens.pt")
    if cache.exists():
        return torch.load(cache)

    banned = gsm8k_test_questions(cfg)
    ds = load_dataset(d["dataset"], split="train")
    prompts, resps, ids = [], [], []
    stats = {"seen": 0, "empty": 0, "too_long": 0, "gsm8k_overlap": 0}
    for i, ex in enumerate(tqdm(ds, desc="selecting MetaMathQA")):
        if len(ids) >= d["n_train"]:
            break
        stats["seen"] += 1
        q, r = (ex.get("query") or "").strip(), (ex.get("response") or "").strip()
        if not q or not r:
            stats["empty"] += 1
            continue
        if normalize_text(q) in banned or normalize_text(ex.get("original_question", "")) in banned:
            stats["gsm8k_overlap"] += 1
            continue
        p_ids, r_ids = encode_example(tok, q, r)
        if len(p_ids) + len(r_ids) > d["max_len"]:
            stats["too_long"] += 1
            continue
        prompts.append(p_ids)
        resps.append(r_ids)
        ids.append(i)
    if len(ids) < d["n_train"]:
        raise RuntimeError(f"only {len(ids)} eligible examples found")
    out = {"source_index": ids, "prompt_ids": prompts, "response_ids": resps, "stats": stats}
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, cache)
    save_json(d["eligible_ids_path"], {"source_index": ids, "stats": stats})
    return out


class SequentialBatcher:
    """Deterministic, in-order stream: update u consumes examples [u*eff, (u+1)*eff)."""

    def __init__(self, data: dict, micro_batch: int, grad_accum: int, pad_id: int):
        self.data, self.mb, self.ga, self.pad_id = data, micro_batch, grad_accum, pad_id
        self.eff = micro_batch * grad_accum

    def update_batches(self, update: int):
        """List of `grad_accum` micro-batches (input_ids, attention_mask, labels) for 0-based update."""
        base = update * self.eff
        out = []
        for m in range(self.ga):
            idx = range(base + m * self.mb, base + (m + 1) * self.mb)
            out.append(self._collate(idx))
        return out

    def _collate(self, idx):
        seqs, labs = [], []
        for i in idx:
            p, r = self.data["prompt_ids"][i], self.data["response_ids"][i]
            seqs.append(p + r)
            labs.append([-100] * len(p) + r)
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.pad_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        lab = torch.full((len(seqs), T), -100, dtype=torch.long)
        for j, (s, l) in enumerate(zip(seqs, labs)):
            ids[j, : len(s)] = torch.tensor(s)
            att[j, : len(s)] = 1
            lab[j, : len(l)] = torch.tensor(l)
        return ids, att, lab


def build_probe_set(cfg) -> dict:
    """32 GSM8K test questions + 32 HellaSwag validation contexts, sampled once (seed 6304)."""
    path = repo_path(cfg["data"]["probe_path"])
    if path.exists():
        return load_json(path)
    rng = np.random.default_rng(cfg["seed"])
    gsm = load_dataset(cfg["data"]["gsm8k_dataset"], "main", split="test")
    gi = sorted(rng.choice(len(gsm), cfg["data"]["probe_n_gsm8k"], replace=False).tolist())
    hs = load_dataset("Rowan/hellaswag", split="validation")
    hi = sorted(rng.choice(len(hs), cfg["data"]["probe_n_hellaswag"], replace=False).tolist())
    probe = {
        "seed": cfg["seed"],
        "gsm8k": [{"index": i, "question": gsm[i]["question"]} for i in gi],
        "hellaswag": [{"index": i, "ind": int(hs[i]["ind"]), "context": hs[i]["ctx"]} for i in hi],
    }
    save_json(path, probe)
    return probe


def load_qualitative_prompts(path: str) -> list[dict]:
    """Parse first_plot_questions.yaml into [{qid, qindex, fmt, prompt}] in file order.

    The assignment ships this file; its exact schema is not documented in the manual, so the
    parser accepts a list (or a dict wrapping one) of mappings with a prompt text and optional
    format / question-id fields. Adjust here if the real file differs.
    """
    import yaml

    raw = yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        lists = [v for v in raw.values() if isinstance(v, list)]
        if not lists:
            raise ValueError(f"cannot find a prompt list in {path}; keys={list(raw)}")
        raw = max(lists, key=len)
    prompt_keys = ["prompt", "text", "input", "content", "question_text"]
    fmt_keys = ["format", "type", "style", "kind"]
    qid_keys = ["question_id", "qid", "base_question", "question", "group"]
    rows, qorder = [], {}
    for n, item in enumerate(raw):
        if isinstance(item, str):
            item = {"prompt": item}
        prompt = next((item[k] for k in prompt_keys if k in item), None)
        if prompt is None:
            raise ValueError(f"item {n} of {path} has no prompt field: {item}")
        fmt = next((str(item[k]) for k in fmt_keys if k in item), ["plain", "json", "template"][n % 3])
        qid = next((str(item[k]) for k in qid_keys if k in item and k != "question"), None)
        if qid is None:
            qid = str(item.get("question", n // 3)) if "question" in item else str(n // 3)
        qorder.setdefault(qid, len(qorder))
        rows.append({"qid": qid, "qindex": qorder[qid], "fmt": fmt, "prompt": prompt})
    if len(rows) != 24:
        print(f"[warn] expected 24 qualitative prompts, parsed {len(rows)}")
    return rows
