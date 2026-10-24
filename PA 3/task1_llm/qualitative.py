"""Greedy qualitative generations for all seven conditions (24 prompts x 7 = 168 responses).

    python -m task1_llm.qualitative --config configs/task1_llm.yaml
Writes results/task1/qualitative/responses.jsonl and an annotation sheet for the fixed 42-response subset.
"""
from __future__ import annotations

import argparse
import csv
import json
import re

import torch

from common.utils import load_yaml, repo_path, write_jsonl
from task1_llm.data import chat_prompt, load_qualitative_prompts, load_tokenizer
from task1_llm.evaluate import load_condition_model


def repetition_score(text: str, n: int = 4) -> float:
    """Fraction of repeated word n-grams (heuristic flag only; manual annotation is authoritative)."""
    w = text.split()
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def valid_json(text: str) -> bool:
    m = re.search(r"\{.*\}|\[.*\]", text, re.S)
    try:
        json.loads(m.group(0) if m else text)
        return True
    except Exception:
        return False


@torch.no_grad()
def generate(model, tok, prompt: str, max_new: int, system: str | None = None) -> tuple[str, int, bool]:
    enc = tok(chat_prompt(tok, prompt, system), return_tensors="pt", add_special_tokens=False).to("cuda")
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                         pad_token_id=tok.pad_token_id)
    gen = out[0, enc["input_ids"].shape[1]:]
    ended = bool((gen == tok.convert_tokens_to_ids("<|im_end|>")).any() or (gen == tok.eos_token_id).any())
    return tok.decode(gen, skip_special_tokens=True), len(gen), not ended


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["model_id"], cfg["model_revision"])
    prompts = load_qualitative_prompts(cfg["data"]["qualitative_questions"])
    max_new = cfg["qualitative"]["max_new_tokens"]
    rows = []
    for cond in cfg["eval"]["conditions"]:
        model = load_condition_model(cfg, cond)
        for p in prompts:
            text, n, trunc = generate(model, tok, p["prompt"], max_new, p["system"])
            rows.append({"condition": cond["name"], **p, "response": text, "n_tokens": n, "truncated": trunc,
                         "repetition": repetition_score(text),
                         "json_valid": valid_json(text) if p["fmt"].lower().startswith("json") else None})
        del model
        torch.cuda.empty_cache()
    out_dir = f"{cfg['paths']['results']}/qualitative"
    write_jsonl(f"{out_dir}/responses.jsonl", rows)

    plain_q = sorted({r["qindex"] for r in rows})
    keep = {plain_q[0], plain_q[-1]}  # first and last underlying questions in the file's order
    sheet = [r for r in rows if r["qindex"] in keep]
    with repo_path(f"{out_dir}/annotation_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "qindex", "fmt", "prompt", "response", "n_tokens", "truncated", "repetition",
                    "format_compliance", "coherence_relevance", "repetition_manual", "math_framing", "inappropriate", "notes"])
        for r in sheet:
            w.writerow([r["condition"], r["qindex"], r["fmt"], r["prompt"], r["response"], r["n_tokens"], r["truncated"],
                        f"{r['repetition']:.3f}", "", "", "", "", "", ""])
    print(f"{len(rows)} responses; {len(sheet)} in annotation subset")


if __name__ == "__main__":
    main()
