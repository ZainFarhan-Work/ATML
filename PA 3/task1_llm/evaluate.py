"""Evaluate the seven Task 1 conditions with lm-evaluation-harness (zero-shot, FP16, dropout off).

    python -m task1_llm.evaluate --config configs/task1_llm.yaml [--only lora16]
"""
from __future__ import annotations

import argparse
import time

import torch
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM

from common.utils import load_yaml, peak_memory_mb, repo_path, reset_peak_memory, save_json, write_jsonl
from task1_llm.data import load_tokenizer
from task1_llm.lora import attach_exported, merge_adapters


class NoThinkHFLM(HFLM):
    """HFLM whose chat template disables Qwen3 thinking."""

    def apply_chat_template(self, chat_history, add_generation_prompt: bool = True) -> str:
        return self.tokenizer.apply_chat_template(
            chat_history, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)


def load_condition_model(cfg: dict, cond: dict, device: str = "cuda"):
    """Fresh FP16 model for a condition. Adapters are merged into a fresh backbone copy."""
    run_dir = repo_path(f"{cfg['paths']['checkpoints']}/{cond['method']}") if cond["method"] else None
    kw = dict(revision=cfg["model_revision"], torch_dtype=torch.float16)
    if cond["method"] is None:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw)
    elif cfg["methods"][cond["method"]]["kind"] == "full":
        model = AutoModelForCausalLM.from_pretrained(run_dir / f"eval_{cond['update']}", **kw)
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw)
        payload = torch.load(run_dir / f"adapter_{cond['update']}.pt", map_location="cpu")
        merge_adapters(model, payload)
    return model.to(device).eval()


@torch.no_grad()
def merge_check(cfg: dict, cond: dict, tok, device: str = "cuda") -> dict:
    """Merged vs unmerged logits on a small batch (adapter conditions only)."""
    run_dir = repo_path(f"{cfg['paths']['checkpoints']}/{cond['method']}")
    payload = torch.load(run_dir / f"adapter_{cond['update']}.pt", map_location="cpu")
    texts = ["Natalia sold clips to 48 of her friends in April.", "The capital of France is",
             "Question: Is the sky blue?\nAnswer:", "def fib(n):"][: cfg["eval"]["merge_check_batch"]]
    enc = tok(texts, return_tensors="pt", padding=True).to(device)
    kw = dict(revision=cfg["model_revision"], torch_dtype=torch.float16)
    merged = merge_adapters(AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw), payload).to(device).eval()
    lm = merged(**enc).logits.float()
    del merged
    unmerged = attach_exported(AutoModelForCausalLM.from_pretrained(cfg["model_id"], **kw).to(device), payload).eval()
    lu = unmerged(**enc).logits.float()
    del unmerged
    m = enc["attention_mask"].bool()
    d = (lm - lu).abs()[m]
    return {"max_abs_diff": d.max().item(), "mean_abs_diff": d.mean().item(),
            "top1_agreement": (lm.argmax(-1) == lu.argmax(-1))[m].float().mean().item()}


def run_benchmarks(cfg: dict, model, tok, name: str) -> dict:
    ecfg = cfg["eval"]
    lm = NoThinkHFLM(pretrained=model, tokenizer=tok, batch_size=ecfg["batch_size"], max_length=4096)
    out, t0 = {}, time.perf_counter()
    reset_peak_memory()
    gen = evaluator.simple_evaluate(
        model=lm, tasks=["gsm8k_cot"], num_fewshot=0, apply_chat_template=True, fewshot_as_multiturn=False,
        gen_kwargs=f"do_sample=False,temperature=0.0,max_gen_toks={ecfg['gsm8k_max_gen_toks']}",
        limit=ecfg["limit"], log_samples=True, random_seed=cfg["seed"], numpy_random_seed=cfg["seed"],
        torch_random_seed=cfg["seed"], fewshot_random_seed=cfg["seed"])
    gen_seconds = time.perf_counter() - t0
    ll = evaluator.simple_evaluate(
        model=lm, tasks=[t for t in ecfg["tasks"] if t != "gsm8k_cot"], num_fewshot=0, apply_chat_template=False,
        limit=ecfg["limit"], log_samples=True, random_seed=cfg["seed"], numpy_random_seed=cfg["seed"],
        torch_random_seed=cfg["seed"], fewshot_random_seed=cfg["seed"])
    out["scores"] = {**gen["results"], **ll["results"]}
    out["eval_seconds"] = time.perf_counter() - t0
    out["gsm8k_generation_seconds"] = gen_seconds
    out["peak_gpu_mb"] = peak_memory_mb()

    # per-example outputs, extracted GSM8K answers, truncation flags
    max_toks = ecfg["gsm8k_max_gen_toks"]
    rows = []
    for s in gen["samples"]["gsm8k_cot"]:
        raw = s["resps"][0][0]
        n_tok = len(tok(raw, add_special_tokens=False)["input_ids"])
        rows.append({"doc_id": s["doc_id"], "target": s.get("target"), "response": raw,
                     "filtered": s["filtered_resps"], "exact_match": {k: v for k, v in s.items() if k.startswith("exact_match")},
                     "n_tokens": n_tok, "truncated": n_tok >= max_toks - 1})
    write_jsonl(f"{cfg['paths']['results']}/eval/{name}_gsm8k_samples.jsonl", rows)
    for task, samples in ll["samples"].items():
        write_jsonl(f"{cfg['paths']['results']}/eval/{name}_{task}_samples.jsonl",
                    [{"doc_id": s["doc_id"], "target": s.get("target"), "resps": s["resps"],
                      **{k: s[k] for k in ("acc", "acc_norm") if k in s}} for s in samples])
    out["gsm8k_truncation_rate"] = sum(r["truncated"] for r in rows) / max(1, len(rows))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="condition names to run")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["model_id"], cfg["model_revision"])
    tok.padding_side = "left"
    for cond in cfg["eval"]["conditions"]:
        if args.only and cond["name"] not in args.only:
            continue
        out_path = repo_path(f"{cfg['paths']['results']}/eval/{cond['name']}.json")
        if out_path.exists() and not args.force:
            print(f"skip {cond['name']} (done)")
            continue
        print(f"=== evaluating {cond['name']}")
        result = {"condition": cond}
        if cond["method"] and cfg["methods"][cond["method"]]["kind"] != "full":
            result["merge_check"] = merge_check(cfg, cond, tok)
            torch.cuda.empty_cache()
        model = load_condition_model(cfg, cond)
        result.update(run_benchmarks(cfg, model, tok, cond["name"]))
        save_json(out_path, result)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
