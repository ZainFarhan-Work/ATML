"""Step 4: live comparison of one long response, parallel sampling, and sequential refinement.

    python -m task3_ttc.live --config configs/task3_ttc.yaml [--n 50]
Same 3,072-token total output allowance per question for every strategy. Models stay fixed (FP16, batch size 1).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common.stats import bootstrap_ci, paired_bootstrap_diff
from common.utils import load_json, load_yaml, peak_memory_mb, read_jsonl, repo_path, reset_peak_memory, save_json, append_jsonl
from task3_ttc.answers import NO_ANSWER, clean_latex, extract_last_boxed, group_math, math_grade
from task3_ttc.data import load_pool
from task3_ttc.scoring import sel_vote


def messages_for(lcfg: dict, problem: str) -> list[dict]:
    return [{"role": "system", "content": lcfg["system_message"]},
            {"role": "user", "content": f"{problem}\n\n{lcfg['prompt_suffix']}"}]


@torch.no_grad()
def generate(model, tok, messages: list[dict], max_new: int, seed: int, sampling: dict) -> dict:
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tok(text, return_tensors="pt").to(model.device)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    reset_peak_memory()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=True, pad_token_id=tok.pad_token_id, **sampling)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = out[0, enc["input_ids"].shape[1]:]
    ended = bool(((gen == tok.eos_token_id) | (gen == tok.convert_tokens_to_ids("<|im_end|>"))).any())
    resp = tok.decode(gen, skip_special_tokens=True)
    return {"response": resp, "input_tokens": int(enc["input_ids"].shape[1]), "output_tokens": int(len(gen)),
            "seconds": dt, "peak_gpu_mb": peak_memory_mb(), "truncated": not ended, "seed": seed,
            "answer": extract_last_boxed(resp)}


def grade(answer: str | None, gold) -> bool:
    return answer is not None and math_grade(clean_latex(answer), str(gold))


def sc_answer(answers: list[str | None]) -> str | None:
    gids, reps = group_math(answers)
    pos = sel_vote(gids)
    return None if gids[pos] == NO_ANSWER else answers[pos]


def run_question(model, tok, lcfg: dict, problem: str, seed0: int) -> dict:
    s = lcfg["sampling"]
    msgs = messages_for(lcfg, problem)
    single = generate(model, tok, msgs, lcfg["single_long_tokens"], seed0, s)
    par = [generate(model, tok, msgs, lcfg["parallel"]["tokens"], seed0 + 1 + i, s) for i in range(lcfg["parallel"]["n"])]
    refine_msgs = msgs + [{"role": "assistant", "content": par[0]["response"]},
                          {"role": "user", "content": lcfg["refine_instruction"]}]
    rev = generate(model, tok, refine_msgs, lcfg["refine"]["revision_tokens"], seed0 + 1 + lcfg["parallel"]["n"], s)
    return {"single": single, "parallel": par, "revision": rev}


def strategy_rows(qid: str, gold, calls: dict) -> list[dict]:
    par, single, rev = calls["parallel"], calls["single"], calls["revision"]
    par_ans = sc_answer([c["answer"] for c in par])
    init = par[0]  # shared initial response: charged to both parallel sampling and refinement
    rows = [
        {"strategy": "single_long", "answer": single["answer"], "calls": [single]},
        {"strategy": "parallel_sc", "answer": par_ans, "calls": par},
        {"strategy": "sequential_refine", "answer": rev["answer"], "calls": [init, rev]},
    ]
    out = []
    for r in rows:
        c = r["calls"]
        out.append({"qid": qid, "strategy": r["strategy"], "answer": r["answer"], "correct": grade(r["answer"], gold),
                    "extracted": r["answer"] is not None,
                    "input_tokens": sum(x["input_tokens"] for x in c), "output_tokens": sum(x["output_tokens"] for x in c),
                    "seconds": sum(x["seconds"] for x in c), "peak_gpu_mb": max(x["peak_gpu_mb"] for x in c),
                    "any_truncated": any(x["truncated"] for x in c), "final_truncated": c[-1]["truncated"]})
    out[2]["initial_correct"] = grade(init["answer"], gold)
    out[2]["initial_answer"] = init["answer"]
    return out


def summarize(rows: list[dict], seed: int) -> dict:
    import pandas as pd
    df = pd.DataFrame(rows)
    res = {}
    for st, g in df.groupby("strategy"):
        m, lo, hi = bootstrap_ci(g.correct.astype(float).to_numpy(), n_boot=1000, seed=seed)
        res[st] = {"accuracy": m, "acc_lo": lo, "acc_hi": hi, "input_tokens": g.input_tokens.mean(), "output_tokens": g.output_tokens.mean(),
                   "seconds": g.seconds.mean(), "peak_gpu_mb": g.peak_gpu_mb.max(), "truncation_rate": g.any_truncated.mean(),
                   "final_truncation_rate": g.final_truncated.mean(), "extraction_rate": g.extracted.mean()}
    piv = df.pivot(index="qid", columns="strategy", values="correct").astype(float)
    for a, b in [("parallel_sc", "single_long"), ("sequential_refine", "single_long"), ("sequential_refine", "parallel_sc")]:
        m, lo, hi = paired_bootstrap_diff(piv[a].to_numpy(), piv[b].to_numpy(), seed=seed)
        res[f"diff_{a}_minus_{b}"] = {"mean": m, "lo": lo, "hi": hi}
    ref = df[df.strategy == "sequential_refine"]
    res["transitions"] = {"wrong_to_correct": int((~ref.initial_correct & ref.correct).sum()),
                          "correct_to_wrong": int((ref.initial_correct & ~ref.correct).sum()),
                          "stay_correct": int((ref.initial_correct & ref.correct).sum()),
                          "stay_wrong": int((~ref.initial_correct & ~ref.correct).sum())}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    lcfg = cfg["live"]
    pool = load_pool(cfg, "math")
    qids = pool.split["eval"][: args.n or cfg["data"]["live_n"]]  # first 50 eval questions in the seeded permutation
    out = repo_path(f"{cfg['paths']['results']}/live")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "question_ids.json", qids)

    texts = {q: pool.questions[q].text for q in qids}
    if any(t is None for t in texts.values()):
        extra = load_json("data/task3_math_questions.json")  # {sample_id: problem}; see README
        texts = {q: extra[q] for q in qids}

    tok = AutoTokenizer.from_pretrained(lcfg["model_id"], revision=lcfg["model_revision"])
    model = AutoModelForCausalLM.from_pretrained(lcfg["model_id"], revision=lcfg["model_revision"], torch_dtype=torch.float16).cuda().eval()
    generate(model, tok, messages_for(lcfg, "What is 1+1?"), 16, 0, lcfg["sampling"])  # warm-up, not recorded

    done = {r["qid"] for r in read_jsonl(out / "calls.jsonl")}
    for i, qid in enumerate(qids):
        if qid in done:
            continue
        calls = run_question(model, tok, lcfg, texts[qid], lcfg["seed"] + 10_000 * i)
        append_jsonl(out / "calls.jsonl", {"qid": qid, "problem": texts[qid], "gold": str(pool.questions[qid].gold), **calls})
        print(f"{i + 1}/{len(qids)} {qid}")

    rows = []
    for rec in read_jsonl(out / "calls.jsonl"):
        rows += strategy_rows(rec["qid"], rec["gold"], rec)
    save_json(out / "strategy_rows.json", rows)
    summ = summarize(rows, cfg["order_seed"])
    save_json(out / "summary.json", summ)
    print({k: round(v["accuracy"], 3) for k, v in summ.items() if isinstance(v, dict) and "accuracy" in v})


if __name__ == "__main__":
    main()
