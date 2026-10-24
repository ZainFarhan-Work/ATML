"""Task 1 trainer: Full FT, Full FT + L2-SP, LoRA (r=4/16), OPLoRA.

    python -m task1_llm.train --config configs/task1_llm.yaml --method full_ft [--resume]
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from transformers import AutoModelForCausalLM

from common.utils import (append_jsonl, read_jsonl, cosine_lr, count_params, dir_size_mb, load_yaml, peak_memory_mb,
                          repo_path, reset_peak_memory, save_json, set_seed, write_jsonl)
from task1_llm.data import SequentialBatcher, load_tokenizer, select_training_examples
from task1_llm.lora import adapter_params, export_adapters, inject_adapters


def build_model(cfg: dict, mcfg: dict, device: str):
    """Returns (model, trainable_params, adapters, setup_info)."""
    kind = mcfg["kind"]
    info: dict = {}
    if kind == "full":
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                                                     torch_dtype=torch.float32).to(device)  # FP32 master weights
        params = [p for p in model.parameters()]  # unique (tied weights counted once)
        adapters = {}
    else:
        model = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                                                     torch_dtype=torch.float16).to(device)  # frozen FP16 backbone
        bases: dict = {}
        adapters, ainfo = inject_adapters(
            model, cfg["lora_targets"], mcfg["r"], mcfg["alpha"], mcfg["dropout"], seed=cfg["seed"],
            protect_k=mcfg.get("protect_k") if kind == "oplora" else None, bases=bases if kind == "oplora" else None)
        info.update(ainfo)
        info["bases"] = bases
        params = adapter_params(adapters)
    model.config.use_cache = False
    if cfg["train"]["grad_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if kind != "full":
            model.enable_input_require_grads()
    model.train()
    return model, params, adapters, info


def make_optimizer(params, tcfg: dict, lr: float):
    decay = [p for p in params if p.ndim >= 2]
    no_decay = [p for p in params if p.ndim < 2]
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(tcfg["betas"]), eps=tcfg["eps"])


def sft_loss_sum(model, ids, att, lab, chunk: int = 512):
    """Sum of token CE over assistant-response + end-of-turn tokens.

    Logits are formed only at supervised positions and in checkpointed chunks, so the [tokens x 152k-vocab]
    logit tensor (and its gradient) never exists in full; this bounds peak memory on long examples.
    """
    hidden = model.model(input_ids=ids, attention_mask=att).last_hidden_state
    mask = lab[:, 1:] != -100
    h = hidden[:, :-1][mask]
    tgt = lab[:, 1:][mask]

    def ce(hc, tc):
        return torch.nn.functional.cross_entropy(model.lm_head(hc).float(), tc, reduction="sum")

    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, h.shape[0], chunk):
        total = total + torch.utils.checkpoint.checkpoint(ce, h[s:s + chunk], tgt[s:s + chunk], use_reentrant=False)
    return total


@torch.no_grad()
def displacement(params, refs) -> float:
    total = 0.0
    for p, r in zip(params, refs):
        total += (p.detach().float() - r.to(p.device).float()).pow(2).sum().item()
    return total ** 0.5


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    if s["cuda"] is not None:
        torch.cuda.set_rng_state_all(s["cuda"])


def latest_recovery(run_dir: Path):
    cks = sorted(run_dir.glob("recovery_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    return cks[-1] if cks else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-updates", type=int, default=None, help="smoke tests only")
    ap.add_argument("--keep-recovery", choices=["latest", "all"], default="latest")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg, mcfg = cfg["train"], cfg["methods"][args.method]
    set_seed(cfg["seed"])
    device = "cuda"
    run_dir = repo_path(f"{cfg['paths']['checkpoints']}/{args.method}")
    res_dir = repo_path(f"{cfg['paths']['results']}/{args.method}")
    run_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path = res_dir / "train_log.jsonl"
    save_json(res_dir / "config.json", {"method": args.method, "mcfg": mcfg, "train": tcfg, "seed": cfg["seed"],
                                        "model_id": cfg["model_id"], "model_revision": cfg["model_revision"]})

    tok = load_tokenizer(cfg["model_id"], cfg["model_revision"])
    data = select_training_examples(cfg, tok)
    batcher = SequentialBatcher(data, tcfg["micro_batch"], tcfg["grad_accum"], tok.pad_token_id)
    total_updates = tcfg["updates"]
    stop_at = min(total_updates, args.max_updates or total_updates)

    reset_peak_memory()
    model, params, adapters, info = build_model(cfg, mcfg, device)
    bases = info.pop("bases", None)
    opt = make_optimizer(params, tcfg, mcfg["lr"])
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    lam = float(mcfg.get("l2sp", 0.0))

    # Reference parameters: used for L2-SP (device configurable) and for displacement logging (Full FT).
    refs, ref_info = None, {}
    if mcfg["kind"] == "full":
        ref_dev = mcfg.get("ref_device", "cuda") if lam > 0 else "cpu"
        t0 = time.perf_counter()
        refs = [p.detach().clone().to(ref_dev) for p in params]
        ref_info = {"ref_device": ref_dev, "ref_bytes": sum(r.numel() * r.element_size() for r in refs),
                    "ref_setup_seconds": time.perf_counter() - t0}

    start_update, train_seconds = 0, 0.0
    if args.resume and (ck := latest_recovery(run_dir)) is not None:
        state = torch.load(ck, map_location="cpu", weights_only=False)
        if mcfg["kind"] == "full":
            model.load_state_dict(state["model"])
        else:
            for p, v in zip(params, state["params"]):
                p.data.copy_(v)
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        set_rng_state(state["rng"])
        start_update, train_seconds = state["update"], state["train_seconds"]
        write_jsonl(log_path, [r for r in read_jsonl(log_path) if r["update"] <= start_update])  # drop rows after the checkpoint
        print(f"resumed from {ck} at update {start_update}")

    n_trainable = count_params(model, trainable_only=True)
    print(f"trainable params: {n_trainable:,}")
    ckpt_updates, eval_updates = set(tcfg["checkpoint_updates"]), set(tcfg["eval_updates"])
    overhead = {"trainable_params": n_trainable, **ref_info, **{k: v for k, v in info.items() if k != "bases"}}
    if bases:
        bpath = run_dir / "bases.pt"
        torch.save(bases, bpath)
        overhead["basis_storage_mb"] = dir_size_mb(bpath)

    t_run = time.perf_counter()
    for u in range(start_update, stop_at):
        t_up = time.perf_counter()
        lr = cosine_lr(u, total_updates, tcfg["warmup_updates"], mcfg["lr"])
        for g in opt.param_groups:
            g["lr"] = lr
        batches = batcher.update_batches(u)
        n_tok = sum(int((lab[:, 1:] != -100).sum()) for _, _, lab in batches)  # valid tokens in effective batch
        opt.zero_grad(set_to_none=True)
        sft = 0.0
        for ids, att, lab in batches:
            ids, att, lab = ids.to(device), att.to(device), lab.to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss_sum = sft_loss_sum(model, ids, att, lab)
            loss = loss_sum / n_tok
            scaler.scale(loss).backward()
            sft += loss.item()
        scaler.unscale_(opt)

        omega = 0.0
        if refs is not None and lam > 0:
            t_x = time.perf_counter()
            omega_t = torch.zeros((), device=device)
            for p, r in zip(params, refs):  # per-parameter: no extra full-model copy
                d = p.detach() - r.to(p.device, non_blocking=True)
                omega_t += 0.5 * d.float().pow(2).sum()
                p.grad.add_(d, alpha=lam)  # grad of lambda * Omega, added after unscaling
            omega = omega_t.item()
            ref_info["ref_transfer_seconds"] = ref_info.get("ref_transfer_seconds", 0.0) + time.perf_counter() - t_x

        gnorm = torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"]).item()
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        skipped = scaler.get_scale() < scale_before
        train_seconds += time.perf_counter() - t_up

        append_jsonl(log_path, {"update": u + 1, "lr": lr, "sft_loss": sft, "omega": omega, "lam_omega": lam * omega,
                                "total_loss": sft + lam * omega, "grad_norm": gnorm, "skipped": bool(skipped),
                                "loss_scale": scaler.get_scale(), "tokens": n_tok})
        if (u + 1) % 25 == 0 or u == 0:
            print(f"[{args.method}] {u + 1}/{total_updates} sft={sft:.4f} lr={lr:.2e} gn={gnorm:.2f} "
                  f"omega={omega:.2f} skipped={skipped}")

        step = u + 1
        if step in ckpt_updates or step in eval_updates:
            save_checkpoint(args, cfg, mcfg, model, params, adapters, opt, scaler, refs, run_dir, res_dir, step,
                            train_seconds, ckpt_updates, eval_updates)

    resources = {"method": args.method, "train_seconds": train_seconds, "wall_seconds": time.perf_counter() - t_run,
                 "peak_gpu_mb": peak_memory_mb(), **overhead,
                 "checkpoint_mb": {p.name: dir_size_mb(p) for p in run_dir.iterdir()},
                 "skipped_updates": sum(1 for r in read_jsonl(log_path) if r["skipped"])}
    save_json(res_dir / "resources.json", resources)


def save_checkpoint(args, cfg, mcfg, model, params, adapters, opt, scaler, refs, run_dir, res_dir, step,
                    train_seconds, ckpt_updates, eval_updates):
    kind = mcfg["kind"]
    extra = {}
    if step in eval_updates:
        t0 = time.perf_counter()
        if kind == "full":
            snap = run_dir / f"eval_{step}"
            snap.mkdir(exist_ok=True)
            model.save_pretrained(snap, safe_serialization=True)  # FP32 snapshot (needed for dW diagnostics)
            extra["displacement_l2"] = displacement(params, refs)
        else:
            meta = {"r": mcfg["r"], "alpha": mcfg["alpha"], "kind": kind, "protect_k": mcfg.get("protect_k"),
                    "update": step}
            extra.update(export_adapters(adapters, str(run_dir / f"adapter_{step}.pt"), meta))
        extra["save_seconds"] = time.perf_counter() - t0
        save_json(res_dir / f"snapshot_{step}.json", extra)
    if step in ckpt_updates:
        state = {"update": step, "opt": opt.state_dict(), "scaler": scaler.state_dict(), "rng": rng_state(),
                 "train_seconds": train_seconds, "data_position": step * cfg["train"]["micro_batch"] * cfg["train"]["grad_accum"]}
        if kind == "full":
            state["model"] = model.state_dict()
        else:
            state["params"] = [p.detach().cpu() for p in params]
        torch.save(state, run_dir / f"recovery_{step}.pt.tmp")
        (run_dir / f"recovery_{step}.pt.tmp").replace(run_dir / f"recovery_{step}.pt")  # atomic: backups never see partial files
        if args.keep_recovery == "latest":
            for old in run_dir.glob("recovery_*.pt"):
                if old.name != f"recovery_{step}.pt":
                    old.unlink()


if __name__ == "__main__":
    main()
