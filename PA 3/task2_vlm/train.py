"""Three-stage VLM curriculum: caption alignment -> visual instruction tuning -> natural VQA.

    python -m task2_vlm.train --config configs/task2_vlm.yaml --run-name main [--compress] [--resume]
"""
from __future__ import annotations

import argparse
import math
import random
import time

import numpy as np
import torch

from common.utils import (append_jsonl, cosine_lr, load_yaml, peak_memory_mb, read_jsonl, repo_path,
                          reset_peak_memory, save_json, set_seed)
from task1_llm.lora import adapter_params
from task2_vlm.evaluation import dev_eval
from task2_vlm.model import VLM
from task2_vlm.vlm_data import CurriculumSampler, FeatureStore, load_splits


def rng_state():
    return {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()}


def set_rng_state(s):
    random.setstate(s["py"])
    np.random.set_state(s["np"])
    torch.set_rng_state(s["torch"])
    torch.cuda.set_rng_state_all(s["cuda"])


def make_optimizer(groups: list[dict], wd: float):
    pg = []
    for g in groups:
        dec = [p for p in g["params"] if p.ndim >= 2]
        nod = [p for p in g["params"] if p.ndim < 2]
        pg.append({"params": dec, "weight_decay": wd, "peak": g["peak"], "lr": g["peak"]})
        pg.append({"params": nod, "weight_decay": 0.0, "peak": g["peak"], "lr": g["peak"]})
    return torch.optim.AdamW(pg, betas=(0.9, 0.999))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--run-name", default="main")
    ap.add_argument("--compress", action="store_true", help="49-token variant (optional Step 7)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max-updates", type=int, default=None, help="smoke tests only")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tcfg = cfg["train"]
    set_seed(cfg["seed"])
    ck_dir = repo_path(f"{cfg['paths']['checkpoints']}/{args.run_name}")
    res_dir = repo_path(f"{cfg['paths']['results']}/{args.run_name}")
    ck_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    log_path, dev_path = res_dir / "train_log.jsonl", res_dir / "dev_metrics.jsonl"
    save_json(res_dir / "config.json", {"cfg": cfg, "compress": args.compress})

    splits, store = load_splits(cfg), FeatureStore(cfg)
    model = VLM(cfg, compress=args.compress)
    sampler = CurriculumSampler(splits, cfg["seed"])
    scaler = torch.amp.GradScaler("cuda", init_scale=2.0 ** 16)
    mb, ga = tcfg["micro_batch"], tcfg["grad_accum"]
    eff = mb * ga
    stages = tcfg["stages"]
    total_updates = sum(s["updates"] for s in stages)
    stage_start = np.cumsum([0] + [s["updates"] for s in stages])
    dev_updates = set(tcfg["dev_eval_updates"])

    # ---- calibration (training data only) -------------------------------------------------------------
    calib_path = res_dir / "calibration.json"
    if not (args.resume and calib_path.exists()):
        cal_rng = np.random.default_rng(cfg["seed"])
        cap_idx = cal_rng.choice(len(splits["caption_train"]), 512, replace=False)
        keys = [f"train2014/{splits['caption_train'][int(i)]['image_id']}" for i in cap_idx]
        texts = [splits["caption_train"][int(i)]["caption"] for i in cap_idx] + [p["question"] for p in splits["vqa_train"][:256]]
        calib = model.calibrate_scale(store.get(keys), texts)
        save_json(calib_path, calib)
        print("calibration:", calib)

    # ---- resume -----------------------------------------------------------------------------------------
    done = 0
    resume_state = None
    if args.resume:
        cks = sorted(ck_dir.glob("recovery_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
        if cks:
            resume_state = torch.load(cks[-1], map_location="cpu", weights_only=False)
            done = resume_state["update"]
            model.connector.load_state_dict(resume_state["connector"])
            print(f"resuming from update {done}")

    def run_dev(u: int):
        if any(r["update"] == u for r in read_jsonl(dev_path)):
            return
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.float16):
            m = dev_eval(model, store, splits, bs=cfg["eval"]["batch_size"])
        append_jsonl(dev_path, {"update": u, "stage": int(np.searchsorted(stage_start[1:], u, side="left")) + 1 if u > 0 else 0,
                                "eval_seconds": time.perf_counter() - t0, **m})
        print(f"[dev @{u}]", {k: round(v, 4) for k, v in m.items()})

    if done == 0 and 0 in dev_updates:
        run_dev(0)

    resources = []
    u = done
    for si, st in enumerate(stages):
        if u >= stage_start[si + 1]:
            continue
        reset_peak_memory()
        t_stage = time.perf_counter()
        groups = [{"params": list(model.connector.parameters()), "peak": st["connector_lr"]}]
        if st["lora"]:
            if not model.adapters:
                model.enable_lora(cfg["seed"])
                if resume_state is not None and resume_state.get("lora") is not None:
                    model.load_lora_state(resume_state["lora"])
            groups.append({"params": adapter_params(model.adapters), "peak": st["lora_lr"]})
        opt = make_optimizer(groups, tcfg["weight_decay"])
        if resume_state is not None:
            if resume_state["stage"] == si:  # mid-stage resume: optimizer / scaler continue
                opt.load_state_dict(resume_state["opt"])
                scaler.load_state_dict(resume_state["scaler"])
            sampler.load_state(resume_state["sampler"])
            set_rng_state(resume_state["rng"])
            resume_state = None
        trainable = [p for g in groups for p in g["params"]]
        warm = math.ceil(tcfg["warmup_frac"] * st["updates"])
        model.train()
        print(f"=== stage {si + 1} ({st['name']}): {st['updates']} updates, trainable={model.trainable_summary()}")

        while u < stage_start[si + 1]:
            if args.max_updates and u >= args.max_updates:
                break
            local = u - stage_start[si]
            for g in opt.param_groups:
                g["lr"] = cosine_lr(local, st["updates"], warm, g["peak"], tcfg["final_lr_frac"])
            exs, counts = sampler.update_batch(st.get("mix"), eff // 2)
            keys = [e["key"] for e in exs]
            opt.zero_grad(set_to_none=True)
            total = 0.0
            for s in range(0, eff, mb):
                chunk = exs[s:s + mb]
                feats = store.get(keys[s:s + mb])
                batch = model.build_batch(chunk, feats, side="right")
                with torch.autocast("cuda", dtype=torch.float16):
                    per_ex = model.per_example_nll(batch)
                loss = per_ex.sum() / eff  # example-weighted: mean over tokens within example, then over batch
                scaler.scale(loss).backward()
                total += loss.item()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(trainable, tcfg["grad_clip"]).item()
            conn_gn = math.sqrt(sum(float(p.grad.float().norm()) ** 2 for p in model.connector.parameters() if p.grad is not None))
            before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            u += 1
            append_jsonl(log_path, {"update": u, "stage": si + 1, "lr_connector": opt.param_groups[0]["lr"], "loss": total,
                                    "grad_norm": gn, "connector_grad_norm": conn_gn, "mix_counts": counts,
                                    "skipped": scaler.get_scale() < before, "loss_scale": scaler.get_scale()})
            if u % 25 == 0:
                print(f"[{u}/{total_updates}] stage={si + 1} loss={total:.4f} gn={gn:.2f} connector_gn={conn_gn:.3g}")
            if u in dev_updates:
                run_dev(u)
            if u % tcfg["checkpoint_every"] == 0:
                torch.save({"update": u, "stage": si, "connector": model.connector.state_dict(),
                            "lora": model.lora_state() if model.adapters else None, "opt": opt.state_dict(),
                            "scaler": scaler.state_dict(), "sampler": sampler.state(), "rng": rng_state()},
                           ck_dir / f"recovery_{u}.pt")
                for old in ck_dir.glob("recovery_*.pt"):
                    if old.name != f"recovery_{u}.pt":
                        old.unlink()
        # stage-end weights
        torch.save(model.connector.state_dict(), ck_dir / f"connector_stage{si + 1}.pt")
        if model.adapters:
            torch.save(model.lora_state(), ck_dir / f"lora_stage{si + 1}.pt")
        resources.append({"stage": si + 1, "name": st["name"], "seconds": time.perf_counter() - t_stage,
                          "peak_gpu_mb": peak_memory_mb(), "trainable": model.trainable_summary()})
        save_json(res_dir / f"resources_stage{si + 1}.json", resources[-1])
        if args.max_updates and u >= args.max_updates:
            break


if __name__ == "__main__":
    main()
