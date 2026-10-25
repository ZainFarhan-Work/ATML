"""Weight-space and activation diagnostics for down_proj in layers 0/13/27 (FP32).

    python -m task1_llm.diagnostics --config configs/task1_llm.yaml
"""
from __future__ import annotations

import argparse

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM

from common.utils import load_yaml, repo_path, save_json
from task1_llm.data import build_probe_set, chat_prompt, load_tokenizer


def weight_name(layer: int, module: str) -> str:
    sub = "mlp" if module in {"gate_proj", "up_proj", "down_proj"} else "self_attn"
    return f"model.layers.{layer}.{sub}.{module}"


def read_full_weight(snap_dir, key: str) -> torch.Tensor:
    layers = snap_dir.parent / f"layers_{snap_dir.name.split('_')[-1]}.pt"  # extracted bundle, if present
    if not snap_dir.exists() and layers.exists():
        return torch.load(layers, map_location="cpu")[key].float()
    for f in sorted(snap_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as sf:
            if key in sf.keys():
                return sf.get_tensor(key).float()
    raise KeyError(f"{key} not found in {snap_dir}")


def delta_w(cfg: dict, cond_files: dict, cond: str, name: str) -> torch.Tensor:
    method, upd = cond_files[cond]
    run_dir = repo_path(f"{cfg['paths']['checkpoints']}/{method}")
    if cfg["methods"][method]["kind"] == "full":
        return None  # filled by caller (needs W0)
    payload = torch.load(run_dir / f"adapter_{upd}.pt", map_location="cpu")
    f = payload["factors"][name]
    return (payload["meta"]["alpha"] / payload["meta"]["r"]) * (f["B"].float() @ f["A"].float())


@torch.no_grad()
def build_probe_prompts(cfg, tok) -> dict[str, list[str]]:
    probe = build_probe_set(cfg)
    return {
        "math_gsm8k_chat": [chat_prompt(tok, p["question"]) for p in probe["gsm8k"]],
        "nonmath_hellaswag_raw": [p["context"] for p in probe["hellaswag"]],
        "nonmath_hellaswag_chat": [chat_prompt(tok, p["context"]) for p in probe["hellaswag"]],
    }


@torch.no_grad()
def cache_down_inputs(base, tok, prompts: dict[str, list[str]], names: list[str], device="cuda") -> dict:
    """z at the last non-padding prompt token (batch size 1 -> no padding). {group: {name: [N, d_in]}}"""
    mods = dict(base.named_modules())
    store: dict[str, list] = {}
    hooks = [mods[n].register_forward_pre_hook(lambda m, a, n=n: store.setdefault(n, []).append(a[0][0, -1].detach().float().cpu()))
             for n in names]
    out = {}
    for group, texts in prompts.items():
        store.clear()
        for t in texts:
            enc = tok(t, return_tensors="pt", add_special_tokens=False).to(device)
            base(**enc)
        out[group] = {n: torch.stack(store[n]) for n in names}
    for h in hooks:
        h.remove()
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    dcfg = cfg["diagnostics"]
    device = "cuda"
    tok = load_tokenizer(cfg["model_id"], cfg["model_revision"])
    base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], revision=cfg["model_revision"],
                                                torch_dtype=torch.float32).to(device).eval()
    COND_FILES = {c["name"]: (c["method"], c["update"]) for c in cfg["eval"]["conditions"] if c["method"]}
    names = [weight_name(l, dcfg["module"]) for l in dcfg["layers"]]
    W0 = {n: dict(base.named_modules())[n].weight.detach().float() for n in names}
    prompts = build_probe_prompts(cfg, tok)
    zs = cache_down_inputs(base, tok, prompts, names, device)
    k = dcfg["protect_k"]
    bases = {}
    for n in names:
        U, S, Vh = torch.linalg.svd(W0[n], full_matrices=False)
        bases[n] = (U[:, :k], Vh[:k].T)

    results = {}
    for cond, (method, upd) in COND_FILES.items():
        results[cond] = {}
        snap = repo_path(f"{cfg['paths']['checkpoints']}/{method}/eval_{upd}")
        for n in names:
            dW = delta_w(cfg, COND_FILES, cond, n)
            dW = (read_full_weight(snap, n + ".weight") - W0[n].cpu()) if dW is None else dW
            dW = dW.to(device)
            Uk, Vk = bases[n]
            nrm2 = dW.pow(2).sum()
            rec = {
                "rho_k": ((Uk @ (Uk.T @ dW)).pow(2).sum() / nrm2.clamp_min(1e-30)).item(),
                "rho_k_input": ((dW @ Vk).pow(2).sum() / nrm2.clamp_min(1e-30)).item(),
                "eta": (dW.norm() / W0[n].norm()).item(),
                "dW_fro": dW.norm().item(),
                "activation": {},
            }
            if method == "oplora16":  # numerical verification of the orthogonality constraints
                rec["oplora_constraint"] = {"||Uk^T dW||_F": (Uk.T @ dW).norm().item(),
                                            "||dW Vk||_F": (dW @ Vk).norm().item()}
            for group, per_layer in zs.items():
                z = per_layer[n].to(device)                               # [N, d_in]
                R = (z @ dW.T).norm(dim=1) / ((z @ W0[n].T).norm(dim=1) + 1e-8)
                rec["activation"][group] = {"mean": R.mean().item(), "median": R.median().item(), "per_prompt": R.tolist()}
            results[cond][n] = rec
        print(cond, {n.split(".")[2]: round(r["rho_k"], 4) for n, r in results[cond].items()})
    save_json(f"{cfg['paths']['results']}/diagnostics.json", results)


if __name__ == "__main__":
    main()
