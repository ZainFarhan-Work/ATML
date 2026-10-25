"""Shrink a Full-FT evaluation snapshot into a portable bundle (run where the FP32 snapshot lives, e.g. Kaggle).

    python -m task1_llm.extract_snapshot --method full_ft --update 625
Writes into checkpoints/task1/<method>/:
    fp16_<update>/        FP16 model (what evaluation uses; identical to casting the FP32 snapshot)
    layers_<update>.pt    FP32 down_proj weights of the diagnostic layers (what the weight-space diagnostics need)
    bundle_<update>.zip   both of the above in a single file, easy to download
Unzip the bundle into the same folder locally; evaluate/diagnostics pick it up automatically.
"""
from __future__ import annotations

import argparse
import shutil

import torch
from transformers import AutoModelForCausalLM

from common.utils import load_yaml, repo_path
from task1_llm.diagnostics import read_full_weight, weight_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task1_llm.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--update", type=int, required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    run = repo_path(f"{cfg['paths']['checkpoints']}/{args.method}")
    snap = run / f"eval_{args.update}"
    dc = cfg["diagnostics"]
    layers = {weight_name(l, dc["module"]) + ".weight": read_full_weight(snap, weight_name(l, dc["module"]) + ".weight")
              for l in dc["layers"]}
    torch.save(layers, run / f"layers_{args.update}.pt")
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float16)
    model.save_pretrained(run / f"fp16_{args.update}", safe_serialization=True)
    zpath = shutil.make_archive(str(run / f"bundle_{args.update}"), "zip", root_dir=run,
                                base_dir=f"fp16_{args.update}")
    with __import__("zipfile").ZipFile(zpath, "a") as z:
        z.write(run / f"layers_{args.update}.pt", f"layers_{args.update}.pt")
    print("wrote", zpath, f"{(run / f'bundle_{args.update}.zip').stat().st_size / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
