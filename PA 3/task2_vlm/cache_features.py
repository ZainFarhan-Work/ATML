"""Cache frozen SigLIP patch features (FP16, [196, 768] per image) for every image used in Task 2.

    python -m task2_vlm.cache_features --config configs/task2_vlm.yaml

SigLIP's `last_hidden_state` already has the final layer normalization (post_layernorm) applied once;
nothing further is applied. Features are stored in a single memory-mappable .npy plus an index.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
from PIL import Image
from transformers import SiglipImageProcessor, SiglipVisionModel

from common.utils import load_json, load_yaml, peak_memory_mb, repo_path, reset_peak_memory, save_json
from task2_vlm.vlm_data import all_image_keys, image_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    keys = all_image_keys(cfg)  # sorted list of "split/image_id"
    out_dir = repo_path(cfg["features"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n, T, D = len(keys), cfg["features"]["n_tokens"], cfg["features"]["vision_dim"]

    proc = SiglipImageProcessor.from_pretrained(cfg["vision_id"], revision=cfg["vision_revision"])
    model = SiglipVisionModel.from_pretrained(cfg["vision_id"], revision=cfg["vision_revision"], torch_dtype=torch.float16).cuda().eval()
    feats = np.lib.format.open_memmap(out_dir / "features.npy", mode="w+", dtype=np.float16, shape=(n, T, D))
    reset_peak_memory()
    t0 = time.perf_counter()
    with torch.no_grad():
        for s in range(0, n, args.batch_size):
            batch = keys[s:s + args.batch_size]
            imgs = [Image.open(image_file(cfg, k)).convert("RGB") for k in batch]
            pix = proc(images=imgs, return_tensors="pt")["pixel_values"].cuda().half()
            h = model(pixel_values=pix).last_hidden_state  # [B, 196, 768], post_layernorm applied once
            assert h.shape[1:] == (T, D), h.shape
            feats[s:s + len(batch)] = h.cpu().numpy()
            if (s // args.batch_size) % 20 == 0:
                print(f"{s}/{n}")
    feats.flush()
    save_json(out_dir / "index.json", {k: i for i, k in enumerate(keys)})
    save_json(f"{cfg['paths']['results']}/feature_cache.json", {
        "n_images": n, "seconds": time.perf_counter() - t0, "peak_gpu_mb": peak_memory_mb(),
        "disk_mb": (out_dir / "features.npy").stat().st_size / 2**20, "vision_id": cfg["vision_id"]})


if __name__ == "__main__":
    main()
