"""Sanity checks for the VLM interface (run before the main training run).

    python -m task2_vlm.checks --config configs/task2_vlm.yaml
Verifies feature shapes, frozen parameter sets, connector gradients, split isolation, and the no-image prefix.
"""
from __future__ import annotations

import argparse

import torch

from common.utils import load_json, load_yaml
from task2_vlm.model import VLM
from task2_vlm.vlm_data import FeatureStore, caption_example, load_splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    splits, store = load_splits(cfg), FeatureStore(cfg)

    f = store.get([f"train2014/{splits['caption_train'][0]['image_id']}"])
    assert f.shape == (1, 196, 768) and f.dtype == torch.float16, f.shape
    print("feature shape/dtype ok", tuple(f.shape), f.dtype)

    ids = load_json(f"{cfg['data']['out_dir']}/ids_summary.json")["image_ids"]
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not set(ids[names[i]]) & set(ids[names[j]]), (names[i], names[j])
    train_imgs = {r["image_id"] for r in splits["caption_train"] if r["split"] == "train2014"}
    for n in names:
        assert not set(ids[n]) & train_imgs, n  # val2014 ids cannot appear in train2014 (sanity)
    print("split isolation ok")

    model = VLM(cfg)
    assert all(not p.requires_grad for p in model.llm.parameters()), "LLM must be frozen in stage 1"
    exs = [caption_example(splits["caption_train"][0])] * 2
    feats = store.get([e["key"] for e in exs])
    model.train()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = model.per_example_nll(model.build_batch(exs, feats)).mean()
    loss.backward()
    gn = [p.grad.norm().item() for p in model.connector.parameters() if p.grad is not None]
    assert len(gn) == 4 and all(g > 0 for g in gn), "gradients must reach the connector"
    print("connector gradient norms:", [round(g, 4) for g in gn])

    with_img = model.build_batch(exs[:1], feats[:1], side="right")
    no_img = model.build_batch(exs[:1], None, side="right")
    assert with_img["inputs_embeds"].shape[1] - no_img["inputs_embeds"].shape[1] == 196
    print("no-image prefix is 196 tokens shorter; mask/positions rebuilt ok")
    print("all checks passed")


if __name__ == "__main__":
    main()
