"""Splits, cached-feature access, and the seeded curriculum sampler for Task 2."""
from __future__ import annotations

import numpy as np
import torch

from common.utils import load_json, repo_path

CAPTION_PROMPT = "Describe this image in one sentence."
SHORT_SUFFIX = "\nAnswer the question using a single word or phrase."

SPLIT_FILES = ["caption_train", "caption_dev", "grounding_pool_train", "grounding_dev", "grounding_test",
               "vqa_train", "vqa_dev", "vqa_test"]


def load_splits(cfg: dict) -> dict:
    d = cfg["data"]["out_dir"]
    return {n: load_json(f"{d}/{n}.json") for n in SPLIT_FILES}


def image_key(split: str, image_id: int) -> str:
    return f"{split}/{image_id}"


def image_file(cfg: dict, key: str):
    split, iid = key.split("/")
    return repo_path(f"{cfg['data']['coco_root']}/{split}/COCO_{split}_{int(iid):012d}.jpg")


def all_image_keys(cfg: dict) -> list[str]:
    sp = load_splits(cfg)
    keys = set()
    for name, recs in sp.items():
        for r in recs:
            if "items" in r:
                keys |= {image_key(r["split"], it["image_id"]) for it in r["items"]}
            else:
                keys.add(image_key(r["split"], r["image_id"]))
    return sorted(keys)


class FeatureStore:
    def __init__(self, cfg: dict):
        d = repo_path(cfg["features"]["dir"])
        self.feats = np.load(d / "features.npy", mmap_mode="r")
        self.index = load_json(d / "index.json")

    def get(self, keys: list[str]) -> torch.Tensor:
        return torch.from_numpy(np.stack([self.feats[self.index[k]] for k in keys]))  # [B, 196, 768] fp16


def caption_example(rec: dict) -> dict:
    return {"key": image_key(rec["split"], rec["image_id"]), "question": CAPTION_PROMPT, "target": rec["caption"],
            "kind": "caption", "max_target": "caption"}


def pair_examples(pair: dict) -> list[dict]:
    out = []
    for slot, it in enumerate(pair["items"]):
        out.append({"key": image_key(pair["split"], it["image_id"]), "question": pair["question"] + SHORT_SUFFIX,
                    "target": it["answer"], "kind": pair["kind"], "max_target": "short", "pair_id": pair["pair_id"],
                    "slot": slot, "partner_key": image_key(pair["split"], pair["items"][1 - slot]["image_id"]),
                    "refs": it.get("refs"), "qtype": pair["qtype"]})
    return out


class CurriculumSampler:
    """Draw-level sampler. A draw yields two examples: a complementary pair, or two caption examples.

    Stage 1 passes over the caption set in seeded shuffled epochs; Stages 2-3 sample components with the stage
    mixture. State (RNG + caption cursor) is serialisable for exact resumption.
    """

    def __init__(self, splits: dict, seed: int):
        self.cap = [caption_example(r) for r in splits["caption_train"]]
        self.pools = {"presence": [p for p in splits["grounding_pool_train"] if p["kind"] == "presence"],
                      "counting": [p for p in splits["grounding_pool_train"] if p["kind"] == "counting"],
                      "vqa": splits["vqa_train"]}
        self.rng = np.random.default_rng(seed)
        self.order = self.rng.permutation(len(self.cap))
        self.cursor = 0

    def _next_caption(self) -> dict:
        if self.cursor >= len(self.order):
            self.order, self.cursor = self.rng.permutation(len(self.cap)), 0
        ex = self.cap[int(self.order[self.cursor])]
        self.cursor += 1
        return ex

    def caption_draw(self) -> list[dict]:
        return [self._next_caption(), self._next_caption()]

    def draw(self, mix: dict | None) -> tuple[str, list[dict]]:
        if mix is None:
            return "caption", self.caption_draw()
        comps, probs = list(mix), np.array(list(mix.values()), dtype=float)
        comp = comps[int(self.rng.choice(len(comps), p=probs / probs.sum()))]
        if comp == "caption":
            return comp, self.caption_draw()
        pool = self.pools[comp]
        return comp, pair_examples(pool[int(self.rng.integers(len(pool)))])

    def update_batch(self, mix: dict | None, draws: int) -> tuple[list[dict], dict]:
        exs, counts = [], {}
        for _ in range(draws):
            comp, e = self.draw(mix)
            exs += e
            counts[comp] = counts.get(comp, 0) + 1
        return exs, counts

    def state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "order": self.order, "cursor": self.cursor}

    def load_state(self, s: dict) -> None:
        self.rng.bit_generator.state = s["rng"]
        self.order, self.cursor = s["order"], s["cursor"]
