"""Build all Task 2 data components from COCO 2014 and VQAv2 complementary pairs.

    python -m task2_vlm.build_data --config configs/task2_vlm.yaml

Outputs (splits/task2/): caption_train/dev, grounding_pool_train, grounding_dev/test, vqa_train/dev/test,
ids_summary.json (all selected image / question / pair IDs), grounding_audit.csv (labels to inspect by hand).
Generated questions are template-based (presence / counting) and come from COCO instance annotations only.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict

import numpy as np

from common.utils import load_yaml, repo_path, save_json
from task2_vlm.vqa_norm import normalize_answer

IRREGULAR_PLURALS = {"person": "people", "knife": "knives", "skis": "skis", "scissors": "scissors",
                     "sheep": "sheep", "mouse": "mice", "tv": "tvs", "sports ball": "sports balls",
                     "wine glass": "wine glasses", "traffic light": "traffic lights", "teddy bear": "teddy bears"}


def pluralize(name: str) -> str:
    if name in IRREGULAR_PLURALS:
        return IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "sh", "ch")):
        return name + "es"
    return name + "s"


def article(name: str) -> str:
    return "an" if name[0] in "aeiou" else "a"


def load_coco(root, split: str):
    ann = repo_path(f"{root}/annotations")
    caps = json.loads((ann / f"captions_{split}.json").read_text())
    inst = json.loads((ann / f"instances_{split}.json").read_text())
    return caps, inst


def index_instances(inst: dict, min_area_frac: float):
    """Per image and category: instance count under the unambiguity rule, plus any-annotation flag."""
    cats = {c["id"]: c["name"] for c in inst["categories"]}
    size = {im["id"]: im["width"] * im["height"] for im in inst["images"]}
    per = defaultdict(lambda: defaultdict(list))
    for a in inst["annotations"]:
        per[a["image_id"]][a["category_id"]].append((a["area"] / size[a["image_id"]], bool(a["iscrowd"])))
    return cats, per, list(size)


def clean_count(entries, min_frac: float):
    """Count if every instance of the category is a non-crowd, non-tiny object; else None (ambiguous)."""
    if any(crowd or frac < min_frac for frac, crowd in entries):
        return None
    return len(entries)


def build_grounding(inst, rng, n_presence, n_counting, cfg, forbid: set[int], split: str, tag: str):
    g = cfg["data"]["grounding"]
    cats, per, images = index_instances(inst, g["min_box_area_frac"])
    pos = defaultdict(list)                  # cat -> images with an unambiguous present object
    by_count = defaultdict(lambda: defaultdict(list))  # cat -> count -> images
    neg_pool = defaultdict(list)
    for im in images:
        if im in forbid:
            continue
        present = per.get(im, {})
        for c in cats:
            if c not in present:
                neg_pool[c].append(im)
        for c, entries in present.items():
            if clean_count(entries, g["min_box_area_frac"]) is not None:
                pos[c].append(im)
                n = clean_count(entries, g["min_box_area_frac"])
                if g["min_instances"] <= n <= g["max_instances"]:
                    by_count[c][n].append(im)
    pcats = [c for c in cats if len(pos[c]) >= 5 and len(neg_pool[c]) >= 5]
    ccats = [c for c in cats if len(by_count[c]) >= 2 and all(len(v) >= 2 for v in by_count[c].values())]
    used: set[int] = set()
    pairs = []

    def take(images):
        for _ in range(50):
            im = int(images[rng.integers(len(images))])
            if im not in used:
                used.add(im)
                return im
        return None

    while sum(p["kind"] == "presence" for p in pairs) < n_presence:
        c = pcats[rng.integers(len(pcats))]
        a, b = take(pos[c]), take(neg_pool[c])
        if a is None or b is None:
            continue
        name = cats[c]
        pairs.append({"pair_id": f"{tag}_presence_{len(pairs)}", "kind": "presence", "qtype": "yes/no", "split": split,
                      "category": name, "question": f"Does this image contain {article(name)} {name}?",
                      "items": [{"image_id": a, "answer": "yes", "refs": None},
                                {"image_id": b, "answer": "no", "refs": None}]})
    while sum(p["kind"] == "counting" for p in pairs) < n_counting:
        c = ccats[rng.integers(len(ccats))]
        n1, n2 = rng.choice(sorted(by_count[c]), 2, replace=False)
        a, b = take(by_count[c][int(n1)]), take(by_count[c][int(n2)])
        if a is None or b is None:
            continue
        name = cats[c]
        pairs.append({"pair_id": f"{tag}_counting_{len(pairs)}", "kind": "counting", "qtype": "number", "split": split,
                      "category": name, "question": f"How many {pluralize(name)} can you see?",
                      "items": [{"image_id": a, "answer": str(int(n1)), "refs": None},
                                {"image_id": b, "answer": str(int(n2)), "refs": None}]})
    return pairs


def build_vqa(cfg, split: str, counts: dict, rng, forbid: set[int], tag: str):
    root = repo_path(cfg["data"]["vqa_root"])
    pairs_raw = json.loads((root / f"v2_mscoco_{split}_complementary_pairs.json").read_text())
    qs = {q["question_id"]: q for q in json.loads((root / f"v2_OpenEnded_mscoco_{split}_questions.json").read_text())["questions"]}
    an = {a["question_id"]: a for a in json.loads((root / f"v2_mscoco_{split}_annotations.json").read_text())["annotations"]}
    min_agree = cfg["data"]["vqa"]["min_agree"]
    cands = defaultdict(list)
    for q1, q2 in pairs_raw:
        if q1 not in an or q2 not in an:
            continue
        a1, a2 = an[q1], an[q2]
        items, ok = [], True
        for a in (a1, a2):
            norm = [normalize_answer(x["answer"]) for x in a["answers"]]
            cons = Counter(norm).most_common(1)[0]
            if len(norm) != 10 or cons[1] < min_agree or a["image_id"] in forbid:
                ok = False
                break
            raw = Counter(x["answer"] for x, n in zip(a["answers"], norm) if n == cons[0]).most_common(1)[0][0]
            items.append({"image_id": a["image_id"], "answer": raw, "canonical": cons[0],
                          "refs": [x["answer"] for x in a["answers"]], "question_id": a["question_id"]})
        if not ok or items[0]["canonical"] == items[1]["canonical"] or items[0]["image_id"] == items[1]["image_id"]:
            continue
        if a1["answer_type"] != a2["answer_type"]:
            continue
        cands[a1["answer_type"]].append({"pair_id": f"{tag}_vqa_{q1}_{q2}", "kind": "vqa", "qtype": a1["answer_type"],
                                         "split": f"{split}", "question": qs[q1]["question"], "items": items})
    out, used = [], set()
    for qtype, n in [("yes/no", counts["yes_no"]), ("number", counts["number"]), ("other", counts["other"])]:
        pool = cands[qtype]
        order = rng.permutation(len(pool))
        got = 0
        for i in order:
            p = pool[int(i)]
            ims = {it["image_id"] for it in p["items"]}
            if ims & used:
                continue
            used |= ims
            out.append(p)
            got += 1
            if got == n:
                break
        if got < n:
            raise RuntimeError(f"only {got}/{n} {qtype} pairs for {tag}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task2_vlm.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    d = cfg["data"]
    rng = np.random.default_rng(d["seed"])
    out = repo_path(d["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    caps_tr, inst_tr = load_coco(d["coco_root"], "train2014")
    caps_va, inst_va = load_coco(d["coco_root"], "val2014")

    # --- evaluation sets first (val2014); every set is image-disjoint from every other one
    forbid: set[int] = set()

    def images_of(pairs):
        return {it["image_id"] for p in pairs for it in p["items"]}

    g = d["grounding"]
    ground_test = build_grounding(inst_va, rng, g["test_pairs"] // 2, g["test_pairs"] // 2, cfg, forbid, "val2014", "gtest")
    forbid |= images_of(ground_test)
    vqa_test = build_vqa(cfg, "val2014", {"yes_no": d["vqa"]["test"]["yes_no"], "number": d["vqa"]["test"]["number"],
                                          "other": d["vqa"]["test"]["other"]}, rng, forbid, "vtest")
    forbid |= images_of(vqa_test)
    ground_dev = build_grounding(inst_va, rng, g["dev_pairs"] // 2, g["dev_pairs"] // 2, cfg, forbid, "val2014", "gdev")
    forbid |= images_of(ground_dev)
    vqa_dev = build_vqa(cfg, "val2014", {"yes_no": d["vqa"]["dev"]["yes_no"], "number": d["vqa"]["dev"]["number"],
                                         "other": d["vqa"]["dev"]["other"]}, rng, forbid, "vdev")
    forbid |= images_of(vqa_dev)

    # dev captions: 96 val2014 images, one caption each, disjoint from the other eval sets
    by_img = defaultdict(list)
    for a in caps_va["annotations"]:
        by_img[a["image_id"]].append(a["caption"].strip())
    cand = sorted(i for i in by_img if i not in forbid)
    dev_imgs = sorted(rng.choice(cand, d["caption"]["dev_images"], replace=False).tolist())
    caption_dev = [{"image_id": int(i), "split": "val2014", "caption": by_img[i][int(rng.integers(len(by_img[i])))]} for i in dev_imgs]
    forbid |= set(dev_imgs)

    # --- training sets (train2014); disjoint from val2014 by construction
    by_img_tr = defaultdict(list)
    for a in caps_tr["annotations"]:
        by_img_tr[a["image_id"]].append(a["caption"].strip())
    tr_imgs = sorted(rng.choice(sorted(by_img_tr), d["caption"]["train_images"], replace=False).tolist())
    caption_train = []
    for i in tr_imgs:
        picks = rng.choice(len(by_img_tr[i]), d["caption"]["captions_per_image"], replace=False)
        caption_train += [{"image_id": int(i), "split": "train2014", "caption": by_img_tr[i][int(k)]} for k in picks]
    ground_pool = build_grounding(inst_tr, rng, g["pool_presence_pairs"], g["pool_counting_pairs"], cfg, set(), "train2014", "gtrain")
    vqa_train = build_vqa(cfg, "train2014", d["vqa"]["train"], rng, set(), "vtrain")

    # --- split isolation check
    eval_sets = {"ground_test": images_of(ground_test), "vqa_test": images_of(vqa_test), "ground_dev": images_of(ground_dev),
                 "vqa_dev": images_of(vqa_dev), "caption_dev": {r["image_id"] for r in caption_dev}}
    names = list(eval_sets)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            assert not (eval_sets[names[i]] & eval_sets[names[j]]), (names[i], names[j])

    for name, obj in [("caption_train", caption_train), ("caption_dev", caption_dev), ("grounding_pool_train", ground_pool),
                      ("grounding_dev", ground_dev), ("grounding_test", ground_test), ("vqa_train", vqa_train),
                      ("vqa_dev", vqa_dev), ("vqa_test", vqa_test)]:
        save_json(out / f"{name}.json", obj)
    save_json(out / "ids_summary.json", {
        "seed": d["seed"],
        "counts": {"caption_train": len(caption_train), "caption_dev": len(caption_dev), "grounding_pool_pairs": len(ground_pool),
                   "grounding_dev_pairs": len(ground_dev), "grounding_test_pairs": len(ground_test),
                   "vqa_train_pairs": len(vqa_train), "vqa_dev_pairs": len(vqa_dev), "vqa_test_pairs": len(vqa_test)},
        "vqa_type_counts": {k: dict(Counter(p["qtype"] for p in v)) for k, v in
                            [("train", vqa_train), ("dev", vqa_dev), ("test", vqa_test)]},
        "image_ids": {k: sorted(v) for k, v in eval_sets.items()},
        "train_caption_images": tr_imgs,
        "vqa_pair_ids": {k: [p["pair_id"] for p in v] for k, v in [("train", vqa_train), ("dev", vqa_dev), ("test", vqa_test)]},
        "grounding_pair_ids": {k: [p["pair_id"] for p in v] for k, v in [("dev", ground_dev), ("test", ground_test)]},
    })

    # label audit sheet: inspect annotation-derived labels independently of any model prediction
    with (out / "grounding_audit.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "split", "question", "image_a", "answer_a", "image_b", "answer_b", "label_ok(manual)"])
        sample = [ground_pool[int(i)] for i in rng.choice(len(ground_pool), 60, replace=False)] + ground_dev + ground_test
        for p in sample:
            w.writerow([p["pair_id"], p["split"], p["question"], p["items"][0]["image_id"], p["items"][0]["answer"],
                        p["items"][1]["image_id"], p["items"][1]["answer"], ""])
    print(json.dumps(json.loads((out / "ids_summary.json").read_text())["counts"], indent=2))


if __name__ == "__main__":
    main()
