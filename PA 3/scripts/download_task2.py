"""Lean Task 2 downloader (official COCO / VQAv2 hosts; no account needed).

    python -m scripts.download_task2 --meta      # annotations + VQAv2 questions/answers/complementary pairs (~0.5 GB)
    python -m scripts.download_task2 --images    # only the images listed in splits/task2/*.json (after build_data), not all 19 GB
"""
from __future__ import annotations

import argparse
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

from common.utils import load_yaml, repo_path
from scripts.download_assets import VQA
from task2_vlm.vlm_data import all_image_keys, image_file

ANN = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
WANT = ("captions_train2014.json", "captions_val2014.json", "instances_train2014.json", "instances_val2014.json")
VQA_NEEDED = ("Questions_Train", "Questions_Val", "Annotations_Train", "Annotations_Val", "Complementary_Pairs_Train", "Complementary_Pairs_Val")


def curl(url: str, dest) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        part = dest.with_name(dest.name + ".part")  # rename only when complete, so an interrupted file is never mistaken for a finished one
        subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(part), url], check=True)
        part.rename(dest)


def meta() -> None:
    coco = repo_path("data/coco")
    z = coco / "_zips" / "annotations_trainval2014.zip"
    curl(ANN, z)
    (coco / "annotations").mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(z) as zf:
        for n in zf.namelist():
            if n.rsplit("/", 1)[-1] in WANT:
                (coco / "annotations" / n.rsplit("/", 1)[-1]).write_bytes(zf.read(n))
    vqa = repo_path("data/vqav2")
    for name, url in VQA.items():
        if any(k in name for k in VQA_NEEDED):
            f = vqa / "_zips" / name
            curl(url, f)
            with zipfile.ZipFile(f) as zf:
                zf.extractall(vqa)


def images(workers: int) -> None:
    cfg = load_yaml("configs/task2_vlm.yaml")
    todo = []
    for key in all_image_keys(cfg):
        f = image_file(cfg, key)
        if not f.exists():
            todo.append((f"http://images.cocodataset.org/{key.split('/')[0]}/{f.name}", f))
    print(f"{len(todo)} images to download")
    with ThreadPoolExecutor(workers) as ex:
        list(ex.map(lambda a: curl(*a), todo))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", action="store_true")
    ap.add_argument("--images", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.meta:
        meta()
    if a.images:
        images(a.workers)
