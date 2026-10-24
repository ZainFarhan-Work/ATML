"""Print library versions, GPU, and whether the expected external assets are present.

    python -m scripts.check_environment
"""
from __future__ import annotations

import importlib
import shutil

from common.utils import repo_path

PKGS = ["torch", "transformers", "datasets", "lm_eval", "math_verify", "pandas", "pyarrow", "scipy", "PIL", "yaml"]
ASSETS = {
    "Task1 qualitative prompts": "data/first_plot_questions.yaml",
    "Task2 COCO train2014": "data/coco/train2014",
    "Task2 COCO val2014": "data/coco/val2014",
    "Task2 COCO annotations": "data/coco/annotations/instances_train2014.json",
    "Task2 VQAv2 complementary pairs": "data/vqav2/v2_mscoco_train2014_complementary_pairs.json",
    "Task3 rollouts (LLM)": "data/ATML-PA3-Data/Qwen2.5-3B-Instruct/generations.parquet",
    "Task3 rollouts (VLM)": "data/ATML-PA3-Data/Qwen2.5-VL-3B-Instruct/attention_metrics.parquet",
    "Task3 MathVista metadata": "data/ATML-PA3-Data/mathvista_question_metadata.json",
}


def main():
    for p in PKGS:
        try:
            m = importlib.import_module(p)
            print(f"{p:14s} {getattr(m, '__version__', 'ok')}")
        except Exception as e:  # noqa: BLE001
            print(f"{p:14s} MISSING ({e.__class__.__name__})")
    try:
        import torch
        if torch.cuda.is_available():
            pr = torch.cuda.get_device_properties(0)
            print(f"GPU: {pr.name}, {pr.total_memory / 2**30:.1f} GiB, bf16={torch.cuda.is_bf16_supported()}")
        else:
            print("GPU: none")
    except Exception:
        pass
    print(f"disk free: {shutil.disk_usage(repo_path('.')).free / 2**30:.0f} GiB")
    for name, rel in ASSETS.items():
        print(f"[{'x' if repo_path(rel).exists() else ' '}] {name}: {rel}")


if __name__ == "__main__":
    main()
