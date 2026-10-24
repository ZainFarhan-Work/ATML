"""Back up / restore run artifacts through a private Hugging Face repo (survives Kaggle/Colab session loss).

    python -m scripts.sync_hub push --repo USER/atml-pa3-backup [--task task1] [--loop 600]
    python -m scripts.sync_hub pull --repo USER/atml-pa3-backup [--task task1]

Needs HF_TOKEN (write access) in the environment. Uploads are incremental (unchanged files are not re-sent).
Only the newest recovery checkpoint is kept remotely; the repo is created private on first push.
"""
from __future__ import annotations

import argparse
import os
import time

from huggingface_hub import HfApi, snapshot_download

from common.utils import REPO_ROOT

PATTERNS = {
    "task1": ["results/task1/**", "splits/task1*", "checkpoints/task1/**", "cached/task1/**", "data/first_plot_questions.yaml"],
    "task2": ["results/task2/**", "splits/task2/**", "checkpoints/task2/**", "cached/task2_features/index.json"],
    "task3": ["results/task3/**", "splits/task3*", "cached/task3/**"],
}


def push(repo: str, tasks: list[str]) -> None:
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(repo, repo_type="model", private=True, exist_ok=True)
    allow = [p for t in tasks for p in PATTERNS[t]]
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(REPO_ROOT), allow_patterns=allow, ignore_patterns=["*.tmp"],
                      delete_patterns=["checkpoints/**/recovery_*.pt"], commit_message=f"sync {time.strftime('%F %T')}")
    print(f"[sync] pushed {tasks} -> {repo}")


def pull(repo: str, tasks: list[str]) -> None:
    allow = [p for t in tasks for p in PATTERNS[t]]
    snapshot_download(repo_id=repo, repo_type="model", local_dir=str(REPO_ROOT), allow_patterns=allow,
                      token=os.environ.get("HF_TOKEN"))
    print(f"[sync] pulled {tasks} <- {repo}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["push", "pull"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--task", nargs="+", default=["task1"], choices=list(PATTERNS))
    ap.add_argument("--loop", type=int, default=0, help="push every N seconds until interrupted")
    args = ap.parse_args()
    if args.mode == "pull":
        pull(args.repo, args.task)
    elif args.loop:
        while True:
            try:
                push(args.repo, args.task)
            except Exception as e:  # noqa: BLE001  keep the loop alive through transient network errors
                print(f"[sync] push failed: {e}")
            time.sleep(args.loop)
    else:
        push(args.repo, args.task)


if __name__ == "__main__":
    main()
