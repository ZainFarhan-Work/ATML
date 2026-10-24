"""Print the schema of the supplied Task 3 parquet tables so COLUMN_CANDIDATES in task3_ttc/data.py can be checked.

    python -m scripts.inspect_task3_data --config configs/task3_ttc.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from common.utils import load_yaml, repo_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/task3_ttc.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    root = repo_path(cfg["data"]["root"])
    for folder in (cfg["data"]["llm_dir"], cfg["data"]["vlm_dir"]):
        for f in sorted((root / folder).glob("*.parquet")):
            df = pd.read_parquet(f)
            print(f"\n== {f.relative_to(root)}  rows={len(df)}")
            for c in df.columns:
                v = df[c].iloc[0]
                desc = f"array(len={len(v)})" if hasattr(v, "__len__") and not isinstance(v, str) else repr(v)[:80]
                print(f"  {c:28s} {str(df[c].dtype):10s} {desc}")


if __name__ == "__main__":
    main()
