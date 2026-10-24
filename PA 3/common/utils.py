from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str | Path) -> dict:
    cfg = yaml.safe_load(repo_path(path).read_text(encoding="utf-8"))
    if cfg.get("base_config"):
        base = load_yaml(cfg["base_config"])  # recursive: configs may chain
        cfg = _deep_merge(base, {k: v for k, v in cfg.items() if k != "base_config"})
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def save_json(path: str | Path, obj: Any) -> None:
    ensure_parent(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    with ensure_parent(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    p = repo_path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    with ensure_parent(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def wall_timer():
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 2**20


def count_params(module: torch.nn.Module, trainable_only: bool = False) -> int:
    seen: set[int] = set()
    total = 0
    for p in module.parameters():
        if id(p) in seen or (trainable_only and not p.requires_grad):
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def dir_size_mb(path: str | Path) -> float:
    p = repo_path(path)
    if p.is_file():
        return p.stat().st_size / 2**20
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 2**20


def cosine_lr(step: int, total: int, warmup: int, peak: float, floor_frac: float = 0.0) -> float:
    """Linear warmup then cosine decay to floor_frac * peak. `step` is 0-based update index."""
    if warmup > 0 and step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(max(progress, 0.0), 1.0)
    floor = peak * floor_frac
    return floor + (peak - floor) * 0.5 * (1.0 + float(np.cos(np.pi * progress)))
