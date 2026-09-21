"""ERM: the Task 2 source-only checkpoint, reused unchanged.

The manual forbids retraining it under a different configuration, so nothing is
trained here; Task 3 loads task2/cache/source_only.pt for the baseline.
"""

from pathlib import Path

ERM_CHECKPOINT = Path(__file__).resolve().parents[2] / "task2" / "cache" / "source_only.pt"
