"""Sampling helpers for task datasets."""

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, TypeVar

from candvg.utils import GroundingSample

RANDOM_SAMPLE_SEED = 42
CANDVG_CANDIDATES_BIN_COUNTS_PATH = Path(
    "data/candvg/bin_counts.json"
)

T = TypeVar("T")


def _bin_sort_key(bin_label: str) -> int:
    """Sort numeric bin labels with 7+ after 6."""
    return 7 if bin_label == "7+" else int(bin_label)


def _bin_label(bin_size: int) -> str:
    """Return the public bin label for grouped sampling."""
    return "7+" if bin_size >= 7 else str(bin_size)


def load_bin_quota(
    path: Path = CANDVG_CANDIDATES_BIN_COUNTS_PATH,
) -> dict[str, int] | None:
    """Load a bin sampling quota if the preprocessing artifact exists."""
    if not path.exists():
        return None

    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(bin_label): int(count)
        for bin_label, count in payload["bin_counts"].items()
        if int(count) > 0
    }


def sample_records(
    records: list[T],
    max_samples: int | None,
    seed: int = RANDOM_SAMPLE_SEED,
) -> list[T]:
    if max_samples is None or max_samples >= len(records):
        return records

    assert max_samples > 0
    return random.Random(seed).sample(records, k=max_samples)


def sample_records_by_bin_quota(
    records: list[GroundingSample],
    quota: dict[str, int],
    seed: int = RANDOM_SAMPLE_SEED,
) -> list[GroundingSample]:
    """Sample records according to per-bin quotas without refilling shortages."""
    groups: dict[str, list[GroundingSample]] = defaultdict(list)
    for record in records:
        groups[_bin_label(int(record.metadata["bin_size"]))].append(record)

    rng = random.Random(seed)
    selected: list[GroundingSample] = []
    for bin_label in sorted(quota, key=_bin_sort_key):
        group_records = groups.get(bin_label, [])
        sample_count = min(quota[bin_label], len(group_records))
        if sample_count == 0:
            continue
        selected.extend(rng.sample(group_records, k=sample_count))

    return sorted(selected, key=lambda record: record.sample_id)
