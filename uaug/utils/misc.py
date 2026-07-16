import dataclasses
import json
import random
import re
import statistics
from pathlib import Path
from typing import Any, Iterable, List, Optional

from omegaconf import DictConfig, OmegaConf
from rich.box import ROUNDED
from rich.console import Console
from rich.style import Style
from rich.table import Table


def print_config(cfg: DictConfig):
    table = Table(title="Hydra Configuration", box=ROUNDED)
    table.row_styles = [Style(color="cyan", dim=True), Style(color="magenta", dim=True)]

    # One column per top-level (or nested group) key, sorted for stable ordering
    keys = set()
    for key, value in cfg.items():
        keys.add(key)
        if isinstance(value, dict):
            for nested_key in value.keys():
                keys.add(nested_key)
    keys = sorted(keys)
    for key in keys:
        table.add_column(key, style="cyan")

    row = {}
    for key, value in cfg.items():
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                row[nested_key] = OmegaConf.to_yaml(nested_value)
        else:
            row[key] = OmegaConf.to_yaml(value)
    table.add_row(*[row.get(key, "") for key in keys])

    console = Console()
    console.print(table)


def read_jsonl(path: str | Path) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: str | Path, rows: Iterable[Any]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if dataclasses.is_dataclass(row):
                row = dataclasses.asdict(row)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def bootstrap_ci(values: List[float], n_resamples: int = 1000, seed: int = 0) -> List[float]:
    """95% percentile bootstrap CI for the mean, resampling over instances."""
    if not values:
        return [None, None]
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(n_resamples))
    return [round(means[int(0.025 * n_resamples)], 3), round(means[int(0.975 * n_resamples) - 1], 3)]


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace for leakage/overlap checks."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


# Words too generic to count as evidence that a detail leaked into a prompt
_STOPWORDS = frozenset(
    "a an the of in on at to and or with is are was were be been its it this that there her his "
    "image picture photo scene left right front back near next".split()
)


def content_words(text: str) -> set:
    return {w for w in normalize_text(text).split() if w not in _STOPWORDS and len(w) > 2}


def leakage_overlap(detail_text: str, prompt_text: str, reference_texts: Optional[List[str]] = None) -> float:
    """Fraction of a detail's content words found in the prompt (1.0 = fully leaked).

    With reference_texts (the kept details), only the detail's *novel* words count.
    A detail with no novel words returns 0.0 - the inferability filter is the
    semantic gate for those.
    """
    detail_words = content_words(detail_text)
    if reference_texts:
        for reference in reference_texts:
            detail_words -= content_words(reference)
    if not detail_words:
        return 0.0
    prompt_words = content_words(prompt_text)
    return len(detail_words & prompt_words) / len(detail_words)
