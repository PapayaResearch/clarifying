import dataclasses
import json
from pathlib import Path
from typing import Any, Iterable, List

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


def clairvoyant_u_rec(importances: Iterable[float], n_questions: int, weighting: str) -> float:
    """Best possible u_rec at a question count: recover the n highest-weighted redacted details."""
    weights = sorted((max(imp, 0.05) if weighting == "llm" else 1.0 for imp in importances), reverse=True)
    return sum(weights[:n_questions]) / sum(weights)
