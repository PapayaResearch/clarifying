import dataclasses
import hashlib
import json
import logging
import os
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, List, Optional

from tqdm import tqdm

from uaug.utils.schema import Detail, TaskInstance
from uaug.utils.misc import read_jsonl, write_jsonl
from uaug.utils.wrappers import LanguageModel


@dataclasses.dataclass
class SynthesisPipeline:
    """Pipeline turning fully-specified sources into redacted task instances."""

    sources_path: str
    num_sources: Optional[int]
    task: str
    k: int
    min_details: int
    split_fractions: Dict[str, float]
    random_seed: int
    resume: bool
    resume_path: Optional[str]
    decomposer: LanguageModel
    eroder: LanguageModel

    def _load_sources(self) -> List[dict]:
        sources = read_jsonl(self.sources_path)
        logging.info("Loaded %d sources from %s", len(sources), self.sources_path)
        if self.num_sources is not None:
            rng = random.Random(self.random_seed)
            rng.shuffle(sources)
            sources = sources[: self.num_sources]
        return sources

    def _assign_split(self, source_id: str) -> str:
        """Split by seeded hash of source_id: a source keeps its split across runs,
        so train sources at one k never reappear in val/test at another."""
        bucket = int(hashlib.sha256(f"{self.random_seed}:{source_id}".encode()).hexdigest(), 16) % 10_000 / 10_000
        cumulative = 0.0
        for split, fraction in self.split_fractions.items():
            cumulative += fraction
            if bucket < cumulative:
                return split
        return list(self.split_fractions)[-1]

    def _decompose(self, source: dict) -> Optional[List[Detail]]:
        output = self.decomposer.get_response(text=source["full_spec"])
        if output is None:
            return None
        return [
            Detail(id=i, text=item.text, category=item.category, importance=item.importance, check=item.check)
            for i, item in enumerate(output.details)
        ]

    def _sample_mask(self, details: List[Detail], rng: random.Random) -> List[int]:
        """Importance-weighted sample of k details, preferring distinct categories."""
        mask: List[int] = []
        used_categories: set = set()
        candidates = list(details)
        while len(mask) < self.k and candidates:
            # Prefer categories not yet redacted so a mask spans distinct detail types
            fresh = [d for d in candidates if d.category not in used_categories]
            choices = fresh if len(fresh) > 0 else candidates
            weights = [max(d.importance, 0.05) for d in choices]
            picked = rng.choices(choices, weights=weights, k=1)[0]
            mask.append(picked.id)
            used_categories.add(picked.category)
            candidates = [d for d in candidates if d.id != picked.id]
        return sorted(mask)

    def _build_instance(self, source: dict) -> tuple[Optional[TaskInstance], str]:
        """Returns (instance, reason); instance is None when dropped, reason explains why."""
        source_id = str(source["source_id"])
        details = self._decompose(source)
        if details is None:
            return None, "decompose_failed"
        if len(details) < self.min_details:
            return None, "too_few_details"

        rng = random.Random(f"{self.random_seed}:{source_id}")
        mask = self._sample_mask(details, rng)
        if len(mask) < self.k:
            return None, "not_enough_redactable"
        mask_set = set(mask)
        kept = [d for d in details if d.id not in mask_set]

        erode_output = self.eroder.get_response(details=[d.text for d in kept])
        if erode_output is None or not erode_output.request.strip():
            return None, "erode_failed"

        instance = TaskInstance(
            instance_id=f"{source_id}.k{self.k}",
            task=self.task,
            source_id=source_id,
            full_spec=source["full_spec"],
            details=details,
            redacted_ids=mask,
            eroded_prompt=erode_output.request.strip(),
            split=self._assign_split(source_id),
        )
        return instance, "ok"

    def run(self, results_dir: str, max_workers: int = 8):
        sources = self._load_sources()

        # Resume: skip sources already present in a previous run's instances
        done_ids: set = set()
        instances: List[TaskInstance] = []
        if self.resume and self.resume_path:
            for name in os.listdir(self.resume_path):
                if name.startswith("instances.") and name.endswith(".jsonl"):
                    for row in read_jsonl(os.path.join(self.resume_path, name)):
                        instances.append(TaskInstance.from_dict(row))
                        done_ids.add(instances[-1].source_id)
            logging.info("Resuming: %d instances loaded, %d sources already done", len(instances), len(done_ids))
        pending = [s for s in sources if str(s["source_id"]) not in done_ids]

        reasons: Counter = Counter()
        lock = Lock()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self._build_instance, source): source for source in pending}
            for future in tqdm(as_completed(futures), total=len(futures), desc="synthesis"):
                source = futures[future]
                try:
                    instance, reason = future.result()
                except Exception:
                    logging.exception("Unhandled error on source %s", source.get("source_id"))
                    instance, reason = None, "error"
                with lock:
                    reasons[reason] += 1
                    if instance is not None:
                        instances.append(instance)

        # Write per-split instance files + a synthesis report
        for split in self.split_fractions:
            split_instances = [inst for inst in instances if inst.split == split]
            write_jsonl(os.path.join(results_dir, f"instances.{split}.jsonl"), split_instances)
            logging.info("Wrote %d %s instances", len(split_instances), split)

        category_counts = Counter(d.category for inst in instances for d in inst.redacted)
        report = {
            "n_sources": len(sources),
            "n_instances": len(instances),
            "outcomes": dict(reasons),
            "redacted_category_counts": dict(category_counts),
            "per_split": {s: sum(1 for i in instances if i.split == s) for s in self.split_fractions},
        }
        with open(os.path.join(results_dir, "synthesis_report.json"), "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logging.info("Synthesis report: %s", json.dumps(report))
