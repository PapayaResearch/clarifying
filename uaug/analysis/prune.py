"""Post-hoc k-sufficiency pruning of synthesized instances using ZeroQ judgments.

The ZeroQ baseline generates the final output from the eroded prompt alone, so any
redacted detail it delivers was inferable without questions — the redaction never
created real uncertainty. This tool drops those details from the redacted mask
(ground-truth replacement for the synthesis-time LLM inferability filter).

Usage:
    python -m uaug.analysis.prune <instances_dir> <judgment_paths...> [--out DIR] [--threshold 0.5] [--min-redacted 1]

<instances_dir> is a synthesis results directory holding instances.<split>.jsonl;
judgment paths are episode run directories (or parent trees) scanned recursively
for judgments.zeroq.jsonl. Pruned instances land in --out (default:
<instances_dir>-pruned) with a prune_report.json alongside.
"""

import argparse
import collections
import glob
import json
import os

from uaug.schema import TaskInstance
from uaug.utils.misc import read_jsonl, write_jsonl


def collect_delivery_rates(paths):
    """Per (instance_id, detail_id): fraction of ZeroQ episodes that delivered the detail."""
    hits = collections.defaultdict(list)
    n_files = 0
    for base in paths:
        pattern = base if base.endswith(".jsonl") else os.path.join(base, "**", "judgments.zeroq.jsonl")
        for path in sorted(glob.glob(pattern, recursive=True)):
            n_files += 1
            for row in read_jsonl(path):
                for detail_id, verdict in row["per_detail"].items():
                    hits[(row["instance_id"], int(detail_id))].append(bool(verdict["delivered"]))
    rates = {key: sum(vals) / len(vals) for key, vals in hits.items()}
    return rates, n_files


def prune_instance(instance: TaskInstance, rates, threshold: float):
    """Remove inferable details from the instance; returns (instance, pruned_details) or (None, ...) if uncovered."""
    covered = any((instance.instance_id, d.id) in rates for d in instance.redacted)
    if not covered:
        return None, []
    pruned = [d for d in instance.redacted if rates.get((instance.instance_id, d.id), 0.0) >= threshold]
    pruned_ids = {d.id for d in pruned}
    instance.details = [d for d in instance.details if d.id not in pruned_ids]
    instance.redacted_ids = [i for i in instance.redacted_ids if i not in pruned_ids]
    return instance, pruned


def main(args):
    rates, n_files = collect_delivery_rates(args.judgments)
    if not rates:
        print("No judgments.zeroq.jsonl found under given paths.")
        return
    out_dir = args.out or args.instances_dir.rstrip("/\\") + "-pruned"
    os.makedirs(out_dir, exist_ok=True)

    report = {"judgment_files": n_files, "threshold": args.threshold, "splits": {}, "pruned_by_category": collections.Counter()}
    for path in sorted(glob.glob(os.path.join(args.instances_dir, "instances.*.jsonl"))):
        split = os.path.basename(path).split(".")[1]
        kept_instances, stats = [], collections.Counter()
        for row in read_jsonl(path):
            instance, pruned = prune_instance(TaskInstance.from_dict(row), rates, args.threshold)
            if instance is None:
                stats["uncovered"] += 1  # no ZeroQ judgment for this instance — dropped, not passed through
                continue
            stats["details_pruned"] += len(pruned)
            for detail in pruned:
                report["pruned_by_category"][detail.category] += 1
            if len(instance.redacted_ids) < args.min_redacted:
                stats["dropped_instances"] += 1
                continue
            kept_instances.append(instance)
        write_jsonl(os.path.join(out_dir, f"instances.{split}.jsonl"), [i.to_dict() for i in kept_instances])
        report["splits"][split] = {"in": stats["uncovered"] + stats["dropped_instances"] + len(kept_instances),
                                   "out": len(kept_instances), **stats}

    report["pruned_by_category"] = dict(report["pruned_by_category"])
    with open(os.path.join(out_dir, "prune_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Pruned instances written to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("instances_dir", help="Synthesis results directory with instances.<split>.jsonl")
    parser.add_argument("judgments", nargs="+", help="Episode run directories (or trees) with judgments.zeroq.jsonl")
    parser.add_argument("--out", default=None, help="Output directory (default: <instances_dir>-pruned)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Delivery rate at or above which a redacted detail counts as inferable")
    parser.add_argument("--min-redacted", type=int, default=1, help="Drop instances left with fewer redacted details than this")
    main(parser.parse_args())
