"""Render each final description of an evaluation run with Gemini into <eval_dir>/images/<instance_id>.jpg.

Every episode is done unless --instances names some. Existing files are reused, so reruns only pay for what is
missing. The instance-level images (reference, request) come from generate_images_for_data.py.

Usage:
    python scripts/generate_images_for_eval.py [<eval_dir> ...] [--instances <id> ...] [--workers 64]

Every run under paper_results/ when no eval_dir is given.
"""
import argparse
import os
from functools import partial

from utils import add_model_args, model_from_args, paper_runs, read_jsonl, run_jobs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("eval_dirs", nargs="*", metavar="eval_dir",
                        help="evaluation run directories (default: every run under paper_results/)")
    parser.add_argument("--instances", nargs="+", metavar="ID", help="instance ids to do instead of every episode")
    add_model_args(parser)
    args = parser.parse_args()
    model = model_from_args(args)

    for eval_dir in args.eval_dirs or paper_runs():
        episodes = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "episodes.jsonl"))}
        if args.instances:
            missing = [i for i in args.instances if i not in episodes]
            if missing:
                raise SystemExit(f"not in {eval_dir}: {', '.join(missing)}")
            chosen = args.instances
        else:
            chosen = sorted(episodes)
        out = os.path.join(eval_dir, "images")
        os.makedirs(out, exist_ok=True)
        jobs = [(os.path.join(out, f"{instance_id}.jpg"), partial(model.generate, episodes[instance_id]["final_output"]))
                for instance_id in chosen]
        n = run_jobs(jobs, args.workers)
        print(f"done {eval_dir}: {n} of {len(jobs)} images generated, the rest existed")


if __name__ == "__main__":
    main()
