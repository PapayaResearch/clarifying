"""Generate the instance-level images of a synthesis run with Gemini into <synthesis_dir>/images/.

Three per instance: the DOCCI reference (_original.jpg, copied from --images-dir), its upscaled copy (_enhanced.jpg)
and the image rendered from the eroded request alone (_request.jpg). The whole --split is done unless --instances
names some. Existing files are reused. Run scripts/setup_docci_images.py once first for the reference images.

Usage:
    python scripts/generate_images_for_data.py [<synthesis_dir>] [--split val] [--instances <id> ...]
        [--images-dir data/docci/images] [--workers 64]
"""
import argparse
import os
from functools import partial

from PIL import Image

from utils import DOCCI_DIR, INSTANCES_DIR, add_model_args, model_from_args, read_jsonl, run_jobs

# Applied once to the reference image
ENHANCEMENT_PROMPT = """Upscale the image and enhance its quality.
Do not make any other changes to the image itself."""


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("synthesis_dir", nargs="?", default=INSTANCES_DIR,
                        help="directory with the instances.<split>.jsonl files (default: the shipped data/docci_instances)")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--instances", nargs="+", metavar="ID", help="instance ids to do instead of the whole split")
    parser.add_argument("--images-dir", default=os.path.join(DOCCI_DIR, "images"), help="DOCCI reference images, <source_id>.jpg")
    add_model_args(parser)
    args = parser.parse_args()

    instances = {r["instance_id"]: r for r in read_jsonl(os.path.join(args.synthesis_dir, f"instances.{args.split}.jsonl"))}
    if args.instances:
        missing = [i for i in args.instances if i not in instances]
        if missing:
            raise SystemExit(f"not in the {args.split} split: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = sorted(instances)
    out = os.path.join(args.synthesis_dir, "images")
    os.makedirs(out, exist_ok=True)
    model = model_from_args(args)

    jobs = []
    for instance_id in chosen:
        instance = instances[instance_id]
        source_id = instance.get("source_id") or instance_id.split(".")[0]
        reference_path = os.path.join(out, f"{instance_id}_original.jpg")
        if not os.path.exists(reference_path):
            src = os.path.join(args.images_dir, f"{source_id}.jpg")
            if os.path.exists(src):
                Image.open(src).convert("RGB").save(reference_path, quality=92)
            else:
                print(f"reference image missing for {source_id} (run scripts/setup_docci_images.py)")
        if os.path.exists(reference_path):
            with open(reference_path, "rb") as f:
                reference_bytes = f.read()
            jobs.append((os.path.join(out, f"{instance_id}_enhanced.jpg"),
                         partial(model.generate, ENHANCEMENT_PROMPT, image_bytes=reference_bytes)))
        jobs.append((os.path.join(out, f"{instance_id}_request.jpg"), partial(model.generate, instance["eroded_prompt"])))
    n = run_jobs(jobs, args.workers)
    print(f"done {args.synthesis_dir}: {n} of {len(jobs)} images generated, the rest existed")


if __name__ == "__main__":
    main()
