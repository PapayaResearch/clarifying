"""Download the DOCCI descriptions and normalize them into data/docci/sources.jsonl.

DOCCI (https://google.github.io/docci/) ships its descriptions as one jsonlines file, no images are needed for the
text-only pipeline.

Usage:
    python scripts/setup_docci.py [--data-dir data/docci] [--min-chars 400] [--limit 2000]
"""
import argparse
import json
import os

from utils import DOCCI_DIR, download

DOCCI_URL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DOCCI_DIR)
    parser.add_argument("--min-chars", type=int, default=400, help="drop descriptions shorter than this")
    parser.add_argument("--limit", type=int, default=2000, help="number of sources to keep (the paper uses the first 2,000, 0 for all)")
    args = parser.parse_args()

    raw_path = os.path.join(args.data_dir, "docci_descriptions.jsonlines")
    if os.path.exists(raw_path):
        print(f"found {raw_path}, skipping the download")
    else:
        download(DOCCI_URL, raw_path)

    sources = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            description = row["description"].strip()
            if len(description) < args.min_chars:
                continue
            sources.append({"source_id": row["example_id"], "docci_split": row.get("split"), "full_spec": description})
            if args.limit and len(sources) >= args.limit:
                break

    out_path = os.path.join(args.data_dir, "sources.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for source in sources:
            f.write(json.dumps(source, ensure_ascii=False) + "\n")
    print(f"wrote {len(sources)} sources to {out_path}")


if __name__ == "__main__":
    main()
