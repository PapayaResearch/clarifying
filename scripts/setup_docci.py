"""Download DOCCI descriptions and normalize them into data/docci/sources.jsonl.

DOCCI (https://google.github.io/docci/) ships descriptions as a standalone jsonlines
file - no images needed for the text-only pipeline.
"""

import argparse
import json
from pathlib import Path

import requests
from tqdm import tqdm

DOCCI_URL = "https://storage.googleapis.com/docci/data/docci_descriptions.jsonlines"


def download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def main(args):
    raw_path = Path(args.data_dir) / "docci_descriptions.jsonlines"
    if not raw_path.exists():
        print(f"Downloading DOCCI descriptions to {raw_path} ...")
        download(DOCCI_URL, raw_path)
    else:
        print(f"Found existing {raw_path}, skipping download.")

    sources = []
    with open(raw_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            description = row["description"].strip()
            if len(description) < args.min_chars:
                continue
            sources.append(
                {
                    "source_id": row["example_id"],
                    "docci_split": row.get("split"),
                    "full_spec": description,
                }
            )
            if args.limit and len(sources) >= args.limit:
                break

    out_path = Path(args.data_dir) / "sources.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for source in sources:
            f.write(json.dumps(source, ensure_ascii=False) + "\n")
    print(f"Wrote {len(sources)} sources to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/docci")
    parser.add_argument("--min-chars", type=int, default=400, help="Drop descriptions shorter than this")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of sources (for quick runs)")
    main(parser.parse_args())
