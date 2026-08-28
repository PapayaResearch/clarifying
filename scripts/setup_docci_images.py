"""Fetch the DOCCI images for the sources in data/docci/sources.jsonl.

Downloads docci_images.tar.gz (7.6 GB) into data/docci/ once unless it is already there, then extracts only the
images of the listed sources into data/docci/images/<source_id>.jpg (the tarball is read in one streaming pass).

Usage:
    python scripts/setup_docci_images.py [--data-dir data/docci] [--ids qual_test_00064 ...] [--all]
"""
import argparse
import json
import os
import tarfile

import requests

TARBALL_URL = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"


def download(url, dest):
    tmp = dest + ".part"
    done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        response.raise_for_status()
        total = done + int(response.headers.get("content-length", 0))
        with open(tmp, "ab") as f:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done % (256 << 20) < (1 << 20):
                    print(f"  {done / 1e9:.2f} of {total / 1e9:.2f} GB", flush=True)
    os.replace(tmp, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data/docci")
    parser.add_argument("--ids", nargs="*", help="source ids to extract (default: every source in sources.jsonl)")
    parser.add_argument("--all", action="store_true", help="extract every image in the tarball")
    args = parser.parse_args()

    tarball = os.path.join(args.data_dir, "docci_images.tar.gz")
    images_dir = os.path.join(args.data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    if not os.path.exists(tarball):
        print(f"downloading {TARBALL_URL} to {tarball}")
        download(TARBALL_URL, tarball)

    if args.all:
        wanted = None
    elif args.ids:
        wanted = set(args.ids)
    else:
        with open(os.path.join(args.data_dir, "sources.jsonl"), encoding="utf-8") as f:
            wanted = {json.loads(line)["source_id"] for line in f}
    missing = None if wanted is None else {i for i in wanted if not os.path.exists(os.path.join(images_dir, f"{i}.jpg"))}
    if missing is not None and not missing:
        print(f"all {len(wanted)} images already in {images_dir}")
        return

    extracted = 0
    with tarfile.open(tarball, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            stem, ext = os.path.splitext(os.path.basename(member.name))
            if missing is not None and stem not in missing:
                continue
            with tar.extractfile(member) as src, open(os.path.join(images_dir, f"{stem}{ext.lower()}"), "wb") as dst:
                dst.write(src.read())
            extracted += 1
            if missing is not None:
                missing.discard(stem)
                if not missing:
                    break
    print(f"extracted {extracted} images to {images_dir}" + (f", {len(missing)} not found" if missing else ""))


if __name__ == "__main__":
    main()
