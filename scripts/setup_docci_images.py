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

from utils import DOCCI_DIR, download

TARBALL_URL = "https://storage.googleapis.com/docci/data/docci_images.tar.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DOCCI_DIR)
    parser.add_argument("--ids", nargs="*", help="source ids to extract (default: every source in sources.jsonl)")
    parser.add_argument("--all", action="store_true", help="extract every image in the tarball")
    args = parser.parse_args()

    tarball = os.path.join(args.data_dir, "docci_images.tar.gz")
    images_dir = os.path.join(args.data_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    if not os.path.exists(tarball):
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
