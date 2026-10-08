"""Shared by the scripts: repository paths, the shipped instances and paper runs, JSONL and image loading, evaluation
runs with or without their config.yaml, the Gemini image model, resumable downloads and the picture cards of the
figures."""
import csv
import glob
import io
import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

import matplotlib
import requests
import yaml
from PIL import Image
from tqdm import tqdm

matplotlib.use("Agg")
matplotlib.rcParams.update({"font.family": ["Roboto", "Segoe UI", "DejaVu Sans"], "pdf.fonttype": 42})
from matplotlib.patches import FancyBboxPatch  # noqa: E402

# Paths
SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
FIGURES = os.path.join(SCRIPTS, "figures")  # where the figure scripts write by default
DOCCI_DIR = os.path.join(ROOT, "data", "docci")  # the DOCCI descriptions and reference images (setup_docci*.py)
INSTANCES_DIR = os.path.join(ROOT, "data", "docci_instances")  # the paper's instances and their images
PAPER_RESULTS = os.path.join(ROOT, "paper_results", "qwen3-8b-thinking")  # the paper's evaluation runs, synced from S3 (see README)
WEIGHTS = {"base_model": "base", "patient_trained": "patient", "impatient_trained": "impatient"}  # their folder names to run labels

# Colors shared by the figures
INK, MUTED, LINE = "#3c4043", "#80868b", "#dadce0"
CAPTION_BG, USER_BG, AGENT_BG = "#f8f9fa", "#f1f3f4", "#e8f0fe"
DELIVERED, MISSED = "#188038", "#c5221f"


def paper_run(agent, user, weights):
    """The paper's evaluation run of an agent under a user with the given weights (a WEIGHTS folder name)."""
    return os.path.join(PAPER_RESULTS, agent, user, "k_15", "test", weights)


def paper_runs():
    """Every evaluation run under paper_results/."""
    return sorted(glob.glob(os.path.join(PAPER_RESULTS, "*", "*", "k_15", "test", "*")))


EXAMPLE_RUN = paper_run("open-asking", "patient", "patient_trained")  # the run behind the paper's example episode


# Files


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_image(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def download(url, dest):
    """Stream url to dest with a progress bar, resuming a partial download left in dest.part."""
    dest = str(dest)
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    part = f"{dest}.part"
    done = os.path.getsize(part) if os.path.exists(part) else 0
    with requests.get(url, stream=True, timeout=60, headers={"Range": f"bytes={done}-"} if done else {}) as response:
        response.raise_for_status()
        if done and response.status_code != 206:  # the server sent the whole file, start over
            done = 0
        total = done + int(response.headers.get("content-length", 0))
        with open(part, "ab" if done else "wb") as f, \
                tqdm(total=total, initial=done, unit="B", unit_scale=True, desc=os.path.basename(dest)) as bar:
            for chunk in response.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    os.replace(part, dest)


# Evaluation runs


def load_config(eval_dir):
    """The run's config.yaml, or its equivalent for a run shipped without one (paper_results/): agent, mode, user and
    split from summary.csv, the instances from data/docci_instances, the weights label from the folder name."""
    path = os.path.join(eval_dir, "config.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    with open(os.path.join(eval_dir, "summary.csv"), encoding="utf-8") as f:
        summary = next(iter(csv.DictReader(f)))
    parts = os.path.normpath(os.path.abspath(eval_dir)).split(os.sep)
    label = WEIGHTS.get(parts[-1], parts[-1])
    k = next((int(p[2:]) for p in parts if re.fullmatch(r"k_\d+", p)), 0)
    agent = {"name": summary["agent"], "mode": summary["mode"]}
    return {"general": {"k": k}, "agent": agent, "user": {"name": summary["user"]}, "run_label": label,
            "model": {"checkpoint": None if label == "base" else label, "thinking": any("thinking" in p for p in parts)},
            "evaluation": {"instances_dir": INSTANCES_DIR, "split": summary["split"], "agent": agent}}


def load_eval(eval_dir):
    """(instances by id, episodes by id, judgments by id, config, synthesis directory) of one evaluation run."""
    config = load_config(eval_dir)
    instances_dir = config["evaluation"]["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(ROOT, instances_dir)
    split = config["evaluation"]["split"]
    instances = {r["instance_id"]: r for r in read_jsonl(os.path.join(instances_dir, f"instances.{split}.jsonl"))}
    if not config["general"].get("k"):
        config["general"]["k"] = max(len(r["redacted_ids"]) for r in instances.values())
    episodes = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "episodes.jsonl"))}
    judgments = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "judgments.jsonl"))}
    return instances, episodes, judgments, config, instances_dir


def run_step(config):
    """The checkpoint step of an evaluation run, or base."""
    checkpoint = config["model"].get("checkpoint")
    return checkpoint.rsplit("step-", 1)[-1].lstrip("0") if checkpoint else "base"


def run_label(config):
    """base, else the user the checkpoint was trained against (patient, impatient) read from the training run that
    holds it (training/<model>/<agent>/<user>/...), the step number when none does."""
    if "run_label" in config:
        return config["run_label"]
    checkpoint = config["model"].get("checkpoint")
    if not checkpoint:
        return "base"
    root = os.path.join(ROOT, "results", config["task"]["name"], config["dataset"]["name"], "training")
    for path in glob.glob(os.path.join(root, *["*"] * 7, "checkpoints.jsonl")):
        if any(r.get("checkpoint") == checkpoint for r in read_jsonl(path)):
            return path.replace("\\", "/").split("/")[-6]
    return run_step(config)


# Gemini image generation (generate_images_for_data.py and generate_images_for_eval.py)

MODEL = "gemini-3.1-flash-image-preview"  # defaults, overridable through the arguments add_model_args adds
MAX_RETRIES = 3
ASPECT_RATIO = "1:1"
RESOLUTION = "512"
WORKERS = 64  # concurrent Gemini calls


class GeminiImageModel:
    """Gemini image generation (text-to-image, or editing when an input image is given)."""

    def __init__(self, model: str, max_retries: int, aspect_ratio: str, resolution: str):
        from dotenv import load_dotenv
        from google.genai import Client, types

        load_dotenv(os.path.join(ROOT, ".env"))
        self.model = model
        self.max_retries = max_retries
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
        self.client = Client()
        self.types = types
        logging.getLogger("google_genai").setLevel(logging.WARNING)

    def generate(self, prompt: str, image_bytes: bytes = None):
        """Text-to-image, or an edit when image_bytes is given."""
        prompt += "\nReturn a single image as output, do not give any options."
        contents = [prompt]
        if image_bytes:
            contents.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))

        response = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=contents,
                    config=self.types.GenerateContentConfig(
                        response_modalities=["Image"],
                        image_config=self.types.ImageConfig(aspect_ratio=self.aspect_ratio, image_size=self.resolution),
                    ),
                )
                if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    break
                logging.warning("Gemini API returned an invalid response structure (attempt %d/%d)", attempt + 1, self.max_retries)
                response = None
            except Exception as e:  # noqa: BLE001
                logging.warning("Gemini API call failed (attempt %d/%d): %s", attempt + 1, self.max_retries, e)
                response = None
        if not response or not response.candidates or not response.candidates[0].content or not response.candidates[0].content.parts:
            logging.error("Gemini API call failed: no valid response after all retries.")
            return None

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                return Image.open(io.BytesIO(part.inline_data.data)).convert("RGB")
        logging.error("Gemini API response carried no image.")
        return None


def add_model_args(parser):
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO)
    parser.add_argument("--resolution", default=RESOLUTION)
    parser.add_argument("--workers", type=int, default=WORKERS, help="concurrent Gemini calls")


def model_from_args(args):
    return GeminiImageModel(args.model, args.max_retries, args.aspect_ratio, args.resolution)


def ensure(path, make):
    """Write make()'s image to path unless it exists. True when a file was written."""
    if os.path.exists(path):
        return False
    image = make()
    if image is None:
        print(f"failed {path}")
        return False
    image.save(path, quality=92)
    print(f"generated {path}", flush=True)
    return True


def run_jobs(jobs, workers):
    """Run (path, make) jobs through ensure, workers at a time. Returns how many were written."""
    if workers <= 1:
        return sum(ensure(path, make) for path, make in jobs)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return sum(pool.map(lambda job: ensure(*job), jobs))


# Picture cards of the figures (visualize_images.py and visualize_instance.py)

CARD = "#f1f3f4"


def squared(image):
    """The image letterboxed on white to a square."""
    w, h = image.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def draw_card(ax, image):
    """The image inside a rounded card with a soft grey frame, "missing" when there is none."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.035,rounding_size=0.09", facecolor=CARD,
                                edgecolor=LINE, linewidth=1.0, clip_on=False, zorder=1))
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color=MUTED, zorder=3)
        return
    shown = ax.imshow(squared(image), extent=(0, 1, 0, 1), zorder=2)
    shown.set_clip_path(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07", transform=ax.transData))
