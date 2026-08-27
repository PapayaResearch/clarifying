"""Image-level comparison figures: the DOCCI reference beside the images a text-to-image model generates from the
eroded request and from each agent's final output, one row per instance.

Usage:
    python scripts/visualize_images.py --naive <eval_dir> --fixed <eval_dir> --open <eval_dir>
        [-n 5] [--seed 42] [--instances <id> ...] [--out visuals_images]

The three evaluation runs must share the instance set. Reference images come from data/docci/images/ (see
scripts/setup_docci_images.py). Every image is saved in the output directory and reused on reruns.
"""
import argparse
import io
import json
import logging
import os
import random
import textwrap

from dotenv import load_dotenv

load_dotenv()

import matplotlib  # noqa: E402
import yaml  # noqa: E402
from google.genai import Client, types  # noqa: E402
from PIL import Image  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# Image model defaults, override with the command-line arguments
MODEL = "gemini-3.1-flash-image-preview"
MAX_RETRIES = 3
ASPECT_RATIO = "1:1"
RESOLUTION = "512"
# Applied once to the reference image
ENHANCEMENT_PROMPT = """Upscale the image and enhance its quality.
Do not make any other changes to the image itself."""

FIG_W, MARGIN, GAP = 15.0, 0.25, 0.52
DIVIDER_GAP = 0.36  # on each side of the divider between the reference and the generated images
TITLE_H, TOP, BOTTOM = 0.74, 0.12, 0.3

# agent columns: (key, title, note), the request column is generated from the instance itself
AGENTS = [
    ("naive-asking", "Naive Asker", "base model, open-asking mode"),
    ("fixed-asking", "Fixed Asker", "fine-tuned model, fixed-asking mode"),
    ("open-asking", "Open Asker", "fine-tuned model, open-asking mode"),
]
INK, MUTED, LINE, CARD = "#3c4043", "#80868b", "#dadce0", "#f1f3f4"


class GeminiImageModel:
    """Gemini image generation (text-to-image, or editing when an input image is given)."""

    def __init__(self, model: str, max_retries: int, aspect_ratio: str, resolution: str):
        self.model = model
        self.max_retries = max_retries
        self.aspect_ratio = aspect_ratio
        self.resolution = resolution
        self.client = Client()
        logging.getLogger("google_genai").setLevel(logging.WARNING)

    def generate(self, prompt: str, image_bytes: bytes = None):
        """Text-to-image, or an edit of image_bytes, always with the same aspect ratio and size."""
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
                    config=types.GenerateContentConfig(
                        response_modalities=["Image"],
                        image_config=types.ImageConfig(aspect_ratio=self.aspect_ratio, image_size=self.resolution),
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
            return None, None

        for part in response.candidates[0].content.parts:
            if part.inline_data:
                data = part.inline_data.data
                return Image.open(io.BytesIO(data)).convert("RGB"), data
        logging.error("Gemini API response carried no image.")
        return None, None


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_eval(eval_dir):
    """(instances by id, episodes by id, judgments by id, summary row) of one evaluation run."""
    with open(os.path.join(eval_dir, "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instances_dir = config["evaluation"]["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), instances_dir)
    split = config["evaluation"]["split"]
    instances = {r["instance_id"]: r for r in read_jsonl(os.path.join(instances_dir, f"instances.{split}.jsonl"))}
    episodes = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "episodes.jsonl"))}
    judgments = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "judgments.jsonl"))}
    return instances, episodes, judgments


def plural(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def metrics_text(episode, judgment, importance):
    return f"{plural(episode['n_questions'], 'question')} (utility recovered = {judgment['u_rec'][importance]:.2f})"


def squared(image):
    """The image letterboxed on white to a square, so every card has the same shape."""
    w, h = image.size
    if w == h:
        return image
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - w) // 2, (side - h) // 2))
    return canvas


def draw_card(ax, image):
    """The image inside a rounded card with a soft grey frame."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    frame = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.035,rounding_size=0.09", facecolor=CARD,
                           edgecolor=LINE, linewidth=1.0, clip_on=False, zorder=1)
    ax.add_patch(frame)
    if image is None:
        ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=8, color=MUTED, zorder=3)
        return
    shown = ax.imshow(squared(image), extent=(0, 1, 0, 1), zorder=2)
    clip = FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.07", transform=ax.transData)
    shown.set_clip_path(clip)


def render(rows, out_base, caption=None):
    """Rows of five cards (reference, then the four generated images), the reference set apart by a divider with a
    full gap on each side, headings above the first row, an optional caption under the last one. Writes png and pdf.

    rows: list of (images, titles, subtitles), caption: list of text lines."""
    n = len(rows)
    card = (FIG_W - 2 * MARGIN - 3 * GAP - 2 * DIVIDER_GAP) / 5
    frame = 0.035 * card  # the card frame reaches this far outside the axes
    caption_h = 0.16 * len(caption) + 0.5 if caption else 0.0  # leaves room for the captions under the cards
    fig_h = TOP + TITLE_H + n * card + (n - 1) * GAP + caption_h + BOTTOM
    fig = plt.figure(figsize=(FIG_W, fig_h))
    xs = [MARGIN] + [MARGIN + card + 2 * DIVIDER_GAP + k * (card + GAP) for k in range(4)]
    for i, (images, titles, subtitles) in enumerate(rows):
        y = fig_h - TOP - TITLE_H - i * (card + GAP) - card
        for x, image, title, subtitle in zip(xs, images, titles, subtitles):
            ax = fig.add_axes([x / FIG_W, y / fig_h, card / FIG_W, card / fig_h])
            draw_card(ax, image)
            if title:
                ax.set_title(title, fontsize=8.5, color=INK, fontweight="bold", pad=22)
            if subtitle:  # just under the frame, well clear of the next row
                ax.text(0.5, -0.035 - 0.05 / card, subtitle, ha="center", va="top", fontsize=7, color=MUTED,
                        transform=ax.transAxes)
        y_mid = (y + card / 2) / fig_h
        for k in range(1, 4):  # arrows between the generated images, none after the reference
            fig.add_artist(FancyArrowPatch(((xs[k] + card + frame + 0.05) / FIG_W, y_mid),
                                           ((xs[k + 1] - frame - 0.05) / FIG_W, y_mid), transform=fig.transFigure,
                                           arrowstyle="-|>", mutation_scale=11, color=INK, linewidth=1.3))
    x = (MARGIN + card + DIVIDER_GAP) / FIG_W
    fig.add_artist(plt.Line2D([x, x], [(BOTTOM + caption_h) / fig_h, (fig_h - TOP) / fig_h], color=LINE,
                              linewidth=1.2, transform=fig.transFigure))
    if caption:
        fig.text(0.5, (BOTTOM + caption_h - 0.45) / fig_h, "\n".join(caption), ha="center", va="top", fontsize=7,
                 color=MUTED)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out_base}.{ext}", dpi=200, facecolor="white")
    plt.close(fig)
    print(f"saved {out_base}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--naive", required=True, help="evaluation run of the untrained asker (open-asking, base weights)")
    parser.add_argument("--fixed", required=True, help="evaluation run of the trained fixed asker")
    parser.add_argument("--open", required=True, help="evaluation run of the trained open asker")
    parser.add_argument("-n", type=int, default=5, help="number of random instances (ignored with --instances)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", metavar="ID",
                        help="instance ids to render instead of a random sample, rows follow this order")
    parser.add_argument("--out", default="visuals_images", help="output directory for images and figures")
    parser.add_argument("--images-dir", default="data/docci/images", help="DOCCI reference images, <source_id>.jpg")
    parser.add_argument("--importance", choices=("llm", "equal"), default="llm", help="weighting shown under each image")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    parser.add_argument("--aspect-ratio", default=ASPECT_RATIO)
    parser.add_argument("--resolution", default=RESOLUTION)
    args = parser.parse_args()

    eval_dirs = {"naive-asking": args.naive, "fixed-asking": args.fixed, "open-asking": args.open}
    evals = {key: load_eval(path) for key, path in eval_dirs.items()}
    instances = evals["naive-asking"][0]
    common = set.intersection(*(set(ep) for _, ep, _ in evals.values()))
    if not common:
        raise SystemExit("the three evaluation runs share no instance, check that they use the same synthesis directory and split")
    if args.instances:  # hand-picked instances, in the order given
        missing = [i for i in args.instances if i not in common]
        if missing:
            raise SystemExit(f"not in every evaluation run: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = random.Random(args.seed).sample(sorted(common), min(args.n, len(common)))
    os.makedirs(args.out, exist_ok=True)
    model = GeminiImageModel(args.model, args.max_retries, args.aspect_ratio, args.resolution)

    rows = []
    for instance_id in chosen:
        instance = instances[instance_id]
        source_id = instance.get("source_id") or instance_id.split(".")[0]
        reference_path = os.path.join(args.out, f"{instance_id}_original.jpg")
        if not os.path.exists(reference_path):
            src = os.path.join(args.images_dir, f"{source_id}.jpg")
            if os.path.exists(src):
                Image.open(src).convert("RGB").save(reference_path, quality=92)
            else:
                print(f"reference image missing for {source_id} (run scripts/setup_docci_images.py)")
        # the reference is enhanced once through the same call as the generations, the figure shows that copy
        enhanced_path = os.path.join(args.out, f"{instance_id}_enhanced.jpg")
        if os.path.exists(reference_path) and not os.path.exists(enhanced_path):
            with open(reference_path, "rb") as f:
                image, _ = model.generate(ENHANCEMENT_PROMPT, image_bytes=f.read())
            if image is not None:
                image.save(enhanced_path, quality=92)
                print(f"enhanced {enhanced_path}")
        shown_path = enhanced_path if os.path.exists(enhanced_path) else reference_path
        reference = Image.open(shown_path) if os.path.exists(shown_path) else None

        # (column key, prompt for the image model, subtitle under the image)
        columns = [("request", instance["eroded_prompt"], "no questions")]
        for key, _, _ in AGENTS:
            _, episodes, judgments = evals[key]
            columns.append((key, episodes[instance_id]["final_output"],
                            metrics_text(episodes[instance_id], judgments[instance_id], args.importance)))
        images, subtitles = [reference], [plural(len(instance["redacted_ids"]), "detail") + " redacted"]
        for key, prompt, subtitle in columns:
            path = os.path.join(args.out, f"{instance_id}_{key}-edit.jpg")
            if not os.path.exists(path):
                image, _ = model.generate(prompt)
                if image is not None:
                    image.save(path, quality=92)
                    print(f"generated {path}")
            images.append(Image.open(path) if os.path.exists(path) else None)
            subtitles.append(subtitle)
        titles = ["Ground Truth\n(DOCCI reference image)", "User Request\n(eroded prompt, no elicitation)"] + \
            [f"{label}\n({note})" for _, label, note in AGENTS]
        rows.append((images, titles, subtitles))
        caption = textwrap.wrap(f"User request (eroded): {instance['eroded_prompt']}", 170)
        render([(images, titles, subtitles)], os.path.join(args.out, f"{instance_id}_figure"), caption)

    if rows:
        render([(images, titles if i == 0 else [""] * 5, subtitles) for i, (images, titles, subtitles) in enumerate(rows)],
               os.path.join(args.out, "figure_all"))


if __name__ == "__main__":
    main()
