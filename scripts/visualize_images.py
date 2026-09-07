"""Image comparisons for the paper, one row per instance: the DOCCI reference, then the images a text-to-image model
made from the user request and from each agent's final output, the redaction as an arrow and the candidates set
apart by dividers:

    [ground truth] -> [user request] || [naive asker] || [fixed asker] (|| [open asker])

Under each image the questions asked and u_rec, plus the CLIP similarity to the reference when the output folder holds
a clip_scores.csv from calculate_clip_similarity.py. Images are read from <synthesis_dir>/images/
(generate_images_for_data.py) and <eval_dir>/images/ (generate_images_for_eval.py), a card reads "missing" where one
is absent.

Usage:
    python scripts/visualize_images.py --naive <eval_dir> --fixed <eval_dir> [--open <eval_dir>]
        [-n 5] [--seed 42] [--instances <id> ...] [--importance llm|equal] [--out <dir>]

The runs must share the instance set. Output: <instance_id>_figure.png/.pdf per instance and figure_all.png/.pdf
stacking them, in scripts/figures/k<k>[-thinking]_naive-<label>_fixed-<label>[_open-<label>]/ or --out.
"""
import argparse
import csv
import glob
import json
import os
import random
import textwrap

import matplotlib
import yaml
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
CARD_SIDE, MARGIN, GAP = 2.44, 0.25, 0.52  # card side in inches, the figure width follows the number of columns
DIVIDER_GAP = 0.36  # on each side of a divider line between two columns
TITLE_H, TOP, BOTTOM = 0.74, 0.12, 0.3
INK, MUTED, LINE, CARD = "#3c4043", "#80868b", "#dadce0", "#f1f3f4"

# agent columns in order: (key, title, note), the naive note takes the mode of its run
AGENTS = [
    ("naive-asking", "Naive Asker", "base model, {mode} mode"),
    ("fixed-asking", "Fixed Asker", "fine-tuned model, fixed-asking mode"),
    ("open-asking", "Open Asker", "fine-tuned model, open-asking mode"),
]
NAME_ORDER = ("naive-asking", "fixed-asking", "open-asking")  # the order the comparison folder names the agents in
# column of clip_scores.csv holding each image's similarity to the reference
CLIP_COLUMN = {"request": "sim_request", "naive-asking": "sim_naive", "fixed-asking": "sim_fixed", "open-asking": "sim_open"}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_eval(eval_dir):
    """(instances by id, episodes by id, judgments by id, config, synthesis directory) of one evaluation run."""
    with open(os.path.join(eval_dir, "config.yaml"), encoding="utf-8") as f:
        config = yaml.safe_load(f)
    instances_dir = config["evaluation"]["instances_dir"]
    if not os.path.isabs(instances_dir):
        instances_dir = os.path.join(ROOT, instances_dir)
    split = config["evaluation"]["split"]
    instances = {r["instance_id"]: r for r in read_jsonl(os.path.join(instances_dir, f"instances.{split}.jsonl"))}
    episodes = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "episodes.jsonl"))}
    judgments = {r["instance_id"]: r for r in read_jsonl(os.path.join(eval_dir, "judgments.jsonl"))}
    return instances, episodes, judgments, config, instances_dir


def run_step(config):
    """The checkpoint step of an evaluation run, or base."""
    checkpoint = config["model"].get("checkpoint")
    return checkpoint.rsplit("step-", 1)[-1].lstrip("0") if checkpoint else "base"


def run_label(config):
    """base for the base weights, else the user the checkpoint was trained against (patient, impatient), read from the
    training run holding the checkpoint (training/<model>/<agent>/<user>/...), the step number when no run does."""
    checkpoint = config["model"].get("checkpoint")
    if not checkpoint:
        return "base"
    root = os.path.join(ROOT, "results", config["task"]["name"], config["dataset"]["name"], "training")
    for path in glob.glob(os.path.join(root, *["*"] * 7, "checkpoints.jsonl")):
        if any(r.get("checkpoint") == checkpoint for r in read_jsonl(path)):
            return path.replace("\\", "/").split("/")[-6]
    return run_step(config)


def comparison_name(k, labels, thinking=False):
    """k<k>[-thinking]_naive-<label>_fixed-<label>[_open-<label>] for the agents in labels (keyed like AGENTS, named in
    NAME_ORDER, the labels from run_label), the tag when the runs' model thinks."""
    return f"k{k}{'-thinking' if thinking else ''}_" + \
        "_".join(f"{key.split('-')[0]}-{labels[key]}" for key in NAME_ORDER if key in labels)


def agent_mode(config):
    """The asking mode of an evaluation run named like the agents: fixed-asking or open-asking."""
    return "fixed-asking" if config["evaluation"]["agent"]["mode"] == "fixed" else "open-asking"


def load_image(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def plural(n, noun):
    return f"{n} {noun}{'' if n == 1 else 's'}"


def load_clip_scores(out_dir):
    """Rows of clip_scores.csv in the output folder by instance id, empty when calculate_clip_similarity.py has not run."""
    path = os.path.join(out_dir, "clip_scores.csv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {row["instance_id"]: row for row in csv.DictReader(f)}


def clip_note(scores, instance_id, key):
    """'CLIP = 0.81' from the clip scores, empty without one."""
    value = scores.get(instance_id, {}).get(CLIP_COLUMN[key])
    return f"CLIP = {float(value):.2f}" if value else ""


def subtitle(questions, *parts):
    """'<questions> (<part>, <part>)', the empty parts dropped and the bracket with them."""
    parts = [p for p in parts if p]
    return questions + (f" ({', '.join(parts)})" if parts else "")


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


def render(rows, out_base, separators, caption=None):
    """Rows of (images, titles, subtitles) as cards, neighbouring columns joined by the separators ("arrow", or
    "divider" with a full gap on each side of the line), an optional caption (a list of lines) under the last row.
    Writes png and pdf."""
    n, card = len(rows), CARD_SIDE
    frame = 0.035 * card  # the card frame reaches this far outside the axes
    xs = [MARGIN]
    for sep in separators:
        xs.append(xs[-1] + card + (2 * DIVIDER_GAP if sep == "divider" else GAP))
    fig_w = xs[-1] + card + MARGIN
    caption_h = 0.16 * len(caption) + 0.5 if caption else 0.0
    fig_h = TOP + TITLE_H + n * card + (n - 1) * GAP + caption_h + BOTTOM
    fig = plt.figure(figsize=(fig_w, fig_h))
    for i, (images, titles, subtitles) in enumerate(rows):
        y = fig_h - TOP - TITLE_H - i * (card + GAP) - card
        for x, image, title, sub in zip(xs, images, titles, subtitles):
            ax = fig.add_axes([x / fig_w, y / fig_h, card / fig_w, card / fig_h])
            draw_card(ax, image)
            if title:
                ax.set_title(title, fontsize=8.5, color=INK, fontweight="bold", pad=22)
            if sub:  # just under the frame, well clear of the next row
                ax.text(0.5, -0.035 - 0.05 / card, sub, ha="center", va="top", fontsize=7, color=MUTED,
                        transform=ax.transAxes)
        y_mid = (y + card / 2) / fig_h
        for k, sep in enumerate(separators):
            if sep == "arrow":
                fig.add_artist(FancyArrowPatch(((xs[k] + card + frame + 0.05) / fig_w, y_mid),
                                               ((xs[k + 1] - frame - 0.05) / fig_w, y_mid), transform=fig.transFigure,
                                               arrowstyle="-|>", mutation_scale=11, color=INK, linewidth=1.3))
    for k, sep in enumerate(separators):  # dividers span every row
        if sep == "divider":
            x = (xs[k] + card + DIVIDER_GAP) / fig_w
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
    parser.add_argument("--naive", required=True, help="evaluation run of the untrained asker (base weights, either mode)")
    parser.add_argument("--fixed", required=True, help="evaluation run of the trained fixed asker")
    parser.add_argument("--open", help="evaluation run of the trained open asker (column left out when absent)")
    parser.add_argument("-n", type=int, default=5, help="number of random instances (ignored with --instances)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", metavar="ID",
                        help="instance ids to render instead of a random sample, rows follow this order")
    parser.add_argument("--importance", choices=("llm", "equal"), default="llm", help="weighting shown under each image")
    parser.add_argument("--out", help="output directory (default scripts/figures/<comparison>)")
    args = parser.parse_args()

    eval_dirs = {key: path for key, path in zip(NAME_ORDER, (args.naive, args.fixed, args.open)) if path}
    agents = [a for a in AGENTS if a[0] in eval_dirs]
    evals = {key: load_eval(path) for key, path in eval_dirs.items()}
    instances, _, _, config, instances_dir = evals["naive-asking"]
    common = set.intersection(*(set(episodes) for _, episodes, _, _, _ in evals.values()))
    if not common:
        raise SystemExit("the evaluation runs share no instance, check that they use the same synthesis directory and split")
    if args.instances:  # hand-picked instances, in the order given
        missing = [i for i in args.instances if i not in common]
        if missing:
            raise SystemExit(f"not in every evaluation run: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = random.Random(args.seed).sample(sorted(common), min(args.n, len(common)))
    labels = {key: run_label(evals[key][3]) for key in eval_dirs}
    out_dir = args.out or os.path.join(SCRIPTS, "figures",
                                       comparison_name(config["general"]["k"], labels, config["model"].get("thinking", False)))
    os.makedirs(out_dir, exist_ok=True)
    clip_scores = load_clip_scores(out_dir)
    data_images = os.path.join(instances_dir, "images")
    titles = ["Ground Truth\n(DOCCI reference image)", "User Request\n(eroded prompt, no elicitation)"] + \
        [f"{title}\n({note.format(mode=agent_mode(evals[key][3]))})" for key, title, note in agents]
    separators = ["arrow"] + ["divider"] * len(agents)  # the redaction, then the candidates set apart as peers

    rows = []
    for instance_id in chosen:
        reference = load_image(os.path.join(data_images, f"{instance_id}_enhanced.jpg")) \
            or load_image(os.path.join(data_images, f"{instance_id}_original.jpg"))
        images = [reference, load_image(os.path.join(data_images, f"{instance_id}_request.jpg"))]
        subtitles = [plural(len(instances[instance_id]["redacted_ids"]), "detail") + " redacted",
                     subtitle("no questions", clip_note(clip_scores, instance_id, "request"))]
        for key, _, _ in agents:
            _, episodes, judgments, _, _ = evals[key]
            images.append(load_image(os.path.join(eval_dirs[key], "images", f"{instance_id}.jpg")))
            subtitles.append(subtitle(plural(episodes[instance_id]["n_questions"], "question"),
                                      f"utility recovered = {judgments[instance_id]['u_rec'][args.importance]:.2f}",
                                      clip_note(clip_scores, instance_id, key)))
        if any(image is None for image in images):
            print(f"images missing for {instance_id} (run scripts/generate_images_for_data.py and _for_eval.py first)")
        caption = textwrap.wrap(f"User request (eroded): {instances[instance_id]['eroded_prompt']}", 150 + 20 * len(agents))
        render([(images, titles, subtitles)], os.path.join(out_dir, f"{instance_id}_figure"), separators, caption)
        rows.append((images, subtitles))

    if rows:
        render([(images, titles if i == 0 else [""] * len(titles), subtitles) for i, (images, subtitles) in enumerate(rows)],
               os.path.join(out_dir, "figure_all"), separators)


if __name__ == "__main__":
    main()
