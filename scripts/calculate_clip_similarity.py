"""Does recovering details make the rendered image closer to what the user wanted? CLIP similarity between each
instance's DOCCI reference image and the images generated from the user's request and from the agents' final outputs.

Two measures per candidate: image-image, the cosine similarity of its embedding to the reference's, and text-image,
its mean similarity to the texts of the instance's redacted details (does the hidden information show up). The
reference is <synthesis_dir>/images/<id>_enhanced.jpg (_original.jpg when missing), the candidates are that folder's
_request.jpg and <eval_dir>/images/<id>.jpg of every run given, all written by the two generate_images_for_*.py
scripts. Instances missing any image are skipped and counted.

Usage:
    python scripts/calculate_clip_similarity.py --fixed <eval_dir> [--naive <eval_dir>] [--no-asking <eval_dir>]
        [--open <eval_dir>] [--instances <id> ...] [--batch-size 64] [--workers 64] [--out <dir>]

Writes clip_scores.csv, clip_summary.md and clip_scores.png into --out, by default the same
scripts/figures/<comparison>/ folder visualize_images.py uses for the same runs.
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# transformers imports torchvision whenever it can find one, and the user-level site-packages hold one built for
# another torch, so make it unfindable. The CLIP processor works without it.
sys.modules["torchvision"] = None

from dotenv import load_dotenv  # noqa: E402

load_dotenv()  # HF_TOKEN, optional

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import stats  # noqa: E402
from transformers import AutoModel, AutoProcessor  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from visualize_images import comparison_name, load_eval, run_step  # noqa: E402

MODEL = "openai/clip-vit-base-patch32"
BATCH_SIZE = 64
WORKERS = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOOTSTRAP = 10_000
INK, MUTED = "#3c4043", "#80868b"
PALETTE = {"request": "#9aa0a6", "no-asking": "#c5a06b", "naive": "#e8a33d", "fixed": "#1a73e8", "open": "#34a853"}

# Candidate columns in order, the agents among them coming from the runs given
CANDIDATES = [
    ("request", "user request"),
    ("no-asking", "no-asking (base)"),
    ("naive", "naive asker (base, open-asking)"),
    ("fixed", "fixed asker (fine-tuned)"),
    ("open", "open asker (fine-tuned)"),
]
AGENT_KEYS = {"no-asking": "no-asking", "naive": "naive-asking", "fixed": "fixed-asking", "open": "open-asking"}


def load_image(path):
    return Image.open(path).convert("RGB") if os.path.exists(path) else None


def features(outputs):
    return outputs if isinstance(outputs, torch.Tensor) else outputs.pooler_output


def encode_images(processor, model, images, batch_size):
    """L2-normalized CLIP image embeddings, one row per image."""
    feats = []
    for i in range(0, len(images), batch_size):
        inputs = processor(images=images[i:i + batch_size], return_tensors="pt")
        with torch.no_grad():
            outputs = model.get_image_features(**{key: value.to(DEVICE) for key, value in inputs.items()})
        feats.append(F.normalize(features(outputs), dim=-1).cpu())
    return torch.cat(feats).numpy()


def encode_texts(processor, model, texts, batch_size):
    """L2-normalized CLIP text embeddings, one row per text (truncated to CLIP's 77 tokens)."""
    feats = []
    for i in range(0, len(texts), batch_size):
        inputs = processor(text=texts[i:i + batch_size], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model.get_text_features(**{key: value.to(DEVICE) for key, value in inputs.items()})
        feats.append(F.normalize(features(outputs), dim=-1).cpu())
    return torch.cat(feats).numpy()


def bootstrap_ci(values, rng, n=BOOTSTRAP):
    """95% bootstrap interval of the mean."""
    means = rng.choice(values, size=(n, len(values)), replace=True).mean(axis=1)
    return np.percentile(means, 2.5), np.percentile(means, 97.5)


def paired(a, b, rng):
    """Mean of a − b with its bootstrap interval, the share of instances where a beats b, Wilcoxon p."""
    diff = a - b
    lo, hi = bootstrap_ci(diff, rng)
    p = stats.wilcoxon(a, b).pvalue if np.any(diff != 0) else 1.0
    return diff.mean(), lo, hi, (diff > 0).mean(), p


def score_table(sims, keys, labels, u_rec, questions, what):
    """Per-candidate means and paired comparisons of one measure, as markdown lines."""
    rng = np.random.default_rng(0)
    lines = ["| candidate | mean | sd | median | u_rec (llm) | questions |", "|---|---|---|---|---|---|"]
    for key in keys:
        s = sims[key]
        u = f"{u_rec[key].mean():.3f}" if key in u_rec else ""
        q = f"{questions[key].mean():.2f}" if key in questions else ""
        lines.append(f"| {labels[key]} | {s.mean():.3f} | {s.std(ddof=1):.3f} | {np.median(s):.3f} | {u} | {q} |")
    lines += ["", f"Paired differences (positive means the first is closer to {what}), 95% bootstrap interval of "
              "the mean, share of instances where the first wins, Wilcoxon signed-rank p:", "",
              "| comparison | mean diff | 95% CI | wins | p |", "|---|---|---|---|---|"]
    for a, b in [(a, b) for a in keys for b in keys if a != b and keys.index(a) > keys.index(b)]:
        m, lo, hi, wins, p = paired(sims[a], sims[b], rng)
        lines.append(f"| {labels[a]} vs {labels[b]} | {m:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {wins:.0%} | {p:.2g} |")
    if "request" in sims:
        lines += ["", "Spearman correlation on instances between an agent's u_rec and its gain over the request image:",
                  "", "| agent | rho | p |", "|---|---|---|"]
        for key in keys:
            if key in u_rec:
                rho, p = stats.spearmanr(u_rec[key], sims[key] - sims["request"])
                lines.append(f"| {labels[key]} | {rho:+.2f} | {p:.2g} |")
    return lines


def summary_lines(sims, detail_sims, keys, labels, u_rec, questions, n_skipped, ceiling, model_name):
    """The markdown summary: image-image first, then text-image."""
    n = len(next(iter(sims.values())))
    lines = [f"# CLIP similarity ({model_name}, {n} instances"
             + (f", {n_skipped} skipped for missing images" if n_skipped else "") + ")", "",
             "## Image-image: cosine similarity of each generated image to the reference image", ""]
    if ceiling is not None:
        lines += [f"Ceiling: the original DOCCI image against its enhanced copy scores {ceiling.mean():.3f} "
                  f"(sd {ceiling.std(ddof=1):.3f}), two renderings of the same scene.", ""]
    lines += score_table(sims, keys, labels, u_rec, questions, "the reference image")
    lines += ["", "## Text-image: cosine similarity of each generated image to the instance's redacted details", "",
              "Each instance's score is the mean over its redacted details of the CLIP similarity between the detail's "
              "text and the image, so it asks whether the hidden details show up in the picture rather than whether "
              "the scene as a whole matches. The reference image is scored the same way as the ceiling.", ""]
    lines += score_table(detail_sims, ["reference"] + keys, {**labels, "reference": "reference image"}, u_rec,
                         questions, "the redacted details")
    return lines


def distribution_panel(ax, sims, keys, labels, ylabel, title):
    """Box and strip plot per candidate, the mean as a bar and in the tick label."""
    rng = np.random.default_rng(1)
    for i, key in enumerate(keys):
        s = sims[key]
        ax.boxplot([s], positions=[i], widths=0.5, showfliers=False, medianprops={"color": INK},
                   boxprops={"color": INK}, whiskerprops={"color": INK}, capprops={"color": INK})
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(s)), s, s=9, color=PALETTE.get(key, INK), alpha=0.55, zorder=3)
        ax.plot([i - 0.28, i + 0.28], [s.mean(), s.mean()], color=PALETTE.get(key, INK), linewidth=2, zorder=4)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([f"{labels[k].replace(' (', chr(10) + '(')}\nmean {sims[k].mean():.3f}" for k in keys],
                       fontsize=7.5)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK)
    ax.grid(axis="y", color="#e8eaed")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def paired_panel(fig, ax, x, y, color, xlabel, ylabel, title, colorbar_label):
    """Scatter of a candidate against the request image, colored by u_rec, the diagonal dashed."""
    sc = ax.scatter(x, y, c=color, cmap="viridis", s=18, alpha=0.85, vmin=0, vmax=1)
    lo, hi = min(x.min(), y.min()) - 0.02, max(x.max(), y.max()) + 0.02
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1, linestyle="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9.5, color=INK)
    if color is not None:
        fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.03).set_label(colorbar_label, fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot(sims, detail_sims, keys, labels, u_rec, out_path):
    """One row per measure: the distribution per candidate, then the fixed asker against the request per instance."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6), gridspec_kw={"width_ratios": [1.35, 1]})
    color = u_rec["fixed"] if "fixed" in u_rec else None
    distribution_panel(axes[0, 0], sims, keys, labels, "CLIP similarity to the reference image",
                       "Image-image: similarity to the reference (mean bar, box = quartiles)")
    distribution_panel(axes[1, 0], detail_sims, ["reference"] + keys, {**labels, "reference": "reference image"},
                       "CLIP similarity to the redacted details", "Text-image: similarity to the redacted details")
    if "request" in sims and "fixed" in sims:
        paired_panel(fig, axes[0, 1], sims["request"], sims["fixed"], color, "request image",
                     "fixed asker's image", f"Fixed asker above the request in {(sims['fixed'] > sims['request']).mean():.0%}"
                     " of instances", "u_rec of the fixed asker (llm)")
        paired_panel(fig, axes[1, 1], detail_sims["request"], detail_sims["fixed"], color, "request image",
                     "fixed asker's image",
                     f"Fixed asker above the request in {(detail_sims['fixed'] > detail_sims['request']).mean():.0%} of instances",
                     "u_rec of the fixed asker (llm)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixed", required=True, help="evaluation run of the trained fixed asker")
    parser.add_argument("--naive", help="evaluation run of the untrained asker (open-asking, base weights)")
    parser.add_argument("--no-asking", dest="no_asking", help="evaluation run of the no-asking agent")
    parser.add_argument("--open", help="evaluation run of the trained open asker")
    parser.add_argument("--instances", nargs="+", metavar="ID", help="instance ids instead of every shared instance")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS, help="threads loading and preprocessing images")
    parser.add_argument("--out", help="output directory (default scripts/figures/<comparison>)")
    args = parser.parse_args()

    eval_dirs = {key: path for key, path in
                 (("no-asking", args.no_asking), ("naive", args.naive), ("fixed", args.fixed), ("open", args.open)) if path}
    evals = {key: load_eval(path) for key, path in eval_dirs.items()}
    instances, _, _, config, instances_dir = evals["fixed"]
    common = set.intersection(*(set(episodes) for _, episodes, _, _, _ in evals.values()))
    if args.instances:
        missing = [i for i in args.instances if i not in common]
        if missing:
            raise SystemExit(f"not in every evaluation run: {', '.join(missing)}")
        chosen = args.instances
    else:
        chosen = sorted(common)
    keys = [key for key, _ in CANDIDATES if key == "request" or key in eval_dirs]
    labels = dict(CANDIDATES)
    steps = {AGENT_KEYS[key]: run_step(evals[key][3]) for key in eval_dirs if key in ("naive", "fixed", "open")}
    out_dir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures",
                                       comparison_name(config["general"]["k"], steps))
    os.makedirs(out_dir, exist_ok=True)

    data_images = os.path.join(instances_dir, "images")

    def paths(instance_id):
        reference = os.path.join(data_images, f"{instance_id}_enhanced.jpg")
        if not os.path.exists(reference):
            reference = os.path.join(data_images, f"{instance_id}_original.jpg")
        row = {"reference": reference, "original": os.path.join(data_images, f"{instance_id}_original.jpg"),
               "request": os.path.join(data_images, f"{instance_id}_request.jpg")}
        for key in eval_dirs:
            row[key] = os.path.join(eval_dirs[key], "images", f"{instance_id}.jpg")
        return row

    def load_all(instance_id):
        return instance_id, {name: load_image(path) for name, path in paths(instance_id).items()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        loaded = list(pool.map(load_all, chosen))
    needed = ["reference"] + keys
    kept = [(i, imgs) for i, imgs in loaded if all(imgs[name] is not None for name in needed)]
    n_skipped = len(loaded) - len(kept)
    if not kept:
        raise SystemExit("no instance has every image, run the two generators first")
    print(f"{len(kept)} instances with every image, {n_skipped} skipped, encoding with {args.model}")

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    processor = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(DEVICE).eval()
    names = needed + (["original"] if all(imgs["original"] is not None for _, imgs in kept) else [])
    flat = [imgs[name] for _, imgs in kept for name in names]
    feats = encode_images(processor, model, flat, args.batch_size).reshape(len(kept), len(names), -1)
    ref = feats[:, 0]
    sims = {name: np.einsum("id,id->i", ref, feats[:, j]) for j, name in enumerate(names) if name != "reference"}
    ceiling = sims.pop("original", None)

    # Text-image: each image against its instance's redacted-detail texts, averaged per instance
    ids = [i for i, _ in kept]
    detail_texts, owner = [], []
    for r, i in enumerate(ids):
        details = {d["id"]: d for d in instances[i]["details"]}
        for did in instances[i]["redacted_ids"]:
            detail_texts.append(details[did]["text"])
            owner.append(r)
    text_feats = encode_texts(processor, model, detail_texts, args.batch_size)
    owner = np.array(owner)
    detail_sims = {}
    for j, name in enumerate(names):
        if name == "original":
            continue
        per_detail = np.einsum("kd,kd->k", text_feats, feats[owner, j])
        detail_sims[name] = np.array([per_detail[owner == r].mean() for r in range(len(ids))])
    u_rec, questions = {}, {}
    for key in eval_dirs:
        _, episodes, judgments, _, _ = evals[key]
        u_rec[key] = np.array([judgments[i]["u_rec"]["llm"] for i in ids])
        questions[key] = np.array([episodes[i]["n_questions"] for i in ids])

    with open(os.path.join(out_dir, "clip_scores.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["instance_id", "redacted"] + [f"sim_{k}" for k in keys] + (["sim_original"] if ceiling is not None else [])
        header += [f"detail_sim_{k}" for k in ["reference"] + keys]
        header += [f"u_rec_{k}" for k in u_rec] + [f"questions_{k}" for k in questions]
        writer.writerow(header)
        for r, i in enumerate(ids):
            row = [i, len(instances[i]["redacted_ids"])] + [f"{sims[k][r]:.4f}" for k in keys]
            row += [f"{ceiling[r]:.4f}"] if ceiling is not None else []
            row += [f"{detail_sims[k][r]:.4f}" for k in ["reference"] + keys]
            row += [f"{u_rec[k][r]:.3f}" for k in u_rec] + [questions[k][r] for k in questions]
            writer.writerow(row)

    lines = summary_lines(sims, detail_sims, keys, labels, u_rec, questions, n_skipped, ceiling, args.model)
    with open(os.path.join(out_dir, "clip_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    plot(sims, detail_sims, keys, labels, u_rec, os.path.join(out_dir, "clip_scores.png"))
    print(f"saved {os.path.join(out_dir, 'clip_scores.csv')}, clip_summary.md and clip_scores.png")


if __name__ == "__main__":
    main()
