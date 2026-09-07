"""Random examples of an evaluation run with the model's reasoning: per instance the episode figure of
visualize_episode.py plus a markdown transcript with the reasoning behind every agent turn.

Usage:
    python scripts/visualize_thinking.py <eval_dir> --out <dir> [-n 8] [--seed 42] [--instances <id> ...]

Writes <out>/<instance_id>.png and .md, and <out>/README.md listing the examples with their scores.
"""
import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from visualize_episode import visualize  # noqa: E402
from visualize_images import load_eval  # noqa: E402


def reasoning_of(turn):
    """The reasoning behind an agent turn: under `reasoning` from 2026-09-04 on, under `thinking` in older records."""
    return turn.get("reasoning") or turn.get("thinking")


def quoted(reasoning, label):
    """The reasoning as a labelled block quote, its word count after the label."""
    return [f"**{label}** ({len(reasoning.split())} words)", "", "> " + reasoning.replace("\n", "\n> "), ""]


def transcript_md(instance, episode, judgment):
    details = {d["id"]: d for d in instance["details"]}
    u = judgment["u_rec"]
    out = [f"# {instance['instance_id']} ({episode['agent_name']}, {episode['n_questions']} questions, "
           f"u_rec llm {u['llm']:.3f} / equal {u['equal']:.3f})", "",
           "## Ground-truth intent", "", instance["full_spec"], "",
           f"## Redacted details ({len(instance['redacted_ids'])})", ""]
    for i in instance["redacted_ids"]:
        d = details[i]
        ok = bool(judgment["per_detail"].get(str(i), {}).get("delivered"))
        out.append(f"- [{'recovered' if ok else 'not recovered'}] ({d['category']}, importance {d['importance']:.2f}) {d['text']}")
    out += ["", "## User request", "", instance["eroded_prompt"], "", "## Interaction", ""]
    q = 0
    for turn in episode["turns"]:
        reasoning = reasoning_of(turn)
        if turn["role"] == "agent_question":
            q += 1
            if reasoning:
                out += quoted(reasoning, f"Agent reasoning before question {q}")
            out += [f"**Agent question {q}:** {turn['text']}", ""]
        elif turn["role"] == "user_answer":
            out += [f"**User answer:** {turn['text']}", ""]
        elif turn["role"] == "agent_final" and reasoning:
            out += quoted(reasoning, "Agent reasoning before the final output")
    out += ["## Agent final output", "", episode["final_output"], ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir")
    parser.add_argument("--out", required=True)
    parser.add_argument("-n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", nargs="+", metavar="ID")
    args = parser.parse_args()

    instances, episodes, judgments, _, _ = load_eval(args.eval_dir)
    chosen = args.instances or random.Random(args.seed).sample(sorted(episodes), min(args.n, len(episodes)))
    os.makedirs(args.out, exist_ok=True)
    index = [f"# {episodes[chosen[0]]['agent_name']} examples from {args.eval_dir}", "",
             "| instance | redacted | questions | u_rec llm | u_rec equal | reasoning words per agent turn |", "|---|---|---|---|---|---|"]
    for iid in chosen:
        inst, ep, j = instances[iid], episodes[iid], judgments[iid]
        visualize(inst, ep, j, os.path.join(args.out, f"{iid}.png"))
        with open(os.path.join(args.out, f"{iid}.md"), "w", encoding="utf-8") as f:
            f.write(transcript_md(inst, ep, j))
        words = [len(reasoning_of(t).split()) for t in ep["turns"] if reasoning_of(t)]
        index.append(f"| [{iid}]({iid}.md) ([figure]({iid}.png)) | {len(inst['redacted_ids'])} | {ep['n_questions']} | "
                     f"{j['u_rec']['llm']:.3f} | {j['u_rec']['equal']:.3f} | {', '.join(str(w) for w in words) or 'none'} |")
        print("saved", iid)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(index) + "\n")


if __name__ == "__main__":
    main()
