import os
import asyncio
import hashlib
import json
import re
import numpy as np
import pandas as pd
import build_gemini_2_embeddings as embedding
from pathlib import Path
from types import SimpleNamespace
from dotenv import load_dotenv
from tqdm import tqdm


def main():
    # SETTINGS
    settings = SimpleNamespace(
        input=Path("survey-data/final"),
        output=Path("analysis/survey/data"),
        participants=None,
        cache=Path("gemini_2_embeddings.cache.jsonl"),
        dotenv=Path(".env"),
        model="gemini-embedding-2-preview",
        task_type="SEMANTIC_SIMILARITY",
        output_dimensionality=768,
        concurrency=12,
        requests_per_minute=600,
        timeout_seconds=120.0,
        max_retries=6,
        api_key_env="GEMINI_API_KEY",
        dry_run=False,
    )
    load_dotenv(settings.dotenv, override=True)
    asyncio.run(preprocess(settings))

async def preprocess(settings: SimpleNamespace) -> None:
    rounds = pd.read_csv(settings.input / "rounds.csv")
    if settings.participants is None:
        settings.participants = sorted(rounds.prolific_pid.unique())
    rounds = rounds.loc[rounds.prolific_pid.isin(settings.participants)].copy()
    rounds["included"] = rounds.finished_at.notna()
    rounds["round_id"] = rounds.session_id + ":" + rounds.round_index.astype(str)
    # SURVEY DATA: INPUT INDEX
    jobs = {}
    records = []
    roles = {
        "reference_text": "ground_truth_text",
        "request_text": "request_text",
        "output_text": "output_text",
        "reference_image": "original_image_path",
        "initial_image": "initial_image_path",
        "final_image": "final_image_path"
    }
    for row in tqdm(rounds.to_dict("records"), desc="Indexing survey inputs"):
        if not row["included"]:
            continue
        for role, column in roles.items():
            value = row[column]
            if role == "initial_image" and pd.isna(value):
                continue
            source = settings.input / value if role.endswith("image") else value
            add_job = embedding.add_image_job if role.endswith("image") else embedding.add_text_job
            job_id = add_job(jobs, source, settings.model, settings.task_type, settings.output_dimensionality)
            records.append({"round_id": row["round_id"], "role": role, "turn": 0, "job_id": job_id})
    # EMBEDDINGS: CACHE AND MISSING INPUTS
    vectors = embedding.load_cache(settings.cache)
    missing = [job for job_id, job in jobs.items() if job_id not in vectors]
    settings.output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "input": str(settings.input),
        "rounds_sha256": hashlib.sha256((settings.input / "rounds.csv").read_bytes()).hexdigest(),
        "participants": settings.participants,
        "selected_rounds": len(rounds),
        "analysis_rounds": int(rounds.included.sum()),
        "missing_initial_images": int(rounds.initial_image_path.isna().sum()),
        "unique_embeddings": len(jobs),
        "cached_before_run": len(jobs) - len(missing),
        "missing_before_run": len(missing),
        "model": settings.model,
        "task_type": settings.task_type,
        "dimensions": settings.output_dimensionality,
        "reference_image": "Exact original target photograph in the survey export; never the enhanced rendering."
    }
    print(json.dumps(metadata, indent=2))
    if settings.dry_run:
        return
    if missing:
        vectors.update(await embedding.acquire_embeddings(missing, settings, os.environ[settings.api_key_env]))
    indexed = pd.DataFrame(records)
    indexed["embedding"] = indexed.job_id.map(lambda key: vectors[key].tolist())
    indexed.to_parquet(settings.output / "embeddings.parquet", index=False)
    # SURVEY OUTCOMES: EMBEDDING SCORES
    outcomes = []
    for round_id, group in tqdm(indexed.groupby("round_id"), desc="Computing survey outcomes"):
        lookup = {row.role: np.array(row.embedding) for row in group[group.turn.eq(0)].itertuples()}
        result = {"round_id": round_id}
        for metric, initial_role, final_role, reference_role in [
            ("text", "request_text", "output_text", "reference_text"),
            ("image", "initial_image", "final_image", "reference_image"),
            ("text_image", "request_text", "output_text", "reference_image"),
            ("image_text", "initial_image", "final_image", "reference_text")
        ]:
            final, reference = lookup[final_role], lookup[reference_role]
            if initial_role not in lookup:
                continue
            initial = lookup[initial_role]
            result["%s_cosine_gain" % metric] = float((final - initial) @ reference)
        outcomes.append(result)
    rounds = rounds.merge(pd.DataFrame(outcomes), on="round_id", how="left", validate="one_to_one")
    for role in ["question", "answer"]:
        lengths = rounds["%s_texts" % role].map(lambda value: [word_count(t) for t in json.loads(value)])
        rounds["%s_words" % role] = lengths.map(sum)
        rounds["mean_%s_words" % role] = lengths.map(lambda x: float(np.mean(x)) if x else np.nan)
    for role in ["request", "output"]:
        rounds["%s_words" % role] = rounds["%s_text" % role].map(lambda x: word_count(x) if isinstance(x, str) else np.nan)
    for name, start, end in [
        ("request_seconds", "started_at", "request_at"),
        ("interaction_seconds", "request_at", "stopped_at"),
        ("render_seconds", "stopped_at", "finished_at")
    ]:
        rounds[name] = (pd.to_datetime(rounds[end]) - pd.to_datetime(rounds[start])).dt.total_seconds()
    # SURVEY DATA: TIMING AND EXPORT
    turns = pd.read_csv(settings.input / "turns.csv").merge(
        rounds[["session_id", "round_index", "prolific_pid", "included", "round_id"]],
        on=["session_id", "round_index"],
        how="inner",
        validate="many_to_one"
    )
    response_times = turns.groupby("round_id").seconds_to_answer.mean().rename("mean_answer_seconds")
    rounds = rounds.merge(response_times, on="round_id", how="left", validate="one_to_one")
    rounds.to_csv(settings.output / "rounds.csv", index=False)
    turns.to_csv(settings.output / "turns.csv", index=False)
    metadata["complete"] = True
    (settings.output / "embedding_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")

def word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


if __name__ == "__main__":
    main()
