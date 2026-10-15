import os
import asyncio
import base64
import hashlib
import json
import mimetypes
import time
import httpx
import numpy
import pandas
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from dotenv import load_dotenv
from tqdm import tqdm


def main():
    # SETTINGS
    settings = SimpleNamespace(
        input=Path("clip_dataset/dataset.csv"),
        output=Path("gemini_2_embeddings.parquet"),
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
    )
    load_dotenv(settings.dotenv, override=True)
    asyncio.run(build_dataset(settings))


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"


class RateLimiter:
    def __init__(self, requests_per_minute: int) -> None:
        self.interval = 60.0 / requests_per_minute
        self.next_request = 0.0
        self.lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_request - now)
            self.next_request = max(now, self.next_request) + self.interval
        await asyncio.sleep(delay)


# INPUT INDEXING

async def build_dataset(settings: SimpleNamespace) -> None:
    # MODEL DATA: INPUT INDEX
    frame = pandas.read_csv(settings.input)
    methods = discover_methods(frame)
    jobs, references = collect_jobs(
        frame,
        settings.input.parent,
        methods,
        settings.model,
        settings.task_type,
        settings.output_dimensionality
    )
    # EMBEDDINGS: CACHE AND MISSING INPUTS
    embeddings = load_cache(settings.cache)
    missing_jobs = [jobs[job_id] for job_id in sorted(jobs) if job_id not in embeddings]
    if missing_jobs:
        api_key = os.environ[settings.api_key_env]
        acquired = await acquire_embeddings(
            missing_jobs,
            settings,
            api_key
        )
        embeddings.update(acquired)
    # EMBEDDINGS: EXPORT
    output = assemble_output(
        frame,
        references,
        embeddings,
        settings.model,
        settings.task_type,
        settings.output_dimensionality
    )
    write_output(
        output,
        settings,
        methods,
        sorted(references),
        len(jobs),
        len(missing_jobs)
    )

def discover_methods(frame: pandas.DataFrame) -> list[str]:
    suffix = "_output_text"
    methods = sorted([
        column[:-len(suffix)]
        for column in frame.columns
        if column.endswith(suffix)
    ])
    return methods

def collect_jobs(
    frame: pandas.DataFrame,
    dataset_directory: Path,
    methods: list[str],
    model: str,
    task_type: str,
    output_dimensionality: int
) -> tuple[dict[str, dict[str, Any]], dict[str, list[Any]]]:
    jobs: dict[str, dict[str, Any]] = {}
    references: dict[str, list[Any]] = {
        "ground_truth_text_embedding": [],
        "ground_truth_enhanced_image_embedding": [],
        "request_text_embedding": [],
        "request_image_embedding": []
    }
    for method in methods:
        references["%s_output_text_embedding" % (method,)] = []
        references["%s_output_image_embedding" % (method,)] = []
    rows = tqdm(
        frame.to_dict(orient="records"),
        total=len(frame),
        desc="Indexing inputs"
    )
    for row in rows:
        references["ground_truth_text_embedding"].append(add_text_job(
            jobs,
            row["ground_truth_text"],
            model,
            task_type,
            output_dimensionality
        ))
        references["ground_truth_enhanced_image_embedding"].append(add_image_job(
            jobs,
            dataset_directory / row["ground_truth_enhanced_image_path"],
            model,
            task_type,
            output_dimensionality
        ))
        references["request_text_embedding"].append(add_text_job(
            jobs,
            row["request_text"],
            model,
            task_type,
            output_dimensionality
        ))
        references["request_image_embedding"].append(add_image_job(
            jobs,
            dataset_directory / row["request_image_path"],
            model,
            task_type,
            output_dimensionality
        ))
        for method in methods:
            references["%s_output_text_embedding" % (method,)].append(add_text_job(
                jobs,
                row["%s_output_text" % (method,)],
                model,
                task_type,
                output_dimensionality
            ))
            references["%s_output_image_embedding" % (method,)].append(add_image_job(
                jobs,
                dataset_directory / row["%s_output_image_path" % (method,)],
                model,
                task_type,
                output_dimensionality
            ))
    return jobs, references

def add_text_job(
    jobs: dict[str, dict[str, Any]],
    text: str,
    model: str,
    task_type: str,
    output_dimensionality: int
) -> str:
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    job_id = make_job_id(
        "text",
        source_hash,
        model,
        task_type,
        output_dimensionality
    )
    jobs[job_id] = {
        "job_id": job_id,
        "kind": "text",
        "source": text,
        "source_hash": source_hash
    }
    return job_id

def add_image_job(
    jobs: dict[str, dict[str, Any]],
    path: Path,
    model: str,
    task_type: str,
    output_dimensionality: int
) -> str:
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    job_id = make_job_id(
        "image",
        source_hash,
        model,
        task_type,
        output_dimensionality
    )
    jobs[job_id] = {
        "job_id": job_id,
        "kind": "image",
        "source": str(path),
        "source_hash": source_hash
    }
    return job_id

def make_job_id(
    kind: str,
    source_hash: str,
    model: str,
    task_type: str,
    output_dimensionality: int
) -> str:
    specification = json.dumps(
        {
            "kind": kind,
            "source_hash": source_hash,
            "model": model,
            "task_type": task_type,
            "output_dimensionality": output_dimensionality
        },
        sort_keys=True
    )
    return hashlib.sha256(specification.encode("utf-8")).hexdigest()

def load_cache(path: Path) -> dict[str, numpy.ndarray]:
    if not path.exists():
        return {}
    entries: dict[str, numpy.ndarray] = {}
    with path.open(encoding="utf-8") as handle:
        for line in tqdm(handle, desc="Loading cache"):
            record = json.loads(line)
            entries[record["job_id"]] = numpy.asarray(
                record["embedding"],
                dtype=numpy.float32
            )
    return entries


# API acquisition

async def acquire_embeddings(
    jobs: list[dict[str, Any]],
    settings: SimpleNamespace,
    api_key: str
) -> dict[str, numpy.ndarray]:
    settings.cache.parent.mkdir(parents=True, exist_ok=True)
    limiter = RateLimiter(settings.requests_per_minute)
    semaphore = asyncio.Semaphore(settings.concurrency)
    timeout = httpx.Timeout(settings.timeout_seconds)
    acquired: dict[str, numpy.ndarray] = {}
    jobs_by_id = {job["job_id"]: job for job in jobs}
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            asyncio.create_task(embed_job(
                client,
                job,
                settings,
                api_key,
                limiter,
                semaphore
            ))
            for job in jobs
        ]
        with settings.cache.open("a", encoding="utf-8") as handle:
            futures = tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="Embedding"
            )
            for future in futures:
                job_id, embedding = await future
                acquired[job_id] = embedding
                job = jobs_by_id[job_id]
                handle.write(json.dumps({
                    "job_id": job_id,
                    "kind": job["kind"],
                    "source_hash": job["source_hash"],
                    "model": settings.model,
                    "task_type": settings.task_type,
                    "output_dimensionality": settings.output_dimensionality,
                    "embedding": embedding.tolist()
                }) + "\n")
                handle.flush()
    return acquired

async def embed_job(
    client: httpx.AsyncClient,
    job: dict[str, Any],
    settings: SimpleNamespace,
    api_key: str,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    attempt: int = 0
) -> tuple[str, numpy.ndarray]:
    url = "%s/%s:embedContent" % (GEMINI_API_BASE, settings.model)
    try:
        async with semaphore:
            await limiter.acquire()
            payload = make_payload(
                job,
                settings.task_type,
                settings.output_dimensionality
            )
            response = await client.post(
                url,
                headers={"x-goog-api-key": api_key},
                json=payload
            )
    except httpx.TransportError:
        return await retry_job(
            client,
            job,
            settings,
            api_key,
            limiter,
            semaphore,
            attempt
        )
    if response.status_code == 429 or response.status_code >= 500:
        return await retry_job(
            client,
            job,
            settings,
            api_key,
            limiter,
            semaphore,
            attempt,
            response
        )
    response.raise_for_status()
    embedding = numpy.asarray(
        response.json()["embedding"]["values"],
        dtype=numpy.float32
    )
    embedding = embedding / numpy.linalg.norm(embedding)
    return job["job_id"], embedding

async def retry_job(
    client: httpx.AsyncClient,
    job: dict[str, Any],
    settings: SimpleNamespace,
    api_key: str,
    limiter: RateLimiter,
    semaphore: asyncio.Semaphore,
    attempt: int,
    response: httpx.Response | None = None
) -> tuple[str, numpy.ndarray]:
    status = None if response is None else response.status_code
    if attempt >= settings.max_retries:
        raise RuntimeError("Embedding retries exhausted for job %s" % job["job_id"])
    retry_after = None if response is None else response.headers.get("retry-after")
    delay = float(retry_after) if retry_after is not None else min(60.0, 2.0 ** attempt)
    await asyncio.sleep(delay)
    return await embed_job(
        client,
        job,
        settings,
        api_key,
        limiter,
        semaphore,
        attempt + 1
    )

def make_payload(
    job: dict[str, Any],
    task_type: str,
    output_dimensionality: int
) -> dict[str, Any]:
    if job["kind"] == "text":
        part = {"text": job["source"]}
    else:
        path = Path(job["source"])
        mime_type = mimetypes.guess_type(path)[0]
        part = {
            "inlineData": {
                "mimeType": mime_type,
                "data": base64.b64encode(path.read_bytes()).decode("ascii")
            }
        }
    return {
        "content": {"parts": [part]},
        "taskType": task_type,
        "outputDimensionality": output_dimensionality
    }


# Output assembly

def assemble_output(
    frame: pandas.DataFrame,
    references: dict[str, list[Any]],
    embeddings: dict[str, numpy.ndarray],
    model: str,
    task_type: str,
    output_dimensionality: int
) -> pandas.DataFrame:
    output = frame.copy()
    embedding_columns = tqdm(
        references.items(),
        total=len(references),
        desc="Assembling output"
    )
    for column, job_ids in embedding_columns:
        output[column] = [embeddings[job_id].tolist() for job_id in job_ids]
    output["embedding_model"] = model
    output["embedding_task_type"] = task_type
    output["embedding_output_dimensionality"] = output_dimensionality
    return output

def write_output(
    frame: pandas.DataFrame,
    settings: SimpleNamespace,
    methods: list[str],
    embedding_columns: list[str],
    n_jobs: int,
    n_acquired_jobs: int
) -> None:
    settings.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = settings.output.with_suffix("%s.tmp" % (settings.output.suffix,))
    frame.to_parquet(temporary_output, index=False)
    os.replace(temporary_output, settings.output)
    metadata_path = settings.output.with_suffix(".metadata.json")
    metadata = {
        "input": str(settings.input),
        "output": str(settings.output),
        "cache": str(settings.cache),
        "model": settings.model,
        "task_type": settings.task_type,
        "output_dimensionality": settings.output_dimensionality,
        "sampling_seed": None,
        "concurrency": settings.concurrency,
        "requests_per_minute": settings.requests_per_minute,
        "instances": len(frame),
        "methods": methods,
        "embedding_columns": embedding_columns,
        "unique_embedding_requests": n_jobs,
        "reused_embedding_requests": n_jobs - n_acquired_jobs,
        "acquired_embedding_requests": n_acquired_jobs
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8"
    )


if __name__ == "__main__":
    main()
