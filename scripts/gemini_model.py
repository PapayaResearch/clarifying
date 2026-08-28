"""Gemini image generation shared by generate_images_for_data.py and generate_images_for_eval.py."""
import io
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from google.genai import Client, types  # noqa: E402
from PIL import Image  # noqa: E402

# Defaults, overridable through the arguments add_model_args adds
MODEL = "gemini-3.1-flash-image-preview"
MAX_RETRIES = 3
ASPECT_RATIO = "1:1"
RESOLUTION = "512"
WORKERS = 64  # concurrent Gemini calls


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
