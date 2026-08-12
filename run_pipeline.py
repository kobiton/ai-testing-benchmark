#!/usr/bin/env python3
"""
Dataset generation pipeline entrypoint.

Runs steps 1-4 for every image in the input directory:
  Step 1. Extract UI elements       1 call per screenshot
  Step 2. Detect bounding boxes     1 call per element  ← dominates the cost
  Step 3. Generate descriptions     1 call per screenshot
  Step 4. Assemble the JSONL        no calls

Usage:
  # Label your own screenshots into a new dataset
  python run_pipeline.py --input path/to/screenshots --output-name my-dataset

  # Pilot a change on 20 screenshots before paying for the whole folder
  python run_pipeline.py --input path/to/screenshots --output-name pilot --limit 20

Output lands in data/<output-name>.jsonl with a stats sidecar beside it, and progress is
checkpointed to data/checkpoints/ so an interrupted run resumes where it stopped.

Every step calls a paid API. There is no flag that runs the pipeline without spending:
--limit is how you keep a trial cheap.
"""
import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipeline import checkpoint
from pipeline.config import config
from pipeline.models import ScreenshotRecord
from pipeline.step1_extract_elements import extract_elements
from pipeline.step2_detect_bboxes import detect_bboxes
from pipeline.step3_generate_descriptions import generate_descriptions
from pipeline.step4_assemble_dataset import assemble_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _load_local_images(input_dir: str, limit: int = 0, skip=frozenset()) -> list[ScreenshotRecord]:
    base = Path(input_dir)
    if not base.is_dir():
        # Returning empty lets main() print its "check --input path" message instead of a FileNotFoundError traceback on a mistyped path.
        logger.error("Input directory does not exist: %s", base)
        return []

    records = []
    for path in sorted(base.iterdir()):
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        screenshot_id = path.stem
        if screenshot_id in skip:
            continue
        records.append(ScreenshotRecord(
            screenshot_id=screenshot_id,
            image_path=str(path),
        ))
        if limit and len(records) >= limit:
            break
    logger.info("Loaded %d images from %s", len(records), input_dir)
    return records


def _process_one(record: ScreenshotRecord) -> tuple[ScreenshotRecord, bool]:
    """Run steps 1-3 for a single screenshot.

    Returns (record, succeeded). The flag exists so the caller can tell a finished screenshot from a failed one:
    checkpointing a failure would record it as done and a resume would skip it, losing it from the dataset for good.
    That is exactly what a dropped network connection produces — every request in flight fails at once.
    """
    try:
        record = extract_elements(record)
        record = detect_bboxes(record)
        record = generate_descriptions(record)
        return record, True
    except Exception as exc:
        logger.error("Failed to process %s: %s — it will be retried on the next run",record.screenshot_id, exc)
        return record, False


def main():
    parser = argparse.ArgumentParser(
        description="Label screenshots into a ground-truth dataset for the benchmark")
    # No default pointing at data/images: that holds the 841 screenshots already labelled in data/dataset-v1.jsonl,
    # and re-labelling them is ~12,000 billable calls that nobody meant to spend. Say which directory you mean.
    parser.add_argument("--input", required=True, help="Directory of screenshots to label")
    parser.add_argument("--output-name", required=True,
                        help="Names the output: data/<name>.jsonl plus a stats sidecar. "
                             "Pick something that is not dataset-v1, which is the corpus "
                             "this repository ships.")
    parser.add_argument("--workers", type=int, default=config.pipeline_workers, help="Screenshots to label in parallel (default: PIPELINE_WORKERS)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N screenshots (0 = all). Takes the first N in key "
                             "order, so the same N are picked every run - use it to pilot a "
                             "change on a handful before paying for the whole folder.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore and discard any checkpoint for this output name, "
                             "re-labeling every screenshot from scratch. Use after changing "
                             "the pipeline, when the earlier labels are no longer wanted.")
    args = parser.parse_args()

    ckpt = checkpoint.path_for(args.output_name)
    if args.no_resume:
        ckpt.unlink(missing_ok=True)
    resumed, done = checkpoint.read(ckpt)

    records = _load_local_images(args.input, args.limit, done)

    if not records and not resumed:
        logger.error("No images found. Check the --input path")
        return

    processed = list(resumed)
    failed = 0
    if records:
        logger.info("Processing %d screenshots with %d workers...", len(records), args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process_one, rec): rec for rec in records}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Screenshots"):
                record, ok = future.result()
                if not ok:
                    failed += 1
                    continue        # left out of the checkpoint so a resume retries it
                # Persist before anything else can go wrong with this run.
                checkpoint.append(ckpt, record)
                processed.append(record)
    else:
        logger.info("Nothing new to label; assembling %d checkpointed screenshot(s)",len(resumed))

    if failed:
        logger.warning("%d screenshot(s) failed and were left out of the checkpoint. Re-run with the same --output-name to retry just those", failed)

    stats = assemble_dataset(processed, output_name=args.output_name)

    # The checkpoint stops being needed only when the dataset is both written and complete.
    # Deleting it while screenshots are still outstanding would throw away the successes too, forcing a re-run to re-label everything.
    if failed:
        logger.info("Keeping the checkpoint at %s so the next run retries only the %d outstanding screenshot(s)", ckpt, failed)
    else:
        ckpt.unlink(missing_ok=True)

    print("\n=== Pipeline complete ===")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
