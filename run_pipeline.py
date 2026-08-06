#!/usr/bin/env python3
"""
Dataset generation pipeline entrypoint.

Runs steps 1-4 for every image in the input directory (or S3 prefix):
  Step 1. Extract UI elements       1 call per screenshot
  Step 2. Detect bounding boxes     1 call per element  ← dominates the cost
  Step 3. Generate descriptions     1 call per screenshot
  Step 4. Assemble the JSONL        no calls

Usage:
  # Label your own screenshots into a new dataset
  python run_pipeline.py --input path/to/screenshots --output-name my-dataset

  # Pilot a prompt change on 20 screenshots before paying for the folder
  python run_pipeline.py --input path/to/screenshots --limit 20

  # Read the screenshots from S3 instead of a local directory
  python run_pipeline.py --s3-input

NOTE --dry-run only skips the S3 upload. Steps 1-3 still make real, billable LLM calls.
"""
import argparse
import json
import logging
import os
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from pipeline import checkpoint
from pipeline.config import config
from pipeline.models import ScreenshotRecord
from pipeline.step1_extract_elements import extract_elements
from pipeline.step2_detect_bboxes import detect_bboxes
from pipeline.step3_generate_descriptions import generate_descriptions
from pipeline.step4_assemble_dataset import assemble_and_upload

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
            s3_key=f"{config.s3_prefix_screenshots}/{path.name}",
            image_path=str(path),
        ))
        if limit and len(records) >= limit:
            break
    logger.info("Loaded %d images from %s", len(records), input_dir)
    return records


def _download_s3_images(local_dir: str, limit: int = 0, skip=frozenset()) -> list[ScreenshotRecord]:
    """Download screenshots from S3 to a local temp directory.

    `limit` stops the walk mid-flight rather than slicing the result, which would still pull every object in the prefix - 290MB to label 20 screenshots.
    `skip` holds screenshots a previous run already labeled; they are filtered before the download so a resume neither re-fetches them nor spends the limit on work that is already done.
    """
    import boto3
    s3 = boto3.client("s3", **config.boto3_kwargs())
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=config.s3_bucket, Prefix=config.s3_prefix_screenshots)

    records = []
    Path(local_dir).mkdir(parents=True, exist_ok=True)

    for page in pages:
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not any(key.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                continue
            filename = Path(key).name
            if Path(filename).stem in skip:
                continue
            local_path = str(Path(local_dir) / filename)
            if not os.path.exists(local_path):
                s3.download_file(config.s3_bucket, key, local_path)
            records.append(ScreenshotRecord(
                screenshot_id=Path(filename).stem,
                s3_key=key,
                image_path=local_path,
            ))
            if limit and len(records) >= limit:
                break
        if limit and len(records) >= limit:
            break

    logger.info("Downloaded %d images from s3://%s/%s", len(records), config.s3_bucket, config.s3_prefix_screenshots)
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
    parser.add_argument("--input", default="", help="Local directory of input images")
    parser.add_argument("--s3-input", action="store_true", help="Download images from S3 instead of --input")
    parser.add_argument("--s3-screenshots-prefix", default="", help="Override S3 screenshots prefix (subfolder)")
    parser.add_argument("--dry-run", action="store_true", help="Skip S3 upload, write JSONL locally")
    parser.add_argument("--workers", type=int, default=config.pipeline_workers, help="Screenshots to label in parallel (default: PIPELINE_WORKERS)")
    parser.add_argument("--output-name", default="", help="Custom output JSONL filename (e.g. dataset-v2.jsonl)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process at most N screenshots (0 = all). Takes the first N in key "
                             "order, so the same N are picked every run - use it to pilot a "
                             "change on a handful before paying for the whole folder.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore and discard any checkpoint for this output name, "
                             "re-labeling every screenshot from scratch. Use after changing "
                             "the pipeline, when the earlier labels are no longer wanted.")
    args = parser.parse_args()

    if not args.input and not args.s3_input:
        parser.error("give --input <dir> (a directory of screenshots) or --s3-input")

    # Allow overriding the S3 screenshots prefix for one run
    if args.s3_screenshots_prefix:
        config.s3_prefix_screenshots = args.s3_screenshots_prefix

    ckpt = checkpoint.path_for(args.output_name)
    if args.no_resume:
        ckpt.unlink(missing_ok=True)
    resumed, done = checkpoint.read(ckpt)

    if args.s3_input:
        records = _download_s3_images("data/s3-downloads", args.limit, done)
    else:
        records = _load_local_images(args.input, args.limit, done)

    if not records and not resumed:
        logger.error("No images found. Check --input path or S3 prefix")
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

    extra = {}
    if args.output_name:
        extra["output_name"] = args.output_name
    stats = assemble_and_upload(processed, dry_run=args.dry_run, **extra)

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
