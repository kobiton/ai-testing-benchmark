"""
Step 4. Dataset assembly and storage.

Merges all processed ScreenshotRecords into a JSONL file and uploads both the dataset and a stats summary to S3.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from .config import config, dataset_filename, stats_filename
from .models import ScreenshotRecord

logger = logging.getLogger(__name__)


def _get_s3():
    import boto3
    return boto3.client("s3", **config.boto3_kwargs())


def assemble_and_upload(records: list[ScreenshotRecord], dry_run: bool = False, output_name: str = "") -> dict:
    """
    Writes one JSONL line per element across all records, uploads to S3.
    Returns a stats dict.

    dry_run=True: writes locally only, skips S3 upload (useful for local testing).
    """
    rows = []
    screenshots_with_no_elements = []

    for rec in records:
        rec_rows = rec.to_jsonl_rows()
        if not rec_rows:
            screenshots_with_no_elements.append(rec.screenshot_id)
            continue
        rows.extend(rec_rows)

    stats = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "llm_provider": config.llm_provider,
        "model": config.active_model,
        "total_screenshots": len(records),
        "screenshots_with_elements": len(records) - len(screenshots_with_no_elements),
        "screenshots_with_no_elements": len(screenshots_with_no_elements),
        "total_elements": len(rows),
        "elements_with_bbox": sum(1 for r in rows if r["bbox"] is not None),
        "elements_with_descriptions": sum(1 for r in rows if len(r["descriptions"]) >= 1),
        "elements_with_two_descriptions": sum(1 for r in rows if len(r["descriptions"]) >= 2),
        "avg_elements_per_screenshot": round(len(rows) / max(len(records), 1), 2),
    }

    logger.info("Dataset stats: %s", json.dumps(stats, indent=2))

    dataset_key = dataset_filename(output_name) if output_name else config.s3_dataset_key
    stats_key = stats_filename(output_name) if output_name else config.s3_stats_key

    if dry_run:
        out_path = Path(dataset_key)
        with open(out_path, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        logger.info("dry_run: wrote %d rows to %s", len(rows), out_path)
        return stats

    s3 = _get_s3()

    # Upload JSONL
    jsonl_body = "\n".join(json.dumps(r) for r in rows)
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=dataset_key,
        Body=jsonl_body.encode("utf-8"),
        ContentType="application/jsonl",
    )
    logger.info("Uploaded %d rows → s3://%s/%s", len(rows), config.s3_bucket, dataset_key)

    # Upload stats
    s3.put_object(
        Bucket=config.s3_bucket,
        Key=stats_key,
        Body=json.dumps(stats, indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    logger.info("Uploaded stats → s3://%s/%s", config.s3_bucket, stats_key)

    return stats
