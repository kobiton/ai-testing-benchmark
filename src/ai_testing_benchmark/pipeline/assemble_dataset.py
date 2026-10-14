"""
Step 4. Dataset assembly.

Merges all processed ScreenshotRecords into one JSONL file under `data/`, beside a stats
sidecar. Makes no LLM calls and touches no network — everything the pipeline produces
stays on this machine.
"""
import json
import logging
from datetime import datetime, timezone

from .config import DATA_DIR, config, dataset_filename, stats_filename
from .models import ScreenshotRecord

logger = logging.getLogger(__name__)


def assemble_dataset(records: list[ScreenshotRecord], output_name: str) -> dict:
    """Write one JSONL line per element, plus a stats sidecar. Returns the stats dict.

    Both land in `data/`, next to the dataset this repository ships, because that is where
    the benchmark's `--dataset` default already looks.
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
        # What produced these labels. A score is read together with the conditions it was
        # measured under; the ground truth those scores are graded against should say who
        # drew its boxes.
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

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dataset_path = DATA_DIR / dataset_filename(output_name)
    stats_path = DATA_DIR / stats_filename(output_name)

    with open(dataset_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    logger.info("Wrote %d rows → %s", len(rows), dataset_path)

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Wrote stats → %s", stats_path)

    return stats
