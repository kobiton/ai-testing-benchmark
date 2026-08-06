"""Per-screenshot checkpointing, so an interrupted labeling run can resume.

Step 4 writes the dataset only once every screenshot is labeled, so without this a
kill, crash or sleeping laptop discarded the whole run — at ~16 LLM calls per
screenshot that is hours of paid work lost at screenshot 800 of 841.

Each finished screenshot is appended here the moment it completes, and a re-run
with the same output name skips whatever is already recorded.

One line per **screenshot**, not per element, so a screenshot that yielded no
elements is still recorded as done rather than re-labeled on every resume.

Paths are relative to the working directory, so run `run_pipeline.py` from the same
directory each time or a resume will not find the checkpoint the earlier run wrote.
"""
import json
import logging
from pathlib import Path

from .config import config, dataset_filename
from .models import BoundingBox, Element, ScreenshotRecord

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/checkpoints")


def path_for(output_name: str) -> Path:
    """One checkpoint per output dataset, so runs producing different datasets
    never read each other's progress.

    Named through `dataset_filename` for the same reason step 4 writes its key that way:
    `Path(name).stem` alone strips everything after the last dot, so `foo-gpt-5.6-terra`
    and `foo-gpt-5.7-x` both checkpointed to `foo-gpt-5.partial.jsonl` and each would have
    resumed from the other's progress. Normalizing first also keeps this agreeing with
    `/api/admin/pipeline/checkpoint`, which probes on the raw name typed into the form."""
    name = dataset_filename(output_name) if output_name else config.s3_dataset_key
    return CHECKPOINT_DIR / f"{Path(name).stem}.partial.jsonl"


def read(path: Path) -> tuple[list[ScreenshotRecord], set]:
    """Rebuild the records an earlier run already labeled.

    Round-trips exactly what `ScreenshotRecord.to_jsonl_rows()` emits, so step 4
    needs no knowledge of checkpoints.
    """
    if not path.exists():
        return [], set()

    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        records.append(ScreenshotRecord(
            screenshot_id=entry["screenshot_id"],
            s3_key=entry["s3_key"],
            image_path=entry.get("image_path", ""),
            elements=[
                Element(
                    element_id=row["element_id"],
                    name=row["name"],
                    type=row["type"],
                    bbox=BoundingBox(**row["bbox"]) if row.get("bbox") else None,
                    descriptions=row.get("descriptions", []),
                )
                for row in entry["rows"]
            ],
        ))

    done = {r.screenshot_id for r in records}
    logger.info("Resuming from %s — %d screenshot(s) already labeled", path, len(done))
    return records, done


def append(path: Path, record: ScreenshotRecord) -> None:
    """Append one finished screenshot. Called only from the `as_completed` loop,
    which is single-threaded, so no lock is needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps({
            "screenshot_id": record.screenshot_id,
            "s3_key": record.s3_key,
            "image_path": record.image_path,
            "rows": record.to_jsonl_rows(),
        }) + "\n")


def count(output_name: str) -> int:
    """How many screenshots a pending checkpoint holds; 0 when there is none.

    Counts lines rather than parsing, so it stays cheap to call repeatedly while a run
    is still appending to the file.
    """
    path = path_for(output_name)
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())
