"""Per-element checkpointing, so an interrupted benchmark can resume.

The benchmark writes its result file once, at the very end. Without this a kill,
crash or sleeping laptop discarded the whole run — at 1-5s per element and 10,307
elements in the shipped dataset that is hours of inference lost at element 9,000.

One line per finished (element, phrasing) pair, appended the moment it completes. A
re-run with the same model and dataset skips whatever is already recorded.

Three rules that matter:

- **Errored elements are not recorded**, so a resume retries them. They would
  otherwise be frozen into the score as permanent failures, which understates the
  model: a transient timeout is not a wrong answer. Whatever still errors on the
  final pass is reported as an error in the output, same as before.
- **The served model is pinned in the header.** Swapping the loaded model and
  resuming would blend two models' predictions into one score — the same class of
  mistake as the 2026-07-21 runs. Reading a checkpoint whose served model differs is
  refused rather than silently discarded, because either choice made silently throws
  away something the operator cares about.
- **The description index is part of the row key, not of the filename.** It used to be
  a `-d0` suffix on the path, which meant a run of `--description-index 0 1 2` opened a
  different file from an earlier run of `0` and re-scored all 10,307 index-0 elements
  it already had. Keyed per row instead, adding phrasings to a dataset already scored
  at one phrasing costs only the phrasings that are new — which is the whole point of
  being able to add them later.
"""
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

META_KEY = "__meta__"


def path_for(output_dir, dataset_path: str, model: str) -> Path:
    """One checkpoint per (dataset, model).

    Not keyed on --limit: an even-stride subset is drawn from the same rows, so a
    2,000-element run's results are reusable by a later full run. Not keyed on the
    description index either — that lives in each row, so phrasings accumulate into
    one file instead of forking it. See the module docstring.
    """
    safe_model = model.replace("/", "-").replace(":", "-")
    stem = Path(dataset_path).stem
    return Path(output_dir) / "checkpoints" / f"{stem}-{safe_model}.partial.jsonl"


def key_of(row: dict) -> tuple:
    """The identity of one scored unit: an element under one phrasing.

    `.get(..., 0)` rather than `[...]` so a checkpoint or result written before
    multi-phrasing existed still reads — those rows are all index 0 by definition.
    """
    return (row["screenshot_id"], row["element_id"], row.get("description_index", 0))


def read(path: Path) -> tuple[list, set, dict]:
    """Return (results, done keys, header). Empty when there is no checkpoint."""
    if not path.exists():
        return [], set(), {}

    meta: dict = {}
    results = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        if META_KEY in entry:
            meta = entry[META_KEY]
            continue
        results.append(entry)

    done = {key_of(r) for r in results}
    logger.info("Resuming from %s — %d (element, phrasing) pair(s) already scored",path, len(done))
    return results, done, meta


def append(path: Path, meta: dict, result: dict) -> None:
    """Append one finished (element, phrasing) pair, writing the header on a new file.
    Called only from the `as_completed` loop, which is single-threaded, so no lock is needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a") as f:
        if new:
            f.write(json.dumps({META_KEY: meta}) + "\n")
        f.write(json.dumps(result) + "\n")


def count(path: Path) -> int:
    """How many scored pairs a pending checkpoint holds; 0 when there is none."""
    if not path.exists():
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip() and META_KEY not in line)
