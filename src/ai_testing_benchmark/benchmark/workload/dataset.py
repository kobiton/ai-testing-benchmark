"""Reading the ground-truth dataset.

Rows without a `bbox` or without `descriptions` are dropped rather than carried: the
benchmark has nothing to ask about an element with no wording and nothing to score it
against with no box, and a row that reaches scoring in that state fails in a way that
reads like a model error.

`origins` selects rows by `annotation.origin` — which labelling first found the element,
`gpt-5.6` or `claude-opus-4-8` in the shipped ground truth. The published figures are over
the `gpt-5.6` rows, the 10,568 elements every model has answers for; the selection is a
config value (`dataset.annotation_origin`) rather than a second label file, so one file
stays the single source of truth.
"""
import json

ORIGIN_KEY = "origin"


def origin_of(row: dict) -> str:
    return str((row.get("annotation") or {}).get(ORIGIN_KEY) or "")


def _load_dataset(path: str, origins=None) -> list:
    rows = []
    seen_origins = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if not (r.get("bbox") and r.get("descriptions")):
                continue
            origin = origin_of(r)
            if origin:
                seen_origins.add(origin)
            if origins and origin not in origins:
                continue
            rows.append(r)
    if origins:
        unknown = sorted(set(origins) - seen_origins)
        if unknown:
            known = ", ".join(sorted(seen_origins)) or "none — no row carries annotation.origin"
            raise ValueError(f"no element has annotation.origin {', '.join(unknown)} (present: {known})")
    return rows


def keys_with_origin(path: str, origins) -> set:
    """(screenshot_id, element_id) of every row whose `annotation.origin` is in `origins`.

    For the scorers that work on result rows rather than on the dataset — the cascade and
    the failure browser — so they can apply the same selection a run made.
    """
    return {(r["screenshot_id"], str(r["element_id"])) for r in _load_dataset(path, origins)}
