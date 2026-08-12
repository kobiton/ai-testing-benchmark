"""Reading the ground-truth dataset.

Rows without a `bbox` or without `descriptions` are dropped rather than carried: the
benchmark has nothing to ask about an element with no wording and nothing to score it
against with no box, and a row that reaches scoring in that state fails in a way that
reads like a model error.
"""
import json


def _load_dataset(path: str) -> list:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("bbox") and r.get("descriptions"):
                rows.append(r)
    return rows
