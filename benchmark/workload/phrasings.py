"""The three ways the labeling pipeline words every element, and the rows they make.

Deliberately free of sibling imports, like `sampling.py` and for the same reason: these
are facts about the *dataset* rather than about a run, so anything that wants to reason
about the work a run will do can import them without pulling in the run itself.
"""


# The three ways step 3 of the labeling pipeline phrases every element, in the order it
# writes them (pipeline/step3_generate_descriptions.py). Benchmarking more than one is
# the point of having them: `name` is usually the element's visible text, so a model that
# grounds by reading text scores well on it and poorly on `intent`, which deliberately
# carries no text handle. One number over `name` alone measures the easiest case and reads
# as if it measured the feature.
DESCRIPTION_STYLES = ("name", "label", "intent")


def _style_of(index: int) -> str:
    """Human name for a description index; the index itself if a dataset has more."""
    return DESCRIPTION_STYLES[index] if index < len(DESCRIPTION_STYLES) else f"d{index}"

def _expand_pairs(dataset: list, description_indices: list) -> list:
    """The work items of a run: one per (element, phrasing) the dataset can supply.

    An element whose labeling produced fewer than three descriptions contributes fewer
    pairs rather than a clamped duplicate. Scoring index 0's text again under the name
    `intent` would put identical predictions on both sides of the agreement figures,
    which would then be measuring a gap in the dataset rather than the model.

    Ordered element-major, so a screenshot's phrasings run close together. That is the
    faster order — the server's prompt cache still holds the image — but it also means a
    multi-phrasing run reports a higher cached share than a single-phrasing one, so
    `cached_input_fraction` is not comparable across the two. `description_indices` is
    recorded in the result for exactly that reason.
    """
    pairs = []
    for row in dataset:
        for idx in description_indices:
            if idx < len(row.get("descriptions", [])):
                pairs.append((row, idx))
    return pairs
