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

    Ordered screenshot-major, then element-major, so every request about one screenshot
    is issued back to back. That is the faster order by a wide margin on a server with a
    prompt cache — llama.cpp keeps the KV cache of the previous request, and every request
    for the same screenshot shares the same image tokens, so the image is encoded once per
    screenshot instead of once per request. Measured on the office Mac mini (M2 Pro, Q4
    GUI-Owl): 22 s for the first request on a screenshot, 0.15 s of prompt work for each one
    after it. A dataset whose rows are not grouped by screenshot — the human ground truth,
    assembled from CVAT tasks, changes screenshot 9,311 times over 841 screenshots — ran
    20x slower than the same rows grouped, which is the whole difference between the
    August runs (~10 h) and the first human-ground-truth run (52 h). Grouping here rather
    than asking callers to sort their dataset means the order of the file never matters.

    Grouping is stable: screenshots keep their first-appearance order and elements keep
    theirs within a screenshot, so a grouped file is unchanged and `--limit`'s even sample
    stays even. It also means a multi-phrasing run reports a higher cached share than a
    single-phrasing one, so `cached_input_fraction` is not comparable across the two;
    `description_indices` is recorded in the result for exactly that reason.
    """
    by_screenshot: dict = {}
    for row in dataset:
        by_screenshot.setdefault(row.get("screenshot_id"), []).append(row)
    pairs = []
    for rows in by_screenshot.values():
        for row in rows:
            for idx in description_indices:
                if idx < len(row.get("descriptions", [])):
                    pairs.append((row, idx))
    return pairs
