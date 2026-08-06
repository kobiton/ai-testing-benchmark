"""Which dataset rows a limited benchmark run uses.

Its own module rather than a helper inside the runner, because anything that pre-fetches
or pre-filters images for a run has to pick the *same* subset. If two sides disagree, the
run asks for images that were never fetched and every element fails with "image not
found", which reads like a storage problem rather than a selection bug.
"""


def even_sample(rows: list, limit: int) -> list:
    """Take `limit` rows on an even stride across `rows`.

    Not the first N: a resumed labeling run writes rows in completion order, so the
    head of the file is a handful of screenshots from one or two apps. Measured on
    dataset-v1, a 200-row stride covers 199 distinct screenshots where the first
    200 rows cover 19. Deterministic, so two pilots are comparable.
    """
    if limit <= 0 or limit >= len(rows):
        return rows
    step = len(rows) / limit
    return [rows[int(i * step)] for i in range(limit)]
