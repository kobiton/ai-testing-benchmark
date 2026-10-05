"""Join an `xml-…` result with a `vision-…` result into the tree-then-screenshot cascade.

A locator that tries the accessibility tree first and falls back to the screenshot only when
the tree yields nothing is scored here without a single new request: both halves have already
been run on their own, so the cascade is a join on (screenshot, element, phrasing) and a
re-count. The eight figures the join produces, in the order they are reported:

    1  xml_correct            XPath resolved; centre of the first node's bounds inside the GT box
    2  xml_incorrect          XPath resolved to a node whose centre is outside the GT box
    3  xml_not_answered       NOT_FOUND, an XPath that does not compile, or one matching no node — every one of these falls through to the screenshot
    4  vision_correct         rows from 3 where the vision answer's centre is inside the GT box
    5  vision_incorrect       rows from 3 with a vision answer whose centre is outside
    6  vision_not_answered    rows from 3 with no vision answer at all
    7  correct                1 + 4
    8  incorrect              2 + 5 + 6

Items 3 and 6 each carry sub-lines, because "not answered" lumps a decision the model made
(`NOT_FOUND`, a declined box) with things it did not decide (an XPath that cannot run, a parse
failure, a transport error), and a reader weighing the two strategies wants them apart.

The population is every XML row that did not end in a transport error — an error is what a
resume retries, not an outcome. The vision file only has to cover the rows that fall through:
a row XML answered needs no partner, so a vision run made with `--only-from` (vision asked
only where XML gave nothing) and a full vision run score the cascade identically, and the
report reads the same for both. A fall-through row with no vision partner cannot be placed
in 4, 5 or 6 and is excluded; the count is returned and should be zero for either kind of
vision file. `vision_alone_*` — the screenshot strategy on its own over the same rows — is
filled only when the vision file covers every row of the population, i.e. a full run.

Pass/fail is read from each row's `pass_centroid`, whatever `--metric` either run was scored
with: the cascade is counted by the centroid rule on both halves, the way `summary.xml_outcomes`
already counts the XML half.
"""
import re
from collections import OrderedDict

XML_FALL_THROUGH = ("not_found", "invalid_xpath", "no_match")
ITEMS = (
    ("xml_correct", "XML answered correctly"),
    ("xml_incorrect", "XML answered incorrectly"),
    ("xml_not_answered", "XML not answered → vision"),
    ("vision_correct", "Vision answered correctly"),
    ("vision_incorrect", "Vision answered incorrectly"),
    ("vision_not_answered", "Vision not answered"),
    ("correct", "Correct overall (1 + 4)"),
    ("incorrect", "Incorrect overall (2 + 5 + 6)"),
)
XML_SUBLINES = XML_FALL_THROUGH
VISION_SUBLINES = ("declined", "non_answer")

# Rows whose description names a key on the on-screen keyboard. The XML dumps this benchmark
# scores against hold the application window only, so a keyboard key can never be found in
# the tree and always falls through; the count is reported as a footnote so the reader can
# tell a limit of the data from a limit of the strategy. A heuristic on the wording, nothing more.
KEYBOARD_PATTERN = re.compile(r"\bkey\b|keypad|keyboard", re.I)


def row_key(r: dict) -> tuple:
    return (r.get("screenshot_id"), r.get("element_id"), r.get("description_index", 0))


def fall_through_keys(xml_result: dict) -> tuple:
    """The (screenshot, element, phrasing) keys an XML result did not answer, for `--only-from`.

    A vision run restricted to these rows is all the cascade needs from the screenshot side
    — a row XML answered never consults vision — so for a model with no full vision run this
    is the run to pay for. Returns (keys, skipped) where `skipped` counts rows that ended in a
    transport error: those are not fall-through yet, a resume of the XML run retries them, and
    asking vision about them now would pay for a row the cascade may never use.
    """
    if xml_result.get("input") != "xml":
        raise ValueError("--only-from needs an --input xml result file")
    keys, skipped = set(), 0
    for r in xml_result.get("results") or []:
        if r.get("error"):
            skipped += 1
        elif (r.get("xml_outcome") or "") in XML_FALL_THROUGH:
            keys.add(row_key(r))
    return keys, skipped


def vision_answered(r: dict) -> bool:
    """Did the vision row return something a tap could be aimed at?

    `answer_class` is authoritative when present: `declined` and `non_answer` are the two
    classes with no usable prediction. Older files without it are judged on the prediction
    fields and the error string directly.
    """
    cls = r.get("answer_class")
    if cls:
        return cls not in VISION_SUBLINES
    if r.get("error"):
        return False
    return bool(r.get("pred_bbox") or r.get("pred_point"))


def vision_subline(r: dict) -> str:
    cls = r.get("answer_class")
    if cls in VISION_SUBLINES:
        return cls
    return "non_answer"


def _bucket(xml_row: dict, vision_row: dict) -> tuple:
    """(primary bucket, sub-line or '') for one joined row."""
    outcome = xml_row.get("xml_outcome") or ""
    if outcome == "answered":
        return ("xml_correct" if xml_row.get("pass_centroid") else "xml_incorrect"), ""
    if outcome not in XML_FALL_THROUGH:
        raise ValueError(f"unexpected xml_outcome {outcome!r} for {row_key(xml_row)}")
    if vision_row is None:
        raise ValueError(f"fall-through row without a vision partner: {row_key(xml_row)}")
    if vision_row.get("pass_centroid"):
        return "vision_correct", outcome
    if vision_answered(vision_row):
        return "vision_incorrect", outcome
    return "vision_not_answered", vision_subline(vision_row)


def _count(pairs: list) -> dict:
    """The eight items plus sub-lines over a list of (xml_row, vision_row)."""
    c = OrderedDict((k, 0) for k, _ in ITEMS)
    xml_sub = OrderedDict((k, 0) for k in XML_SUBLINES)
    vis_sub = OrderedDict((k, 0) for k in VISION_SUBLINES)
    keyboard_fall_through = 0
    vision_alone_correct = 0
    for x, v in pairs:
        bucket, sub = _bucket(x, v)
        c[bucket] += 1
        if bucket.startswith("vision_"):
            c["xml_not_answered"] += 1
            xml_sub[x["xml_outcome"]] += 1
            if KEYBOARD_PATTERN.search(x.get("description") or ""):
                keyboard_fall_through += 1
        if bucket == "vision_not_answered":
            vis_sub[sub] += 1
        if v is not None and v.get("pass_centroid"):
            vision_alone_correct += 1
    c["correct"] = c["xml_correct"] + c["vision_correct"]
    c["incorrect"] = c["xml_incorrect"] + c["vision_incorrect"] + c["vision_not_answered"]
    n = len(pairs)
    full_vision = all(v is not None for _, v in pairs)
    return {
        "total": n,
        "vision_rows_used": c["xml_not_answered"],
        "counts": dict(c),
        "shares": {k: (round(v / n, 4) if n else 0.0) for k, v in c.items()},
        "xml_not_answered_by": dict(xml_sub),
        "vision_not_answered_by": dict(vis_sub),
        "keyboard_rows_in_fall_through": keyboard_fall_through,
        "keyboard_rows": sum(1 for x, _ in pairs if KEYBOARD_PATTERN.search(x.get("description") or "")),
        "vision_alone_correct": vision_alone_correct if full_vision else None,
        "vision_alone_accuracy": (round(vision_alone_correct / n, 4) if n else 0.0) if full_vision else None,
    }


def join_cascade(xml_rows: list, vision_rows: list, source: str = "") -> dict:
    """Join the two result lists and count the cascade, overall and per phrasing.

    `source` keeps only element ids with that prefix (`gpt`, `opus`) — the same split
    `summary.by_source` makes, for the same reason: the two labellings are not equally hard.
    Pairs carry `None` as the vision row where XML answered; see the module docstring.
    """
    if source:
        pref = source + "-e"
        xml_rows = [r for r in xml_rows if str(r.get("element_id", "")).startswith(pref)]
        vision_rows = [r for r in vision_rows if str(r.get("element_id", "")).startswith(pref)]
    vision_by_key = {row_key(r): r for r in vision_rows}

    xml_errors = [r for r in xml_rows if r.get("error")]
    candidates = [r for r in xml_rows if not r.get("error")]
    # A row XML answered keeps `None` as its partner: the vision side is never consulted for it.
    pairs, unmatched = [], []
    for r in candidates:
        v = vision_by_key.get(row_key(r))
        if v is None and (r.get("xml_outcome") or "") in XML_FALL_THROUGH:
            unmatched.append(r)
        else:
            pairs.append((r, v))

    out = {
        "xml_rows": len(xml_rows),
        "vision_rows": len(vision_rows),
        "excluded_xml_errors": len(xml_errors),
        "excluded_fall_through_no_vision_row": len(unmatched),
        "vision_covers_all_rows": all(row_key(r) in vision_by_key for r in candidates),
        "overall": _count(pairs),
        "by_description": OrderedDict(),
    }
    styles = {}
    for x, _ in pairs:
        styles.setdefault(x.get("description_index", 0), x.get("description_style", ""))
    for idx in sorted(styles):
        sub = [(x, v) for x, v in pairs if x.get("description_index", 0) == idx]
        out["by_description"][str(idx)] = {"style": styles[idx], **_count(sub)}
    return out
