"""Turning a model's reply into a bounding box, or refusing to.

Eleven output shapes seen across models — clean `[x1,y1,x2,y2]`, nested arrays,
JSON-encoded strings, markdown fences, `{x,y,width,height}` dicts, list-wrapped values,
centre-point-only `{x,y}`, GUI-Owl's native `{"x": [cx, cy]}`, several JSON objects in
one reply, 5+ numbers, and preamble prose. Malformed JSON gets one repair attempt.

**The hard requirement is that prose is a failure, not a box.** A model reasoning in a
numbered list once handed over `1. … 2. … 3. … 4. …` and got a box back with no error
recorded: 34 identical copies of one artefact in the Gemma 4 12B run. Two rules keep
that out — numbers must arrive as a *group* (`_numeric_runs`), and a window gets **one**
scale decision for all four values (`_normalize_window`) rather than one per value.
Size cannot be the test: the artefact box was wider than the smallest real element in
the dataset.

`img_w`/`img_h` here are whatever the caller decided a pixel-scale answer divides by —
the screenshot's own dimensions, or a fixed grid twice over. See `coords._norm_dims`.
"""
import json
import re
from typing import Optional


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)

def _scalar(v) -> float:
    """Unwrap a value that may be a list (take first element) or a plain number."""
    if isinstance(v, list):
        return float(v[0]) if v else 0.0
    return float(v)


def _normalize_val(v: float, dim: int) -> float:
    """Normalize a single coordinate: if > 1.0 assume pixel and divide by dim."""
    if v > 1.0 and dim > 0:
        v = v / dim
    return max(0.0, min(1.0, v))


def _first_json(text: str):
    """Return the first valid JSON value from text (handles extra trailing data)."""
    text = text.strip()
    m = JSON_FENCE_RE.search(text)
    if m:
        text = m.group(1).strip()
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text)
        return obj, None
    except json.JSONDecodeError:
        pass
    # Fallback: try json_repair for malformed JSON (missing commas, quotes, etc.)
    try:
        from json_repair import repair_json
        repaired = repair_json(text, return_objects=True)
        if repaired is not None:
            return repaired, None
    except Exception:
        pass
    # Last resort: re-raise original parse error
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return None, str(e)
    return None, "unknown parse failure"


def _flatten_nums(obj) -> list:
    """Recursively extract all numbers from nested lists/dicts."""
    result = []
    if isinstance(obj, (int, float)):
        result.append(float(obj))
    elif isinstance(obj, list):
        for item in obj:
            result.extend(_flatten_nums(item))
    elif isinstance(obj, dict):
        for v in obj.values():
            result.extend(_flatten_nums(v))
    return result


def _extract_second_pair(content: str):
    """
    When content starts with a 2-element list like [69, 299], look for a second
    [x, y] pair in the remainder (handles '[69, 299],[957, 357]' and
    '[69, 299], 957, 357]' malformed variants).
    Returns (x2, y2) or None.
    """
    # Try wrapping full content in outer brackets and flatten
    try:
        from json_repair import repair_json
        wrapped = repair_json("[" + content.strip() + "]", return_objects=True)
        nums = _flatten_nums(wrapped)
        if len(nums) == 4:
            return nums[2], nums[3]
    except Exception:
        pass
    # Fallback: scan for a second number pair with a regex
    pairs = re.findall(r'\[?\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]?', content)
    if len(pairs) >= 2:
        return float(pairs[1][0]), float(pairs[1][1])
    return None


def _normalize_window(a, b, c, d, img_w: int, img_h: int) -> list:
    """Normalize four raw numbers under ONE scale decision for the whole window.

    Normalizing each value on its own is what let prose become a bounding box.
    `_normalize_val` divides by the image dimension only when a value exceeds 1.0, so
    in the window [1, 2, 3, 4] the `1` was read as an already-normalized 1.0 — the
    right-hand edge of the screen — while 2, 3 and 4 were read as pixels. Two
    conventions inside one box, and the result was a legal-looking box reported with
    no error: exactly the artefact that turned up 34 times in the Gemma 4 12B run.

    Deciding once, from the window's maximum, leaves a legitimate edge-hugging box
    like [0, 812, 300, 906] intact — its max is 906, so all four are pixels including
    the 0 — while [1, 2, 3, 4] becomes the 2x2-pixel box it actually describes.

    `img_w`/`img_h` are whatever `_parse_response` was handed: the screenshot's pixel
    dimensions for a model that answers in pixels, or the same grid value twice for one
    that answers on a fixed grid. See `_norm_dims`.
    """
    vals = (a, b, c, d)
    dims = (img_w, img_h, img_w, img_h)
    if max(vals) <= 1.0:
        return [max(0.0, min(1.0, v)) for v in vals]
    return [max(0.0, min(1.0, v / dim)) if dim > 0 else max(0.0, min(1.0, v))
            for v, dim in zip(vals, dims)]


def _try_bbox_from_nums(nums: list, img_w: int, img_h: int):
    """
    Given a flat list of numbers, try each consecutive window of 4 to find a valid
    normalized bbox.  Prefers windows where ALL four values are already in [0,1]
    (no pixel-scale division needed) so ordinal prefixes like "1." are skipped first.
    Falls back to pixel-scale windows if nothing better exists.
    """
    def _window_score(a, b, c, d):
        """Return (valid, already_normalized, normalized_bbox)."""
        an, bn, cn, dn = _normalize_window(a, b, c, d, img_w, img_h)
        if not (0 <= an <= 1 and 0 <= bn <= 1 and 0 < cn <= 1 and 0 < dn <= 1):
            return False, False, None
        already = max(a, b, c, d) <= 1.0
        # Corner reading only. A window whose third and fourth values do not exceed
        # the first two is rejected rather than reinterpreted as x/y/width/height —
        # dropping that rejection makes the scan accept the first window it is handed
        # instead of sliding past a leading stray number, which is the whole point of
        # sliding. `[1, 0.05, 0.32, 0.48, 0.38]` has to skip the 1 to find the box.
        if not (cn > an and dn > bn):
            return False, False, None
        return True, already, {"x": an, "y": bn,
                               "width": cn - an, "height": dn - bn}

    # First pass: prefer windows already in [0,1]
    for i in range(len(nums) - 3):
        valid, already, bbox = _window_score(*nums[i:i+4])
        if valid and already:
            return bbox, None, ""
    # Second pass: accept pixel-scale windows
    for i in range(len(nums) - 3):
        valid, already, bbox = _window_score(*nums[i:i+4])
        if valid:
            return bbox, None, ""
    return None, None, f"No valid bbox found in {len(nums)} numbers"


# Largest number of letters allowed between two numbers for them still to count as
# one group of coordinates. Enough for "x1=" or ", y:", nowhere near a clause.
MAX_LETTERS_BETWEEN_COORDS = 3


def _numeric_runs(text: str) -> list:
    """Split the numbers in `text` into groups adjacent enough to be coordinates.

    Real coordinates arrive as a group — "[0.24, 0.81, 0.83, 0.90]", "x1=245, y1=812,
    x2=835, y2=906" — while prose puts a sentence between them. A run therefore breaks
    wherever more than MAX_LETTERS_BETWEEN_COORDS letters separate two numbers, which
    turns "1. Scan the image … 2. Read the label … 3. … 4. …" into four runs of one
    number each, yielding no window of four at all.

    Size is deliberately not the test. The Gemma artefact box was 0.00411 wide and the
    smallest real element in dataset-v1 is 0.00370, so any floor that rejects the
    fabrication also rejects genuine 4px icons. Adjacency separates them cleanly;
    magnitude cannot.
    """
    runs: list = []
    current: list = []
    prev_end = None
    # The \b anchors are load-bearing, not decoration: they stop a digit glued to a
    # letter from counting as a number, so "x1=24, y1=345" yields 24 and 345 rather
    # than 1, 24, 1, 345 — where the label digits would be read as the box's corner.
    for m in re.finditer(r'\b\d+\.?\d*\b', text):
        if prev_end is not None:
            gap = text[prev_end:m.start()]
            if sum(ch.isalpha() for ch in gap) > MAX_LETTERS_BETWEEN_COORDS:
                if current:
                    runs.append(current)
                current = []
        current.append(float(m.group()))
        prev_end = m.end()
    if current:
        runs.append(current)
    return runs


def _no_coords_error(content: str) -> str:
    """The error for an answer that contains no group of numbers to read as a box."""
    snippet = " ".join(content.split())[:160]
    return f"no coordinates in response — the model answered in prose: {snippet!r}"


def _regex_extract_bbox(content: str, img_w: int, img_h: int):
    """
    Last-resort: find a group of numbers in the text and read it as a box.
    Handles cases where the model preambles before outputting coordinates.

    Only *groups* count — see `_numeric_runs`. Scanning the whole text indiscriminately
    is what let a prose answer score as a prediction, and a wrong number reported
    confidently is worse than a failure reported honestly. Same rule the truncation
    check applies one step earlier: an answer that never arrived is a failure, not a bbox.
    """
    for run in _numeric_runs(content):
        if len(run) < 4:
            continue
        bbox, point, _err = _try_bbox_from_nums(run, img_w, img_h)
        if bbox is not None or point is not None:
            return bbox, point, ""
    return None, None, _no_coords_error(content)


def _parse_response(content: str, img_w: int, img_h: int) -> tuple[Optional[dict], Optional[dict], str]:
    """
    Parse the model's raw text into a normalized bbox and/or click point.

    `img_w`/`img_h` are the numbers a pixel-scale answer is divided by, which are the
    screenshot's dimensions only for a model that answers in the screenshot's pixels.
    Callers pass `_norm_dims(img_w, img_h, coord_grid)`, so for a model on a fixed
    0-1000 grid both arrive as 1000 — see `COORD_GRIDS`. Nothing below needs to know
    which case it is in, and that is the point.

    Returns:
        pred_bbox  — {x, y, width, height} all 0-1 (top-left corner + dims), or None
        pred_point — {cx, cy} all 0-1 (center click), or None
        error      — non-empty string on failure

    Handles all output variants observed from GUI-Owl (Qwen3-VL), Gemma, and similar models:
      1. [x1, y1, x2, y2]                    — corner list (preferred / new prompt format)
      2. [[x1, y1, x2, y2]]                  — nested corner list
      3. {"x": f, "y": f, "width": f, ...}   — standard xywh dict
      4. {"x": f, "y": f, "width": [f], ...} — list-wrapped values
      5. {"x": f, "y": f}                    — center-point only
      6. {"x": [cx, cy]}                     — GUI-Owl native: both coords in one list key
      7. {"x": [cx, cy], "width": w, ...}    — GUI-Owl native with bbox dimensions
      8. Multiple JSON objects → take first
      9. 5+ numbers in a list → slide window of 4
     10. String-encoded JSON → unwrap and re-parse
     11. Preamble text + numbers → regex fallback extraction
    """
    if not content or not content.strip():
        return None, None, "empty response"

    parsed, err = _first_json(content)

    # Case 10: model returned a JSON-encoded string e.g. '"[0.1, 0.2, 0.3, 0.4]"'
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except Exception:
            # Fall through to regex fallback
            parsed = None

    if parsed is None:
        # Last resort: regex scan the raw text for numbers
        return _regex_extract_bbox(content, img_w, img_h)

    # Case 1/2/9: top-level list — [x1,y1,x2,y2], [[x1,y1],[x2,y2]], or [cx,cy]
    if isinstance(parsed, list):
        # Flatten all numbers from arbitrarily nested lists / mixed structures
        nums = _flatten_nums(parsed)
        if len(nums) == 4:
            a, b, c, d = nums
            an, bn, cn, dn = _normalize_window(a, b, c, d, img_w, img_h)
            bbox = ({"x": an, "y": bn, "width": cn - an, "height": dn - bn}
                    if (cn > an and dn > bn)
                    else {"x": an, "y": bn, "width": cn, "height": dn})
            return bbox, None, ""
        if len(nums) == 2:
            # Single click point [cx, cy] — but check if content has a SECOND pair
            extra = _extract_second_pair(content)
            if extra is not None:
                a, b = nums
                c, d = extra
                raw = {"x": a, "y": b, "width": c - a, "height": d - b} if (c > a and d > b) \
                      else {"x": a, "y": b, "width": c, "height": d}
                return _normalize_bbox(raw, img_w, img_h), None, ""
            # True click point
            cx_n = _normalize_val(nums[0], img_w)
            cy_n = _normalize_val(nums[1], img_h)
            return None, {"cx": cx_n, "cy": cy_n}, ""
        if len(nums) >= 5:
            # Model leaked extra numbers (preamble index, confidence, etc.) — slide window
            result = _try_bbox_from_nums(nums, img_w, img_h)
            if result[0] is not None:
                return result
        # Last resort on the raw text
        return _regex_extract_bbox(content, img_w, img_h)

    if not isinstance(parsed, dict):
        return _regex_extract_bbox(content, img_w, img_h)

    x_val = parsed.get("x")

    # Case 6/7: GUI-Owl native {"x": [cx_pixel, cy_pixel], ...}
    # The model encodes BOTH x and y as a 2-element list under the "x" key.
    if isinstance(x_val, list) and len(x_val) == 2 and "y" not in parsed:
        try:
            cx_raw, cy_raw = float(x_val[0]), float(x_val[1])
        except (TypeError, ValueError) as e:
            return None, None, f"Cannot parse [cx,cy] list: {e}"

        has_w = "width" in parsed
        has_h = "height" in parsed
        if has_w and has_h:
            try:
                w = _scalar(parsed["width"])
                h = _scalar(parsed["height"])
            except (TypeError, ValueError) as e:
                return None, None, f"Cannot parse width/height: {e}"
            # Convert from center to corner
            cx_n = _normalize_val(cx_raw, img_w)
            cy_n = _normalize_val(cy_raw, img_h)
            w_n  = _normalize_val(w, img_w)
            h_n  = _normalize_val(h, img_h)
            xl = max(0.0, cx_n - w_n / 2)
            yt = max(0.0, cy_n - h_n / 2)
            bbox = {
                "x": xl, "y": yt,
                "width":  min(1.0, cx_n + w_n / 2) - xl,
                "height": min(1.0, cy_n + h_n / 2) - yt,
            }
            return bbox, None, ""
        else:
            # Click point only
            cx_n = _normalize_val(cx_raw, img_w)
            cy_n = _normalize_val(cy_raw, img_h)
            return None, {"cx": cx_n, "cy": cy_n}, ""

    # Cases 3/4/5: standard dict with "x" and "y" keys
    if "x" not in parsed or "y" not in parsed:
        return None, None, f"Missing x/y in response: {content[:120]}"

    try:
        cx = _scalar(parsed["x"])
        cy = _scalar(parsed["y"])
    except (TypeError, ValueError) as e:
        return None, None, f"Cannot parse x/y: {e}"

    has_w = "width" in parsed
    has_h = "height" in parsed

    # Case 5: center-point only {x, y}
    if not has_w and not has_h:
        return None, {"cx": _normalize_val(cx, img_w), "cy": _normalize_val(cy, img_h)}, ""

    # Cases 3/4: full bbox {x, y, width, height}
    if has_w and has_h:
        try:
            w = _scalar(parsed["width"])
            h = _scalar(parsed["height"])
        except (TypeError, ValueError) as e:
            return None, None, f"Cannot parse width/height: {e}"
        raw = {"x": cx, "y": cy, "width": w, "height": h}
        return _normalize_bbox(raw, img_w, img_h), None, ""

    return None, None, f"Partial bbox (missing width or height): {content[:120]}"


def _normalize_bbox(raw: dict, img_w: int, img_h: int) -> dict:
    """Normalize {x, y, width, height} under one scale decision for all four fields.

    Per-field normalization had a latent version of the bug `_normalize_window`
    documents: `_normalize_val` leaves a value of 1.0 alone, so a one-pixel-wide box
    at x=245 came back with width=1.0 — the full screen — because 245 was divided by
    the image width and the 1 was not. Deciding once from the largest field keeps a
    model that answers in pixels and one that answers normalized both intact, and
    makes the mixed reading impossible.
    """
    xn, yn, wn, hn = _normalize_window(
        raw["x"], raw["y"], raw["width"], raw["height"], img_w, img_h)
    return {"x": xn, "y": yn, "width": wn, "height": hn}
