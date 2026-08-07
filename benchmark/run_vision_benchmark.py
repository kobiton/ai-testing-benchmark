"""
Vision benchmark - can a model locate a UI element from a natural-language description?

For each (element, phrasing) pair in the dataset:
  1. Send the screenshot + the description to an OpenAI-compatible /v1/chat/completions
  2. Parse the answer, handles bbox {x,y,w,h} and bare click point {x,y}, plus nine other shapes models have been observed to emit
  3. centroid (primary): does the centre of the predicted box fall inside the GT box?
  4. IoU (secondary): PASS at IoU >= threshold, default 0.5

Both metrics are always computed; --metric only picks the headline number.

Path defaults resolve against the repository root, so these work from anywhere:
    python benchmark/run_vision_benchmark.py --dry-run
    python benchmark/run_vision_benchmark.py --model qwen2.5-vl
    python benchmark/run_vision_benchmark.py --model qwen2.5-vl --description-index 0 1 2

Point it at any OpenAI-compatible server with --base-url (vLLM, llama.cpp, Ollama, OpenAI, or a gateway). See the repository README for endpoint recipes.
"""
import argparse
import base64
import io
import json
import logging
import os
import re
import signal
import statistics
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

import checkpoint
from sampling import even_sample

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# This file lives in <repo>/benchmark/, so the root is one level up. Every path default
# is built from this rather than from the caller's CWD — `python benchmark/run_...py`
# from the repository root is the documented way to run it, and relative defaults would
# resolve against wherever the shell happens to be.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = "dataset-v1.jsonl"

IOU_THRESHOLD = 0.5
WORST_CASES_N = 20

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

# Set by SIGTERM/SIGINT so the run finishes the elements already in flight and still
# writes a result file. Without it a stop produced nothing at all: the checkpoint held the
# scored elements, but the result JSON — the artifact everything downstream reads — is
# written once, at the very end.
_STOP = threading.Event()


def _install_stop_handlers() -> None:
    """First SIGTERM/SIGINT asks for a graceful stop; a second one kills the run.

    A graceful stop cancels whatever is still queued, lets the in-flight requests
    land, and writes a PARTIAL result. The escape hatch matters because "in flight"
    is up to --timeout seconds per worker, which on a slow endpoint is minutes.
    """
    def _handler(signum, _frame):
        if _STOP.is_set():
            logger.warning("Second signal — exiting now. Elements in flight are lost; "
                           "everything already scored is in the checkpoint.")
            os._exit(130)
        _STOP.set()
        logger.warning("Signal %d — cancelling the queue, finishing the %s in flight, "
                       "then writing a PARTIAL result. Send it again to exit now.",
                       signum, "requests")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


SYSTEM_PROMPT = """\
You are a visual grounding assistant for mobile UI screenshots.
Your only job: find the element described and output its bounding box.

Output format — a JSON array of four floats, nothing else:
[x1, y1, x2, y2]

Rules:
- x1,y1 = top-left corner; x2,y2 = bottom-right corner
- All values normalized 0.0–1.0 relative to image width/height
- x1 < x2, y1 < y2
- Output ONLY the array — no explanation, no preamble, no markdown

Good response:  [0.05, 0.32, 0.48, 0.38]
Bad response:   "The element is located at..." or ```json [...]```
"""

USER_PROMPT_TEMPLATE = 'Find the bounding box of: "{description}"\nAnswer:'

JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def _load_image_b64(path: str, max_dim: int = 0) -> tuple[str, str, int, int]:
    """Return (base64_data, media_type, width_px, height_px).

    `max_dim` downscales the longest side first; 0 — the default — sends the original.

    Downscaling to 1080 used to be the default, to match a consumer that downscales
    before sending. That was the wrong trade for this harness: the benchmark exists to
    measure how well a model can ground an element, and shrinking the screenshot first
    only handicaps it. Measuring a specific consumer's preprocessing is a different
    question, and `--max-image-dim 1080` still answers it.

    The number is not free. A screenshot in this corpus is 1080x2400, which a patch-based
    vision model turns into ~3300 image tokens against ~900 at max-dim 1080 — enough to
    overflow a 4096-token context before the prompt is counted, which is what "the request
    exceeds the available context size" means on a llama.cpp server. Native also multiplies
    input tokens per element by roughly four, so results either side of this default change
    are not comparable on cost; `max_image_dim` is recorded in every result for that reason.

    Predictions and ground truth are both compared in normalised coordinates, so the
    resolution the model saw does not change what is measured — but the *returned*
    dimensions must be the resized ones, since that is what pixel answers get normalised
    against.
    """
    if max_dim:
        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                w, h = round(w * scale), round(h * scale)
                buf = io.BytesIO()
                im.convert("RGB").resize((w, h), Image.LANCZOS).save(
                    buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode(), "image/jpeg", w, h

    suffix = Path(path).suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()
    w, h = 0, 0
    try:
        if suffix == ".png":
            w, h = struct.unpack(">II", raw[16:24])
        else:
            i = 2
            while i < len(raw) - 9:
                if raw[i] != 0xFF:
                    break
                marker = raw[i + 1]
                length = struct.unpack(">H", raw[i + 2:i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                    break
                i += 2 + length
    except Exception:
        pass
    return b64, media, w, h


# ---------------------------------------------------------------------------
# Coordinate space
# ---------------------------------------------------------------------------

# The grid a model's pixel-scale answers are expressed on, per model name. 0 — the
# default for anything not listed — means the model answers in the screenshot's own
# pixels, which is what the parser has always assumed.
#
# GUI-Owl (Qwen-VL family) does not. It answers on a fixed 0-1000 grid, so dividing its
# numbers by a 1080x2400 screenshot shrinks x by 1.08x and y by 2.4x — and 2.4x on one
# axis is the difference between a working locator and a useless one. Measured over a full
# three-phrasing GUI-Owl run on this dataset: **10.56% centroid as recorded, ~87.9% once
# the grid is applied.** The reverse holds too — applying the grid to Qwen2.5-VL, which
# really does answer in pixels, drops it from 84.40% to 0.31%. So this
# is a property of the model and there is no safe way to infer it per answer: on a
# 1080-wide screenshot `[67, 91]` is a legal pixel pair *and* a legal grid pair, which is
# exactly why the bug survived a full 30,921-call run without tripping anything.
#
# 1000 rather than 999 on purpose. Both fit — the sweep gives 85.62% at /1000 and 85.91%
# at /999 — but 1000 is the documented Qwen-VL convention (pixels scaled into [0,1000)),
# while 999 is the value that happens to score highest on this dataset. Fitting a
# constant to the benchmark it is then evaluated on is how a harness starts flattering
# itself; the 0.29pp is not worth that.
#
# Keys are matched as substrings of `_normalize_model_name(requested + served)`, so one
# entry covers both `gui-owl-1.5-8b` and a served id like
# `/models/GUI-Owl-1.5-8B-Instruct.Q4_K_M.gguf`.
COORD_GRIDS = {"guiowl": 1000}


def _coord_grid_for(model: str, served_model: str = "", override: int = -1) -> int:
    """The grid `model` answers on; 0 means the screenshot's own pixels.

    The served id is matched as well as the requested name because a single-model
    llama.cpp endpoint ignores the name it is asked for — the same reason
    `_probe_served_model` exists. A run asking for `qwen2.5-vl` against a box that has
    GUI-Owl loaded is scored under GUI-Owl's convention, which is the one that produced
    the numbers.
    """
    if override >= 0:
        return override
    hay = _normalize_model_name(f"{model} {served_model}")
    for key, grid in COORD_GRIDS.items():
        if key in hay:
            return grid
    return 0


def _norm_dims(img_w: int, img_h: int, coord_grid: int = 0) -> tuple[int, int]:
    """What a pixel-scale answer should be divided by to reach [0,1].

    Handing the grid over as *both* dimensions is the whole implementation: every place
    that normalises divides by these two numbers, so a model on a fixed grid needs no
    separate code path and cannot drift from the pixel path. The "already in [0,1]"
    branches keep working untouched, which is load-bearing — GUI-Owl mixes normalised
    floats (`[0.19, 0.81, 0.81, 0.85]`) into the same run as grid integers
    (`[67, 91]`), so both readings have to stay available within one model.
    """
    return (coord_grid, coord_grid) if coord_grid > 0 else (img_w, img_h)


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _compute_iou(pred: dict, gt: dict) -> float:
    px1, py1 = pred["x"], pred["y"]
    px2, py2 = px1 + pred["width"], py1 + pred["height"]
    gx1, gy1 = gt["x"], gt["y"]
    gx2, gy2 = gx1 + gt["width"], gy1 + gt["height"]
    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = pred["width"] * pred["height"] + gt["width"] * gt["height"] - inter
    return inter / union if union > 0 else 0.0


def _point_inside_bbox(cx: float, cy: float, gt: dict) -> bool:
    return (gt["x"] <= cx <= gt["x"] + gt["width"] and
            gt["y"] <= cy <= gt["y"] + gt["height"])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Endpoint identity
# ---------------------------------------------------------------------------

def _normalize_model_name(name: str) -> str:
    """Lowercase alphanumerics only, so `qwen2.5-vl` matches a served id like
    `/Users/x/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _auth_headers(api_key: str) -> dict:
    """Both auth conventions, because the endpoints this runs against disagree.

    `Authorization: Bearer` is the OpenAI-compatible standard and is what vLLM
    (`--api-key`), llama.cpp (`--api-key`), Ollama, OpenAI and every hosted gateway
    read. `X-API-Key` is what some self-hosted proxies use instead. Sending both costs
    one header and means the same command works against any of them; a server that does
    not recognise one simply ignores it.

    An empty key sends neither, which is the common local case — llama.cpp, vLLM and
    Ollama started without an API key reject a request carrying `Bearer ` with nothing
    after it on some builds, and have nothing to check on the rest.
    """
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}


def _list_loaded_models(client: httpx.Client, proxy_url: str, api_key: str) -> list:
    """Model ids the endpoint admits to having. Empty list if it won't say."""
    try:
        resp = client.get(f"{proxy_url.rstrip('/')}/v1/models",
                          headers=_auth_headers(api_key), timeout=15)
        if resp.status_code != 200:
            return []
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]
    except Exception as exc:
        logger.debug("Could not list models: %s", exc)
        return []


def _probe_served_model(client: httpx.Client, proxy_url: str, api_key: str,
                        model: str) -> str:
    """Ask for `model` and report which model actually answers.

    A single-model server (llama.cpp serving one .gguf) does not route by name: it
    answers every request with whatever is loaded and never says so. Asking for
    `gemma-4-12b` on a box running Qwen returns a perfectly normal Qwen answer, which is
    how two runs here once produced result files with byte-identical predictions under two
    different model names. The response's own `model` field is the only thing that tells
    the truth, so read it before spending hours on the run.
    """
    try:
        resp = client.post(
            f"{proxy_url.rstrip('/')}/v1/chat/completions",
            json={"model": model,
                  "messages": [{"role": "user", "content": "ping"}],
                  "max_tokens": 1},
            headers=_auth_headers(api_key), timeout=30,
        )
        if resp.status_code != 200:
            return ""
        return resp.json().get("model", "") or ""
    except Exception as exc:
        logger.debug("Could not probe served model: %s", exc)
        return ""


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

# How much of the model's own text to keep per element. **0 keeps all of it**, which is
# the default: `raw` is the only field that can tell a model's mistake from the parser's,
# and every coordinate-convention bug this project has hit was diagnosed by reading it —
# most recently GUI-Owl's 0-1000 grid, which was found by re-parsing `raw` out of a
# finished result file rather than by re-running 30,921 calls. Truncating it is
# truncating the evidence.
#
# The cost is bounded by `--max-tokens`, not by the model's mood: at the 2048 default a row
# can hold ~8 KB, so a pathological full run of this dataset would be ~250 MB against the
# ~27 MB it comes to in practice, where Qwen2.5-VL writes ~25 characters per answer and
# GUI-Owl ~12. Set this to a positive number to cap it again if a chatty model ever makes a
# result file unwieldy.
RAW_RESPONSE_CHARS = 0


# Set once, for the whole process, the first time a server refuses `temperature`.
#
# The benchmark sends temperature=0 because its main targets are self-hosted llama.cpp and
# vLLM endpoints, which honour it — and greedy decoding is what makes a coordinate
# reproducible. llama.cpp's own default is 0.8, so *not* sending it would put sampling noise
# straight into digit tokens, where a flip turns x=432 into x=332.
#
# Frontier models reject it outright: GPT-5.x answers "Unsupported value: 'temperature' does
# not support 0.0 with this model", and Claude Opus 4.7 and later return 400 for any
# non-default value. Since benchmark/README.md documents pointing this script at
# api.openai.com for a baseline, an unconditional temperature would fail every element of
# that run. Dropping it for the run instead costs reproducibility on that endpoint only,
# which is the lesser loss and is recorded in the result.
_TEMPERATURE_REJECTED = threading.Event()

# Matched against the response body, so a 400 about something else — a malformed image, a
# context overflow — is not silently treated as a temperature problem and retried blind.
_TEMPERATURE_ERR_RE = re.compile(r"temperature", re.I)


def _temperature_rejected(resp) -> bool:
    """True when this response is a refusal of the `temperature` parameter specifically."""
    if resp.status_code not in (400, 422):
        return False
    try:
        return bool(_TEMPERATURE_ERR_RE.search(resp.text))
    except Exception:
        return False


def _response_meta(usage: dict, msg: dict) -> dict:
    """Per-element diagnostics: token counts and the model's own words.

    `reasoning_content` is preferred when `content` is empty, because that is the
    case worth looking at — a thinking model that spent its budget and never answered.

    `cached_input_tokens` is recorded because prompt caching makes `input_tokens` swing
    for reasons that have nothing to do with the model: the system prompt is identical on
    every element, and so is the *image* across every element of the same screenshot, so
    whether a given call pays for those depends on what the server happened to serve
    before it — which the worker interleaving decides. Measured on the llama.cpp endpoint,
    14 of 21 prompt tokens came back cached on a bare text call. Without this field the
    variation looks like noise; with it, it is explained.
    """
    raw = msg.get("content") or msg.get("reasoning_content") or ""
    text = " ".join(str(raw).split())
    details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0) or 0,
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cached_input_tokens": details.get("cached_tokens", 0) or 0,
        "raw": text[:RAW_RESPONSE_CHARS] if RAW_RESPONSE_CHARS > 0 else text,
    }

def _call_model(
    client: httpx.Client,
    proxy_url: str,
    api_key: str,
    model: str,
    image_path: str,
    description: str,
    timeout: int,
    coord_format: str = "corner",
    thinking_budget: int = -1,
    max_image_dim: int = 0,
    max_tokens: int = 2048,
    temperature: float = 0,
    coord_grid: int = 0,
) -> tuple[Optional[dict], Optional[dict], float, str, dict]:
    """
    Returns (pred_bbox or None, pred_point or None, latency_ms, error, meta).
    pred_bbox  = {x, y, width, height} normalized 0-1
    pred_point = {cx, cy} normalized 0-1 (click-point models like GUI-Owl)
    meta       = {"input_tokens", "output_tokens", "raw", "img_w", "img_h"} — a dict
                 rather than five more positional values, so the next thing worth
                 recording does not change this signature again.

    `img_w`/`img_h` are in `meta` on **every** path, including the failures that never
    reach a response: they are known as soon as the image is loaded, and a row that
    records the dimensions its answer was normalised against is a row that can be
    re-scored later from `raw` alone. Without them a rescore has to re-derive them from
    the image file, which only works while the file is still in the cache and silently
    gives the wrong answer if the run used `--max-image-dim`.
    """
    b64, media, img_w, img_h = _load_image_b64(image_path, max_image_dim)
    # Repeated into every return below rather than merged once at the end, because most
    # of those returns are early exits.
    dims = {"img_w": img_w, "img_h": img_h}

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                    {"type": "text", "text": USER_PROMPT_TEMPLATE.format(description=description)},
                ],
            },
        ],
        # Four floats need a handful of tokens; the budget is for models that reason
        # first. Measured on Gemma 4 12B: 428-512 output tokens before the array, so 512
        # truncated roughly half of all calls. Default 2048 covers that with headroom;
        # Qwen2.5-VL answers directly and is unaffected. Raise it (--max-tokens) if a
        # heavier thinking model still truncates.
        "max_tokens": max_tokens,
        # thinking_budget=0 is meant to disable chain-of-thought on Qwen3/Gemma4-style
        # models. It only works where the server understands it: `budget_tokens` is a
        # vLLM/Anthropic-shaped parameter and llama.cpp ignores it outright — verified,
        # output tokens stayed at 430-512 with budget_tokens=0 — so against llama.cpp
        # this flag does nothing at all. Use it only with vLLM.
        **({"budget_tokens": thinking_budget} if thinking_budget >= 0 else {}),
        # Sent only where it is accepted — see _TEMPERATURE_REJECTED.
        **({"temperature": temperature}
           if temperature >= 0 and not _TEMPERATURE_REJECTED.is_set() else {}),
    }

    t0 = time.monotonic()
    last_err = ""

    def _post():
        return client.post(
            f"{proxy_url.rstrip('/')}/v1/chat/completions",
            json=payload,
            headers=_auth_headers(api_key),
            timeout=timeout,
        )

    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 ** attempt)  # 2s, 4s back-off
        try:
            resp = _post()
            # A server that rejects `temperature` rejects it on every call, so drop it for
            # the whole run rather than burning a wasted round-trip per element. Retried
            # inline rather than by `continue`, which would spend one of the three
            # connection attempts and report "Connection failed" if this were the last.
            if "temperature" in payload and _temperature_rejected(resp):
                logger.warning(
                    "Endpoint rejected temperature=%s (HTTP %d): %s. Dropping it for the "
                    "rest of the run and letting the model use its own sampling default. "
                    "Coordinates are digit tokens, so a non-zero default can move a "
                    "prediction between runs — see --temperature.",
                    payload["temperature"], resp.status_code, resp.text[:160])
                _TEMPERATURE_REJECTED.set()
                payload.pop("temperature")
                resp = _post()
            break
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_err = str(e)
            logger.debug("Attempt %d failed: %s", attempt + 1, e)
            continue
        except httpx.TimeoutException:
            return None, None, (time.monotonic() - t0) * 1000, "timeout", dict(dims)
        except Exception as e:
            return None, None, (time.monotonic() - t0) * 1000, str(e), dict(dims)
    else:
        return None, None, (time.monotonic() - t0) * 1000, f"Connection failed after 3 attempts: {last_err}", dict(dims)

    latency_ms = (time.monotonic() - t0) * 1000

    try:
        if resp.status_code != 200:
            return None, None, latency_ms, f"HTTP {resp.status_code}: {resp.text[:200]}", dict(dims)

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        finish = choice.get("finish_reason", "unknown")
        usage = data.get("usage") or {}

        # A truncated answer is a failure, not a prediction — check before falling back
        # to reasoning_content. A thinking model spends its whole budget reasoning and
        # returns content="" with finish_reason="length"; the fallback then handed the
        # raw reasoning prose to _parse_response, which scraped the "1." and "2." of a
        # numbered list into bbox {x: 1.0, y: 0.002, w: 0.004, h: 0.002} and reported no
        # error. That one bogus box, repeated, is what a Gemma 4 12B run recorded as 68%
        # zero-overlap and 16% centroid accuracy. Silently manufacturing a wrong number
        # is worse than reporting the failure.
        meta = _response_meta(usage, msg)
        meta.update(dims)

        if finish == "length":
            return None, None, latency_ms, (
                f"Truncated at max_tokens={payload['max_tokens']} "
                f"(completion_tokens={meta['output_tokens']}): the answer never arrived. "
                f"Thinking models need a larger cap — raise --max-tokens."
            ), meta

        content = msg.get("content") or msg.get("reasoning_content", "")
        if not content:
            return None, None, latency_ms, (
                f"Empty content (finish_reason={finish}, "
                f"prompt_tokens={meta['input_tokens']}). "
                "Image may not be reaching the model."
            ), meta

        norm_w, norm_h = _norm_dims(img_w, img_h, coord_grid)
        pred_bbox, pred_point, parse_err = _parse_response(content, norm_w, norm_h)
        if parse_err:
            return None, None, latency_ms, parse_err, meta

        # If the model returns (x,y) as the CENTER (not top-left), convert to corner.
        if coord_format == "center" and pred_bbox:
            cx = pred_bbox["x"]
            cy = pred_bbox["y"]
            w  = pred_bbox["width"]
            h  = pred_bbox["height"]
            xl = max(0.0, cx - w / 2)
            yt = max(0.0, cy - h / 2)
            pred_bbox = {
                "x": xl, "y": yt,
                "width":  min(1.0, cx + w / 2) - xl,
                "height": min(1.0, cy + h / 2) - yt,
            }

        return pred_bbox, pred_point, latency_ms, "", meta

    except httpx.TimeoutException:
        return None, None, (time.monotonic() - t0) * 1000, "timeout", dict(dims)
    except Exception as e:
        return None, None, (time.monotonic() - t0) * 1000, str(e), dict(dims)


# ---------------------------------------------------------------------------
# Single-element benchmark
# ---------------------------------------------------------------------------

def _benchmark_element(
    client: httpx.Client,
    proxy_url: str,
    api_key: str,
    model: str,
    row: dict,
    images_dir: str,
    description_index: int,
    timeout: int,
    coord_format: str = "corner",
    thinking_budget: int = -1,
    max_image_dim: int = 0,
    max_tokens: int = 2048,
    temperature: float = 0,
    coord_grid: int = 0,
) -> dict:
    screenshot_id = row["screenshot_id"]
    element_id = row["element_id"]
    gt_bbox = row["bbox"]
    descriptions = row["descriptions"]
    # The caller only ever passes an index this element actually has — `_expand_pairs`
    # drops the rest rather than clamping. Clamping is what the old single-index path
    # did, and under multi-phrasing it would score index 0's text three times over and
    # file the copies as `label` and `intent`, so agreement between phrasings would be
    # measuring a dataset gap. Kept as a floor only so a stray index cannot IndexError.
    description_index = min(description_index, len(descriptions) - 1)
    desc = descriptions[description_index]
    # Repeated on every row because the row is the unit everything downstream reads:
    # the compare API groups by it, and a result row that cannot say which phrasing
    # produced it is not attributable to anything.
    ident = {"screenshot_id": screenshot_id, "element_id": element_id,
             "description_index": description_index,
             "description_style": _style_of(description_index)}

    image_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = Path(images_dir) / f"{screenshot_id}{ext}"
        if candidate.exists():
            image_path = str(candidate)
            break

    if not image_path:
        # `pass_centroid` is spelled out even though it is always False here. It is the
        # key `build_result` selects worst_cases on under the default metric, and leaving
        # it off made a single missing screenshot end the whole run in a KeyError — at
        # write time, after every other element had already been paid for.
        return {
            **ident,
            "description": desc, "gt_bbox": gt_bbox,
            "pred_bbox": None, "pred_point": None,
            "iou": 0.0, "pass_iou": False,
            "click_inside": False, "pass_centroid": False,
            "error": "image not found", "latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "raw": "",
            "img_w": 0, "img_h": 0,
        }

    pred_bbox, pred_point, latency_ms, error, meta = _call_model(
        client, proxy_url, api_key, model, image_path, desc, timeout, coord_format,
        thinking_budget, max_image_dim, max_tokens, temperature, coord_grid
    )

    iou = _compute_iou(pred_bbox, gt_bbox) if pred_bbox else 0.0
    pass_iou = iou >= IOU_THRESHOLD

    # Centroid metric (aka click accuracy): the centroid of the predicted bbox must
    # fall inside the ground-truth bbox. Use bbox center if a bbox was returned,
    # otherwise use the raw predicted click point (GUI-Owl-style models).
    click_inside = False
    if pred_bbox:
        cx = pred_bbox["x"] + pred_bbox["width"] / 2
        cy = pred_bbox["y"] + pred_bbox["height"] / 2
        click_inside = _point_inside_bbox(cx, cy, gt_bbox)
    elif pred_point:
        click_inside = _point_inside_bbox(pred_point["cx"], pred_point["cy"], gt_bbox)

    return {
        **ident,
        "description": desc,
        "gt_bbox": gt_bbox,
        "pred_bbox": pred_bbox,
        "pred_point": pred_point,
        "iou": round(iou, 4),
        "pass_iou": pass_iou,
        "click_inside": click_inside,     # kept for backward compat
        "pass_centroid": click_inside,    # centroid metric — the primary one
        "error": error,
        "latency_ms": round(latency_ms, 1),
        # Token counts for the cost-versus-accuracy comparison, and the model's own
        # text so a bad score can be attributed to the model or to the parser.
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "cached_input_tokens": meta.get("cached_input_tokens", 0),
        "raw": meta.get("raw", ""),
        # The dimensions this answer was normalised against — the resized ones when
        # --max-image-dim is in force, not the file's. Recorded per row so `--rescore`
        # can re-read `raw` without the images, and without having to guess whether a
        # downscale happened. See `_call_model`.
        "img_w": meta.get("img_w", 0),
        "img_h": meta.get("img_h", 0),
    }


# ---------------------------------------------------------------------------
# Run benchmark for one model
# ---------------------------------------------------------------------------

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


def run_benchmark(
    dataset: list,
    model: str,
    proxy_url: str,
    api_key: str,
    images_dir: str,
    workers: int,
    description_indices: list,
    timeout: int,
    dry_run: bool,
    coord_format: str = "corner",
    thinking_budget: int = -1,
    metric: str = "centroid",
    max_image_dim: int = 0,
    max_tokens: int = 2048,
    temperature: float = 0,
    output_dir: str = "",
    dataset_path: str = "",
    no_resume: bool = False,
    coord_grid: int = -1,
) -> dict:
    pairs = _expand_pairs(dataset, description_indices)
    styles = ", ".join(_style_of(i) for i in description_indices)
    logger.info(
        "Benchmarking model=%s on %d element(s) x %d phrasing(s) [%s] = %d call(s), "
        "metric=%s", model, len(dataset), len(description_indices), styles,
        len(pairs), metric)
    short = len(dataset) * len(description_indices) - len(pairs)
    if short:
        logger.info("  %d pair(s) skipped: the element has fewer descriptions than that",
                    short)

    if dry_run:
        logger.info("Dry-run: skipping API calls")
        return {"model": model, "dry_run": True,
                "total_elements": len(pairs), "unique_elements": len(dataset),
                "results": []}

    ckpt = checkpoint.path_for(output_dir or ".", dataset_path, model)
    if no_resume:
        ckpt.unlink(missing_ok=True)

    results = []
    with httpx.Client() as client:
        loaded_models = _list_loaded_models(client, proxy_url, api_key)
        served_model = _probe_served_model(client, proxy_url, api_key, model)
        mismatch = bool(served_model) and (
            _normalize_model_name(model) not in _normalize_model_name(served_model))
        if served_model:
            logger.info("  endpoint served: %s", served_model)
        if mismatch:
            logger.warning(
                "MODEL MISMATCH — asked for %r, the endpoint answered as %r. This "
                "endpoint does not route by model name, so these results describe "
                "whatever model is currently loaded, NOT %s. Restart the server on the "
                "model you want and re-run.",
                model, served_model, model)
        elif not served_model:
            logger.warning("Could not determine which model the endpoint serves; "
                           "results will not record it.")

        # Decided here rather than in main() because the served id is only known after
        # the probe, and on a single-model endpoint that id is the one that decides the
        # convention — see `_coord_grid_for`.
        grid = _coord_grid_for(model, served_model, coord_grid)
        if grid:
            logger.info("  coordinate grid: 0-%d (pixel-scale answers are divided by "
                        "%d, not by the screenshot's dimensions)", grid, grid)

        resumed, done, meta = checkpoint.read(ckpt)
        if resumed and meta.get("served_model", served_model) != served_model:
            logger.error(
                "Checkpoint %s was scored by %r but the endpoint now serves %r. "
                "Resuming would blend two models into one score. Re-load that model, "
                "or pass --no-resume to discard %d scored element(s) and start over.",
                ckpt, meta.get("served_model"), served_model, len(done))
            sys.exit(1)

        # Same refusal, same reason, for the coordinate grid. Rows scored under two
        # conventions in one file is worse than the served-model case, not better: there
        # is no field on a row that says which convention produced it, so the blend is
        # undetectable afterwards. A checkpoint written before the grid existed reports
        # None and is read as this run's value — those runs were all pixel-scale, which
        # is what `grid` is for every model except the ones in COORD_GRIDS.
        prior_grid = meta.get("coord_grid")
        if resumed and prior_grid is not None and prior_grid != grid:
            logger.error(
                "Checkpoint %s was scored on coordinate grid %s but this run uses %s. "
                "Resuming would put two coordinate conventions in one score and nothing "
                "downstream could tell them apart. Pass --coord-grid %s to match it, or "
                "--no-resume to discard %d scored pair(s) and start over.",
                ckpt, prior_grid or "pixel", grid or "pixel", prior_grid, len(done))
            sys.exit(1)

        # A checkpoint may hold pairs outside this run's selection — a smaller --limit,
        # or the phrasings of an earlier run this one does not ask for — so keep only
        # what belongs to the current set.
        wanted = {(row["screenshot_id"], row["element_id"], idx) for row, idx in pairs}
        results = [r for r in resumed if checkpoint.key_of(r) in wanted]
        done &= wanted
        todo = [(row, idx) for row, idx in pairs
                if (row["screenshot_id"], row["element_id"], idx) not in done]
        if done:
            logger.info("  %d already scored, %d to go", len(results), len(todo))

        # max_tokens rides along for the same reason served_model does: --finalize-only
        # rebuilds a result from this header alone, and the honest value is the one the
        # run actually used, not whatever the flag says whenever someone gets round to
        # recovering it. It also decides whether a truncation counts as an error.
        meta = {"served_model": served_model, "model": model,
                "dataset": Path(dataset_path).name,
                "description_indices": description_indices,
                "max_tokens": max_tokens,
                "workers": workers,
                "temperature": temperature,
                "coord_grid": grid}

        futures = {}
        stopped_early = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, idx in todo:
                f = pool.submit(
                    _benchmark_element,
                    client, proxy_url, api_key, model,
                    row, images_dir, idx, timeout, coord_format,
                    thinking_budget, max_image_dim, max_tokens, temperature,
                    grid,
                )
                futures[f] = (row, idx)

            with tqdm(total=len(futures), desc=f"{model}") as pbar:
                for f in as_completed(futures):
                    # Cancelled by the stop below. as_completed yields them straight
                    # away and .result() would raise CancelledError.
                    if f.cancelled():
                        pbar.update(1)
                        continue
                    r = f.result()
                    results.append(r)
                    # Errors stay out so a resume retries them — see checkpoint.py.
                    if not r["error"]:
                        checkpoint.append(ckpt, meta, r)
                    pbar.update(1)

                    # Cancel the queue but keep draining rather than breaking out:
                    # the futures still running would otherwise finish into nothing,
                    # losing up to `workers` elements from both the result and the
                    # checkpoint. Cancelled ones come back immediately, so the loop
                    # ends as soon as the last in-flight request lands.
                    if _STOP.is_set() and not stopped_early:
                        stopped_early = True
                        cancelled = sum(1 for g in futures if g.cancel())
                        logger.warning(
                            "Stopping early — %d queued element(s) cancelled, "
                            "waiting for the ones in flight", cancelled)

    return build_result(
        results=results,
        model=model,
        served_model=served_model,
        mismatch=mismatch,
        loaded_models=loaded_models,
        metric=metric,
        description_indices=description_indices,
        thinking_budget=thinking_budget,
        max_image_dim=max_image_dim,
        max_tokens=max_tokens,
        workers=workers,
        temperature=temperature,
        coord_grid=grid,
        unique_elements=len(dataset),
        # Pairs, not elements: `completed_fraction` divides the rows scored by this, and
        # a 3-phrasing run produces three rows per element. Counting elements here would
        # report a finished run as 300% complete.
        selected_elements=len(pairs),
        stopped_early=stopped_early,
        checkpoint_path=str(ckpt),
    )


# Below this many predictions the scale check reports its numbers but reaches no verdict:
# a handful of elements from one screen can genuinely sit in the top third, and a false
# alarm on the field that exists to catch a silent bug is worse than no field.
SCALE_CHECK_MIN_ROWS = 100

# A per-axis ratio outside this band is the fingerprint. The band is wide on purpose —
# the failure it is looking for was 0.45, and a model with an honest bias toward the top
# of the screen should not be accused of a parser bug.
SCALE_RATIO_BAND = (0.8, 1.25)


def _predicted_centroid(r: dict):
    """The point a prediction would tap, or None. Bbox centre, else the click point."""
    b = r.get("pred_bbox")
    if b:
        return b["x"] + b["width"] / 2, b["y"] + b["height"] / 2
    p = r.get("pred_point")
    if p:
        return p["cx"], p["cy"]
    return None


def _scale_check(results: list) -> dict:
    """Are the predictions distributed like the ground truth on each axis?

    This is the check that was missing. GUI-Owl's 0-1000 grid read as 1080x2400 pixels
    produced a full, clean, error-free 30,921-row run reporting 10.56% centroid accuracy,
    and nothing in the artifact said the number described the harness rather than the
    model. Every existing guard looked at individual answers, where a grid value and a
    pixel value are indistinguishable; the tell only exists in aggregate. Over thousands
    of rows a model that finds elements at all must scatter its answers roughly the way
    the targets are scattered, so a median predicted centroid at 0.45x the median ground
    truth on one axis and 0.92x on the other is not a model being wrong — it is a divisor
    being wrong, and the two axes disagree because the screenshot is not square.

    Deliberately only a flag. It does not rescale anything: guessing a convention from a
    statistic and silently applying it is how this codebase produced its worst data, and
    a wrong guess here would be indistinguishable from a correct one downstream.
    """
    pairs = [(_predicted_centroid(r), r["gt_bbox"]) for r in results
             if _predicted_centroid(r) and r.get("gt_bbox")]
    if not pairs:
        return {}
    pred_cx = [p[0] for p, _ in pairs]
    pred_cy = [p[1] for p, _ in pairs]
    gt_cx = [g["x"] + g["width"] / 2 for _, g in pairs]
    gt_cy = [g["y"] + g["height"] / 2 for _, g in pairs]

    def _ratio(pred, gt):
        m = statistics.median(gt)
        return round(statistics.median(pred) / m, 4) if m else 0.0

    x_ratio, y_ratio = _ratio(pred_cx, gt_cx), _ratio(pred_cy, gt_cy)
    lo, hi = SCALE_RATIO_BAND
    enough = len(pairs) >= SCALE_CHECK_MIN_ROWS
    off_band = not (lo <= x_ratio <= hi) or not (lo <= y_ratio <= hi)
    # Axis disagreement is the sharper signal of the two, because a single wrong divisor
    # hits the longer side harder: on 1080x2400 a 0-1000 grid is 1.08x out on x and 2.4x
    # on y. A model that is merely biased is wrong in the same direction on both.
    axes_disagree = bool(min(x_ratio, y_ratio)) and (
        max(x_ratio, y_ratio) / min(x_ratio, y_ratio) > 1.25)
    return {
        "predictions": len(pairs),
        "pred_median_cx": round(statistics.median(pred_cx), 4),
        "pred_median_cy": round(statistics.median(pred_cy), 4),
        "gt_median_cx": round(statistics.median(gt_cx), 4),
        "gt_median_cy": round(statistics.median(gt_cy), 4),
        "x_ratio": x_ratio,
        "y_ratio": y_ratio,
        "axes_disagree": axes_disagree,
        "suspect": bool(enough and (off_band or axes_disagree)),
    }


def build_result(
    results: list,
    model: str,
    served_model: str,
    mismatch: bool,
    loaded_models: list,
    metric: str,
    description_indices: list,
    thinking_budget: int,
    max_image_dim: int,
    max_tokens: int,
    workers: int,
    temperature: float,
    unique_elements: int,
    selected_elements: int,
    stopped_early: bool = False,
    checkpoint_path: str = "",
    coord_grid: int = 0,
) -> dict:
    """Score `results` and shape them into the result file.

    Split out of `run_benchmark` because three paths need it and they must produce
    the same shape: a completed run, a run stopped part-way, and `--finalize-only`
    rebuilding a file from a checkpoint left behind by a run that was killed.
    """
    total = len(results)
    ious = [r["iou"] for r in results if r["pred_bbox"] is not None]
    pass_iou = sum(1 for r in results if r["pass_iou"])
    pass_click = sum(1 for r in results if r["click_inside"])
    error_count = sum(1 for r in results if r["error"])
    timeout_count = sum(1 for r in results if r["error"] == "timeout")
    point_only = sum(1 for r in results if r["pred_point"] and not r["pred_bbox"])
    latencies = [r["latency_ms"] for r in results]

    # Did the model answer in the shape it was asked for at all, and — separately — were the
    # shapes it did produce any good? Two different failures got merged into one low IoU and
    # the distinction is the whole model-selection question:
    #
    #   iou_accuracy = bbox_coverage x iou_accuracy_given_bbox
    #
    # A model can miss the IoU bar because our ground truth boxes the padded control while it
    # boxes the glyph — a convention disagreement, which is why `--metric centroid` exists —
    # or because it never returned a box. GUI-Owl is the second: 15.1% coverage against
    # Qwen2.5-VL's 100%, because it is an agentic model trained to emit click points. Both
    # land in `iou_accuracy` as 2.9% vs 49.2% and neither says which happened.
    #
    # Derivable from `point_only_count` before this, and so it appeared in no report: a
    # centroid of 87.8% next to Qwen's 85.2% read as GUI-Owl winning, when Appium AI needs a
    # bounding box and GUI-Owl supplies one in one request out of seven. Coverage is a gate,
    # not a metric — a model below it should not be compared on accuracy at all.
    bbox_rows = [r for r in results if r["pred_bbox"]]
    bbox_iou_pass = sum(1 for r in bbox_rows if r["pass_iou"])

    # Cost side of the comparison. `elements_with_token_counts` is separate on purpose:
    # a server that reports no `usage` block leaves the totals at 0, which is not the
    # same finding as a model that is genuinely cheap.
    in_toks = [r.get("input_tokens", 0) for r in results]
    out_toks = [r.get("output_tokens", 0) for r in results]
    cached_toks = [r.get("cached_input_tokens", 0) for r in results]
    counted = sum(1 for r in results
                  if r.get("input_tokens") or r.get("output_tokens"))

    # A prediction pushed against the [0,1] edge is the fingerprint of a coordinate
    # convention the parser read wrongly, not of a model aiming at the screen edge.
    # 0.4% on the Qwen2.5-VL run, 18.2% on Gemma 4 12B — where it was prose being
    # scraped into a box. A model answering on a 0-999 scale would show most of its
    # predictions here, so read this before believing a low accuracy.
    clamped = sum(1 for r in results if r["pred_bbox"] and max(
        r["pred_bbox"]["x"], r["pred_bbox"]["y"],
        r["pred_bbox"]["width"], r["pred_bbox"]["height"]) >= 0.999)

    scale_check = _scale_check(results)

    iou_accuracy = round(pass_iou / total, 4) if total else 0
    centroid_accuracy = round(pass_click / total, 4) if total else 0
    # Primary metric drives the headline accuracy + which cases count as "failed".
    primary_accuracy = centroid_accuracy if metric == "centroid" else iou_accuracy

    def _passed(r: dict) -> bool:
        if metric == "centroid":
            return bool(r.get("pass_centroid", r.get("click_inside")))
        return bool(r.get("pass_iou"))

    multi = len(description_indices) > 1

    # Per-phrasing breakdown. The headline accuracy averages the phrasings, which is the
    # honest single number for "how well does this model do on the ways a tester might
    # actually ask" — but it hides the spread, and the spread is the finding. `name` is
    # usually the element's visible text while `intent` deliberately carries none, so a
    # model that grounds by reading text scores far apart on the two. One number over
    # `name` alone measures the easiest case and reads as if it measured the feature.
    by_description = {}
    if multi:
        for idx in sorted({r.get("description_index", 0) for r in results}):
            rows = [r for r in results if r.get("description_index", 0) == idx]
            n = len(rows)
            d_ious = [r["iou"] for r in rows if r["pred_bbox"] is not None]
            d_in = [r.get("input_tokens", 0) for r in rows]
            d_out = [r.get("output_tokens", 0) for r in rows]
            d_cent = sum(1 for r in rows if r["click_inside"])
            d_iou_pass = sum(1 for r in rows if r["pass_iou"])
            by_description[str(idx)] = {
                "style": _style_of(idx),
                "total": n,
                "primary_accuracy": round(
                    sum(1 for r in rows if _passed(r)) / n, 4) if n else 0,
                "centroid_pass_count": d_cent,
                "centroid_accuracy": round(d_cent / n, 4) if n else 0,
                "iou_pass_count": d_iou_pass,
                "iou_accuracy": round(d_iou_pass / n, 4) if n else 0,
                "mean_iou": round(statistics.mean(d_ious), 4) if d_ious else 0,
                "median_iou": round(statistics.median(d_ious), 4) if d_ious else 0,
                "avg_input_tokens": round(statistics.mean(d_in), 1) if d_in else 0,
                "avg_output_tokens": round(statistics.mean(d_out), 1) if d_out else 0,
                "avg_latency_ms": round(
                    statistics.mean([r["latency_ms"] for r in rows]), 1) if rows else 0,
                "error_count": sum(1 for r in rows if r["error"]),
            }

    # Agreement between phrasings, over the elements that have every phrasing scored.
    # Restricted that way on purpose: a stopped run leaves elements holding one or two of
    # their three, and folding those in would score a 1-of-1 as unanimous and push
    # `all_accuracy` up exactly when the run is least complete.
    #
    # `any` is the ceiling a better prompt could reach — the element was findable, some
    # phrasing found it. `all` is robustness. `mixed` is the population worth inspecting
    # element by element, because those elements say which phrasing the model cannot use.
    agreement = {}
    if multi:
        per_element: dict = {}
        for r in results:
            per_element.setdefault(
                (r["screenshot_id"], r["element_id"]), []).append(_passed(r))
        full = [v for v in per_element.values() if len(v) == len(description_indices)]
        any_pass = sum(1 for v in full if any(v))
        all_pass = sum(1 for v in full if all(v))
        agreement = {
            "phrasings": len(description_indices),
            # Elements with a complete set. Below unique_elements on a stopped run, and
            # on a dataset whose labeling gave some elements fewer than three descriptions.
            "scored_elements": len(full),
            "any_pass_count": any_pass,
            "any_accuracy": round(any_pass / len(full), 4) if full else 0,
            "all_pass_count": all_pass,
            "all_accuracy": round(all_pass / len(full), 4) if full else 0,
            "mixed_count": any_pass - all_pass,
        }

    summary = {
        # Rows, i.e. how many times a model was asked to ground something. On a
        # multi-phrasing run this is elements x phrasings, and every accuracy here is
        # over it — `unique_elements` is the element count those rows describe.
        "total_elements": total,
        "unique_elements": unique_elements,
        # How much of the run actually happened. Equal to total_elements on a normal
        # run; smaller on a stopped one, and the only place the shortfall is visible
        # as a number — every accuracy above is over what was scored, not what was
        # asked for, so a 12%-complete run can look as authoritative as a full one.
        "selected_elements": selected_elements,
        "completed_fraction": round(total / selected_elements, 4) if selected_elements else 0,
        # Which metric is primary for this run
        "metric": metric,
        "primary_accuracy": primary_accuracy,
        # IoU metric
        "iou_pass_count": pass_iou,
        "iou_fail_count": total - pass_iou,
        "iou_accuracy": iou_accuracy,
        "mean_iou": round(statistics.mean(ious), 4) if ious else 0,
        "median_iou": round(statistics.median(ious), 4) if ious else 0,
        "iou_threshold": IOU_THRESHOLD,
        # Centroid metric (centroid of predicted bbox inside gt bbox). click_* kept
        # as aliases for backward compatibility with older result files / UI.
        "centroid_pass_count": pass_click,
        "centroid_accuracy": centroid_accuracy,
        "click_pass_count": pass_click,
        "click_accuracy": centroid_accuracy,
        "point_only_count": point_only,
        # Did it answer in the requested shape, and were those shapes any good — see above.
        # `bbox_count` is spelled out rather than left as a subtraction because the errored
        # rows mean total - point_only is not it.
        "bbox_count": len(bbox_rows),
        "bbox_coverage": round(len(bbox_rows) / total, 4) if total else 0,
        "iou_accuracy_given_bbox": (round(bbox_iou_pass / len(bbox_rows), 4)
                                    if bbox_rows else 0),
        "clamped_pred_count": clamped,
        # Read this before believing a low accuracy — see `_scale_check`. Omitted when
        # there was nothing to predict against.
        **({"scale_check": scale_check} if scale_check else {}),
        # Both omitted entirely on a single-phrasing run, where they would only restate
        # the figures above: `by_description` would hold one entry copying the summary,
        # and `any`/`all` would both equal the accuracy. Absent means "not measured".
        **({"by_description": by_description} if by_description else {}),
        **({"agreement": agreement} if agreement else {}),
        # Tokens — the cost column. Self-hosted models have no per-token price, so the
        # figure to compare is throughput: tokens and seconds per element at a known
        # concurrency, against the hourly cost of the GPU.
        "total_input_tokens": sum(in_toks),
        "total_output_tokens": sum(out_toks),
        "avg_input_tokens": round(statistics.mean(in_toks), 1) if in_toks else 0,
        "avg_output_tokens": round(statistics.mean(out_toks), 1) if out_toks else 0,
        # Of the input above, how much the server served from its prompt cache. High and
        # variable is normal — the system prompt repeats on every element and the image
        # repeats across every element of one screenshot — so this is what stops the
        # input figure looking erratic.
        "total_cached_input_tokens": sum(cached_toks),
        "cached_input_fraction": (round(sum(cached_toks) / sum(in_toks), 4)
                                  if sum(in_toks) else 0),
        "p95_output_tokens": (round(sorted(out_toks)[int(len(out_toks) * 0.95)], 1)
                              if out_toks else 0),
        "elements_with_token_counts": counted,
        # Errors
        "error_count": error_count,
        "timeout_count": timeout_count,
        # Latency
        "avg_latency_ms": round(statistics.mean(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(statistics.median(latencies), 1) if latencies else 0,
        "p95_latency_ms": round(sorted(latencies)[int(total * 0.95)], 1) if latencies else 0,
    }

    # Worst cases = failures under the PRIMARY metric, ordered worst-first by IoU.
    # Read the same defensive way `_pass` above does: `--rescore` and `--finalize-only`
    # take their rows from a file that may predate `pass_centroid`, where the equivalent
    # is `click_inside`. A subscript here crashed the run rather than the row.
    def _failed_primary(r):
        if metric == "centroid":
            return not bool(r.get("pass_centroid", r.get("click_inside")))
        return not bool(r.get("pass_iou"))

    failed = sorted([r for r in results if _failed_primary(r)], key=lambda r: r["iou"])
    worst_cases = failed[:WORST_CASES_N]

    logger.info(
        "  primary(%s)=%.1f%%  IoU=%.1f%%  centroid=%.1f%%  mean_iou=%.3f  errors=%d",
        metric, primary_accuracy * 100,
        iou_accuracy * 100,
        centroid_accuracy * 100,
        summary["mean_iou"],
        summary["error_count"],
    )
    if point_only:
        logger.warning(
            "  bbox coverage %.1f%% — %d of %d answer(s) were a bare click point, so they "
            "carry no box and score 0 on IoU by construction. The IoU %.1f%% above is "
            "%.1f%% coverage x %.1f%% among the answers that did return a box. Where the "
            "consumer needs a bounding box, coverage is the gate: these accuracies are not "
            "comparable with a model that always returns one.",
            summary["bbox_coverage"] * 100, point_only, total, iou_accuracy * 100,
            summary["bbox_coverage"] * 100, summary["iou_accuracy_given_bbox"] * 100)
    for idx, d in by_description.items():
        logger.info("    %-7s %s=%.1f%%  IoU=%.1f%%  (%d scored, %d errors)",
                    d["style"], metric, d["primary_accuracy"] * 100,
                    d["iou_accuracy"] * 100, d["total"], d["error_count"])
    if agreement:
        logger.info("    agreement over %d element(s) with all %d phrasing(s): "
                    "any=%.1f%%  all=%.1f%%  mixed=%d",
                    agreement["scored_elements"], agreement["phrasings"],
                    agreement["any_accuracy"] * 100, agreement["all_accuracy"] * 100,
                    agreement["mixed_count"])
    if stopped_early:
        logger.warning(
            "  PARTIAL — %d of %d selected element(s) scored (%.1f%%). The accuracies "
            "above describe only those.",
            total, selected_elements, summary["completed_fraction"] * 100)
    if scale_check.get("suspect"):
        logger.warning(
            "  SCALE CHECK FAILED — median predicted centroid is %.3f/%.3f against a "
            "ground truth of %.3f/%.3f, i.e. x is %.2fx and y is %.2fx. Over %d "
            "predictions that is a coordinate convention being read wrongly, not a model "
            "aiming somewhere odd, and the accuracies above measure this harness rather "
            "than %s. Check what grid the model answers on and re-score with "
            "--rescore <this file> --coord-grid <N>; no inference is needed.",
            scale_check["pred_median_cx"], scale_check["pred_median_cy"],
            scale_check["gt_median_cx"], scale_check["gt_median_cy"],
            scale_check["x_ratio"], scale_check["y_ratio"],
            scale_check["predictions"], model)

    return {
        "model": model,                      # what was asked for
        "served_model": served_model,        # what actually answered — trust this one
        "model_mismatch": mismatch,
        "loaded_models": loaded_models,
        # True when the run was cut short (stop button, SIGTERM, or rebuilt from a
        # checkpoint with --finalize-only). Consumers should treat the scores as a
        # sample of the dataset, not a verdict on it.
        "stopped_early": stopped_early,
        "checkpoint_path": checkpoint_path,  # popped by main() once the file is written
        "date": datetime.now(timezone.utc).isoformat(),
        "dataset_path": "",
        # Which of the labeling pipeline's three phrasings this run asked. `metric` and
        # `workers` are top-level for the same reason: they decide what the accuracies
        # mean. The old scalar `description_index` is still written when there is exactly
        # one, so a consumer of an older file keeps working — but it is deliberately
        # absent on a multi-phrasing run rather than being given a misleading single
        # value, since no one index describes the file.
        "description_indices": list(description_indices),
        **({"description_index": description_indices[0]}
           if len(description_indices) == 1 else {}),
        "metric": metric,
        "max_image_dim": max_image_dim,
        # 0 = the model's pixel answers were divided by the screenshot's own dimensions;
        # N = by N on both axes, because the model answers on a 0-N grid. It belongs
        # beside max_image_dim because it is the same kind of fact: it changes every
        # coordinate in the file, so two runs at different values are not comparable and
        # a reader has no way to know which they are holding without it. See COORD_GRIDS.
        "coord_grid": coord_grid,
        # Part of what error_count means: a run whose cap was too low for a thinking
        # model records truncations as errors, so the cap has to be readable off the
        # artifact to tell that apart from a model that answers badly.
        "max_tokens": max_tokens,
        # Every latency figure in the summary is measured at this concurrency, and a
        # thinking model wants a lower one than a direct-answering model. Two runs at
        # different values are not comparable on latency, and on a single-slot server
        # the extra workers only add queue wait to each request rather than throughput,
        # so the number has to travel with the result.
        "workers": workers,
        # What was asked for, and whether the endpoint accepted it. -1 means it was never
        # sent. `temperature_dropped` matters because coordinates are digit tokens: with
        # greedy decoding a re-run reproduces them, and under the server's own default
        # (0.8 on llama.cpp) it need not — so two runs of the same dataset are only
        # strictly comparable when this is false.
        "temperature": temperature,
        "temperature_dropped": _TEMPERATURE_REJECTED.is_set(),
        # -1 means the model's own default, which for a thinking model like Gemma 4
        # means full chain-of-thought. Recorded because its absence made a 100-element
        # Gemma run untraceable: the scores were poor and there was no way to tell from
        # the artifact whether thinking had been left on, which changes what the number
        # means. Same reasoning as served_model.
        "thinking_budget": thinking_budget,
        "summary": summary,
        "results": results,
        "worst_cases": worst_cases,
    }


def finalize_from_checkpoint(
    dataset: list,
    model: str,
    metric: str,
    description_indices: list,
    thinking_budget: int,
    max_image_dim: int,
    max_tokens: int,
    workers: int,
    temperature: float,
    output_dir: str,
    dataset_path: str,
) -> dict:
    """Rebuild a result file from an existing checkpoint, calling no model at all.

    The recovery path for a run that died without writing one: a crash, an OOM kill, a
    container being deleted under it, or — the case this was written for — a Ctrl+C that
    reaches the whole process group and so kills a benchmark something else had launched.
    Everything the run had finished is already on disk one line per element; this turns it
    into the same JSON a completed run writes.

    `--limit` still applies, because the checkpoint is not keyed on it and may hold
    elements from a wider run.
    """
    ckpt = checkpoint.path_for(output_dir or ".", dataset_path, model)
    resumed, _done, meta = checkpoint.read(ckpt)
    if not resumed:
        logger.error("No checkpoint at %s — nothing to finalize.", ckpt)
        sys.exit(1)

    # Same rule as max_tokens below: the phrasings are the dead run's, not this command
    # line's. Recovering with a different set would divide the scored rows by the wrong
    # denominator in `agreement` and silently drop rows the checkpoint does hold.
    description_indices = meta.get("description_indices", description_indices)

    pairs = _expand_pairs(dataset, description_indices)
    wanted = {(row["screenshot_id"], row["element_id"], idx) for row, idx in pairs}
    results = [r for r in resumed if checkpoint.key_of(r) in wanted]
    if not results:
        logger.error("Checkpoint %s holds %d scored pair(s), none of them in this "
                     "dataset selection. Wrong --dataset or --limit?", ckpt, len(resumed))
        sys.exit(1)

    # No endpoint is contacted, so the served model can only come from the header the
    # interrupted run wrote. Trust it: it is what actually produced these predictions.
    served_model = meta.get("served_model", "")
    mismatch = bool(served_model) and (
        _normalize_model_name(model) not in _normalize_model_name(served_model))

    # Same rule: the cap that shaped these predictions, and the concurrency the
    # latencies were measured at, are the dead run's — not whatever is on this command
    # line. Checkpoints written before the header carried them fall back to the flags,
    # the only values available and better than recording nothing.
    max_tokens = meta.get("max_tokens", max_tokens)
    workers = meta.get("workers", workers)
    temperature = meta.get("temperature", temperature)
    # Not a flag at all on this path: the rows in the checkpoint were already normalised
    # by the dead run, so the grid here only records which convention produced them.
    # Reading it off the command line would let a recovered file claim a convention its
    # own numbers were not scored under.
    coord_grid = meta.get("coord_grid", 0) or 0

    logger.info("Finalizing %d of %d selected (element, phrasing) pair(s) from %s",
                len(results), len(pairs), ckpt)
    return build_result(
        results=results,
        model=model,
        served_model=served_model,
        mismatch=mismatch,
        loaded_models=[],
        metric=metric,
        description_indices=description_indices,
        thinking_budget=thinking_budget,
        max_image_dim=max_image_dim,
        max_tokens=max_tokens,
        workers=workers,
        temperature=temperature,
        coord_grid=coord_grid,
        unique_elements=len(dataset),
        selected_elements=len(pairs),
        stopped_early=True,
        checkpoint_path=str(ckpt),
    )


# ---------------------------------------------------------------------------
# Re-scoring a finished result
# ---------------------------------------------------------------------------

# Errors a re-parse is allowed to overturn, matched as prefixes of the recorded message.
# Everything else — `timeout`, `HTTP 500: …`, `Truncated at max_tokens=…`, `Empty content
# (…)`, `image not found`, a bare connection error — describes something that happened
# *instead of* an answer, and must survive untouched.
#
# The truncation case is why this list exists rather than a blanket re-parse of every row
# that has text. A thinking model that spends its budget reasoning returns prose in
# `raw`, and handing that prose to the parser is precisely the bug that recorded a Gemma
# run as 16% centroid accuracy. Re-parsing it here would reintroduce it, one release
# later and much harder to see.
_REPARSEABLE_ERROR_PREFIXES = (
    "no coordinates in response",
    "No valid bbox found",
    "Missing x/y in response",
    "Cannot parse",
    "Partial bbox",
    "empty response",
)


def _is_reparseable(error: str) -> bool:
    """True when this row's error came from reading the answer, not from not getting one."""
    if not error:
        return True
    return any(error.startswith(p) for p in _REPARSEABLE_ERROR_PREFIXES)


def _image_dims(images_dir: str, screenshot_id: str) -> tuple[int, int]:
    """(width, height) of a screenshot, read from its header only. (0, 0) if missing."""
    for ext in (".png", ".jpg", ".jpeg"):
        p = Path(images_dir) / f"{screenshot_id}{ext}"
        if p.exists():
            try:
                with Image.open(p) as im:
                    return im.size
            except Exception as exc:
                logger.debug("Could not read dimensions of %s: %s", p, exc)
                return 0, 0
    return 0, 0


def _row_dims(row: dict, images_dir: str, max_image_dim: int, cache: dict) -> tuple[int, int]:
    """The dimensions this row's answer was normalised against, however we can get them.

    The row's own `img_w`/`img_h` first, which is why they are now recorded. Falling back
    to the image on disk is correct only after re-applying `--max-image-dim` the way
    `_load_image_b64` does: a run that downscaled to 1080 normalised its pixel answers
    against 486x1080, and dividing them by the file's 1080x2400 instead would rescore the
    run into a different, quieter version of the same class of bug this whole change is
    about.
    """
    w, h = row.get("img_w") or 0, row.get("img_h") or 0
    if w and h:
        return w, h
    sid = row.get("screenshot_id", "")
    if sid not in cache:
        cache[sid] = _image_dims(images_dir, sid)
    w, h = cache[sid]
    if w and h and max_image_dim and max(w, h) > max_image_dim:
        scale = max_image_dim / max(w, h)
        w, h = round(w * scale), round(h * scale)
    return w, h


def rescore_result(prior: dict, images_dir: str, coord_grid: int,
                   metric: str = "") -> dict:
    """Re-score a finished result file from each row's `raw`, calling no model at all.

    The fourth path into `build_result`, alongside a completed run, a stopped one and
    `--finalize-only`. It exists because a coordinate-convention bug is a *parsing* bug:
    the model's answers were fine and are all still there in `raw`, so fixing the parser
    and re-reading them is the whole repair. Re-running is not merely slower — 30,921
    calls is ~10 hours against ~2 seconds — it is *worse evidence*, because the model
    would answer afresh and nobody could then tell a harness fix from a model that
    happened to reply differently.

    What is preserved rather than recomputed: tokens, latency, `served_model`,
    `stopped_early`, the phrasings, and every non-parse error. Those are facts about the
    run that happened. Only the coordinates and the pass/fail derived from them move.
    """
    rows = prior.get("results") or []
    if not rows:
        # Reached by pointing --rescore at a summary-only file, which the published
        # reference results are — and the README sends readers to that directory, so this
        # is a normal mistake and gets a sentence rather than a traceback.
        logger.error(
            "%s carries a summary but no `results` rows, so there is nothing to re-score. "
            "Re-scoring re-reads each row's own `raw` text, which only a file written by a "
            "real run contains.", prior.get("model") or "that file")
        sys.exit(1)

    max_image_dim = prior.get("max_image_dim", 0) or 0
    dim_cache: dict = {}
    reparsed = changed = kept_error = missing_dims = 0

    out_rows = []
    for row in rows:
        r = dict(row)
        raw = r.get("raw") or ""
        if not _is_reparseable(r.get("error", "")) or not raw:
            kept_error += 1
            out_rows.append(r)
            continue

        img_w, img_h = _row_dims(r, images_dir, max_image_dim, dim_cache)
        norm_w, norm_h = _norm_dims(img_w, img_h, coord_grid)
        # With a grid in force the dimensions are not consulted at all, so a missing
        # image is harmless — which is what makes the GUI-Owl rescore need nothing but
        # the result file. Without one, a pixel answer cannot be normalised.
        if not (norm_w and norm_h):
            missing_dims += 1
            out_rows.append(r)
            continue

        before = _predicted_centroid(r)
        pred_bbox, pred_point, err = _parse_response(raw, norm_w, norm_h)
        gt = r.get("gt_bbox") or {}
        iou = _compute_iou(pred_bbox, gt) if (pred_bbox and gt) else 0.0
        centroid = _predicted_centroid(
            {"pred_bbox": pred_bbox, "pred_point": pred_point})
        r.update({
            "pred_bbox": pred_bbox,
            "pred_point": pred_point,
            "iou": round(iou, 4),
            "pass_iou": iou >= IOU_THRESHOLD,
            "click_inside": bool(centroid and gt and
                                 _point_inside_bbox(centroid[0], centroid[1], gt)),
            "error": err,
            "img_w": img_w,
            "img_h": img_h,
        })
        r["pass_centroid"] = r["click_inside"]
        reparsed += 1
        if before != centroid:
            changed += 1
        out_rows.append(r)

    logger.info("Re-scored %d row(s): %d re-parsed, %d prediction(s) moved, "
                "%d error(s) left as they were", len(out_rows), reparsed, changed,
                kept_error)
    if missing_dims:
        logger.warning(
            "%d row(s) left untouched: no img_w/img_h recorded and no image on disk to "
            "read it from, so a pixel-scale answer could not be normalised. Point "
            "--images-dir at the screenshots, or pass --coord-grid, which needs neither.",
            missing_dims)

    summary = prior.get("summary") or {}
    indices = prior.get("description_indices")
    if not indices:
        indices = [prior.get("description_index", 0)]
    result = build_result(
        results=out_rows,
        model=prior.get("model", ""),
        served_model=prior.get("served_model", ""),
        mismatch=bool(prior.get("model_mismatch")),
        loaded_models=prior.get("loaded_models") or [],
        metric=metric or prior.get("metric", "centroid"),
        description_indices=indices,
        thinking_budget=prior.get("thinking_budget", -1),
        max_image_dim=max_image_dim,
        max_tokens=prior.get("max_tokens", 0),
        workers=prior.get("workers", 0),
        temperature=prior.get("temperature", 0),
        coord_grid=coord_grid,
        unique_elements=summary.get("unique_elements")
        or len({(r.get("screenshot_id"), r.get("element_id")) for r in out_rows}),
        selected_elements=summary.get("selected_elements") or len(out_rows),
        stopped_early=bool(prior.get("stopped_early")),
    )
    result.pop("checkpoint_path", None)
    # `date` is when the inference happened, and that is still the original run — the
    # filename is stamped with it too. `build_result` set it to now, which would have
    # made a rescore look like a fresh run of the model in every list that sorts on it.
    result["date"] = prior.get("date") or result["date"]
    result["dataset_path"] = prior.get("dataset_path", "")
    # Read off the original rather than this process, which never called the endpoint and
    # so never had the chance to have `temperature` refused.
    result["temperature_dropped"] = bool(prior.get("temperature_dropped"))
    result["rescored_at"] = datetime.now(timezone.utc).isoformat()
    result["rescored_from_coord_grid"] = prior.get("coord_grid", 0) or 0
    return result


RESCORE_INFIX = "-RESCORED"
_TRAILING_TS_RE = re.compile(r"-(\d{8}-\d{6})")


def rescored_filename(original: str) -> str:
    """`vision-x-<timestamp>.json` -> `vision-x-RESCORED-<timestamp>.json`.

    The infix goes *before* the timestamp, like `-PARTIAL-` and `-SERVED-`, so the run
    time stays the last one in the name. That is not cosmetic: anything reading a run's
    time out of its filename takes the last match. Stamping the rescore's own date there
    would sort a re-read of an old file above runs that genuinely
    came after it, and file it under a day no model was called. It also keeps the pair
    adjacent in a sorted listing, which is where you want it while deciding which number
    to quote.
    """
    stem = Path(original).stem
    if RESCORE_INFIX in stem:
        return f"{stem}.json"
    matches = list(_TRAILING_TS_RE.finditer(stem))
    if not matches:
        return f"{stem}{RESCORE_INFIX}.json"
    last = matches[-1]
    return f"{stem[:last.start()]}{RESCORE_INFIX}{stem[last.start():]}.json"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Vision benchmark — natural-language element localisation")
    # Defaults are absolute, anchored on the repository root rather than on the caller's
    # CWD. They used to be `../dataset-v1.jsonl`, which resolved correctly only when you
    # had cd'd into this directory first and otherwise produced a "dataset not found"
    # that reads as a missing download rather than as a wrong working directory.
    parser.add_argument("--dataset", default=str(REPO_ROOT / "data" / DEFAULT_DATASET),
                        help="Labelled dataset to score against (default: the corpus this "
                             "repository ships)")
    parser.add_argument("--images-dir", default=str(REPO_ROOT / "data" / "images"),
                        help="Where the screenshots are, resolved as "
                             "<images-dir>/<screenshot_id>.png")
    # `--base-url` is the OpenAI-compatible name and the one the docs use; `--proxy-url`
    # is kept as an alias so existing scripts and checkpoints keep working.
    parser.add_argument("--base-url", "--proxy-url", dest="proxy_url",
                        default=os.getenv("BASE_URL") or os.getenv("PROXY_URL")
                        or "http://localhost:8080",
                        help="Base URL of an OpenAI-compatible server; /v1/chat/completions "
                             "is appended")
    # Empty by default, which is right for a local vLLM/llama.cpp/Ollama started without
    # one — `_auth_headers` then sends no auth header at all. Deliberately does NOT fall
    # back to OPENAI_API_KEY: that would forward a hosted credential to whatever
    # --base-url happens to point at.
    parser.add_argument("--api-key",
                        default=os.getenv("API_KEY") or os.getenv("PROXY_API_KEY") or "",
                        help="Sent as both `Authorization: Bearer` and `X-API-Key`. Leave "
                             "empty for a local server started without one — an empty value "
                             "sends no auth header at all")
    parser.add_argument("--model", nargs="+", default=["qwen2.5-vl"],
                        help="Model name(s) to ask for, space-separated. Whichever model "
                             "actually answers is recorded as `served_model`")
    # No env fallback on purpose, unlike the pipeline's PIPELINE_WORKERS. Concurrency here
    # is a property of the endpoint you are pointed at right now, not of your machine — a
    # single-slot llama.cpp wants 1-3 where a vLLM deployment wants 8-16 — and `workers`
    # rides along in every result because two runs at different values are not comparable
    # on latency. A set-and-forget default would work against both.
    parser.add_argument("--workers", type=int, default=3,
                        help="Concurrent requests (default: 3). On a single-slot server "
                             "extra workers add queue wait, not throughput; every latency "
                             "figure in the result was measured at this value")
    parser.add_argument("--description-index", type=int, nargs="+", default=[0],
                        metavar="N",
                        help="Which of the labeling pipeline's three phrasings to ask: "
                             "0=name ('the login button'), 1=label ('the button with the "
                             "text login'), 2=intent ('the button that will allow the "
                             "user to login'). Several may be given — `0 1 2` scores every "
                             "element under all three and reports them separately plus "
                             "their agreement. Default 0 alone, which is the phrasing that "
                             "most often repeats the element's visible text and so flatters "
                             "a model that grounds by reading text. NOTE: --limit still "
                             "counts elements, so N phrasings means N times the calls.")
    parser.add_argument("--timeout", type=int, default=120,
                        help="Per-request timeout in seconds")
    parser.add_argument("--coord-format", choices=["corner", "center"], default="corner",
                        help="How to interpret (x,y) from the model response. "
                             "'corner' (default) = top-left corner. "
                             "'center' = center of element (GUI-Owl native).")
    parser.add_argument("--metric", choices=["centroid", "iou"], default="centroid",
                        help="Primary pass/accuracy metric. "
                             "'centroid' (default) = centroid of predicted bbox falls "
                             "inside the ground-truth bbox. "
                             "'iou' = IoU(pred, gt) >= threshold. "
                             "Both are always computed; this selects the headline number.")
    parser.add_argument("--thinking-budget", type=int, default=-1,
                        help="budget_tokens for chain-of-thought models (Qwen3/Gemma4). "
                             "Set to 0 to disable thinking (faster but less accurate). "
                             "Default -1 leaves the model's own default unchanged.")
    parser.add_argument("--max-image-dim", type=int, default=0,
                        help="Downscale the longest side to this before sending. Default 0 "
                             "sends the original: the benchmark's job is to measure how well "
                             "a model can ground, and downscaling only handicaps it. Pass "
                             "1080 to reproduce a consumer that downscales before sending, "
                             "or a smaller value for a server whose context cannot hold a full "
                             "screenshot (~3300 image tokens for 1080x2400).")
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Output-token cap per request. Default 2048 covers thinking "
                             "models (Gemma 4 12B reasons ~430-512 tokens before its answer, "
                             "so 512 truncated ~half of all calls); a truncated answer is now "
                             "recorded as an error. Raise it if a heavier model still "
                             "truncates; direct answerers like Qwen2.5-VL are unaffected.")
    parser.add_argument("--temperature", type=float, default=0,
                        help="Sampling temperature. Default 0 is greedy, which is what makes "
                             "a coordinate reproducible — llama.cpp's own default is 0.8, and "
                             "coordinates are digit tokens where a flip turns x=432 into "
                             "x=332. Pass -1 to send no temperature at all, for a frontier "
                             "model that rejects it (GPT-5.x, Claude Opus 4.7+); the script "
                             "also detects that refusal and drops it for the run by itself.")
    parser.add_argument("--limit", type=int, default=0,
                        help="Benchmark at most N elements (0 = all), spread evenly "
                             "across the dataset so a pilot still covers the whole app "
                             "mix. Deterministic: the same N every run.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Discard the checkpoint for this dataset/model and score "
                             "everything again. Use after changing the prompt, or when the "
                             "earlier scores are no longer wanted. Note the checkpoint "
                             "covers every phrasing, so this throws away the other "
                             "phrasings' work too.")
    parser.add_argument("--finalize-only", action="store_true",
                        help="Write a result file from the existing checkpoint and "
                             "exit, without benchmarking anything. Recovers a run "
                             "that was killed before it could write its results — "
                             "the output is marked PARTIAL. Same --dataset, --model, "
                             "--limit and --description-index as the run that died.")
    parser.add_argument("--coord-grid", type=int, default=-1, metavar="N",
                        help="Divide the model's pixel-scale answers by N on both axes "
                             "instead of by the screenshot's dimensions, for a model that "
                             "answers on a fixed 0-N grid. Default -1 picks it per model "
                             "from COORD_GRIDS (GUI-Owl answers on 0-1000); pass 0 to "
                             "force the screenshot's own pixels. Getting this wrong is not "
                             "a small error: read as 1080x2400 pixels, GUI-Owl's grid "
                             # `%%` because argparse runs help text through `%`-formatting;
                             # a bare `%` here made `--help` itself raise ValueError.
                             "scored 10.6%% where it should have scored ~88%%.")
    parser.add_argument("--rescore", default="", metavar="RESULT.json",
                        help="Re-score an existing result file from each row's `raw` text "
                             "and exit, calling no model. The repair path for a parsing or "
                             "coordinate-convention bug: the answers are already in the "
                             "file, so nothing has to be inferred again — 30,921 rows take "
                             "~2s against ~10h of GPU. Combine with --coord-grid. Token "
                             "counts, latency, served_model and every non-parse error are "
                             "carried over untouched; only the coordinates and the pass/fail "
                             "drawn from them are recomputed. Writes a new -RESCORED- file "
                             "and never overwrites the input.")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "benchmark-results"),
                        help="Where result JSON files and resume checkpoints go "
                             "(default: benchmark-results/)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Load and validate the dataset, print how many requests a real "
                             "run would make, then stop. Contacts no endpoint and costs "
                             "nothing. Note it checks that the images directory exists, not "
                             "that the images are in it")
    args = parser.parse_args()

    # Deduped and sorted so the set is canonical: it goes into the checkpoint header and
    # into `agreement.phrasings`, where a repeated index would count one phrasing twice
    # and make a unanimous element look like it agreed across more evidence than it did.
    description_indices = sorted(set(args.description_index))
    if any(i < 0 for i in description_indices):
        logger.error("--description-index must be >= 0, got %s", args.description_index)
        sys.exit(1)

    _install_stop_handlers()

    output_dir = Path(args.output_dir)

    # Handled before anything else because it needs none of what follows: no dataset (the
    # ground truth is already on every row), no images unless a row is missing its
    # dimensions, and no endpoint at all.
    if args.rescore:
        src = Path(args.rescore)
        if not src.exists():
            src = output_dir / args.rescore
        if not src.exists():
            logger.error("Result file not found: %s", args.rescore)
            sys.exit(1)
        with open(src) as f:
            prior = json.load(f)
        grid = _coord_grid_for(prior.get("model", ""), prior.get("served_model", ""),
                               args.coord_grid)
        logger.info("Re-scoring %s — model=%s, coordinate grid %s -> %s",
                    src.name, prior.get("model", "?"),
                    prior.get("coord_grid", 0) or "pixel", grid or "pixel")
        result = rescore_result(prior, str(Path(args.images_dir).resolve()), grid,
                               metric=args.metric)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / rescored_filename(src.name)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        s, ps = result["summary"], (prior.get("summary") or {})
        logger.info("Results written to %s", out_path)
        logger.info("  centroid %.2f%% -> %.2f%%   IoU>=%.1f %.2f%% -> %.2f%%",
                    (ps.get("centroid_accuracy") or 0) * 100,
                    s["centroid_accuracy"] * 100, IOU_THRESHOLD,
                    (ps.get("iou_accuracy") or 0) * 100, s["iou_accuracy"] * 100)
        return

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        logger.error("Dataset not found: %s", dataset_path)
        sys.exit(1)

    # --finalize-only reads no images, and the run being recovered may well have been
    # the thing that populated the cache dir this points at.
    images_dir = Path(args.images_dir).resolve()
    if not images_dir.exists() and not args.finalize_only:
        logger.error("Images directory not found: %s", images_dir)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = _load_dataset(str(dataset_path))
    logger.info("Loaded %d elements from %s", len(dataset), dataset_path.name)
    if not dataset:
        logger.error("Dataset empty or no elements with bbox + descriptions")
        sys.exit(1)

    was_limited = 0 < args.limit < len(dataset)
    if was_limited:
        dataset = even_sample(dataset, args.limit)
        logger.info("Sampling %d elements evenly across the dataset (--limit)", len(dataset))

    # One model per run is not a style preference on a single-model endpoint: it
    # cannot tell the requests apart, so N models would produce N identically-scored
    # result files and a comparison table of one model against itself.
    if len(args.model) > 1 and not args.dry_run and not args.finalize_only:
        with httpx.Client() as probe_client:
            loaded = _list_loaded_models(probe_client, args.proxy_url, args.api_key)
        if len(loaded) == 1:
            logger.error(
                "Asked to benchmark %d models but the endpoint has exactly one loaded "
                "(%s) and does not route by name — every run would score that same "
                "model. Benchmark one model per run, reloading the endpoint in "
                "between.", len(args.model), loaded[0])
            sys.exit(1)

    date_str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    all_summaries = []

    for model in args.model:
        if args.finalize_only:
            result = finalize_from_checkpoint(
                dataset=dataset, model=model,
                description_indices=description_indices,
                metric=args.metric,
                thinking_budget=args.thinking_budget,
                max_image_dim=args.max_image_dim,
                max_tokens=args.max_tokens,
                workers=args.workers,
                temperature=args.temperature,
                output_dir=str(output_dir.resolve()),
                dataset_path=str(dataset_path),
            )
        else:
            result = run_benchmark(
                dataset=dataset, model=model,
                proxy_url=args.proxy_url, api_key=args.api_key,
                images_dir=str(images_dir), workers=args.workers,
                description_indices=description_indices,
                timeout=args.timeout, dry_run=args.dry_run,
                coord_format=args.coord_format,
                thinking_budget=args.thinking_budget,
                metric=args.metric,
                max_image_dim=args.max_image_dim,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                output_dir=str(output_dir.resolve()),
                dataset_path=str(dataset_path),
                no_resume=args.no_resume,
                coord_grid=args.coord_grid,
            )
        result["dataset_path"] = str(dataset_path)
        ckpt_path = result.pop("checkpoint_path", "")
        stopped_early = result.get("stopped_early", False)

        safe_model = model.replace("/", "-").replace(":", "-")
        # A multi-phrasing run's headline accuracy averages every phrasing, two of which
        # are harder than the `name` every earlier run used on its own — so the same model
        # scores lower here than in its own single-phrasing result sitting next to it in
        # the list. Say so in the filename, for the same reason -PARTIAL- and -SERVED- are
        # there: a directory listing shows names, not summaries, and a number that moved
        # for a reason other than model quality is the kind that gets quoted without it.
        if len(description_indices) > 1:
            safe_model = f"{safe_model}-{len(description_indices)}phrasing"
        # A mismatch goes in the filename, not just the JSON: two runs that silently hit
        # the same loaded model would otherwise sit side by side in the results directory
        # looking like two different models.
        if result.get("model_mismatch"):
            served_stem = Path(result["served_model"]).stem.replace("/", "-").replace(":", "-")
            safe_model = f"{safe_model}-SERVED-{served_stem}"
        # Same reasoning for a partial run: a run stopped at 12% otherwise looks in the
        # listing exactly like one that finished.
        if stopped_early:
            safe_model = f"{safe_model}-PARTIAL"
        out_path = output_dir / f"vision-{safe_model}-{date_str}.json"
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Results written to %s", out_path)

        # The checkpoint is expendable only once this dataset is fully and cleanly
        # scored. Errored elements were never recorded, so keeping it lets the next run
        # retry just those. A --limit run is also incomplete by definition: deleting its
        # checkpoint would throw the pilot away, and on a slow endpoint a 200-element
        # pilot is over an hour of inference that a later full run should reuse rather
        # than pay for twice.
        errors = result.get("summary", {}).get("error_count", 0)
        if ckpt_path:
            if stopped_early:
                logger.info("Keeping %s — this run was stopped part-way, so re-running "
                            "it carries on from the %d scored element(s) instead of "
                            "paying for them again", ckpt_path, len(result["results"]))
            elif errors:
                logger.info("Keeping %s so the next run retries the %d errored "
                            "element(s)", ckpt_path, errors)
            elif was_limited:
                logger.info("Keeping %s — this was a --limit run, so its %d scored "
                            "pair(s) carry over to the next run of this dataset",
                            ckpt_path, len(result["results"]))
            else:
                Path(ckpt_path).unlink(missing_ok=True)

        if "summary" in result:
            all_summaries.append({"model": model, **result["summary"]})

    if len(all_summaries) > 1:
        print(f"\n=== Model Comparison (primary metric: {args.metric}) ===")
        # `Box%` sits between the two accuracies because it is what makes them comparable or
        # not. A model at 15% answered a different question from one at 100%, and without the
        # column the higher Centroid reads as the better model — see `bbox_coverage`.
        print(f"{'Model':<25} {'Centroid':>10} {'Box%':>7} {'IoU Acc':>9}"
              f" {'IoU|box':>8} {'Errors':>8}")
        print("-" * 78)
        for s in all_summaries:
            cov = s.get("bbox_coverage")
            given = s.get("iou_accuracy_given_bbox")
            print(
                f"{s['model']:<25} {s['centroid_accuracy']*100:>9.1f}%"
                f" {(f'{cov*100:.0f}%' if cov is not None else '—'):>7}"
                f" {s['iou_accuracy']*100:>8.1f}%"
                f" {(f'{given*100:.1f}%' if given is not None else '—'):>8}"
                f" {s['error_count']:>8}"
            )
        print("\n  Box%    — share of requests that returned a bounding box at all.")
        print("  IoU|box — IoU ≥ 0.5 among only those. IoU Acc = Box% x IoU|box.")
        print("  Where the consumer requires a bounding box, Box% is a gate, not a metric:")
        print("  a model below it is not comparable on Centroid with one above it.")

    if len(all_summaries) == 1 and not args.dry_run:
        s = all_summaries[0]
        primary = s.get("metric", args.metric)
        print(f"\n=== {s['model']} (primary metric: {primary}) ===")
        star_c = " *" if primary == "centroid" else ""
        star_i = " *" if primary == "iou" else ""
        print(f"  Centroid Accuracy (center inside gt): {s['centroid_accuracy']*100:.1f}%  "
              f"({s['centroid_pass_count']}/{s['total_elements']} pass){star_c}")
        print(f"  IoU Accuracy  (IoU≥{IOU_THRESHOLD}): {s['iou_accuracy']*100:.1f}%  "
              f"({s['iou_pass_count']}/{s['total_elements']} pass){star_i}")
        # Printed right under the two accuracies, and only when it has something to say, so a
        # model that always returns a box adds no noise here.
        if s.get("point_only_count"):
            print(f"  Bbox coverage: {s.get('bbox_coverage', 0)*100:.1f}%  "
                  f"({s.get('bbox_count', 0)}/{s['total_elements']} returned a box; "
                  f"{s['point_only_count']} were a bare click point)")
            print(f"  IoU≥{IOU_THRESHOLD} among those: "
                  f"{s.get('iou_accuracy_given_bbox', 0)*100:.1f}%   "
                  f"<- the IoU above is this x coverage")
        print(f"  Mean IoU:     {s['mean_iou']:.3f}")
        print(f"  Median IoU:   {s['median_iou']:.3f}")
        print(f"  Errors:       {s['error_count']} (timeout: {s['timeout_count']})")
        print(f"  Avg latency:  {s['avg_latency_ms']:.0f}ms  "
              f"(P50={s['p50_latency_ms']:.0f}ms, P95={s['p95_latency_ms']:.0f}ms)")

        # The point of a multi-phrasing run. The figures above average these, and the
        # average is the least interesting of the numbers on this screen.
        by_desc = s.get("by_description") or {}
        if by_desc:
            print(f"\n  By phrasing — {s.get('unique_elements', 0)} element(s) "
                  f"x {len(by_desc)} phrasing(s):")
            print(f"    {'Phrasing':<9} {'Centroid':>9} {'IoU':>8} {'MeanIoU':>8}"
                  f" {'In tok':>8} {'Out tok':>8} {'Errors':>7}")
            print("    " + "-" * 61)
            for idx in sorted(by_desc, key=int):
                d = by_desc[idx]
                print(f"    {d['style']:<9} {d['centroid_accuracy']*100:>8.1f}%"
                      f" {d['iou_accuracy']*100:>7.1f}% {d['mean_iou']:>8.3f}"
                      f" {d['avg_input_tokens']:>8.0f} {d['avg_output_tokens']:>8.0f}"
                      f" {d['error_count']:>7}")
        ag = s.get("agreement") or {}
        if ag:
            print(f"\n  Agreement over the {ag['scored_elements']} element(s) scored "
                  f"under all {ag['phrasings']} phrasings:")
            print(f"    any  {ag['any_accuracy']*100:>5.1f}%  ({ag['any_pass_count']})"
                  f"  — at least one phrasing found it")
            print(f"    all  {ag['all_accuracy']*100:>5.1f}%  ({ag['all_pass_count']})"
                  f"  — every phrasing found it")
            print(f"    mixed {ag['mixed_count']} element(s) pass under some phrasings "
                  f"and fail under others")


if __name__ == "__main__":
    main()
