"""Scoring a prediction against ground truth, and one guard on the scores.

Both metrics are always computed; `--metric` only chooses the headline number and which
rows count as worst cases. Centroid is the default because GUI-Owl is trained to emit
click points rather than boxes, so its IoU is structurally low and says nothing about
whether it found the right element.

`_scale_check` is the guard, and it only works in aggregate: it compares the median
predicted centroid against the median ground-truth centroid per axis. It **flags and
never corrects** — guessing a convention from a statistic and applying it silently is
how this harness produced its worst data. `suspect: false` means "no evidence here",
not "correct": the signal scales with the gap between the true divisor and the one used,
so it depends on the screenshot's aspect ratio.
"""
import statistics


IOU_THRESHOLD = 0.5

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


def _offset_check(results: list) -> dict:
    """Which *way* is the model wrong: displaced, mis-sized, or just scattered?

    `_scale_check` above compares the median of *positions*, which catches a wrong divisor
    and little else — shifting a whole distribution barely moves a ratio of medians. It is
    therefore blind to a model that finds every element but places it consistently low, and
    that is not hypothetical: a frontier model asked for normalized floats put its
    predictions ~0.10 of the screen height too low on every one of them, and `scale_check`
    reported `y_ratio` 0.857 — inside its 0.8-1.25 band, `suspect: false`. This takes the
    median of the *per-element difference* instead, which is where a constant bias lives.

    Three faults that all arrive as a low accuracy, told apart here:

    * **Displacement** — `median_dx`/`median_dy` far from 0. A fixed aim error. Signed on
      purpose: the sign says which way, and a fix is cheap once the direction is known.
    * **Scatter** — the signed medians near 0 while `median_abs_*` is large. The model is
      wrong in no particular direction, i.e. it is not finding the element. Both are
      reported because the signed figure alone cannot distinguish this from being right.
    * **Mis-sizing** — the box lands on the element but is the wrong size.
      `median_height_ratio` 1.60 means boxes 60% too tall; 0.82 means 18% too short. This
      is what separates "centroid passes, IoU fails" into a cause.

    Position uses `_predicted_centroid`, so a click-point model is measured too; the size
    ratios necessarily cover only the rows that returned a box.
    """
    pairs = [(_predicted_centroid(r), r["gt_bbox"]) for r in results
             if _predicted_centroid(r) and r.get("gt_bbox")]
    if not pairs:
        return {}
    dx = [p[0] - (g["x"] + g["width"] / 2) for p, g in pairs]
    dy = [p[1] - (g["y"] + g["height"] / 2) for p, g in pairs]

    boxes = [r for r in results if r.get("pred_bbox") and r.get("gt_bbox")]
    w_ratio = [r["pred_bbox"]["width"] / r["gt_bbox"]["width"]
               for r in boxes if r["gt_bbox"]["width"]]
    h_ratio = [r["pred_bbox"]["height"] / r["gt_bbox"]["height"]
               for r in boxes if r["gt_bbox"]["height"]]

    out = {
        "predictions": len(pairs),
        # Signed: predicted centre minus ground-truth centre, in screen fractions.
        # Positive x = too far right, positive y = too far down.
        "median_dx": round(statistics.median(dx), 4),
        "median_dy": round(statistics.median(dy), 4),
        # Magnitude. Read beside the signed pair: near-zero signed with a large absolute
        # is scatter, not bias, and the two call for completely different responses.
        "median_abs_dx": round(statistics.median([abs(v) for v in dx]), 4),
        "median_abs_dy": round(statistics.median([abs(v) for v in dy]), 4),
    }
    if w_ratio:
        out["boxes"] = len(w_ratio)
        out["median_width_ratio"] = round(statistics.median(w_ratio), 4)
    if h_ratio:
        out["median_height_ratio"] = round(statistics.median(h_ratio), 4)
    return out
