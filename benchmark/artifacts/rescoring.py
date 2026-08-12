"""Rebuilding a finished result by re-reading the answers it already holds.

A coordinate-convention bug is a *parsing* bug: the model's own text is in `raw`, so
re-reading it is the whole repair — 30,921 rows in about two seconds against ~10 hours
and 85M input tokens for a re-run. Cost is not the main argument. The model is not
deterministic, so a re-run answers afresh and nobody can then tell a harness fix from a
model that simply replied differently; re-scoring the same `raw` proves which one moved.

Only parse-family errors are re-parsed. A `timeout`, an `HTTP 500` or a
`Truncated at max_tokens=…` describes something that happened *instead of* an answer,
and re-parsing a truncated thinking model's reasoning prose is exactly the bug that
recorded a Gemma run at 16%.
"""
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from benchmark.scoring.coords import _norm_dims
from benchmark.scoring.metrics import IOU_THRESHOLD, _compute_iou, _point_inside_bbox, _predicted_centroid
from benchmark.scoring.parsing import _parse_response
from benchmark.artifacts.builder import build_result

logger = logging.getLogger(__name__)


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
            "This file carries a summary but no `results` rows, so there is nothing to "
            "re-score. A re-score reads each row's `raw` — the model's own text — and "
            "parses it again; the published files in reference-results/ are summaries "
            "with the rows stripped, so only a result file from your own run can be "
            "re-scored.")
        sys.exit(1)

    max_image_dim = prior.get("max_image_dim", 0) or 0
    dim_cache: dict = {}
    reparsed = changed = kept_error = missing_dims = 0

    out_rows = []
    for row in rows:
        r = dict(row)
        # Dropped for every row, on every path, before anything branches. Rows come from
        # the input file, so one written while `click_inside` still existed carries it —
        # and on the re-parse path `update()` below would leave it holding the *previous*
        # verdict. Nothing reads it any more, and a stale field contradicting
        # `pass_centroid` in the same row is worse than an absent one. Doing it here
        # rather than per-branch is what stops the carried-over rows keeping it.
        r.pop("click_inside", None)
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
            "pass_centroid": bool(centroid and gt and
                                  _point_inside_bbox(centroid[0], centroid[1], gt)),
            "error": err,
            "img_w": img_w,
            "img_h": img_h,
        })
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
        # Off the original file, never off this process: a rescore re-parses `raw` and
        # calls no model, so it cannot change which prompt produced these answers. A file
        # predating the field was run when `normalized` was the only prompt, so the
        # default states a fact rather than guessing one.
        prompt_style=prior.get("prompt_style") or "normalized",
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
    """`vision-x-20260802-124638.json` -> `vision-x-RESCORED-20260802-124638.json`.

    The infix goes *before* the timestamp, like `-PARTIAL-` and `-SERVED-`, so the run
    time stays the last thing in the name. That is not cosmetic: anything reading a run's
    date off its filename takes the last match, and stamping the rescore's own date there
    would sort a re-read of an old file above runs that genuinely came after it, filed
    under a day no model was called. It also keeps the pair adjacent in a sorted listing,
    which is where you want it while deciding which number to quote.
    """
    stem = Path(original).stem
    if RESCORE_INFIX in stem:
        return f"{stem}.json"
    matches = list(_TRAILING_TS_RE.finditer(stem))
    if not matches:
        return f"{stem}{RESCORE_INFIX}.json"
    last = matches[-1]
    return f"{stem[:last.start()]}{RESCORE_INFIX}{stem[last.start():]}.json"
