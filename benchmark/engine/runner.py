"""Driving one model over the work items, and stopping cleanly when asked.

The concurrency lives here: one `httpx.Client` outside the pool (never one per call —
that is what exhausted the descriptor table on the labeling side), `--workers` threads,
and a checkpoint append the moment each pair finishes.

**Stopping is a first-class path, not an abort.** The first SIGTERM/SIGINT sets `_STOP`;
the `as_completed` loop then cancels what is still queued and **keeps draining** instead
of breaking, so the requests already in flight still land in both the results and the
checkpoint. A second signal exits immediately. A stopped run is therefore not instant —
it lives until the slowest in-flight request returns, up to one `--timeout`.
"""
import logging
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx
from tqdm import tqdm

from benchmark.artifacts import checkpoint
from benchmark.scoring.coords import _coord_grid_for, _normalize_model_name
from benchmark.scoring.metrics import IOU_THRESHOLD, _compute_iou, _point_inside_bbox
from benchmark.model.client import _call_model, _call_model_text, _list_loaded_models, _probe_served_model
from benchmark.model.xml_tree import ViewTreeStore, build_xml_prompt
from benchmark.scoring.xpath import xml_prediction
from benchmark.workload.phrasings import _expand_pairs, _style_of
from benchmark.artifacts.builder import build_result, gt_boxes_from_dataset

logger = logging.getLogger(__name__)


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
            logger.warning("Second signal — exiting now. Elements in flight are lost; everything already scored is in the checkpoint.")
            os._exit(130)
        _STOP.set()
        logger.warning("Signal %d — cancelling the queue, finishing the %s in flight, then writing a PARTIAL result. Send it again to exit now.",signum, "requests")
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, _handler)


class RatePacer:
    """Spaces request starts evenly so a run never exceeds `rpm` requests a minute.

    One instance is shared by every worker. Each caller takes the next free slot under the
    lock and sleeps until it arrives, so the interval between any two starts is at least
    60/rpm seconds no matter how many workers there are — a strict ceiling, not an average
    that bursts and then idles. The sleep is `_STOP.wait`, so a stop signal wakes the sleeper
    at once; the request it was about to make still goes out, like any other in-flight one.

    It counts model calls, not HTTP attempts: the back-off retries inside
    `_post_with_retries` are not paced, because they are the endpoint already telling us to
    wait. A `--rpm` well under the account's limit is what keeps those from happening.
    """

    def __init__(self, rpm: int):
        self.interval = 60.0 / rpm
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            start = max(now, self._next_slot)
            self._next_slot = start + self.interval
        delay = start - now
        if delay > 0:
            _STOP.wait(delay)


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
    api_flavor: str = "proxy",
    prompt_style: str = "pixels",
    pacer: "RatePacer | None" = None,
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
    ident = {"screenshot_id": screenshot_id,
             "element_id": element_id,
             "description_index": description_index,
             "description_style": _style_of(description_index)}

    image_path = None
    for ext in (".png", ".jpg", ".jpeg"):
        candidate = Path(images_dir) / f"{screenshot_id}{ext}"
        if candidate.exists():
            image_path = str(candidate)
            break

    if not image_path:
        return {
            **ident,
            "description": desc, "gt_bbox": gt_bbox,
            "pred_bbox": None, "pred_point": None,
            "iou": 0.0, "pass_iou": False, "pass_centroid": False,
            "error": "image not found", "latency_ms": 0,
            "input_tokens": 0, "output_tokens": 0,
            "cached_input_tokens": 0, "cache_creation_input_tokens": 0, "raw": "",
            "img_w": 0, "img_h": 0,
        }

    if pacer:
        pacer.wait()
    pred_bbox, pred_point, latency_ms, error, meta = _call_model(
        client, proxy_url, api_key, model, image_path, desc, timeout, coord_format,
        thinking_budget, max_image_dim, max_tokens, temperature, coord_grid,
        api_flavor, prompt_style,
        # The image is what every request about this screenshot shares; see `_cache_routing`.
        cache_key=screenshot_id,
    )

    iou = _compute_iou(pred_bbox, gt_bbox) if pred_bbox else 0.0
    pass_iou = iou >= IOU_THRESHOLD

    # The centroid metric: the centre of the predicted bbox must fall inside the
    # ground-truth bbox. Use the bbox centre when a bbox came back, otherwise the raw
    # predicted click point (GUI-Owl-style models return only that).
    pass_centroid = False
    if pred_bbox:
        cx = pred_bbox["x"] + pred_bbox["width"] / 2
        cy = pred_bbox["y"] + pred_bbox["height"] / 2
        pass_centroid = _point_inside_bbox(cx, cy, gt_bbox)
    elif pred_point:
        pass_centroid = _point_inside_bbox(pred_point["cx"], pred_point["cy"], gt_bbox)

    return {
        **ident,
        "description": desc,
        "gt_bbox": gt_bbox,
        "pred_bbox": pred_bbox,
        "pred_point": pred_point,
        "iou": round(iou, 4),
        "pass_iou": pass_iou,
        "pass_centroid": pass_centroid,   # would the tap have landed on the control
        "error": error,
        "latency_ms": round(latency_ms, 1),
        # Token counts for the cost-versus-accuracy comparison, and the model's own
        # text so a bad score can be attributed to the model or to the parser.
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "cached_input_tokens": meta.get("cached_input_tokens", 0),
        "cache_creation_input_tokens": meta.get("cache_creation_input_tokens", 0),
        "raw": meta.get("raw", ""),
        # The dimensions this answer was normalised against — the resized ones when
        # --max-image-dim is in force, not the file's. Recorded per row so `--rescore`
        # can re-read `raw` without the images, and without having to guess whether a
        # downscale happened. See `_call_model`.
        "img_w": meta.get("img_w", 0),
        "img_h": meta.get("img_h", 0),
    }

def _benchmark_element_xml(
    client: httpx.Client,
    proxy_url: str,
    api_key: str,
    model: str,
    row: dict,
    description_index: int,
    timeout: int,
    max_tokens: int,
    temperature: float,
    api_flavor: str,
    trees: ViewTreeStore,
    pacer: "RatePacer | None" = None,
) -> dict:
    """One (element, phrasing) under `--input xml`: the tree in, an XPath out, a box graded.

    Same row shape as `_benchmark_element` plus four fields — `input`, `xpath`,
    `xml_outcome`, `xpath_matches` — so every reader of a result file (Result Analysis,
    `--rescore`, the cascade scorer) sees one schema. The screenshot is never opened;
    `img_w`/`img_h` are the tree's own `<hierarchy width height>`, which is what its
    `bounds` are normalised against.

    `NOT_FOUND`, an invalid XPath and an XPath matching nothing all come back with
    `error == ""`: they are the model's answer, recorded in `xml_outcome`, and a resume must
    not pay to ask again. Only a transport failure is an error here.
    """
    screenshot_id = row["screenshot_id"]
    element_id = row["element_id"]
    gt_bbox = row["bbox"]
    descriptions = row["descriptions"]
    description_index = min(description_index, len(descriptions) - 1)
    desc = descriptions[description_index]
    base = {
        "screenshot_id": screenshot_id,
        "element_id": element_id,
        "description_index": description_index,
        "description_style": _style_of(description_index),
        "description": desc, "gt_bbox": gt_bbox,
        "pred_bbox": None, "pred_point": None,
        "iou": 0.0, "pass_iou": False, "pass_centroid": False,
        "error": "", "latency_ms": 0,
        "input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0,
        "cache_creation_input_tokens": 0, "raw": "",
        "img_w": 0, "img_h": 0,
        "input": "xml", "xpath": "", "xml_outcome": "", "xpath_matches": 0,
    }

    vt = trees.get(screenshot_id)
    if vt is None:
        return {**base, "error": "xml dump not found"}
    base["img_w"], base["img_h"] = vt.width, vt.height

    prefix, suffix = build_xml_prompt(desc, vt.filtered_xml)
    if pacer:
        pacer.wait()
    content, latency_ms, error, meta = _call_model_text(
        client, proxy_url, api_key, model, prefix, suffix, timeout,
        max_tokens, temperature, api_flavor,
        # The tree is what every request about this screenshot shares; see `_cache_routing`.
        # `_build_text_payload` sends it as the system message, where OpenAI's cache can actually pick it up.
        cache_key=screenshot_id)
    base.update({
        "latency_ms": round(latency_ms, 1),
        "input_tokens": meta.get("input_tokens", 0),
        "output_tokens": meta.get("output_tokens", 0),
        "cached_input_tokens": meta.get("cached_input_tokens", 0),
        "cache_creation_input_tokens": meta.get("cache_creation_input_tokens", 0),
        "raw": meta.get("raw", ""),
    })
    if error:
        base["error"] = error
        return base

    pred = xml_prediction(content, vt.tree, vt.width, vt.height)
    pred_bbox = pred["pred_bbox"]
    iou = _compute_iou(pred_bbox, gt_bbox) if pred_bbox else 0.0
    pass_centroid = False
    if pred_bbox:
        cx = pred_bbox["x"] + pred_bbox["width"] / 2
        cy = pred_bbox["y"] + pred_bbox["height"] / 2
        pass_centroid = _point_inside_bbox(cx, cy, gt_bbox)
    base.update({
        "pred_bbox": pred_bbox,
        "iou": round(iou, 4),
        "pass_iou": iou >= IOU_THRESHOLD,
        "pass_centroid": pass_centroid,
        "xpath": pred["xpath"],
        "xml_outcome": pred["xml_outcome"],
        "xpath_matches": pred["xpath_matches"],
    })
    return base


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
    api_flavor: str = "proxy",
    prompt_style: str = "pixels",
    input_mode: str = "screenshot",
    xml_dir: str = "",
    gt_boxes: dict = None,
    rpm: int = 0,
    max_requests: int = 0,
    only_keys: set = None,
) -> dict:
    """Score `dataset` under one model; see `cli/commands.py` for how the flags arrive.

    `gt_boxes` is every labelled box on the screenshots being scored, keyed as
    `gt_boxes_from_dataset` keys them. The caller passes it on a `--limit` run, where
    `dataset` is the sampled subset: `_answer_classes` decides "chose another element" by
    looking for *any* labelled element under the predicted centre, and a 200-element sample
    spread over ~199 screenshots carries one box per screen, so every wrong answer read as
    `empty_space` and `other_element` was 0 on both pilots. Absent, the subset's own boxes
    are used, which is complete for a full run.

    `rpm` and `max_requests` are the two cost controls, for a paid run spread over days
    under an account's limits. `rpm` caps request starts per minute across all workers
    (`RatePacer`); `max_requests` caps the model calls this invocation makes — the pairs
    still to do are cut to that many, the rest wait for the next invocation. A capped run
    ends as a stopped one does: `stopped_early`, a `-PARTIAL-` file, the checkpoint kept, so
    the same command the next night carries on where this one left off. Both count model
    calls, not HTTP attempts, and 0 means no cap.

    `only_keys` (from `--only-from`) restricts the run to those (screenshot, element,
    phrasing) keys — the rows an XML run did not answer. Everything else is a plain run on
    a smaller work list: same prompt, same checkpoint, same scoring, so a later full run
    reuses every row this one bought.
    """
    pairs = _expand_pairs(dataset, description_indices)
    if only_keys is not None:
        before = len(pairs)
        pairs = [(row, idx) for row, idx in pairs if (row["screenshot_id"], row["element_id"], idx) in only_keys]
        logger.info("  --only-from: %d of %d pair(s) kept — the rows the XML run did not answer", len(pairs), before)
        dataset = [row for row in dataset if any((row["screenshot_id"], row["element_id"], idx) in only_keys for idx in description_indices)]
    styles = ", ".join(_style_of(i) for i in description_indices)
    logger.info("Benchmarking model=%s on %d element(s) x %d phrasing(s) [%s] = %d call(s), metric=%s, input=%s", model, len(dataset), len(description_indices), styles, len(pairs), metric, input_mode)
    short = len(dataset) * len(description_indices) - len(pairs)
    if short:
        logger.info("  %d pair(s) skipped: the element has fewer descriptions than that",short)

    xml = input_mode == "xml"
    trees = ViewTreeStore(xml_dir) if xml else None
    if xml:
        # The coordinate prompt never goes out on this track, so the result must not claim one. Recorded as the prompt that did: the XPath template.
        prompt_style = "xpath"
        # Checked up front, whether or not this is a dry run: a missing dump is per screenshot, and finding out at a paid run is the expensive way.
        sids = {row["screenshot_id"] for row in dataset}
        missing = sorted(s for s in sids if not trees.path_for(s).exists())
        if missing:
            logger.warning("  %d of %d screenshot(s) have no XML dump in %s — their rows will error with 'xml dump not found'. First: %s",len(missing), len(sids), xml_dir, missing[0])
        else:
            logger.info("  XML dumps present for all %d screenshot(s) (%s)", len(sids), xml_dir)

    if dry_run:
        logger.info("Dry-run: skipping API calls")
        return {"model": model, "dry_run": True, "total_elements": len(pairs), "unique_elements": len(dataset), "results": []}

    ckpt = checkpoint.path_for(output_dir or ".", dataset_path, model, prompt_style, input_mode)
    if no_resume:
        ckpt.unlink(missing_ok=True)

    results = []
    with httpx.Client() as client:
        loaded_models = _list_loaded_models(client, proxy_url, api_key, api_flavor)
        served_model = _probe_served_model(client, proxy_url, api_key, model, api_flavor)
        mismatch = bool(served_model) and (
            _normalize_model_name(model) not in _normalize_model_name(served_model))
        if served_model:
            logger.info("  endpoint served: %s", served_model)
        if mismatch:
            logger.warning("MODEL MISMATCH — asked for %r, the endpoint answered as %r. This endpoint does not route by model name, so these results describe "
                "whatever model is currently loaded, NOT %s. Load the model you want (ssh to the host and restart the server with it) and re-run.",
                model, served_model, model)
        elif not served_model:
            logger.warning("Could not determine which model the endpoint serves; results will not record it.")

        # Decided here rather than in main() because the served id is only known after
        # the probe, and on a single-model endpoint that id is the one that decides the
        # convention — see `_coord_grid_for`.
        grid = _coord_grid_for(model, served_model, coord_grid)
        if grid:
            logger.info("  coordinate grid: 0-%d (pixel-scale answers are divided by %d, not by the screenshot's dimensions)", grid, grid)

        resumed, done, meta = checkpoint.read(ckpt)
        if resumed and meta.get("served_model", served_model) != served_model:
            logger.error("Checkpoint %s was scored by %r but the endpoint now serves %r. Resuming would blend two models into one score."
                         "Re-load that model, or pass --no-resume to discard %d scored element(s) and start over.",
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
                "Resuming would put two coordinate conventions in one score and nothing downstream could tell them apart."
                "Pass --coord-grid %s to match it, or --no-resume to discard %d scored pair(s) and start over.",
                ckpt, prior_grid or "pixel", grid or "pixel", prior_grid, len(done))
            sys.exit(1)

        # A checkpoint may hold pairs outside this run's selection — a smaller --limit,
        # or the phrasings of an earlier run this one does not ask for — so keep only
        # what belongs to the current set.
        wanted = {(row["screenshot_id"], row["element_id"], idx) for row, idx in pairs}
        results = [r for r in resumed if checkpoint.key_of(r) in wanted]
        done &= wanted
        todo = [(row, idx) for row, idx in pairs if (row["screenshot_id"], row["element_id"], idx) not in done]
        if done:
            logger.info("  %d already scored, %d to go", len(results), len(todo))

        # The per-invocation cap. Pairs are ordered screenshot-major, so the cut keeps every phrasing of the screenshots it does take and the cache prefix they share.
        capped = 0 < max_requests < len(todo)
        if capped:
            logger.info("  --max-requests %d: scoring the next %d pair(s) this invocation, %d left for the next",max_requests, max_requests, len(todo) - max_requests)
            todo = todo[:max_requests]
        pacer = RatePacer(rpm) if rpm > 0 else None
        if pacer:
            logger.info("  --rpm %d: at most one request start every %.1fs across %d worker(s)", rpm, pacer.interval, workers)

        # max_tokens rides along for the same reason served_model does: --finalize-only
        # rebuilds a result from this header alone, and the honest value is the one the
        # run actually used, not whatever the flag says whenever someone gets round to
        # recovering it. It also decides whether a truncation counts as an error.
        # prompt_style belongs here for a sharper reason than the rest: two runs that
        # differ only by it produce filenames differing only by timestamp, and on a
        # 10-element pilot one frontier model scored 0/10 under one style and 9/10 under
        # the other. A result that cannot say which prompt produced it is not attributable
        # to anything — the same failure as the `served_model` one, one field later.
        meta = {"served_model": served_model, "model": model,
                "dataset": Path(dataset_path).name,
                "description_indices": description_indices,
                "max_tokens": max_tokens,
                "workers": workers,
                "temperature": temperature,
                "prompt_style": prompt_style,
                "coord_grid": grid,
                # Which track wrote these rows: --finalize-only rebuilds from this header alone, and the track changes what the rows mean.
                "input": input_mode}

        futures = {}
        stopped_early = False
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for row, idx in todo:
                if xml:
                    f = pool.submit(
                        _benchmark_element_xml,
                        client, proxy_url, api_key, model,
                        row, idx, timeout, max_tokens, temperature, api_flavor,
                        trees, pacer,
                    )
                else:
                    f = pool.submit(
                        _benchmark_element,
                        client, proxy_url, api_key, model,
                        row, images_dir, idx, timeout, coord_format,
                        thinking_budget, max_image_dim, max_tokens, temperature,
                        grid, api_flavor, prompt_style, pacer,
                    )
                futures[f] = (row, idx)

            with tqdm(total=len(futures), desc=f"{model}") as pbar:
                for f in as_completed(futures):
                    # Cancelled by the stop below. as_completed yields them straight away and .result() would raise CancelledError.
                    if f.cancelled():
                        pbar.update(1)
                        continue
                    r = f.result()
                    results.append(r)
                    # Errors stay out so a resume retries them — see checkpoint.py.
                    if not r["error"]:
                        checkpoint.append(ckpt, meta, r)
                    pbar.update(1)

                    # Cancel the queue but keep draining rather than breaking out: the futures still running would otherwise finish into nothing,
                    # losing up to `workers` elements from both the result and the checkpoint.
                    # Cancelled ones come back immediately, so the loop ends as soon as the last in-flight request lands.
                    if _STOP.is_set() and not stopped_early:
                        stopped_early = True
                        cancelled = sum(1 for g in futures if g.cancel())
                        logger.warning("Stopping early — %d queued element(s) cancelled, waiting for the ones in flight", cancelled)

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
        prompt_style=prompt_style,
        unique_elements=len(dataset),
        # Pairs, not elements: `completed_fraction` divides the rows scored by this, and
        # a 3-phrasing run produces three rows per element. Counting elements here would
        # report a finished run as 300% complete.
        selected_elements=len(pairs),
        # A run cut short by --max-requests is as partial as a stopped one, and must look it: PARTIAL in the name, checkpoint kept, completed_fraction below 1.
        stopped_early=stopped_early or capped,
        checkpoint_path=str(ckpt),
        gt_boxes=gt_boxes or gt_boxes_from_dataset(dataset),
        input_mode=input_mode,
    )
