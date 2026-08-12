"""Assembling the result JSON every other surface reads.

`build_result` is shared by all four paths that can produce a file:
- a completed run
- a run stopped by a signal
- `--finalize-only` off a checkpoint
- `--rescore`.

**Keep it that way.** A partial result shaped differently from a full one is a partial
result nothing can read, and everything downstream reads this file rather than the
checkpoint.

Several fields exist only to stop a number being believed too easily — `bbox_coverage`,
`clamped_pred_count`, `scale_check`, `elements_with_token_counts`. The reasoning for each
is in `benchmark/README.md` under "Output JSON format". They are cheap to compute and were
each added after a figure was misread.
"""
import logging
import statistics
import sys
from datetime import datetime, timezone

from benchmark.artifacts import checkpoint
from benchmark.scoring.coords import _normalize_model_name
from benchmark.scoring.metrics import IOU_THRESHOLD, _scale_check
from benchmark.model.client import _TEMPERATURE_REJECTED
from benchmark.workload.phrasings import _expand_pairs, _style_of

logger = logging.getLogger(__name__)


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
    pass_centroid = sum(1 for r in results if r["pass_centroid"])
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
    # centroid of 87.8% next to Qwen's 85.2% read as GUI-Owl winning, when a consumer that
    # needs a bounding box gets one from GUI-Owl in one request out of seven. Coverage is a
    # gate, not a metric — a model below it should not be compared on accuracy at all.
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
    centroid_accuracy = round(pass_centroid / total, 4) if total else 0
    # Primary metric drives the headline accuracy + which cases count as "failed".
    primary_accuracy = centroid_accuracy if metric == "centroid" else iou_accuracy

    def _passed(r: dict) -> bool:
        if metric == "centroid":
            return bool(r.get("pass_centroid"))
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
            d_cent = sum(1 for r in rows if r["pass_centroid"])
            d_iou_pass = sum(1 for r in rows if r["pass_iou"])
            by_description[str(idx)] = {
                "style": _style_of(idx),
                "total": n,
                "primary_accuracy": round(sum(1 for r in rows if _passed(r)) / n, 4) if n else 0,
                "centroid_pass_count": d_cent,
                "centroid_accuracy": round(d_cent / n, 4) if n else 0,
                "iou_pass_count": d_iou_pass,
                "iou_accuracy": round(d_iou_pass / n, 4) if n else 0,
                "mean_iou": round(statistics.mean(d_ious), 4) if d_ious else 0,
                "median_iou": round(statistics.median(d_ious), 4) if d_ious else 0,
                "avg_input_tokens": round(statistics.mean(d_in), 1) if d_in else 0,
                "avg_output_tokens": round(statistics.mean(d_out), 1) if d_out else 0,
                "avg_latency_ms": round(statistics.mean([r["latency_ms"] for r in rows]), 1) if rows else 0,
                "error_count": sum(1 for r in rows if r["error"]),
            }

    # Agreement between phrasings, over the elements that have every phrasing scored.
    # Restricted that way on purpose: a stopped run leaves elements holding one or two of
    # their three, and folding those in would score a 1-of-1 as unanimous and push
    # `all_accuracy` up exactly when the run is least complete.
    #
    # `any` is the ceiling a better prompt could reach — the element was findable, some
    # phrasing found it. `all` is robustness. `mixed` is the population worth looking at
    # element by element, because those rows say which phrasing the model cannot use.
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
        # Centroid metric: the centre of the predicted bbox inside the gt bbox.
        "centroid_pass_count": pass_centroid,
        "centroid_accuracy": centroid_accuracy,
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
        # Every latency figure in the summary is measured at this concurrency. Two runs at
        # different values are not comparable on latency, and on a single-slot server the
        # extra workers only add queue wait to each request rather than throughput, so the
        # number has to travel with the result.
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

    The recovery path for a run that died without writing one: a crash, an OOM kill, the
    machine going to sleep, or a Ctrl+C that reached the process before its own handler
    could write a PARTIAL. Everything the run had finished is already on disk, one line
    per element; this turns it into the result JSON everything downstream expects.

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
