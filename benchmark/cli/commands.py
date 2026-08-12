"""Dispatch: which of the four things that can produce a result file to run.

    a run          model calls, scored as they land
    --finalize-only  rebuild from a checkpoint after a crash, no inference
    --rescore      re-read the answers a finished result already holds
    --dry-run      validate the dataset and call nothing

All four end in `build_result`, and they have to: a partial result shaped differently
from a full one is a partial result nothing can read.
"""
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv

from benchmark.artifacts.builder import finalize_from_checkpoint
from benchmark.artifacts.rescoring import rescore_result, rescored_filename
from benchmark.cli.args import build_parser
from benchmark.engine.runner import _install_stop_handlers, run_benchmark
from benchmark.model.client import _list_loaded_models
from benchmark.scoring.coords import _coord_grid_for
from benchmark.scoring.metrics import IOU_THRESHOLD
from benchmark.workload.dataset import _load_dataset
from benchmark.workload.sampling import even_sample

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = build_parser()
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
        # there: the dropdown shows names, and a number that moved for a reason other than
        # model quality is the kind that gets quoted without its reason.
        if len(description_indices) > 1:
            safe_model = f"{safe_model}-{len(description_indices)}phrasing"
        # A mismatch goes in the filename, not just the JSON: the results dropdown
        # lists files by name, and two runs that silently hit the same loaded model
        # would otherwise sit there looking like two different models.
        if result.get("model_mismatch"):
            served_stem = Path(result["served_model"]).stem.replace("/", "-").replace(":", "-")
            safe_model = f"{safe_model}-SERVED-{served_stem}"
        # Same reasoning for a partial run: the dropdown shows filenames, and a run
        # stopped at 12% otherwise looks exactly like one that finished.
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
