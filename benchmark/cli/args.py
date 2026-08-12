"""The command line, in one place.

Separated from dispatch because the flags are the harness's public contract while what
happens next is internal. The help text carries the reasoning for the non-obvious
defaults (`--max-image-dim 0`, `--metric centroid`, `--description-index 0`), so read it
before changing one — the README calls `--help` the place every flag is listed.
"""
import argparse
import os
from pathlib import Path

# This file is <repo>/benchmark/cli/args.py, so the root is three levels up. Every path
# default is built from this rather than from the caller's CWD: `python
# benchmark/run_vision_benchmark.py` from the repository root is the documented way to
# run it, and a relative default would resolve against wherever the shell happens to be.
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "dataset-v1.jsonl"


def build_parser() -> argparse.ArgumentParser:
    """Every flag the benchmark accepts."""
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
    return parser
