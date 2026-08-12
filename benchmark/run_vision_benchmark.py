"""
Vision benchmark — can a model locate a UI element from a natural-language description?

For each (element, phrasing) pair in the dataset:
  1. Send the screenshot + the description to an OpenAI-compatible /v1/chat/completions
     (or to Anthropic's /v1/messages — see --api-flavor)
  2. Parse the answer: bbox {x,y,w,h}, bare click point {x,y}, and nine other shapes
     models have been observed to emit
  3. centroid (primary): does the centre of the predicted box fall inside the GT box?
  4. IoU (secondary): PASS at IoU >= threshold, default 0.5

Both metrics are always computed; --metric only picks the headline number.

Path defaults resolve against the repository root, so these work from anywhere:
    python benchmark/run_vision_benchmark.py --dry-run
    python benchmark/run_vision_benchmark.py --model qwen2.5-vl
    python benchmark/run_vision_benchmark.py --model qwen2.5-vl --description-index 0 1 2

Point it at any OpenAI-compatible server with --base-url (vLLM, llama.cpp, Ollama,
OpenAI, or a gateway). See the repository README for endpoint recipes.

**This file is deliberately almost empty.** Everything lives in a layered package beside
it, and the layers depend downwards only:

    cli/        argument parsing and dispatch
    engine/     the run: thread pool, per-element step, stop handling
    artifacts/  checkpoint, the result JSON, re-scoring
    model/      the system under test: prompt, image encoding, HTTP
    scoring/    coordinate conventions, response parsing, metrics
    workload/   dataset rows, phrasings, --limit sampling

The one thing this file does do is fix the import root. Run as a script, `sys.path[0]`
is `benchmark/` — the package itself would then not be importable by name — so the
parent goes on the path before the first `benchmark.` import. Don't switch to relative
imports instead: a plain script has no package context to resolve them against.
"""
import sys
from pathlib import Path

# Before any `benchmark.` import, for the reason in the docstring above.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.cli.commands import main  # noqa: E402  (must follow the path insert)

if __name__ == "__main__":
    main()
