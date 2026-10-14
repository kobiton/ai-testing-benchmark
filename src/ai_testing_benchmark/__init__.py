"""AI Testing Benchmark: can a model find a UI element from a plain-English description?

Two packages, one command:

    benchmark/   the harness: both tracks (screenshot, accessibility tree),
                 the cascade scorer, the failure browser, re-scoring and checkpoints
    pipeline/    the labelling pipeline that produced the dataset

`ai-testing-benchmark <command>` (or `python -m ai_testing_benchmark`) dispatches to them; see `cli.py`.
Run configurations are JSON files, see `config.py`.
"""

__version__ = "1.0.0"
