"""Where the repository is.

The package is installed, so `__file__` says nothing about where `data/`, `config/` and `benchmark-results/` are.

Every relative path (a CLI default, a path in a config file) resolves against the project root instead:
the nearest directory at or above the current working directory holding this project's `pyproject.toml`,
or the working directory when there is none (an installation without a checkout).

`AI_TESTING_BENCHMARK_ROOT` overrides both, for a run started from a service whose working directory is elsewhere.

Paths recorded in a result are written relative to that root when they fall inside it,
so a result file names `data/labels/ground-truth.jsonl` rather than one machine's absolute path.
"""
import os
from functools import lru_cache
from pathlib import Path

ROOT_ENV = "AI_TESTING_BENCHMARK_ROOT"
_MARKER = "pyproject.toml"
_PROJECT = 'name = "ai-testing-benchmark"'


@lru_cache(maxsize=1)
def project_root() -> Path:
    env = os.getenv(ROOT_ENV)
    if env:
        return Path(env).resolve()
    cwd = Path.cwd().resolve()
    for candidate in (cwd, *cwd.parents):
        marker = candidate / _MARKER
        try:
            if marker.is_file() and _PROJECT in marker.read_text(encoding="utf-8"):
                return candidate
        except OSError:
            continue
    return cwd


def resolve(path: str) -> Path:
    """A path from a flag or a config file, made absolute against the project root."""
    p = Path(path).expanduser()
    return p if p.is_absolute() else (project_root() / p)


def display_path(path) -> str:
    """The path as a result file should record it: relative to the root when inside it."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(project_root()))
    except ValueError:
        return str(p)
