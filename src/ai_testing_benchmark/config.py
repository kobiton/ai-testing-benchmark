"""JSON run configuration - the command line as a file.

A config names everything a run depends on in one place that can be committed beside the
result it produced: the model, the endpoint, which rows of the dataset, which phrasings,
how the model is asked, how its coordinates are read, and the inference limits. The files
under `config/<track>/<model>.json` are the configurations the published results were
made with.

Rules:

- **Every key is checked before anything is loaded.** An unknown key, a missing required
  key or a value of the wrong type stops the run with the key named. A config that is
  silently half-read produces a result that looks right and is not.
- **No credential in the file.** `endpoint.api_key_env` names the environment variable the
  key is read from; the run refuses to start when that variable is unset, except under
  `--dry-run` and in `cascade`, which call no model.
- **Flags override the file**, flag by flag: `--config x.json --limit 50` is x.json's run
  as a pilot.
- **The result records what was used**, under `config`: the file (name and hash) and the
  resolved values after overrides and defaults, with the key's variable name in place of
  the key.

Relative paths in a config resolve against the project root (see `paths.py`), not the
working directory, so `config/xml/gpt-5-6-terra.json` means the same thing from anywhere
inside the checkout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

from ai_testing_benchmark.benchmark.model.client import API_FLAVORS as _API_FLAVORS
from ai_testing_benchmark.benchmark.model.prompts import PROMPT_STYLES as _PROMPT_STYLES
from ai_testing_benchmark.paths import display_path, resolve

API_FLAVORS = tuple(_API_FLAVORS)
PROMPT_STYLES = tuple(_PROMPT_STYLES)
INPUTS = ("screenshot", "xml")
COORD_FORMATS = ("corner", "center")
METRICS = ("centroid", "iou")

# Field: (type, argparse dest or None, choices or None). A nested dict is a section.
_STR, _INT, _NUM, _INTS, _STRS = "str", "int", "number", "int list", "str list"
SCHEMA = {
    "model": (_STR, "model", None),
    "input": (_STR, "input", INPUTS),
    "endpoint": {
        "base_url": (_STR, "proxy_url", None),
        "api_flavor": (_STR, "api_flavor", API_FLAVORS),
        "api_key_env": (_STR, None, None),
    },
    "dataset": {
        "path": (_STR, "dataset", None),
        "annotation_origin": (_STRS, "annotation_origin", None),
        "images_dir": (_STR, "images_dir", None),
        "xml_dir": (_STR, "xml_dir", None),
        "limit": (_INT, "limit", None),
    },
    "description_indices": (_INTS, "description_index", None),
    "prompt": {
        "style": (_STR, "prompt_style", PROMPT_STYLES),
        "xml_prompt": (_STR, "xml_prompt", None),
    },
    "coordinates": {
        "format": (_STR, "coord_format", COORD_FORMATS),
        "grid": (_INT, "coord_grid", None),
    },
    "inference": {
        "max_tokens": (_INT, "max_tokens", None),
        "temperature": (_NUM, "temperature", None),
        "thinking_budget": (_INT, "thinking_budget", None),
        "max_image_dim": (_INT, "max_image_dim", None),
        "timeout": (_INT, "timeout", None),
        "workers": (_INT, "workers", None),
        "rpm": (_INT, "rpm", None),
        "max_requests": (_INT, "max_requests", None),
    },
    "metric": (_STR, "metric", METRICS),
    "output_dir": (_STR, "output_dir", None),
}
REQUIRED = ("model", "input", "endpoint.base_url", "endpoint.api_flavor", "dataset.path",
            "description_indices", "inference.max_tokens", "inference.temperature",
            "inference.workers")
# Which config keys hold paths — these resolve against the project root.
PATH_KEYS = ("dataset.path", "dataset.images_dir", "dataset.xml_dir", "prompt.xml_prompt", "output_dir")


class ConfigError(ValueError):
    pass


def _is(kind: str, value) -> bool:
    if kind == _STR:
        return isinstance(value, str)
    if kind == _INT:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind == _NUM:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind == _INTS:
        return isinstance(value, list) and value and all(_is(_INT, v) for v in value)
    if kind == _STRS:
        return isinstance(value, list) and value and all(_is(_STR, v) for v in value)
    raise AssertionError(kind)


def _flatten(data: dict, schema: dict, prefix: str = "") -> dict:
    """{'endpoint.base_url': value, …}, raising on an unknown key or a wrong type."""
    if not isinstance(data, dict):
        raise ConfigError(f"'{prefix.rstrip('.') or '<root>'}' must be an object")
    out = {}
    for key, value in data.items():
        full = f"{prefix}{key}"
        if key not in schema:
            raise ConfigError(f"unknown key '{full}' (known: {', '.join(sorted(schema))})")
        spec = schema[key]
        if isinstance(spec, dict):
            out.update(_flatten(value, spec, f"{full}."))
            continue
        kind, _dest, choices = spec
        if not _is(kind, value):
            raise ConfigError(f"'{full}' must be a {kind}, got {json.dumps(value)}")
        if choices and value not in choices:
            raise ConfigError(f"'{full}' must be one of {', '.join(choices)}, got {json.dumps(value)}")
        out[full] = value
    return out


def validate(data: dict) -> dict:
    """The config as a flat {'section.key': value} dict, or ConfigError."""
    flat = _flatten(data, SCHEMA)
    missing = [k for k in REQUIRED if k not in flat]
    if missing:
        raise ConfigError(f"missing required key(s): {', '.join(missing)}")
    if flat["input"] == "xml":
        if "prompt.style" in flat:
            raise ConfigError("'prompt.style' applies to input=screenshot; the tree track has no coordinate prompt")
    else:
        if "prompt.xml_prompt" in flat:
            raise ConfigError("'prompt.xml_prompt' applies to input=xml")
        if "prompt.style" not in flat:
            raise ConfigError("missing required key(s): prompt.style (input=screenshot)")
    if any(i < 0 for i in flat["description_indices"]):
        raise ConfigError("'description_indices' must be >= 0")
    return flat


def load(path, require_key: bool = True) -> tuple[dict, dict]:
    """Read, validate and resolve a run config.

    Returns (flat config, file info) where file info is `{"file": …, "sha256": …}` as the
    result records it. Paths are resolved against the project root; `api_key_env` is read
    into the flat key `api_key`, and the variable's name is kept under `endpoint.api_key_env`.
    `require_key=False` lets an unset variable pass: for `--dry-run` and for `cascade`, which
    read the config without calling the model.
    """
    p = resolve(str(path))
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p}: not valid JSON ({e})") from e
    flat = validate(data)
    for key in PATH_KEYS:
        if key in flat:
            flat[key] = str(resolve(flat[key]))
    env = flat.get("endpoint.api_key_env")
    if env:
        value = os.getenv(env, "")
        if value:
            flat["api_key"] = value
        elif require_key:
            raise ConfigError(f"'endpoint.api_key_env' names {env}, which is not set in the environment")
    info = {"file": display_path(p), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}
    return flat, info


def _dests(schema: dict = SCHEMA) -> dict:
    """{'endpoint.base_url': 'proxy_url', …} for every key that maps to a flag."""
    out = {}
    for key, spec in schema.items():
        if isinstance(spec, dict):
            out.update({f"{key}.{k}": v for k, v in _dests(spec).items()})
        elif spec[1]:
            out[key] = spec[1]
    return out


def apply_to_parser(parser: argparse.ArgumentParser, argv) -> tuple:
    """Pre-read `--config` from argv and install the file's values as the parser's defaults.

    Done before the real parse so that flags given on the command line still win: argparse
    only falls back to a default when the flag is absent. Returns (flat config or None,
    file info or None).
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default="")
    pre.add_argument("--dry-run", action="store_true")
    known, _ = pre.parse_known_args(argv)
    if not known.config:
        return None, None
    flat, info = load(known.config, require_key=not known.dry_run)
    defaults = {}
    for key, dest in _dests().items():
        if key in flat:
            defaults[dest] = [flat[key]] if dest == "model" else flat[key]
    if "api_key" in flat:
        defaults["api_key"] = flat["api_key"]
    parser.set_defaults(**defaults)
    return flat, info


def resolved(args: argparse.Namespace, model: str, info: dict = None, api_key_env: str = "") -> dict:
    """The configuration a run actually used, in the file's own shape, for the result.

    Built from the parsed arguments rather than from the file, so a flag that overrode the
    file is what gets recorded. Never holds the key: `endpoint.api_key_env` names the
    variable it came from (empty when it was passed as a flag or by the default variables),
    and `api_key_set` says whether one was sent at all.
    """
    xml = args.input == "xml"
    out = {
        "model": model,
        "input": args.input,
        "endpoint": {
            "base_url": args.proxy_url,
            "api_flavor": args.api_flavor,
            "api_key_env": api_key_env,
            "api_key_set": bool(args.api_key),
        },
        "dataset": {
            "path": display_path(args.dataset),
            "annotation_origin": list(args.annotation_origin or []),
            "images_dir": display_path(args.images_dir),
            **({"xml_dir": display_path(args.xml_dir)} if xml else {}),
            "limit": args.limit,
        },
        "description_indices": sorted(set(args.description_index)),
        "prompt": ({"xml_prompt": display_path(args.xml_prompt)} if xml
                   else {"style": args.prompt_style}),
        "coordinates": {"format": args.coord_format, "grid": args.coord_grid},
        "inference": {
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "thinking_budget": args.thinking_budget,
            "max_image_dim": args.max_image_dim,
            "timeout": args.timeout,
            "workers": args.workers,
            "rpm": args.rpm,
            "max_requests": args.max_requests,
        },
        "metric": args.metric,
        "output_dir": display_path(args.output_dir),
    }
    if info:
        out = {"file": info["file"], "file_sha256": info["sha256"], **out}
    return out


# ── cascade configs ──────────────────────────────────────────────────────────
# A cascade is a join of two results, so its config names the two run configs (or two
# result files outright) and how to pick the rows.
CASCADE_SCHEMA = {
    "model": (_STR, None, None),
    "xml_config": (_STR, None, None),
    "vision_config": (_STR, None, None),
    "xml_result": (_STR, None, None),
    "vision_result": (_STR, None, None),
    "annotation_origin": (_STRS, None, None),
    "with_vision_alone": ("bool", None, None),
    "results_dir": (_STR, None, None),
}
CASCADE_REQUIRED = ("model",)


def load_cascade(path) -> tuple[dict, dict]:
    p = resolve(str(path))
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    text = p.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p}: not valid JSON ({e})") from e
    if not isinstance(data, dict):
        raise ConfigError("the config must be an object")
    flat = {}
    for key, value in data.items():
        if key not in CASCADE_SCHEMA:
            raise ConfigError(f"unknown key '{key}' (known: {', '.join(sorted(CASCADE_SCHEMA))})")
        kind = CASCADE_SCHEMA[key][0]
        ok = isinstance(value, bool) if kind == "bool" else _is(kind, value)
        if not ok:
            raise ConfigError(f"'{key}' must be a {kind}, got {json.dumps(value)}")
        flat[key] = value
    missing = [k for k in CASCADE_REQUIRED if k not in flat]
    if missing:
        raise ConfigError(f"missing required key(s): {', '.join(missing)}")
    if not (("xml_config" in flat or "xml_result" in flat) and ("vision_config" in flat or "vision_result" in flat)):
        raise ConfigError("name the two halves: xml_config or xml_result, and vision_config or vision_result")
    for key in ("xml_config", "vision_config", "xml_result", "vision_result", "results_dir"):
        if key in flat:
            flat[key] = str(resolve(flat[key]))
    info = {"file": display_path(p), "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]}
    return flat, info
