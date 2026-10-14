"""How to read a model's coordinate numbers.

Two answers can be byte-identical and mean different things: on a 1080-wide screenshot
`[67, 91]` is a legal pixel pair *and* a legal 0-1000 grid pair. Nothing in a single
response distinguishes them, so the convention has to be a per-model fact — which is
what this module holds, and why model *identity* lives here too rather than beside the
HTTP call. Everything that compares model names in this project is ultimately asking
the same question: did I get the model I asked for, and therefore does its grid apply?

Getting this wrong is not a rounding error. A full, clean, error-free 30,921-row run
reported 10.56% centroid for a model whose real score was 87.80%.
"""
import re


def _normalize_model_name(name: str) -> str:
    """Lowercase alphanumerics only, so `qwen2.5-vl` matches a served id like
    `/Users/x/models/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf`."""
    return re.sub(r"[^a-z0-9]", "", name.lower())

# The grid a model's pixel-scale answers are expressed on, per model name. 0 — the
# default for anything not listed — means the model answers in the screenshot's own
# pixels, which is what the parser has always assumed.
#
# GUI-Owl (Qwen-VL family) does not. It answers on a fixed 0-1000 grid, so dividing its
# numbers by a 1080x2400 screenshot shrinks x by 1.08x and y by 2.4x — and 2.4x on one
# axis is the difference between a working locator and a useless one. Measured over a full
# three-phrasing GUI-Owl run on this dataset: **10.56% centroid as recorded, ~87.9% once
# the grid is applied.** The reverse holds too — applying the grid to Qwen2.5-VL, which
# really does answer in pixels, drops it from 84.40% to 0.31%. So this
# is a property of the model and there is no safe way to infer it per answer: on a
# 1080-wide screenshot `[67, 91]` is a legal pixel pair *and* a legal grid pair, which is
# exactly why the bug survived a full 30,921-call run without tripping anything.
#
# 1000 rather than 999 on purpose. Both fit — the sweep gives 85.62% at /1000 and 85.91%
# at /999 — but 1000 is the documented Qwen-VL convention (pixels scaled into [0,1000)),
# while 999 is the value that happens to score highest on this dataset. Fitting a
# constant to the benchmark it is then evaluated on is how a harness starts flattering
# itself; the 0.29pp is not worth that.
#
# Keys are matched as substrings of `_normalize_model_name(requested + served)`, so one
# entry covers both `gui-owl-1.5-8b` and a served id like
# `/models/GUI-Owl-1.5-8B-Instruct.Q4_K_M.gguf`.
COORD_GRIDS = {"guiowl": 1000}


def _coord_grid_for(model: str, served_model: str = "", override: int = -1) -> int:
    """The grid `model` answers on; 0 means the screenshot's own pixels.

    The served id is matched as well as the requested name because a single-model
    llama.cpp endpoint ignores the name it is asked for — the same reason
    `_probe_served_model` exists. A run asking for `qwen2.5-vl` against a box that has
    GUI-Owl loaded is scored under GUI-Owl's convention, which is the one that produced
    the numbers.
    """
    if override >= 0:
        return override
    hay = _normalize_model_name(f"{model} {served_model}")
    for key, grid in COORD_GRIDS.items():
        if key in hay:
            return grid
    return 0


def _norm_dims(img_w: int, img_h: int, coord_grid: int = 0) -> tuple[int, int]:
    """What a pixel-scale answer should be divided by to reach [0,1].

    Handing the grid over as *both* dimensions is the whole implementation: every place
    that normalises divides by these two numbers, so a model on a fixed grid needs no
    separate code path and cannot drift from the pixel path. The "already in [0,1]"
    branches keep working untouched, which is load-bearing — GUI-Owl mixes normalised
    floats (`[0.19, 0.81, 0.81, 0.85]`) into the same run as grid integers
    (`[67, 91]`), so both readings have to stay available within one model.
    """
    return (coord_grid, coord_grid) if coord_grid > 0 else (img_w, img_h)
