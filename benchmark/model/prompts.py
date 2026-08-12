"""What the model under test is asked for.

Deliberately blunt and format-first, and it states **no box convention at all**. Loading
the labeller's rules into it — where a control's padding ends, what counts as the
element — would coach a model toward our annotator's answer and measure agreement with
them rather than element localisation.

Screenshots are sent raw, so a prompt that referred to any preprocessing (a drawn grid,
a resize) would be describing something that is not there.
"""
import logging

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a visual grounding assistant for mobile UI screenshots.
Your only job: find the element described and output its bounding box.

Output format - a JSON array of four floats, nothing else:
[x1, y1, x2, y2]

Rules:
- x1,y1 = top-left corner; x2,y2 = bottom-right corner
- All values normalized 0.0–1.0 relative to image width/height
- x1 < x2, y1 < y2
- Output ONLY the array — no explanation, no preamble, no markdown

Good response:  [0.05, 0.32, 0.48, 0.38]
Bad response:   "The element is located at..." or ```json [...]```
"""

USER_PROMPT_TEMPLATE = 'Find the bounding box of: "{description}"\nAnswer:'


# The same instruction in pixels instead of [0,1] floats, and **the default**. Note it is
# not a single-variable change: asking for pixels requires stating the image size, so this
# prompt also hands the model a concrete coordinate frame the normalized one withholds.
# Both halves turned out to matter — see benchmark/README.md for the table.
#
# What it fixed: on a 10-element pilot a frontier model scored 0/10 under `normalized`,
# with a systematic +0.0988 median vertical offset and boxes 1.41x too tall — while the
# *same model* asked for pixel integers by `pipeline/step2_detect_bboxes.py` put the same
# element at the ground-truth y to four decimals. Under this prompt it scores 10/10, median
# dy +0.0025. The model was never bad at grounding; it was obeying an instruction that
# withheld the frame it needed.
#
# What it does not settle, and must not be assumed: this is a per-model trait. Qwen2.5-VL
# *ignores* the normalized instruction and returns pixels anyway (5 of 6 raws on a pilot),
# which the parser rescales — so the style is close to a no-op for it, and the published
# 85.19% is not understated. GUI-Owl answers on a 0-1000 grid and has not been measured
# under this prompt at all. Measure a model before concluding anything about it from this
# note.
#
# Note what is NOT copied from the labelling pipeline's step 2: its "zero margin, stop at
# the outermost visible pixels" guidance. Those boxes were also 1.60x too tall, which that
# text plausibly fixes — but changing both at once would leave neither attributable. That
# is the next variable to try, not this one.
#
# The parser needs no change: `_normalize_window` decides scale from the window's maximum,
# so a pixel answer is divided by the image dimensions and a normalized one is not.
SYSTEM_PROMPT_PIXELS = """\
You are a visual grounding assistant for mobile UI screenshots.
Your only job: find the element described and output its bounding box.

IMAGE SIZE: {img_w} x {img_h} pixels

Output format - a JSON array of four integers, nothing else:
[x1, y1, x2, y2]

Rules:
- x1,y1 = top-left corner; x2,y2 = bottom-right corner
- All values are PIXEL coordinates in this image, integers, origin (0,0) at top-left
- 0 <= x1 < x2 <= {img_w} and 0 <= y1 < y2 <= {img_h}
- Output ONLY the array — no explanation, no preamble, no markdown

Good response:  [54, 768, 518, 912]
Bad response:   "The element is located at..." or ```json [...]```
"""

# A third style, `normalized-dims` — SYSTEM_PROMPT plus the one IMAGE SIZE line and nothing
# else — existed briefly to separate the two things `pixels` changes at once. It scored 40%
# where `normalized` scored 0% and `pixels` 100%, which is what established that both changes
# matter and neither alone explains the result. Removed once it had answered that: it is a
# control, not something to run a benchmark under, and a third branch here is a third thing
# to keep correct. The measurement it produced is kept in `benchmark/README.md` — if it ever
# needs reproducing, it is the two-line insertion this comment describes.
PROMPT_STYLES = ("normalized", "pixels")

_dims_warned = False


def system_prompt_for(style: str, img_w: int, img_h: int) -> str:
    """The system prompt for one output format.

    Falls back to `normalized` when `pixels` is asked for but the dimensions are unknown —
    `_load_image_b64` returns 0x0 when it cannot parse the header. Sending
    `IMAGE SIZE: 0 x 0 pixels` would be worse than falling back, and raising would kill the
    run: `engine/runner.py` calls `f.result()` with no `except`, so one unreadable file
    would end a paid run at whatever element it reached. The warning is emitted once, not
    per element.
    """
    global _dims_warned
    if style not in PROMPT_STYLES:
        # argparse's `choices` already rejects this, so reaching here is a coding error
        # in a programmatic caller. Raise rather than quietly answer in another format —
        # a silent fallback is how the wrong-model results in this repo happened.
        raise ValueError(f"unknown prompt style {style!r}; expected one of {PROMPT_STYLES}")

    if style == "pixels":
        if not (img_w and img_h):
            if not _dims_warned:
                _dims_warned = True
                logger.warning(
                    "prompt-style 'pixels' states the image size but the dimensions could "
                    "not be read (%sx%s); falling back to 'normalized' for those images. "
                    "Results are then a mix of two prompts — check the image files before "
                    "trusting this run.", img_w, img_h)
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT_PIXELS.format(img_w=img_w, img_h=img_h)
    return SYSTEM_PROMPT
