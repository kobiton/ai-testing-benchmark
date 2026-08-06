"""
Step 2.  Bounding box detection.

For each element, ask the LLM to locate its bounding box (pixel coords → normalized to [0, 1]). One LLM call per element
for precise localization, which makes this step the dominant cost of the whole pipeline.

**This step produces ground truth, so its prompt is tuned for labelling accuracy and is deliberately not the prompt a
consumer of the model would send.** Keep the two separate. A labelling prompt is free to describe a box convention,
state that the answer must cover the full control including its padding, and take as many tokens as it needs; a
consumer's prompt is whatever the product actually sends. Copying one into the other in either direction is a mistake:

 - putting this prompt in the benchmark coaches the model toward the labeller's own convention, which is the one thing
   a benchmark must not do — the model would be scored on agreeing with the labeller rather than on finding the element;
 - putting a consumer prompt here degrades the ground truth every later measurement is compared against.

If you want to measure a specific consumer's prompt, do it in the benchmark harness together with whatever image
preprocessing makes that prompt valid.
"""
import json
import logging
from PIL import Image

from .llm_client import call_llm, extract_json
from .models import BoundingBox, ScreenshotRecord

logger = logging.getLogger(__name__)

_NOT_FOUND_RESPONSE = '{"found": false, "element": null}'

# __DESCRIPTION__, __IMG_W__, __IMG_H__ and __NOT_FOUND__ are substitution placeholders rather than str.format fields, so
# the JSON braces in the body need no escaping.
#
# The prompt asks for exactly what `_parse_bbox_response` consumes and nothing else. Element identity and type were
# settled in step 1, so this call is pure localization: no candidate selection, no scoring, no reasoning trace.
#
# **The box convention below is the load-bearing part.** Every measurement made against this dataset compares a model's
# box to one produced by these rules, and the headline finding of the whole benchmark — that models box the glyph while
# ground truth boxes the padded control — is a statement about this convention. Reword it freely; do not change what it
# asks for. The three edge cases (bounded control / icon / bare text) are not decoration: each is a case where two
# annotators would otherwise disagree by more than the metric's tolerance.
#
# The coordinate range is deliberately not restated as inequalities. `_parse_bbox_response` clamps to [0, 1] and rejects
# a non-positive width or height, so a violation costs nothing and spending prompt on it buys nothing.
_SYSTEM_PROMPT_TEMPLATE = """\
You are annotating ground truth for a UI-grounding dataset. Given a mobile screenshot and one element known to appear in it, report the pixel rectangle that element occupies.

Element: "__DESCRIPTION__"
Screenshot: __IMG_W__ x __IMG_H__ pixels. Origin top-left, x grows right, y grows down.

Where the rectangle's edges belong:
- A control drawn with its own boundary - button, input, chip, toggle, tab, list row - is boxed at the outer edge of that drawn boundary. Include the control's background and border, and stop where they stop.
- An icon is boxed around the graphic itself, stopping at its outermost drawn pixels and not at the whitespace around it.
- Text with no boundary of its own is boxed at the glyphs: tallest ascender to lowest descender, first character to last.
- No edge may sit in empty space. Each of the four edges rests on the element's own outermost drawn pixels.
- Box the element named above and nothing that encloses it. A section, card, form or panel containing it is a different rectangle.

Check all four edges before answering.

Answer with this JSON and nothing else, in whole pixels measured from the top-left corner:
{"found": true, "element": {"x": <int>, "y": <int>, "width": <int>, "height": <int>}}

The element is expected to be present. Only if it is genuinely not visible, answer:
__NOT_FOUND__"""


def _build_system_prompt(description: str, img_w: int, img_h: int) -> str:
    return (
        _SYSTEM_PROMPT_TEMPLATE
        .replace("__DESCRIPTION__", description)
        .replace("__IMG_W__", str(img_w))
        .replace("__IMG_H__", str(img_h))
        .replace("__NOT_FOUND__", _NOT_FOUND_RESPONSE)
    )


def _parse_bbox_response(raw: str, img_w: int, img_h: int):
    """Parse the LLM response; return a normalized BoundingBox or None."""
    try:
        data = json.loads(extract_json(raw))
    except (json.JSONDecodeError, ValueError):
        return None

    if not data.get("found"):
        return None

    el = data.get("element")
    if not el:
        return None

    x_px = el.get("x", 0)
    y_px = el.get("y", 0)
    w_px = el.get("width", 0)
    h_px = el.get("height", 0)

    if w_px <= 0 or h_px <= 0:
        return None

    x = max(0.0, min(1.0, x_px / img_w))
    y = max(0.0, min(1.0, y_px / img_h))
    w = max(0.0, min(1.0 - x, w_px / img_w))
    h = max(0.0, min(1.0 - y, h_px / img_h))

    if w <= 0 or h <= 0:
        return None

    return BoundingBox(x=x, y=y, width=w, height=h)


def detect_bboxes(record: ScreenshotRecord) -> ScreenshotRecord:
    """
    Calls the LLM once per element to locate its bounding box.
    Pixel output is normalized to [0, 1] against the image's own dimensions.
    """
    if not record.elements:
        return record

    img = Image.open(record.image_path)
    img_w, img_h = img.size

    logger.info("Detecting bboxes for %s (%d elements, %dx%d)",record.screenshot_id, len(record.elements), img_w, img_h)

    detected = 0
    skipped = 0
    for el in record.elements:
        description = f"{el.name} ({el.type})"
        system_prompt = _build_system_prompt(description, img_w, img_h)
        raw = call_llm(system_prompt, "Locate the element described above.", image_path=record.image_path)

        bbox = _parse_bbox_response(raw, img_w, img_h)
        if bbox:
            el.bbox = bbox
            detected += 1
            logger.debug("  ✓ %s → bbox(%.3f,%.3f,%.3f,%.3f)", el.element_id, bbox.x, bbox.y, bbox.width, bbox.height)
        else:
            logger.debug("  ✗ no bbox for %s (%s)", el.element_id, el.name)
            skipped += 1

    logger.info("  → %d detected, %d skipped", detected, skipped)
    return record
