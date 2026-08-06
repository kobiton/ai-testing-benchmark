"""
Step 3. Alternative description generation.

For each element, generate 3 natural-language descriptions that a tester might use to identify it:
  1. Short common name. E.g. "the login button"
  2. Structural label. E.g. "the button with the text login"
  3. Functional intent. E.g. "the button that will allow the user to login to the system"
"""
import json
import logging

from .llm_client import call_llm, extract_json
from .models import ScreenshotRecord

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a mobile QA engineer writing element descriptions for automated test selectors.

Given a screenshot and a list of UI elements, generate exactly 3 natural-language descriptions per element that a tester would use to locate it.

Description styles (one of each, in order):
1. name - short common name, e.g. "the login button"
2. label - structural description referencing visible text or icon, e.g. "the button with the text login"
3. intent - functional purpose from the user's perspective, e.g. "the button that will allow the user to login to the system"

Return ONLY valid JSON in this exact format:
{
  "descriptions": [
    {
      "element_id": "e1",
      "name": "the login button",
      "label": "the button with the text login",
      "intent": "the button that will allow the user to login to the system"
    }
  ]
}

Rules:
- All 3 descriptions must be distinct and written in plain English
- Start each description with "the"
- Do not include element_id, technical type names, or coordinates in the text
- name: 3-6 words; label: 6-12 words; intent: 10-20 words
"""


def _build_user_prompt(elements) -> str:
    lines = ["Generate 3 descriptions for each element below:\n"]
    for el in elements:
        bbox_hint = ""
        if el.bbox:
            cx = el.bbox.x + el.bbox.width / 2
            cy = el.bbox.y + el.bbox.height / 2
            h_pos = "left" if cx < 0.33 else ("right" if cx > 0.66 else "center")
            v_pos = "top" if cy < 0.33 else ("bottom" if cy > 0.66 else "middle")
            bbox_hint = f" (located at {v_pos}-{h_pos})"
        lines.append(f"- {el.element_id}: {el.name} [{el.type}]{bbox_hint}")
    return "\n".join(lines)


def generate_descriptions(record: ScreenshotRecord) -> ScreenshotRecord:
    """
    Generates 3 descriptions per element. Elements that already have descriptions are skipped (idempotent).
    """
    elements_to_describe = [el for el in record.elements if not el.descriptions]
    if not elements_to_describe:
        return record

    logger.info("Generating descriptions for %s (%d elements)", record.screenshot_id, len(elements_to_describe))

    user_prompt = _build_user_prompt(elements_to_describe)
    raw = call_llm(SYSTEM_PROMPT, user_prompt, image_path=record.image_path)

    try:
        data = json.loads(extract_json(raw))
        desc_list = data.get("descriptions", [])
    except json.JSONDecodeError:
        logger.error("Failed to parse description response for %s", record.screenshot_id)
        return record

    desc_map = {d["element_id"]: d for d in desc_list if "element_id" in d}

    generated = 0
    for el in elements_to_describe:
        d = desc_map.get(el.element_id)
        if not d:
            logger.debug("  No description returned for %s", el.element_id)
            continue
        name = d.get("name", "").strip()
        label = d.get("label", "").strip()
        intent = d.get("intent", "").strip()
        el.descriptions = [s for s in [name, label, intent] if s]
        generated += 1

    logger.info("  → descriptions generated for %d elements", generated)
    return record
