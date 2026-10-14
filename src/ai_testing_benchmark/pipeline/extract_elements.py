"""
Step 1. Element extraction.

For each screenshot, ask the LLM to identify all interactive UI elements and return them as a structured JSON list.
"""
import json
import logging

from .llm_client import call_llm, extract_json
from .models import Element, ScreenshotRecord

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a mobile UI analyst. Your job is to identify all interactive UI elements visible in a mobile app screenshot.

Return ONLY valid JSON in this exact format:
{
  "elements": [
    {
      "element_id": "e1",
      "name": "Login",
      "type": "button"
    }
  ]
}

Element types to use: button, text_field, link, icon, checkbox, radio, toggle, dropdown, tab, image, label, list_item.

Rules:
- Include only elements a user can interact with (tap, type, scroll)
- Use short, descriptive names (e.g. "Login button", "Email input")
- Assign sequential element_ids: e1, e2, e3, ...
- Do not include decorative elements (backgrounds, dividers, illustrations)
"""

USER_PROMPT = "Identify all interactive UI elements in this screenshot."


def extract_elements(record: ScreenshotRecord) -> ScreenshotRecord:
    """
    Calls the LLM on the screenshot and populates record.elements.
    Returns the same record with elements filled in.
    """
    logger.info("Extracting elements from %s", record.screenshot_id)

    raw = call_llm(SYSTEM_PROMPT, USER_PROMPT, image_path=record.image_path)

    try:
        data = json.loads(extract_json(raw))
        elements = data.get("elements", [])
    except json.JSONDecodeError:
        logger.error("Failed to parse LLM response for %s: %s", record.screenshot_id, raw)
        return record

    record.elements = [
        Element(
            element_id=el.get("element_id", f"e{i+1}"),
            name=el.get("name", ""),
            type=el.get("type", "unknown"),
        )
        for i, el in enumerate(elements)
    ]

    logger.info("  → found %d elements", len(record.elements))
    return record
