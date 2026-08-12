"""What the model under test is asked for.

Deliberately blunt and format-first, and it states **no box convention at all**. Loading
the labeller's rules into it — where a control's padding ends, what counts as the
element — would coach a model toward our annotator's answer and measure agreement with
them rather than element localisation.

Screenshots are sent raw, so a prompt that referred to any preprocessing (a drawn grid,
a resize) would be describing something that is not there.
"""

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
