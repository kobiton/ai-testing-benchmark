"""Turning the text model's XPath into a box, or recording why it could not be.

The vision track's answer is a box; the XML track's answer is an **XPath**, and an XPath
is not gradable until it is executed. This module does what a WebDriver client does with
it — runs it against the tree and takes the first node — and then what the tree does for
us: a node has `bounds`, so it becomes a box and the centroid metric applies unchanged.

Four outcomes, recorded per row as `xml_outcome`, because "no box" has three different
causes here. An XML-then-vision cascade counts them as one bucket ("not answered, sent
to vision") while the report keeps them apart:

    answered       the XPath selected at least one node; graded on the first one's bounds
    not_found      the model said NOT_FOUND — the only outcome the model *decided*
    invalid_xpath  the text does not compile as XPath 1.0 (a driver would reject it)
    no_match       valid XPath, zero nodes on this tree (a driver would find no element)

None of the four is an `error`: they are what the model answered, and a resume must not
pay to ask again. Errors stay what they were — transport, HTTP, truncation.
"""
import re
from typing import Optional

from lxml import etree

# What the prompt tells the model to answer when nothing matches. Lives here, in
# `scoring`, which depends on nothing internal by the package's layering rule.
NOT_FOUND_RESPONSE = "NOT_FOUND"

XML_OUTCOMES = ("answered", "not_found", "invalid_xpath", "no_match")

# In an XML-then-vision cascade every outcome but `answered` falls through to vision.
XML_NOT_ANSWERED = ("not_found", "invalid_xpath", "no_match")

# Markdown stripping as a locator does it: one fenced block or one inline code span
# wrapping the whole reply, nothing else.
_CODE_FENCE_RE = re.compile(r"^```(?:json|xpath|xml|css)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"^`(.+)`$", re.DOTALL)

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


def strip_markdown(text: str) -> Optional[str]:
    if not text or not text.strip():
        return None
    cleaned = text.strip()
    m = _CODE_FENCE_RE.match(cleaned)
    if m:
        return m.group(1).strip()
    m = _INLINE_CODE_RE.match(cleaned)
    if m:
        return m.group(1).strip()
    return cleaned


def is_not_found(text: str) -> bool:
    return bool(text) and text.strip().casefold() == NOT_FOUND_RESPONSE.casefold()


def sanitize_xpath(raw: str) -> tuple[Optional[str], bool]:
    """`(xpath, not_found)` from the model's reply.

    A strict reader checks for NOT_FOUND on the trimmed reply *before* stripping markdown,
    so a fenced ```NOT_FOUND``` reaches the XPath compiler and fails there. Here it is
    checked both before and after, so the row is filed under what the model meant rather
    than under a formatting slip. Both readings fall through to vision in a cascade; only
    the sub-line differs, and the more informative one is the point of having sub-lines.
    """
    if is_not_found(raw):
        return None, True
    cleaned = strip_markdown(raw)
    if cleaned is None:
        return None, False
    if is_not_found(cleaned):
        return None, True
    return cleaned, False


def resolve_xpath(xpath: str, tree) -> tuple[list, str]:
    """The element nodes `xpath` selects on the document, or why none could be.

    Evaluated on the *document* (`_ElementTree.xpath`), so `//x` and `hierarchy/x` both
    mean what they mean when a driver evaluates the expression on the whole page source.
    A result that is not a node-set — `count(//x)`, `//x/@text` — selects no *element*, and
    a driver would report no element for it either.
    """
    try:
        found = tree.xpath(xpath)
    except (etree.XPathSyntaxError, etree.XPathEvalError) as exc:
        return [], f"{type(exc).__name__}: {exc}"
    if not isinstance(found, list):
        return [], ""
    return [n for n in found if isinstance(n, etree._Element)], ""


def node_bbox(node, screen_w: int, screen_h: int) -> Optional[dict]:
    """The node's `bounds` as a normalised `{x, y, width, height}`, clamped to the screen.

    UiAutomator writes `bounds="[x1,y1][x2,y2]"`; a tree that carries `x/y/width/height`
    instead (iOS page source) is read too. A zero-area node still yields a box — a driver
    would tap its centre all the same — and scores 0 on IoU by construction.
    """
    if not (screen_w and screen_h):
        return None
    m = _BOUNDS_RE.match(node.get("bounds") or "")
    if m:
        x1, y1, x2, y2 = (int(v) for v in m.groups())
    else:
        try:
            x1, y1 = int(float(node.get("x"))), int(float(node.get("y")))
            x2 = x1 + int(float(node.get("width")))
            y2 = y1 + int(float(node.get("height")))
        except (TypeError, ValueError):
            return None
    x1, x2 = sorted((max(0, min(screen_w, x1)), max(0, min(screen_w, x2))))
    y1, y2 = sorted((max(0, min(screen_h, y1)), max(0, min(screen_h, y2))))
    return {"x": x1 / screen_w, "y": y1 / screen_h,
            "width": (x2 - x1) / screen_w, "height": (y2 - y1) / screen_h}


def xml_prediction(raw: str, tree, screen_w: int, screen_h: int) -> dict:
    """Everything the XML track records about one reply, from the reply and the tree.

    Shared by the live runner and `--rescore`, so a re-read of `raw` reproduces a run
    exactly — the same reason the vision track routes both through `_parse_response`.
    `tree` is the parsed full dump (an lxml `_ElementTree`); `screen_w`/`screen_h` are what
    its `bounds` are normalised against. Returns `xpath`, `xml_outcome`, `xpath_matches`
    and `pred_bbox` (None unless answered).
    """
    xpath, not_found = sanitize_xpath(raw)
    if not_found or xpath is None:
        return {"xpath": "", "xml_outcome": "not_found" if not_found else "invalid_xpath",
                "xpath_matches": 0, "pred_bbox": None}
    nodes, err = resolve_xpath(xpath, tree)
    if err:
        return {"xpath": xpath, "xml_outcome": "invalid_xpath", "xpath_matches": 0, "pred_bbox": None}
    if not nodes:
        return {"xpath": xpath, "xml_outcome": "no_match", "xpath_matches": 0, "pred_bbox": None}
    # The first match, as a driver's find-element takes it.
    bbox = node_bbox(nodes[0], screen_w, screen_h)
    if bbox is None:
        return {"xpath": xpath, "xml_outcome": "no_match", "xpath_matches": len(nodes), "pred_bbox": None}
    return {"xpath": xpath, "xml_outcome": "answered", "xpath_matches": len(nodes), "pred_bbox": bbox}
