"""The XML input: the accessibility tree, filtered and worded for a text model.

The second way a natural-language locator can find an element does not look at the screenshot at all.
It takes the screen's accessibility tree, drops what carries no information,
and asks a text model for **one XPath or `NOT_FOUND`**;
a WebDriver client then runs the XPath.

The `--input xml` track measures a model's skill at that task. The filter is the one such a
locator applies before the tree goes out. The prompt is read from a file — `prompts/xpath-android.txt`
beside this module unless `--xml-prompt` names another — so the words a run was measured with are
data the result records (file name and hash) rather than code it was built from.

Two things about the input are worth knowing, and both are recorded in the result:

- **The tree source.** A locator running on a device reads every window on screen except
  the system bars, so the on-screen keyboard is part of its tree. The dataset carries the
  UiAutomator page source captured beside each screenshot, which covers the app window
  only. Same attribute vocabulary (`class`, `text`, `resource-id`, `content-desc`,
  `bounds`), not the same bytes — a keyboard key has no node here and would on a device.

- **The prompt order.** The original prompt puts the description *before* the tree. With
  that order no two requests share a prefix longer than the ~900-token instructions, which
  is under OpenAI's 1,024-token caching floor, so the 3k–26k-token tree is paid for on
  every one of the ~38 requests about the same screenshot. This track sends the tree first
  and the description last and changes no other character. The swap was checked before it
  became the only order: on a 200-element pilot the two layouts gave the same verdict on
  97.5% of rows (58.0% vs 57.5% right), and the original layout cached nothing where this
  one cached 70% of input. A flag for the original order existed for that pilot and was
  removed once it had answered; reproducing it is `build_xml_prompt` with its two trailing
  sections swapped, sent as one user message.

"""
import copy
import hashlib
import logging
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from lxml import etree

logger = logging.getLogger(__name__)


# --- The tree filter ------------------------------------------------------------------

# Attributes kept for the model, compared case-insensitively. The set is the union over
# Android, iOS and HarmonyOS vocabularies, so one filter serves any platform's tree.
ESSENTIAL_ATTRIBUTES = frozenset(a.lower() for a in (
    # Identifiers
    "type", "class", "name", "resource-id", "id", "key",
    # Text content
    "label", "text", "content-desc", "description", "value", "hint",
    # State
    "visible", "enabled", "selected", "checked", "focused",
    # Bounds (needed to verify elements exist)
    "x", "y", "width", "height", "bounds",
))

# Deep branches beyond this level are removed unless they contain meaningful text.
MAX_DEPTH = 50

# `HasMeaningfulContent` for Android: any of these non-blank keeps the node.
_MEANINGFUL_ANDROID = ("text", "content-desc", "resource-id", "hint")


def _has_meaningful_content(el) -> bool:
    return any((el.get(a) or "").strip() for a in _MEANINGFUL_ANDROID)


def _filter_nodes(el, depth: int) -> None:
    # Children first (bottom-up), so a parent emptied by its children's removal is seen as empty.
    for child in list(el):
        _filter_nodes(child, depth + 1)
    if el.getparent() is None:
        return
    # Invisible elements are *kept* on purpose: they can still be found and queried.
    # Only a leaf with nothing to say, or an over-deep branch, goes.
    if not _has_meaningful_content(el) and (depth > MAX_DEPTH or len(el) == 0):
        el.getparent().remove(el)


def _strip_attributes(el) -> None:
    for name in list(el.attrib):
        if name.lower() not in ESSENTIAL_ATTRIBUTES:
            del el.attrib[name]
    for child in el:
        _strip_attributes(child)


def filter_view_tree(root):
    """A filtered *copy* of the tree.

    The original is left intact because the XPath the model returns is executed against the **full** tree,
     the driver runs it on the live device, which knows nothing of the filter.
     And that is also where positional indices like `(//X)[2]` may differ between what the model saw and what gets selected.
     A deployed locator has that same gap; keeping it is the point.
    """
    filtered = copy.deepcopy(root)
    _filter_nodes(filtered, 0)
    _strip_attributes(filtered)
    return filtered


def serialize(el) -> str:
    """The tree as the model sees it: indented, no XML declaration."""
    return etree.tostring(el, encoding="unicode", pretty_print=True).rstrip()


# --- The XPath prompt, Android ----------------------------------------------------------

# The instructions the model reads before the tree, kept as a text file beside this module
# rather than as a string in it: the words a run was measured with are then data the result
# records — file name and a hash of the text — and `--xml-prompt` can point a run at another
# file without touching code. The two trailing sections (tree, then description) are fixed
# here, because their order is what makes the request cacheable; see `build_xml_prompt`.
DEFAULT_XML_PROMPT = Path(__file__).with_name("prompts") / "xpath-android.txt"


@dataclass
class XmlPrompt:
    path: str
    name: str
    sha256: str          # first 12 hex digits over `instructions`, as the result records it
    instructions: str    # the file's text, ending in exactly one blank line


def load_xml_prompt(path: str = "") -> XmlPrompt:
    """Read the instructions file; `path` empty means the default beside this module.

    Trailing newlines are normalised to one blank line so the hash does not depend on how an
    editor ended the file, and so the tree section always starts after exactly one blank line.
    """
    p = Path(path) if path else DEFAULT_XML_PROMPT
    text = p.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"XML prompt file is empty: {p}")
    text = text.rstrip("\n") + "\n\n"
    return XmlPrompt(path=str(p.resolve()), name=p.name,
                     sha256=hashlib.sha256(text.encode("utf-8")).hexdigest()[:12],
                     instructions=text)


_DESCRIPTION_SECTION = "USER DESCRIPTION:\n{description}"
_HIERARCHY_SECTION = "UI ELEMENT HIERARCHY:\n{hierarchy}"

def build_xml_prompt(description: str, hierarchy_xml: str, prompt: XmlPrompt) -> tuple[str, str]:
    """The prompt as `(prefix, suffix)`; the request sends `prefix` then `suffix`.

    Split where the reusable part ends, so a provider that caches on request can be told:
    the prefix is the instructions plus the tree — identical for every element and phrasing
    of one screenshot — and the suffix is the description alone. `_build_text_payload`
    sends the prefix as the system message and the suffix as the user message.
    """
    desc = _DESCRIPTION_SECTION.format(description=description)
    tree = _HIERARCHY_SECTION.format(hierarchy=hierarchy_xml)
    return f"{prompt.instructions}{tree}\n\n", desc


# --- The dumps on disk ------------------------------------------------------------------

_BOUNDS_RE = re.compile(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]")


@dataclass
class ViewTree:
    screenshot_id: str
    tree: object            # lxml _ElementTree — XPath is evaluated on the document
    filtered_xml: str       # what the model is shown
    width: int
    height: int
    node_count: int
    filtered_node_count: int


def _screen_size(root) -> tuple[int, int]:
    """`<hierarchy width height>` is the authority; the outermost bounds the fallback."""
    try:
        w, h = int(root.get("width") or 0), int(root.get("height") or 0)
    except ValueError:
        w = h = 0
    if w and h:
        return w, h
    for el in root.iter():
        m = _BOUNDS_RE.match(el.get("bounds") or "")
        if m:
            w = max(w, int(m.group(3))) # x2
            h = max(h, int(m.group(4))) # y2
    return w, h


class ViewTreeStore:
    """Parsed dumps by screenshot id, filtered once each and cached.

    Requests are issued screenshot-major (`_expand_pairs`), so the working set is a few trees at a time;
    the LRU bound is there so a 30,921-request run does not hold 841 parsed documents at once.
    Thread-safe because `--workers` threads share one store.
    """

    def __init__(self, xml_dir: str, capacity: int = 64):
        self.xml_dir = Path(xml_dir)
        self.capacity = capacity
        self._cache: "OrderedDict[str, Optional[ViewTree]]" = OrderedDict()
        self._lock = threading.Lock()
        self._parser = etree.XMLParser(huge_tree=True, recover=True)

    def path_for(self, screenshot_id: str) -> Path:
        return self.xml_dir / f"{screenshot_id}.xml"

    def get(self, screenshot_id: str) -> Optional[ViewTree]:
        with self._lock:
            if screenshot_id in self._cache:
                self._cache.move_to_end(screenshot_id)
                return self._cache[screenshot_id]
        vt = self._load(screenshot_id)
        with self._lock:
            self._cache[screenshot_id] = vt
            self._cache.move_to_end(screenshot_id)
            while len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
        return vt

    def _load(self, screenshot_id: str) -> Optional[ViewTree]:
        p = self.path_for(screenshot_id)
        if not p.exists():
            return None
        try:
            tree = etree.parse(str(p), self._parser)
        except (etree.XMLSyntaxError, OSError) as exc:
            logger.warning("Could not parse %s: %s", p, exc)
            return None
        root = tree.getroot()
        if root is None:
            return None
        filtered = filter_view_tree(root)
        w, h = _screen_size(root)
        return ViewTree(
            screenshot_id=screenshot_id,
            tree=tree,
            filtered_xml=serialize(filtered),
            width=w, height=h,
            node_count=sum(1 for _ in root.iter()),
            filtered_node_count=sum(1 for _ in filtered.iter()),
        )
