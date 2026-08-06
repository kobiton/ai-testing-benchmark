"""Shared data models for the dataset pipeline."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BoundingBox:
    x: float       # normalized [0, 1] relative to image width
    y: float       # normalized [0, 1] relative to image height
    width: float   # normalized [0, 1]
    height: float  # normalized [0, 1]


@dataclass
class Element:
    element_id: str
    name: str
    type: str                       # button, text_field, link, icon, checkbox, ...
    bbox: Optional[BoundingBox] = None
    descriptions: list[str] = field(default_factory=list)


@dataclass
class ScreenshotRecord:
    """One fully-labeled screenshot, written as one JSONL line per element."""
    screenshot_id: str
    s3_key: str                     # where the screenshot was read from, for this run only
    image_path: str                 # local path used during processing
    elements: list[Element] = field(default_factory=list)

    def to_jsonl_rows(self) -> list[dict]:
        """Expand into one dict per element for JSONL output.

        **`s3_key` is deliberately not among them.** It records where this particular run happened to read the screenshot
        from, which is a fact about the machine that labelled it rather than about the element, a storage layout nobody
        downstream can resolve, repeated on every row.
        `screenshot_id` is the join key everything actually uses, and the benchmark resolves images as `<images-dir>/<screenshot_id>.png`.
        It stays on the dataclass because the pipeline's own checkpoint round-trips it.
        """
        rows = []
        for el in self.elements:
            bbox_dict = None
            if el.bbox:
                bbox_dict = {
                    "x": el.bbox.x,
                    "y": el.bbox.y,
                    "width": el.bbox.width,
                    "height": el.bbox.height,
                }
            rows.append({
                "screenshot_id": self.screenshot_id,
                "element_id": el.element_id,
                "name": el.name,
                "type": el.type,
                "bbox": bbox_dict,
                "descriptions": el.descriptions,
            })
        return rows
