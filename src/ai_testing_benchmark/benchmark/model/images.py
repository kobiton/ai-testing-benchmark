"""Getting the screenshot into the request.

**`--max-image-dim` defaults to 0, i.e. the original resolution**, and no benchmark
run should change that: a comparison across models is only meaningful when every
model saw the screenshot the consumer will send it. The flag exists to reproduce a
consumer that downscales before sending, which is a different question.

When it *does* resize, the returned dimensions must be the resized ones — pixel
answers are normalised against them, so returning the originals silently rescales
every prediction.
"""
import base64
import io
import struct
from pathlib import Path

from PIL import Image


def _load_image_b64(path: str, max_dim: int = 0) -> tuple[str, str, int, int]:
    """Return (base64_data, media_type, width_px, height_px).

    `max_dim` downscales the longest side first; 0 — the default — sends the original.

    Downscaling to 1080 used to be the default, to match a consumer that downscales
    before sending. That was the wrong trade for this harness: the benchmark exists to
    measure how well a model can ground an element, and shrinking the screenshot first
    only handicaps it. Measuring a specific consumer's preprocessing is a different
    question, and `--max-image-dim 1080` still answers it.

    The number is not free. A screenshot in this corpus is 1080x2400, which a patch-based
    vision model turns into ~3300 image tokens against ~900 at max-dim 1080 — enough to
    overflow a 4096-token context before the prompt is counted, which is what "the request
    exceeds the available context size" means on a llama.cpp server. Native also multiplies
    input tokens per element by roughly four, so results either side of this default change
    are not comparable on cost; `max_image_dim` is recorded in every result for that reason.

    Predictions and ground truth are both compared in normalised coordinates, so the
    resolution the model saw does not change what is measured — but the *returned*
    dimensions must be the resized ones, since that is what pixel answers get normalised
    against.
    """
    if max_dim:
        with Image.open(path) as im:
            w, h = im.size
            if max(w, h) > max_dim:
                scale = max_dim / max(w, h)
                w, h = round(w * scale), round(h * scale)
                buf = io.BytesIO()
                im.convert("RGB").resize((w, h), Image.LANCZOS).save(
                    buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode(), "image/jpeg", w, h

    suffix = Path(path).suffix.lower()
    media = "image/png" if suffix == ".png" else "image/jpeg"
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.b64encode(raw).decode()
    w, h = 0, 0
    try:
        if suffix == ".png":
            w, h = struct.unpack(">II", raw[16:24])
        else:
            i = 2
            while i < len(raw) - 9:
                if raw[i] != 0xFF:
                    break
                marker = raw[i + 1]
                length = struct.unpack(">H", raw[i + 2:i + 4])[0]
                if marker in (0xC0, 0xC1, 0xC2):
                    h, w = struct.unpack(">HH", raw[i + 5:i + 9])
                    break
                i += 2 + length
    except Exception:
        pass
    return b64, media, w, h
