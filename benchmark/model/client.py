"""One request to the model, and finding out who actually answered.

Two behaviours worth knowing before changing anything:

- **A truncated reply is an error, not a partial answer.** `finish_reason == "length"`
  returns an explicit truncation error before the `reasoning_content` fallback runs. A
  reasoning model spends its output budget before answering, so with a small
  `--max-tokens` roughly half of Gemma 4 12B's calls returned empty content and the
  fallback then handed raw reasoning prose to the parser. That is what recorded a Gemma
  run at 16% — a number that measured the token cap, not the model.
- **The served model is probed, not assumed.** A single-model llama.cpp endpoint ignores
  the model name it is asked for, so the id that answered is the one that decides which
  coordinate convention applies and it is recorded in the result filename.
"""
import logging
import re
import threading
import time
from typing import Optional

import httpx

from benchmark.model.images import _load_image_b64
from benchmark.model.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from benchmark.scoring.coords import _norm_dims
from benchmark.scoring.parsing import _parse_response

logger = logging.getLogger(__name__)


def _list_loaded_models(client: httpx.Client, proxy_url: str, api_key: str) -> list:
    """Model ids the endpoint admits to having. Empty list if it won't say."""
    try:
        resp = client.get(f"{proxy_url.rstrip('/')}/v1/models",
                          headers=_auth_headers(api_key), timeout=15)
        if resp.status_code != 200:
            return []
        return [m["id"] for m in resp.json().get("data", []) if m.get("id")]
    except Exception as exc:
        logger.debug("Could not list models: %s", exc)
        return []


def _probe_served_model(client: httpx.Client, proxy_url: str, api_key: str,
                        model: str) -> str:
    """Ask for `model` and report which model actually answers.

    A single-model server (llama.cpp serving one .gguf) does not route by name: it
    answers every request with whatever is loaded and never says so. Asking for
    `gemma-4-12b` on a box running Qwen returns a perfectly normal Qwen answer, which
    is how one afternoon produced two result files with byte-identical predictions
    under two different model names. The response's own `model` field is the only
    thing that tells the truth, so read it before spending hours on the run.

    Against a hosted provider this is a formality — they route by name and echo it
    back — but it is still worth the one token, because it is also the cheapest possible
    check that the key and the model id are both right before a run that costs real
    money starts.
    """
    ping = {"model": model, "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1}
    try:
        resp = client.post(
            f"{proxy_url.rstrip('/')}/v1/chat/completions",
            json=ping,
            headers=_auth_headers(api_key), timeout=30,
        )
        if resp.status_code != 200:
            logger.debug("Probe returned HTTP %d: %s", resp.status_code, resp.text[:200])
            return ""
        return resp.json().get("model", "") or ""
    except Exception as exc:
        logger.debug("Could not probe served model: %s", exc)
        return ""

# How much of the model's own text to keep per element. **0 keeps all of it**, which is
# the default: `raw` is the only field that can tell a model's mistake from the parser's,
# and every coordinate-convention bug this project has hit was diagnosed by reading it —
# most recently GUI-Owl's 0-1000 grid, which was found by re-parsing `raw` out of a
# finished result file rather than by re-running 30,921 calls. Truncating it is
# truncating the evidence.
#
# The cost is bounded by `--max-tokens`, not by the model's mood: at the 2048 default a
# row can hold ~8 KB, so a pathological 30,921-row run is ~250 MB against the 27 MB it
# is today. In practice Qwen2.5-VL writes ~25 characters and GUI-Owl ~12. Set this to a
# positive number to cap it again if a chatty model ever makes a result file unusable.
RAW_RESPONSE_CHARS = 0


# Set once, for the whole process, the first time a server refuses `temperature`.
#
# The benchmark sends temperature=0 because its main targets are self-hosted llama.cpp and
# vLLM endpoints, which honour it — and greedy decoding is what makes a coordinate
# reproducible. llama.cpp's own default is 0.8, so *not* sending it would put sampling noise
# straight into digit tokens, where a flip turns x=432 into x=332.
#
# Hosted frontier models reject it outright: GPT-5.x answers "Unsupported value:
# 'temperature' does not support 0.0 with this model", and recent Claude models return 400
# for any non-default value. Since the README documents pointing this script at
# api.openai.com for a baseline, an unconditional temperature would fail every element of
# that run. Dropping it for the run instead costs reproducibility on that endpoint only,
# which is the lesser loss and is recorded in the result.
_TEMPERATURE_REJECTED = threading.Event()

# Matched against the response body, so a 400 about something else — a malformed image, a
# context overflow — is not silently treated as a temperature problem and retried blind.
_TEMPERATURE_ERR_RE = re.compile(r"temperature", re.I)


def _temperature_rejected(resp) -> bool:
    """True when this response is a refusal of the `temperature` parameter specifically."""
    if resp.status_code not in (400, 422):
        return False
    try:
        return bool(_TEMPERATURE_ERR_RE.search(resp.text))
    except Exception:
        return False


def _auth_headers(api_key: str) -> dict:
    """Both auth conventions, because the endpoints this runs against disagree.

    `Authorization: Bearer` is the OpenAI-compatible standard and is what vLLM
    (`--api-key`), llama.cpp (`--api-key`), Ollama, OpenAI and every hosted gateway
    read. `X-API-Key` is what some self-hosted proxies use instead. Sending both costs
    one header and means the same command works against any of them; a server that does
    not recognise one simply ignores it.

    An empty key sends neither, which is the common local case — llama.cpp, vLLM and
    Ollama started without an API key reject a request carrying `Bearer ` with nothing
    after it on some builds, and have nothing to check on the rest.
    """
    if not api_key:
        return {}
    return {"Authorization": f"Bearer {api_key}", "X-API-Key": api_key}


def _build_payload(model: str, b64: str, media: str, description: str,
                   max_tokens: int, thinking_budget: int, temperature: float,
                   send_temperature: bool) -> dict:
    """The request body."""
    user_text = USER_PROMPT_TEMPLATE.format(description=description)
    temp = ({"temperature": temperature} if send_temperature else {})

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                    {"type": "text", "text": user_text},
                ],
            },
        ],
        # Four floats need a handful of tokens; the budget is for models that reason
        # first. Measured on Gemma 4 12B: 428-512 output tokens before the array, so 512
        # truncated roughly half of all calls. Default 2048 covers that with headroom.
        "max_tokens": max_tokens,
        **({"budget_tokens": thinking_budget} if thinking_budget >= 0 else {}),
        **temp,
    }


def _response_meta(usage: dict, msg: dict) -> dict:
    """Per-element diagnostics: token counts and the model's own words.

    `reasoning_content` is preferred when `content` is empty, because that is the
    case worth looking at — a thinking model that spent its budget and never answered.

    `cached_input_tokens` is recorded because prompt caching makes `input_tokens` swing
    for reasons that have nothing to do with the model: the system prompt is identical on
    every element, and so is the *image* across every element of the same screenshot, so
    whether a given call pays for those depends on what the server happened to serve
    before it — which the worker interleaving decides. Measured on a llama.cpp endpoint,
    14 of 21 prompt tokens came back cached on a bare text call. Without this field the
    variation looks like noise; with it, it is explained.
    """
    raw = msg.get("content") or msg.get("reasoning_content") or ""
    text = " ".join(str(raw).split())
    details = usage.get("prompt_tokens_details") or {}
    return {
        "input_tokens": usage.get("prompt_tokens", 0) or 0,
        "output_tokens": usage.get("completion_tokens", 0) or 0,
        "cached_input_tokens": details.get("cached_tokens", 0) or 0,
        "raw": text[:RAW_RESPONSE_CHARS] if RAW_RESPONSE_CHARS > 0 else text,
    }

def _call_model(
    client: httpx.Client,
    proxy_url: str,
    api_key: str,
    model: str,
    image_path: str,
    description: str,
    timeout: int,
    coord_format: str = "corner",
    thinking_budget: int = -1,
    max_image_dim: int = 0,
    max_tokens: int = 2048,
    temperature: float = 0,
    coord_grid: int = 0,
) -> tuple[Optional[dict], Optional[dict], float, str, dict]:
    """
    Returns (pred_bbox or None, pred_point or None, latency_ms, error, meta).
    pred_bbox  = {x, y, width, height} normalized 0-1
    pred_point = {cx, cy} normalized 0-1 (click-point models like GUI-Owl)
    meta       = {"input_tokens", "output_tokens", "raw", "img_w", "img_h"} — a dict
                 rather than five more positional values, so the next thing worth
                 recording does not change this signature again.

    `img_w`/`img_h` are in `meta` on **every** path, including the failures that never
    reach a response: they are known as soon as the image is loaded, and a row that
    records the dimensions its answer was normalised against is a row that can be
    re-scored later from `raw` alone. Without them a rescore has to re-derive them from
    the image file, which only works while the file is still on disk and silently
    gives the wrong answer if the run used `--max-image-dim`.
    """
    b64, media, img_w, img_h = _load_image_b64(image_path, max_image_dim)
    # Repeated into every return below rather than merged once at the end, because most
    # of those returns are early exits.
    dims = {"img_w": img_w, "img_h": img_h}

    payload = _build_payload(
        model, b64, media, description, max_tokens, thinking_budget, temperature,
        # Sent only where it is accepted — see _TEMPERATURE_REJECTED.
        send_temperature=(temperature >= 0 and not _TEMPERATURE_REJECTED.is_set()),
    )
    url = f"{proxy_url.rstrip('/')}/v1/chat/completions"
    headers = _auth_headers(api_key)

    t0 = time.monotonic()
    last_err = ""

    def _post():
        return client.post(url, json=payload, headers=headers, timeout=timeout)

    for attempt in range(3):
        if attempt > 0:
            time.sleep(2 ** attempt)  # 2s, 4s back-off
        try:
            resp = _post()
            # A server that rejects `temperature` rejects it on every call, so drop it for
            # the whole run rather than burning a wasted round-trip per element. Retried
            # inline rather than by `continue`, which would spend one of the three
            # connection attempts and report "Connection failed" if this were the last.
            if "temperature" in payload and _temperature_rejected(resp):
                logger.warning(
                    "Endpoint rejected temperature=%s (HTTP %d): %s. Dropping it for the "
                    "rest of the run and letting the model use its own sampling default. "
                    "Coordinates are digit tokens, so a non-zero default can move a "
                    "prediction between runs — see --temperature.",
                    payload["temperature"], resp.status_code, resp.text[:160])
                _TEMPERATURE_REJECTED.set()
                payload.pop("temperature")
                resp = _post()
            break
        except (httpx.ConnectError, httpx.RemoteProtocolError) as e:
            last_err = str(e)
            logger.debug("Attempt %d failed: %s", attempt + 1, e)
            continue
        except httpx.TimeoutException:
            return None, None, (time.monotonic() - t0) * 1000, "timeout", dict(dims)
        except Exception as e:
            return None, None, (time.monotonic() - t0) * 1000, str(e), dict(dims)
    else:
        return None, None, (time.monotonic() - t0) * 1000, f"Connection failed after 3 attempts: {last_err}", dict(dims)

    latency_ms = (time.monotonic() - t0) * 1000

    try:
        if resp.status_code != 200:
            return None, None, latency_ms, f"HTTP {resp.status_code}: {resp.text[:200]}", dict(dims)

        data = resp.json()
        choice = data["choices"][0]
        finish = choice.get("finish_reason", "unknown")
        msg = choice["message"]
        usage = data.get("usage") or {}

        # A truncated answer is a failure, not a prediction — check before falling back
        # to reasoning_content. A thinking model spends its whole budget reasoning and
        # returns content="" with finish_reason="length"; the fallback then handed the
        # raw reasoning prose to _parse_response, which scraped the "1." and "2." of a
        # numbered list into bbox {x: 1.0, y: 0.002, w: 0.004, h: 0.002} and reported no
        # error. That one bogus box, repeated, is what a Gemma 4 12B run recorded as 68%
        # zero-overlap and 16% centroid accuracy. Silently manufacturing a wrong number
        # is worse than reporting the failure.
        meta = _response_meta(usage, msg)
        meta.update(dims)

        if finish == "length":
            return None, None, latency_ms, (
                f"Truncated at max_tokens={max_tokens} "
                f"(completion_tokens={meta['output_tokens']}): the answer never arrived. "
                f"Thinking models need a larger cap — raise --max-tokens."
            ), meta

        content = msg.get("content") or msg.get("reasoning_content", "")
        if not content:
            return None, None, latency_ms, (
                f"Empty content (finish_reason={finish}, "
                f"prompt_tokens={meta['input_tokens']}). "
                "Image may not be reaching the model."
            ), meta

        norm_w, norm_h = _norm_dims(img_w, img_h, coord_grid)
        pred_bbox, pred_point, parse_err = _parse_response(content, norm_w, norm_h)
        if parse_err:
            return None, None, latency_ms, parse_err, meta

        # If the model returns (x,y) as the CENTER (not top-left), convert to corner.
        if coord_format == "center" and pred_bbox:
            cx = pred_bbox["x"]
            cy = pred_bbox["y"]
            w  = pred_bbox["width"]
            h  = pred_bbox["height"]
            xl = max(0.0, cx - w / 2)
            yt = max(0.0, cy - h / 2)
            pred_bbox = {
                "x": xl, "y": yt,
                "width":  min(1.0, cx + w / 2) - xl,
                "height": min(1.0, cy + h / 2) - yt,
            }

        return pred_bbox, pred_point, latency_ms, "", meta

    except httpx.TimeoutException:
        return None, None, (time.monotonic() - t0) * 1000, "timeout", dict(dims)
    except Exception as e:
        return None, None, (time.monotonic() - t0) * 1000, str(e), dict(dims)
