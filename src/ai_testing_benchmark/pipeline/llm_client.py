"""LLM client supporting OpenAI (GPT) and Anthropic (Claude) with retry logic."""
import base64
import logging
import re
import threading
from pathlib import Path
from typing import Optional

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .config import config

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def extract_json(text: str) -> str:
    """Strip markdown code fences if present; return raw JSON string."""
    m = _JSON_FENCE_RE.search(text)
    return m.group(1) if m else text.strip()


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return "image/png" if suffix == ".png" else "image/jpeg"


# ---------------------------------------------------------------------------
# Shared clients
# ---------------------------------------------------------------------------

# One client per provider for the whole process. Building one per call is what
# killed the first full 841-screenshot run: a client owns an httpx pool whose
# sockets outlive it (the SDK finalizer only runs at GC), and at PIPELINE_WORKERS
# threads the discard rate outruns the collector until the process hits its
# descriptor ceiling — 256 by default on macOS (`launchctl limit maxfiles`). Measured
# under that cap: per-call peaks at 245 descriptors, shared holds 28 no matter how long
# the run. 382 of 841 screenshots failed as a mix of `[Errno 24] Too many open files`
# and bare `Connection error.`, which look like two faults but are one exhausted
# descriptor table. Both SDKs document their clients as thread-safe.
_clients: dict = {}
_clients_lock = threading.Lock()


def _shared_client(provider: str):
    with _clients_lock:
        if provider not in _clients:
            if provider == "claude":
                import anthropic
                _clients[provider] = anthropic.Anthropic(api_key=config.anthropic_api_key)
            else:
                from openai import OpenAI
                _clients[provider] = OpenAI(api_key=config.openai_api_key)
        return _clients[provider]


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------


def _call_openai(system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
    from openai import RateLimitError, APIConnectionError

    @retry(
        # APIConnectionError covers APITimeoutError, which subclasses it, and also
        # the plain connection drop that used to abandon a screenshot outright.
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        before_sleep=lambda rs: logger.warning(
            "OpenAI call failed (%s), retrying (attempt %d)…",
            type(rs.outcome.exception()).__name__, rs.attempt_number),
    )
    def _inner():
        client = _shared_client("openai")
        messages = [{"role": "system", "content": system_prompt}]

        if image_path:
            b64 = _encode_image(image_path)
            media = _image_media_type(image_path)
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                    {"type": "text", "text": user_prompt},
                ],
            })
        else:
            messages.append({"role": "user", "content": user_prompt})

        model = config.openai_model
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        # Same reasoning as the Claude branch: a truncated answer would surface as a
        # JSON parse error rather than as the length limit it is.
        if response.choices[0].finish_reason == "length":
            raise RuntimeError(
                f"{model} stopped at the output length limit: the "
                f"response is truncated and will not parse.")
        return response.choices[0].message.content

    return _inner()


# ---------------------------------------------------------------------------
# Anthropic / Claude backend
# ---------------------------------------------------------------------------

def _call_claude(system_prompt: str, user_prompt: str, image_path: Optional[str]) -> str:
    import anthropic

    @retry(
        # APIConnectionError covers APITimeoutError, which subclasses it, and also
        # the plain connection drop that used to abandon a screenshot outright.
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(5),
        before_sleep=lambda rs: logger.warning(
            "Claude call failed (%s), retrying (attempt %d)…",
            type(rs.outcome.exception()).__name__, rs.attempt_number),
    )
    def _inner():
        client = _shared_client("claude")

        if image_path:
            b64 = _encode_image(image_path)
            media = _image_media_type(image_path)
            user_content = [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media, "data": b64},
                },
                {"type": "text", "text": user_prompt},
            ]
        else:
            user_content = user_prompt

        model = config.anthropic_model
        response = client.messages.create(
            model=model,
            # An output cap, not a context limit. Step 3 answers for every element on
            # a screenshot in one response at ~48 output tokens each, so 4096 ran out
            # at roughly 84 elements — the densest pilot screenshot had 33. 8192 keeps
            # the headroom well clear without inflating the value the output-token
            # rate limit is estimated against.
            max_tokens=8192,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
        )

        # Truncated output is invalid JSON, and every caller reports that as a parse
        # failure — indistinguishable from a genuinely malformed answer, and step 3
        # then returns a record with no descriptions at all, which the benchmark
        # silently skips. Say so instead of letting it look like a bad response.
        if response.stop_reason == "max_tokens":
            raise RuntimeError(
                f"{model} hit max_tokens=8192: the response is "
                f"truncated and will not parse. A screenshot with very many elements "
                f"can do this in step 3, raise max_tokens in llm_client.py.")
        return response.content[0].text

    return _inner()


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def call_llm(system_prompt: str, user_prompt: str, image_path: Optional[str] = None) -> str:
    """
    Call the configured LLM (OpenAI or Claude) with an optional image.
    Provider is selected by the LLM_PROVIDER env var ("openai" or "claude").
    Returns the raw text response (expected to be JSON).
    """
    provider = config.llm_provider.lower()
    if provider == "claude":
        logger.debug("Using Claude (%s)", config.anthropic_model)
        return _call_claude(system_prompt, user_prompt, image_path)
    else:
        logger.debug("Using OpenAI (%s)", config.openai_model)
        return _call_openai(system_prompt, user_prompt, image_path)
