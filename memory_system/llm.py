"""
Thin Anthropic SDK wrapper for pipeline distillation calls.

All distillation (Stages 1-3) uses a cheap model (Haiku by default).
The frontier model is reserved for the user-facing turn.

call_model() returns the raw text response.
call_model_json() parses and returns a dict; raises ValueError on bad JSON.
"""

from __future__ import annotations
import json
import re
import urllib.request
from typing import Any

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

# Default cheap model for distillation — matches config.compression.distillation_model
_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 1024


def call_model(
    prompt: str,
    system: str = "You are a precise extraction assistant. Follow instructions exactly.",
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _MAX_TOKENS,
    endpoint: str | None = None,
    _json_mode: bool = False,
) -> str:
    """
    Single-turn call. Routes to Ollama when model starts with 'ollama/';
    otherwise uses the Anthropic SDK.
    """
    if model.startswith("ollama/"):
        local_model = model[len("ollama/"):]
        base = endpoint or "http://localhost:11434"
        return _call_ollama(prompt, system, local_model, base, max_tokens, _json_mode)

    if not _HAS_ANTHROPIC:
        raise ImportError("pip install anthropic  to use the distillation pipeline")

    client = _anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def call_model_json(
    prompt: str,
    system: str = "You are a precise extraction assistant. Always respond with valid JSON only.",
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _MAX_TOKENS,
    endpoint: str | None = None,
) -> dict[str, Any]:
    """
    Like call_model() but extracts and parses the first JSON object/array.
    Raises ValueError if no valid JSON is found in the response.
    """
    raw = call_model(prompt, system=system, model=model, max_tokens=max_tokens,
                     endpoint=endpoint, _json_mode=True)
    return _extract_json(raw)


def _call_ollama(
    prompt: str,
    system: str,
    model: str,
    endpoint: str,
    max_tokens: int,
    json_mode: bool,
) -> str:
    body: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"num_predict": max_tokens},
        # thinking: false disables <think> blocks for models that support it (qwen3, etc.)
        # This is the safe default for structured extraction calls.
        "think": False,
    }
    # Do NOT use Ollama's json format mode — it silences output on thinking models.
    # Instead rely on _extract_json's regex-based extraction.

    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{endpoint}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        result = json.loads(resp.read())
    return result["message"]["content"]


def _extract_json(text: str) -> dict[str, Any]:
    # Strip <think>...</think> blocks (thinking models like qwen3)
    text = re.sub(r"<think>[\s\S]*?</think>", "", text).strip()
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    # Try the whole string first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Find the first {...} or [...] block
    for pattern in (r"\{[\s\S]*\}", r"\[[\s\S]*\]"):
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No valid JSON found in model response:\n{text[:300]}")
