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
import threading
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

# §5.4 — thread-safe LLM call counter (cost meter)
_counter_lock = threading.Lock()
_call_counts: dict[str, int] = {}    # model → call count
_token_counts: dict[str, int] = {}   # model → input tokens (Anthropic only)
_output_token_counts: dict[str, int] = {}  # model → output tokens (Anthropic only)


def _record_call(model: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
    with _counter_lock:
        _call_counts[model] = _call_counts.get(model, 0) + 1
        _token_counts[model] = _token_counts.get(model, 0) + input_tokens
        _output_token_counts[model] = _output_token_counts.get(model, 0) + output_tokens


def get_llm_stats() -> dict[str, Any]:
    """Return cumulative LLM call counts since process start."""
    with _counter_lock:
        return {
            "calls_by_model": dict(_call_counts),
            "input_tokens_by_model": dict(_token_counts),
            "output_tokens_by_model": dict(_output_token_counts),
            "total_calls": sum(_call_counts.values()),
        }


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
        _record_call(model)
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
    _record_call(model, input_tokens=msg.usage.input_tokens,
                 output_tokens=msg.usage.output_tokens)
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


# ── §6.2 LLMRouter — role-based dispatch ────────────────────────────────────

# §5.6 producer provenance — bump when distillation prompts / logic change so a
# future model can tell which generation of the pipeline wrote a fragment.
PRODUCER_VERSION = "1"


class LLMRouter:
    """
    Dispatches LLM calls by role (cheap / medium / strong) rather than by
    hard-coded model strings per pipeline stage.  Config supplies one model
    per role; the router picks the right one and falls through if unset.

    Usage:
        router = LLMRouter(cfg.llm_roles)
        result = router.call("cheap", prompt)
        result = router.call_json("medium", prompt)
    """

    # Fallback chain when a role is not configured
    _FALLBACK: dict[str, str] = {
        "cheap":  _DEFAULT_MODEL,
        "medium": "claude-sonnet-4-6",
        "strong": "claude-opus-4-7",
    }

    def __init__(self, roles_cfg: "Any | None" = None):
        self._roles_cfg = roles_cfg  # LLMRolesConfig or None

    def _resolve(self, role: str) -> tuple[str, str | None]:
        """Return (model, endpoint) for the given role."""
        if self._roles_cfg is not None:
            model = getattr(self._roles_cfg, f"{role}_model", None)
            endpoint = getattr(self._roles_cfg, f"{role}_endpoint", None)
            if model:
                return model, endpoint
        return self._FALLBACK.get(role, _DEFAULT_MODEL), None

    def model_for(self, role: str) -> str:
        """Public: the model string call()/call_json() would use for this role.
        Used to stamp producer_model on fragments this router produced."""
        return self._resolve(role)[0]

    def call(self, role: str, prompt: str, system: str = "", max_tokens: int = _MAX_TOKENS) -> str:
        model, endpoint = self._resolve(role)
        return call_model(prompt, system=system, model=model,
                          max_tokens=max_tokens, endpoint=endpoint)

    def call_json(self, role: str, prompt: str, system: str = "", max_tokens: int = _MAX_TOKENS) -> dict[str, Any]:
        model, endpoint = self._resolve(role)
        return call_model_json(prompt, system=system, model=model,
                               max_tokens=max_tokens, endpoint=endpoint)
