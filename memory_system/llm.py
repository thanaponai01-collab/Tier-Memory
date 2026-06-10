"""
Thin Anthropic SDK wrapper for pipeline distillation calls.

All distillation (Stages 1-3) uses a cheap model (Haiku by default).
The frontier model is reserved for the user-facing turn.

call_model() returns the raw text response.
call_model_json() parses and returns a dict; raises ValueError on bad JSON.
"""

from __future__ import annotations
import json
import os
import re
import subprocess
import threading
import time
import urllib.request
from typing import Any, Callable, Optional

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


# ── Budget fallback ─────────────────────────────────────────────────────────
# When a paid (Anthropic) call fails because the API credit is exhausted — or
# the key is otherwise unusable — keep the pipeline working by degrading to a
# local model instead of letting distillation/re-synthesis fail silently. The
# fallback "latches" for a cooldown so we don't re-hit (and re-fail on) the dead
# paid API for every call; after the cooldown we probe it again in case credit
# was topped up. Configure via LLMRolesConfig.budget_fallback_model.
# The preferred fallback target. "claude-code" routes LLM work through the
# Claude Code CLI (`claude -p`), which uses the user's *subscription* (OAuth) —
# free at point of use and full quality — so the flywheel keeps IMPROVING memory
# when the metered API credit is gone. Any "ollama/..." value routes to a local
# model instead. If "claude-code" is unreachable (CLI not installed, or the
# runaway circuit-breaker trips), it degrades to _BUDGET_FALLBACK_LOCAL_MODEL.
_BUDGET_FALLBACK_MODEL = "claude-code"
_BUDGET_FALLBACK_ENDPOINT: Optional[str] = "http://localhost:11434"
_BUDGET_FALLBACK_LOCAL_MODEL = "ollama/qwen3:8b"  # last-resort local
_BUDGET_COOLDOWN_SECS = 1800  # 30 min before re-probing the paid API
_budget_lock = threading.Lock()
_budget_exhausted_until = 0.0           # monotonic deadline; >now ⇒ use fallback
_on_budget_fallback: Optional[Callable[[str, str], None]] = None  # (model, detail)
# Fired when the paid API rejects a call with a persistent error we did NOT
# recognize as out-of-credit (e.g. a reworded billing 400) — the "can't be sure"
# alarm. Throttled by _paid_error_alarm_until so a tight failing loop alarms once.
_on_paid_error: Optional[Callable[[str, str], None]] = None       # (model, detail)
_paid_error_alarm_until = 0.0

# Circuit-breaker for the CLI path. The hook guard (TIER_MEMORY_NO_INGEST) should
# already stop the spawn→ingest→spawn loop, but this caps the blast radius if it
# ever fails: too many CLI calls in the window ⇒ degrade to local until it cools.
_CLI_WINDOW_SECS = 300
_CLI_MAX_IN_WINDOW = 25
_cli_call_times: list[float] = []


def set_budget_fallback(
    model: Optional[str],
    endpoint: Optional[str] = None,
    local_model: Optional[str] = None,
) -> None:
    """Configure the budget fallback. Called once at daemon start.
    `model` is the preferred target ("claude-code" or "ollama/..."), `local_model`
    is the last-resort local model used if "claude-code" is unreachable."""
    global _BUDGET_FALLBACK_MODEL, _BUDGET_FALLBACK_ENDPOINT, _BUDGET_FALLBACK_LOCAL_MODEL
    if model:
        _BUDGET_FALLBACK_MODEL = model
        _BUDGET_FALLBACK_ENDPOINT = endpoint or _BUDGET_FALLBACK_ENDPOINT
    if local_model:
        _BUDGET_FALLBACK_LOCAL_MODEL = local_model


def register_budget_fallback_hook(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register a callback fired once when the paid API first trips the fallback
    (used to surface it on the Issue Catcher). Decoupled so llm.py needn't import
    the daemon."""
    global _on_budget_fallback
    _on_budget_fallback = fn


def register_paid_error_hook(fn: Optional[Callable[[str, str], None]]) -> None:
    """Register a callback for the 'can't be sure' alarm: the paid API rejected a
    call with a persistent error we did NOT classify as out-of-credit. Surfaces
    on the Issue Catcher so a reworded billing message can't fail silently."""
    global _on_paid_error
    _on_paid_error = fn


def _alarm_paid_error(model: str, err: Exception) -> None:
    """Fire the 'can't be sure' alarm at most once per cooldown window."""
    global _paid_error_alarm_until
    with _budget_lock:
        if time.monotonic() < _paid_error_alarm_until:
            return
        _paid_error_alarm_until = time.monotonic() + _BUDGET_COOLDOWN_SECS
    if _on_paid_error is not None:
        status = _http_status(err)
        detail = f"HTTP {status}: {str(getattr(err, 'message', '') or err)[:200]}"
        try:
            _on_paid_error(model, detail)
        except Exception:
            pass  # the alarm must never break the caller's own error handling


def budget_fallback_active() -> bool:
    """True while LLM work is degraded to the local fallback model."""
    with _budget_lock:
        return time.monotonic() < _budget_exhausted_until


# Billing/out-of-credit signals. A real out-of-credit error is a 400 whose body
# names the credit balance — Anthropic does NOT (today) give it a dedicated
# error .type, so recognition can't hang on one exact phrase or it would
# silently stop opening the parachute the day they reword the message. We gather
# every text source the SDK exposes (top-level .message, str(e), and the parsed
# .body dict's type+message) and match any of several billing phrasings. This
# phrase list is the LAST resort: status code (402) and exception type catch the
# unambiguous cases first, so the list only has to cover today's 400 wording —
# and a reworded 400 we miss still surfaces via the uncertainty alarm below.
_BILLING_SIGNALS = (
    "billing_error", "credit balance", "insufficient credit", "out of credit",
    "purchase credits", "buy credits", "add credits", "too low to access",
    "billing", "payment required", "payment method", "spending limit",
    "upgrade your plan", "plan and billing", "quota exceeded",
    "usage limits", "api usage",  # Anthropic: "You have reached your specified API usage limits"
)


def _http_status(e: Exception) -> Optional[int]:
    """Best-effort HTTP status for an SDK/HTTP exception. Checks the common
    attributes the Anthropic SDK / httpx expose. None if there is no status."""
    for attr in ("status_code", "status"):
        v = getattr(e, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(e, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _is_transient_error(e: Exception) -> bool:
    """True for failures that clear on their own — rate limits (429) and server
    errors (5xx). These must NEVER degrade us to local: the paid API is still the
    right place to retry. Checked by exception TYPE and status so it is robust to
    message rewording (and so broadening _BILLING_SIGNALS can't cause a false
    positive on a 429 whose prose happens to mention a 'limit')."""
    if _HAS_ANTHROPIC:
        for name in ("RateLimitError", "InternalServerError", "APIConnectionError",
                     "APITimeoutError"):
            if isinstance(e, getattr(_anthropic, name, ())):
                return True
    status = _http_status(e)
    return status == 429 or (isinstance(status, int) and status >= 500)


def _is_budget_or_auth_error(e: Exception) -> bool:
    """Distinguish 'paid API unusable' (out of credit / bad key) from transient
    failures (rate limit, 5xx) that should NOT trigger a local downgrade.

    Recognition is layered so it does not hang on one phrase: (1) transient
    errors are excluded first by TYPE/status; (2) an unusable key (401/403) and a
    402 Payment Required are caught by type/status regardless of wording — 402 is
    the semantically-correct billing status, so this survives Anthropic moving
    out-of-credit off the current 400; (3) only then do we fall back to matching
    billing PHRASES, which today's out-of-credit 400 still needs."""
    if _is_transient_error(e):
        return False
    if _HAS_ANTHROPIC:
        for name in ("AuthenticationError", "PermissionDeniedError"):
            if isinstance(e, getattr(_anthropic, name, ())):
                return True  # 401/403 — key missing/invalid/revoked: API unusable
    status = _http_status(e)
    if status == 402:
        return True  # Payment Required — unambiguous billing wall, any wording
    parts = [
        str(getattr(e, "type", "") or ""),
        str(getattr(e, "message", "") or ""),
        str(e),
    ]
    body = getattr(e, "body", None)
    if isinstance(body, dict):
        parts.append(str(body.get("type", "") or ""))
        parts.append(str(body.get("message", "") or ""))
    haystack = " ".join(parts).lower()
    return any(sig in haystack for sig in _BILLING_SIGNALS)


def _is_unrecognized_paid_rejection(e: Exception) -> bool:
    """True when the paid API rejected the call with a persistent client error
    (a non-transient 4xx) that we did NOT classify as budget/auth. This is the
    'can't be sure' case — most importantly a reworded out-of-credit 400 that
    slipped past _BILLING_SIGNALS. The caller surfaces it on the Issue Catcher so
    a wording change becomes VISIBLE instead of silently stalling the pipeline."""
    if _is_transient_error(e) or _is_budget_or_auth_error(e):
        return False
    status = _http_status(e)
    return isinstance(status, int) and 400 <= status < 500


def _latch_budget(failed_model: str, err: Exception) -> None:
    global _budget_exhausted_until
    with _budget_lock:
        first = time.monotonic() >= _budget_exhausted_until
        _budget_exhausted_until = time.monotonic() + _BUDGET_COOLDOWN_SECS
    if first and _on_budget_fallback is not None:
        try:
            _on_budget_fallback(failed_model, str(getattr(err, "message", "") or err)[:200])
        except Exception:
            pass  # the alarm must never break the fallback


def _cli_circuit_open() -> bool:
    """True if the CLI fallback has fired too often recently (runaway guard)."""
    with _budget_lock:
        cutoff = time.monotonic() - _CLI_WINDOW_SECS
        recent = [t for t in _cli_call_times if t >= cutoff]
        _cli_call_times[:] = recent
        return len(recent) >= _CLI_MAX_IN_WINDOW


def _note_cli_call() -> None:
    with _budget_lock:
        _cli_call_times.append(time.monotonic())


def _call_claude_cli(prompt: str, system: str, max_tokens: int, timeout: int = 180) -> str:
    """Run one LLM call through the Claude Code CLI on the user's subscription.

    Two things make this safe to call from inside the daemon:
      - ANTHROPIC_API_KEY is stripped from the child env, so the CLI authenticates
        with the OAuth subscription (not the out-of-credit metered key it inherited).
      - TIER_MEMORY_NO_INGEST is set, so this helper session's Stop/UserPromptSubmit
        hooks no-op and it is never ingested → no spawn→ingest→spawn recursion.
    Raises FileNotFoundError if the `claude` CLI isn't on PATH (caller degrades)."""
    env = dict(os.environ)
    env.pop("ANTHROPIC_API_KEY", None)
    env["TIER_MEMORY_NO_INGEST"] = "1"

    cmd = ["claude", "-p", prompt, "--output-format", "json", "--no-session-persistence"]
    if system:
        cmd += ["--system-prompt", system]

    _note_cli_call()
    result = subprocess.run(
        cmd, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, env=env,
    )
    out = (result.stdout or "").strip()
    if not out:
        return ""
    try:
        envelope = json.loads(out)
    except json.JSONDecodeError:
        return out  # not the expected envelope — return raw, best effort
    usage = envelope.get("usage") or {}
    _record_call(
        "claude-code",
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
    )
    text = envelope.get("result", "")
    return text.strip() if isinstance(text, str) else ""


def _local_fallback_call(prompt: str, system: str, max_tokens: int, json_mode: bool,
                         model: Optional[str] = None) -> str:
    model = model or _BUDGET_FALLBACK_LOCAL_MODEL
    if not model.startswith("ollama/"):
        # Only local models are a safe, free last resort. Refuse to silently
        # route the overflow to another paid model.
        raise RuntimeError(f"local fallback model must be ollama/..., got {model!r}")
    _record_call(model)
    local = model[len("ollama/"):]
    base = _BUDGET_FALLBACK_ENDPOINT or "http://localhost:11434"
    return _call_ollama(prompt, system, local, base, max_tokens, json_mode)


def _budget_fallback_call(prompt: str, system: str, max_tokens: int, json_mode: bool) -> str:
    model = _BUDGET_FALLBACK_MODEL
    if model == "claude-code":
        # Prefer the subscription via the CLI; degrade to local if it's
        # unreachable or the runaway circuit-breaker has tripped.
        if not _cli_circuit_open():
            try:
                out = _call_claude_cli(prompt, system, max_tokens)
                if out.strip():
                    return out
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass  # CLI missing/slow — fall through to local
        return _local_fallback_call(prompt, system, max_tokens, json_mode)
    # Configured directly at a local model.
    return _local_fallback_call(prompt, system, max_tokens, json_mode, model=model)


def call_model(
    prompt: str,
    system: str = "You are a precise extraction assistant. Follow instructions exactly.",
    model: str = _DEFAULT_MODEL,
    max_tokens: int = _MAX_TOKENS,
    endpoint: str | None = None,
    _json_mode: bool = False,
) -> str:
    """
    Single-turn call. Routes based on model string:
    - 'ollama/...' → local Ollama
    - 'claude-code' → Claude Code CLI (user's subscription, no metered API)
    - anything else → Anthropic SDK (paid)
    """
    if model.startswith("ollama/"):
        local_model = model[len("ollama/"):]
        base = endpoint or "http://localhost:11434"
        _record_call(model)
        return _call_ollama(prompt, system, local_model, base, max_tokens, _json_mode)

    if model == "claude-code":
        # Route through the Claude Code CLI on the user's subscription —
        # no metered API cost. Falls back to local if CLI is missing or the
        # runaway circuit-breaker has tripped.
        if not _cli_circuit_open():
            try:
                out = _call_claude_cli(prompt, system, max_tokens)
                if out.strip():
                    return out
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass
        return _local_fallback_call(prompt, system, max_tokens, _json_mode)

    # Paid (Anthropic) path. If a prior call already found the budget exhausted,
    # skip straight to the local fallback until the cooldown elapses — no point
    # re-hitting a dead API on every call.
    if budget_fallback_active():
        return _budget_fallback_call(prompt, system, max_tokens, _json_mode)

    if not _HAS_ANTHROPIC:
        raise ImportError("pip install anthropic  to use the distillation pipeline")

    try:
        return _call_anthropic(model, prompt, system, max_tokens)
    except Exception as e:
        if _is_budget_or_auth_error(e):
            _latch_budget(model, e)
            return _budget_fallback_call(prompt, system, max_tokens, _json_mode)
        # The paid API rejected the call with a persistent error we couldn't pin
        # on out-of-credit. Could be a reworded billing message we failed to
        # match — surface it so it doesn't stall the pipeline silently — then
        # re-raise (transient errors still bubble up untouched for normal retry).
        if _is_unrecognized_paid_rejection(e):
            _alarm_paid_error(model, e)
        raise


def _call_anthropic(model: str, prompt: str, system: str, max_tokens: int) -> str:
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
PRODUCER_VERSION = "2"   # v2: distillation emits a crisp standalone "recall" headline (answer-shaped fragments)


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
