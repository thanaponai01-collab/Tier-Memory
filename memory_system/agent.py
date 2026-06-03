"""
memory_system.agent — LLM inference loop with memory retrieval, tool use,
reflection, and transcript ingestion.

One full agent turn:
  retrieve → prompt → stream → tools (up to 10 rounds) → reflect → ingest

Step-8 additions vs Step-7:
  - Git dirty-file diff injected as L3 file_contents on first turn.
  - Global user profile loaded from ~/.agent/user_profile.md for L1.
  - Session summary derived from reflection and written back via ingest.
  - Token usage accumulated per turn and persisted to the sessions table.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

import anthropic

from memory_system.daemon import MemoryClientError, get_client
from memory_system.project import git_dirty_files, load_global_profile
from memory_system.prompt import SYSTEM_PROMPT, assemble
from memory_system.tools import TOOL_DEFINITIONS, execute

_MAX_TOOL_ROUNDS = 10
_TOOL_RESULT_MAX = 8_000

_REFLECTION_PROMPT = """\
Before this session ends, reflect on what you learned:
{
  "new_facts": ["..."],
  "corrected_assumptions": [{"was": "...", "now": "..."}],
  "lessons": ["..."],
  "proactive_suggestions": ["..."]
}
Output JSON only. Do not explain."""


class AgentRunner:
    def __init__(
        self,
        project_id: str,
        session_id: str,
        cwd: Path,
        model: str = "claude-opus-4-7-20251101",
    ):
        self.project_id = project_id
        self.session_id = session_id
        self.cwd = cwd
        self.model = model
        self.messages: list[dict] = []
        self._claude = anthropic.Anthropic()
        self._input_tokens = 0
        self._output_tokens = 0
        self._cache_read_tokens = 0
        self._cache_creation_tokens = 0
        self._ingest_thread: threading.Thread | None = None

    def run(self, query: str) -> str:
        """Execute one full agent turn. Returns the final text response."""
        # 1. Retrieve from memory
        fragments: list[dict] = []
        project_summary: str | None = None
        try:
            with get_client() as client:
                result = client.retrieve(self.project_id, query_text=query)
            fragments = result.get("fragments", [])
            project_summary = result.get("project_summary")
            n = len(fragments)
            suffix = ", 1 project summary" if project_summary else ""
            print(f"[memory: {n} fragments retrieved{suffix}]")
        except MemoryClientError:
            print("[memory: unavailable]")

        # 2. Assemble on first turn; append query on subsequent REPL turns
        if not self.messages:
            global_profile = load_global_profile()
            file_contents = git_dirty_files(self.cwd)
            if file_contents:
                print("[context: git diff injected]")
            self.messages = assemble(
                system=SYSTEM_PROMPT,
                global_profile=global_profile,
                project_summary=project_summary,
                fragments=fragments,
                file_contents=file_contents,
                query=query,
                project_id=self.project_id,
            )
        else:
            self.messages.append({"role": "user", "content": query})

        # 3. Stream with tool use loop
        final_text = self._inference_loop()

        # 4. Reflect (hidden, non-streaming, best-effort)
        reflection = self._reflect()

        # 5. Ingest transcript (fire-and-forget)
        segments = build_segments(self.messages, reflection)
        session_summary = _reflection_to_summary(reflection) if reflection else None
        self._ingest_async(segments, session_summary=session_summary)

        return final_text

    # ── Inference loop ────────────────────────────────────────────────────────

    def _inference_loop(self) -> str:
        final_text = ""
        for _ in range(_MAX_TOOL_ROUNDS):
            with self._claude.messages.stream(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=_apply_cache_breakpoint(self.messages),
                tools=TOOL_DEFINITIONS,
                max_tokens=4096,
            ) as stream:
                for chunk in stream.text_stream:
                    print(chunk, end="", flush=True)
                final_msg = stream.get_final_message()

            print()  # newline after streamed output

            # Accumulate token usage
            if final_msg.usage:
                self._input_tokens += final_msg.usage.input_tokens
                self._output_tokens += final_msg.usage.output_tokens
                self._cache_read_tokens += getattr(
                    final_msg.usage, "cache_read_input_tokens", 0) or 0
                self._cache_creation_tokens += getattr(
                    final_msg.usage, "cache_creation_input_tokens", 0) or 0

            text_parts = [
                block.text
                for block in final_msg.content
                if hasattr(block, "text")
            ]
            final_text = "".join(text_parts)

            # Append assistant turn
            self.messages.append({
                "role": "assistant",
                "content": [_block_to_dict(b) for b in final_msg.content],
            })

            if final_msg.stop_reason == "end_turn":
                break

            if final_msg.stop_reason == "tool_use":
                tool_results = []
                for block in final_msg.content:
                    if block.type == "tool_use":
                        label = block.input.get("path") or block.input.get("command", "")
                        print(f"[tool: {block.name} {label}]")
                        result_str = execute(block.name, block.input, self.cwd)
                        if len(result_str) > _TOOL_RESULT_MAX:
                            result_str = result_str[:_TOOL_RESULT_MAX] + "\n... [truncated]"
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        })
                self.messages.append({"role": "user", "content": tool_results})
            else:
                break  # unexpected stop reason

        tok_in, tok_out = self._input_tokens, self._output_tokens
        cr, cw = self._cache_read_tokens, self._cache_creation_tokens
        print(f"[tokens: {tok_in} in / {tok_out} out | cache: {cr} read / {cw} write]")
        return final_text

    # ── Reflection ────────────────────────────────────────────────────────────

    def _reflect(self) -> dict | None:
        reflect_messages = self.messages + [
            {"role": "user", "content": _REFLECTION_PROMPT}
        ]
        try:
            resp = self._claude.messages.create(
                model=self.model,
                system=SYSTEM_PROMPT,
                messages=_apply_cache_breakpoint(reflect_messages),
                max_tokens=500,
            )
            if resp.usage:
                self._input_tokens += resp.usage.input_tokens
                self._output_tokens += resp.usage.output_tokens
                self._cache_read_tokens += getattr(
                    resp.usage, "cache_read_input_tokens", 0) or 0
                self._cache_creation_tokens += getattr(
                    resp.usage, "cache_creation_input_tokens", 0) or 0
            text = "".join(
                block.text for block in resp.content if hasattr(block, "text")
            )
            return json.loads(text)
        except Exception:
            return None

    # ── Ingest ────────────────────────────────────────────────────────────────

    def _ingest_async(
        self,
        segments: list[dict],
        session_summary: str | None = None,
    ) -> None:
        input_tokens = self._input_tokens
        output_tokens = self._output_tokens
        cache_read_tokens = self._cache_read_tokens
        cache_creation_tokens = self._cache_creation_tokens

        def _do() -> None:
            try:
                with get_client(timeout=30.0) as client:
                    resp = client.ingest(
                        project_id=self.project_id,
                        session_id=self.session_id,
                        transcript_segments=segments,
                        model_id=self.model,
                        session_summary=session_summary,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        cache_read_tokens=cache_read_tokens,
                        cache_creation_tokens=cache_creation_tokens,
                        priority="immediate",
                    )
                episodes = resp.get("episodes_created", 0)
                facts = resp.get("facts_created", 0)
                queue = resp.get("queue_depth", 0)
                ep_label = f"{episodes} episode distilled" if episodes == 1 else f"{episodes} episodes distilled"
                print(
                    f"\n[memory] + ingested | {ep_label} | {facts} new facts | queue: {queue} pending"
                )
            except Exception as e:
                print(f"[warning: ingest failed: {e}]")

        self._ingest_thread = threading.Thread(target=_do, daemon=True)
        self._ingest_thread.start()

    def wait_for_ingest(self, timeout: float = 10.0) -> None:
        if self._ingest_thread is not None:
            self._ingest_thread.join(timeout=timeout)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _apply_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Return a shallow copy of `messages` with an ephemeral cache_control marker
    on the final content block, so the whole stable prefix (system + tools + the
    memory msg0 + every prior turn) is served from Anthropic's prompt cache on the
    next call instead of being re-billed each tool round.

    The originals are never mutated — build_segments and REPL re-sending rely on
    message content staying plain strings. Below the model's minimum cacheable
    size (~1024 tok) the API silently ignores the marker and charges nothing, so
    this is safe to apply unconditionally on every call.
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content", "")
    if isinstance(content, str):
        if not content:
            return messages  # nothing to mark
        blocks: list[dict] = [{
            "type": "text",
            "text": content,
            "cache_control": {"type": "ephemeral"},
        }]
    else:
        blocks = [dict(b) for b in content]
        if not blocks:
            return messages
        blocks[-1] = {**blocks[-1], "cache_control": {"type": "ephemeral"}}
    last["content"] = blocks
    out[-1] = last
    return out


def _block_to_dict(block) -> dict:
    """Convert an SDK content block to a plain dict suitable for re-sending."""
    if hasattr(block, "model_dump"):
        return block.model_dump()
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return {"type": block.type}


def _reflection_to_summary(reflection: dict) -> str:
    """Convert a reflection dict to a plain-text session summary."""
    parts: list[str] = []
    if reflection.get("new_facts"):
        parts.append("Facts: " + "; ".join(str(f) for f in reflection["new_facts"]))
    if reflection.get("lessons"):
        parts.append("Lessons: " + "; ".join(str(l) for l in reflection["lessons"]))
    if reflection.get("corrected_assumptions"):
        corrections = [
            f"{c['was']} → {c['now']}"
            for c in reflection["corrected_assumptions"]
            if isinstance(c, dict) and "was" in c and "now" in c
        ]
        if corrections:
            parts.append("Corrections: " + "; ".join(corrections))
    if reflection.get("proactive_suggestions"):
        parts.append("Suggestions: " + "; ".join(str(s) for s in reflection["proactive_suggestions"]))
    return "\n".join(parts)


def build_segments(messages: list[dict], reflection: dict | None) -> list[dict]:
    """
    Convert the accumulated messages list into transcript_segments for ingest.
    Heuristics detect corrections and explicit remember requests.
    """
    now = datetime.now(timezone.utc).isoformat()
    segments: list[dict] = []
    tool_chain_len = 0

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "assistant" and isinstance(content, list):
            tool_count = sum(
                1 for b in content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            )
            tool_chain_len = tool_count
            text = " ".join(
                b.get("text", "")
                for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            ).strip()
            if text or tool_count:
                segments.append({
                    "role": "assistant",
                    "content": text,
                    "timestamp": now,
                    "is_correction": False,
                    "is_explicit_remember": False,
                    "tool_call_chain_len": tool_chain_len,
                })

        elif role == "user" and isinstance(content, list):
            text = " ".join(
                str(b.get("content", ""))
                for b in content
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ).strip()
            if text:
                segments.append({
                    "role": "tool",
                    "content": text,
                    "timestamp": now,
                    "is_correction": False,
                    "is_explicit_remember": False,
                    "tool_call_chain_len": tool_chain_len,
                })

        elif role == "user" and isinstance(content, str):
            lower = content.lower()
            is_correction = any(w in lower for w in ("no ", "wrong", "incorrect", "that's not"))
            is_remember = "remember" in lower
            segments.append({
                "role": "user",
                "content": content,
                "timestamp": now,
                "is_correction": is_correction,
                "is_explicit_remember": is_remember,
                "tool_call_chain_len": 0,
            })

    if reflection:
        segments.append({
            "role": "assistant",
            "content": f"[reflection] {json.dumps(reflection)}",
            "timestamp": now,
            "is_correction": False,
            "is_explicit_remember": False,
            "tool_call_chain_len": 0,
        })

    return segments
