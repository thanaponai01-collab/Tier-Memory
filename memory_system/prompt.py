"""
memory_system.prompt — L0/L1/L2/L3 prompt assembly for the agent runner.

Layer ordering (most stable → least stable):
  L0: SYSTEM_PROMPT        → passed as system= kwarg to the API, never in messages
  L1: user_profile         → first block in the single user message
      project_memory       → second block in same user message
  L2: recalled_memories    → fragments, CRS-sorted descending
  L3: active_files         → then files, then the actual query

All L1–L3 content is collapsed into one user message so the stable prefix
(L0+L1+L2) can be matched by the Anthropic prompt cache on subsequent calls.
"""

from __future__ import annotations

# Frozen between CLI versions — changing this string breaks the prompt cache.
SYSTEM_PROMPT = (
    "You are a capable software engineering assistant with persistent memory.\n"
    "You have access to tools for reading files, writing files, and running shell commands.\n"
    "Think step by step. Use tools when you need to inspect or modify the project.\n"
    "Be direct and concise. Format code in fenced blocks. Make surgical changes."
)


def format_fragments(fragments: list[dict]) -> str:
    """Render memory fragments as <memory> tags, CRS-sorted descending."""
    sorted_frags = sorted(fragments, key=lambda f: f.get("crs", 0), reverse=True)
    parts = []
    for f in sorted_frags:
        fid = f.get("id", "")
        scope = f.get("scope", "")
        crs = f.get("crs", "")
        content = f.get("content", "")
        parts.append(f'<memory id="{fid}" scope="{scope}" crs="{crs}">{content}</memory>')
    return "\n".join(parts)


def assemble(
    *,
    system: str,
    global_profile: str | None,
    project_summary: str | None,
    fragments: list[dict],
    file_contents: str | None,
    query: str,
    project_id: str,
) -> list[dict]:
    """
    Build the Anthropic messages list. Returns a single-element list containing
    one user message with all context layers concatenated. The system prompt is
    not included here — pass it as system= when calling the API.
    """
    blocks: list[str] = []

    if global_profile:
        blocks.append(f"<user_profile>\n{global_profile}\n</user_profile>")

    if project_summary:
        blocks.append(
            f'<project_memory project_id="{project_id}">\n{project_summary}\n</project_memory>'
        )

    if fragments:
        blocks.append(
            f"<recalled_memories>\n{format_fragments(fragments)}\n</recalled_memories>"
        )

    if file_contents:
        blocks.append(f"<active_files>\n{file_contents}\n</active_files>")

    blocks.append(query)

    return [{"role": "user", "content": "\n\n".join(blocks)}]
