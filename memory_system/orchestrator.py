"""
memory_system.orchestrator — Autonomous multi-agent orchestration layer.

One orchestrator run:
  retrieve -> plan -> execute subagents (parallel + sequential) -> synthesize -> ingest lesson

Uses the Claude Code CLI (`claude -p`) for all LLM calls — no API key needed,
inherits whichever model is active in the current Claude Code session.

Usage (via CLI):
    python -m memory_system.cli orchestrate "your goal here"
    python -m memory_system.cli orchestrate "your goal here" --dry-run
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from memory_system.daemon import MemoryClientError, get_client
from memory_system.project import new_session_id

_MAX_SUBAGENTS   = 6
_PRINT_LOCK      = threading.Lock()


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SubagentTask:
    id: int
    brief: str
    parallel: bool = True
    depends_on: list[int] = field(default_factory=list)


@dataclass
class SubagentResult:
    task_id: int
    output: str
    elapsed: float
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


# ── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(
        self,
        project_id: str,
        cwd: Path,
        max_workers: int = 4,
        dry_run: bool = False,
    ):
        self.project_id  = project_id
        self.cwd         = cwd
        self.max_workers = max_workers
        self.dry_run     = dry_run
        self._t0         = time.monotonic()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(self, goal: str) -> str:
        self._t0 = time.monotonic()
        self._log_header(goal)

        fragments = self._retrieve_context(goal)
        tasks     = self._plan(goal, fragments)

        if not tasks:
            self._log("plan", "no tasks produced -- single-task fallback")
            tasks = [SubagentTask(id=0, brief=goal)]

        n_parallel  = sum(1 for t in tasks if t.parallel and not t.depends_on)
        n_remaining = len(tasks) - n_parallel
        self._log("plan", f"{len(tasks)} tasks -- {n_parallel} parallel -> {n_remaining} sequential")

        if self.dry_run:
            self._log("dry-run", "plan only, skipping execution")
            for t in tasks:
                deps = f" [needs: {','.join(str(d) for d in t.depends_on)}]" if t.depends_on else ""
                mode = "parallel" if t.parallel else "sequential"
                self._log(f"task-{t.id}", f"{mode}{deps}: {t.brief[:80]}")
            self._log_footer()
            return "[dry-run: plan only]"

        results = self._execute_tasks(tasks, fragments)

        self._log("synth", f"synthesizing {len(results)} result(s)...")
        answer, lesson = self._synthesize(goal, results)

        if lesson:
            self._write_back(goal, lesson)

        elapsed = time.monotonic() - self._t0
        errors  = sum(1 for r in results if not r.ok)
        self._log("done", f"total {elapsed:.1f}s | {len(results)} tasks | {errors} errors")
        self._log_footer()
        return answer

    # ── CLI LLM backend ───────────────────────────────────────────────────────

    def _call_cli(self, prompt: str, system: str = None, timeout: int = 120) -> str:
        """
        Call `claude -p` with the given prompt. Uses whichever model is active
        in the current Claude Code session. --bare avoids recursive memory
        ingestion and hook interference.
        """
        cmd = ["claude", "-p", prompt, "--output-format", "text"]
        if system:
            cmd += ["--system-prompt", system]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                cwd=str(self.cwd),
                timeout=timeout,
            )
            out = result.stdout.strip()
            if result.returncode != 0 and not out:
                self._log("warn", f"claude -p exited {result.returncode}: {result.stderr.strip()[:200]}")
            return out
        except subprocess.TimeoutExpired:
            self._log("warn", f"claude -p timed out after {timeout}s")
            return ""
        except FileNotFoundError:
            raise RuntimeError(
                "claude CLI not found. Make sure Claude Code is installed and 'claude' is on PATH."
            )

    def _call_cli_json(self, prompt: str, system: str = None, timeout: int = 120) -> dict:
        raw = self._call_cli(prompt, system=system, timeout=timeout)
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        s = raw.find("{")
        e = raw.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except json.JSONDecodeError:
                pass
        return {}

    # ── Memory ────────────────────────────────────────────────────────────────

    def _retrieve_context(self, goal: str) -> list[dict]:
        try:
            with get_client() as c:
                resp = c.retrieve(self.project_id, query_text=goal, max_tokens=3000)
            fragments = resp.get("fragments", [])
            self._log("memory", f"{len(fragments)} fragments retrieved")
            return fragments
        except MemoryClientError as e:
            self._log("memory", f"unavailable ({str(e).splitlines()[0]}) -- continuing without context")
            return []

    def _write_back(self, goal: str, lesson: str) -> None:
        session_id = new_session_id()
        now = datetime.now(timezone.utc).isoformat()
        segments = [
            {
                "role": "user",
                "content": f"Orchestrator goal: {goal}",
                "timestamp": now,
                "is_correction": False,
                "is_explicit_remember": False,
                "tool_call_chain_len": 0,
            },
            {
                "role": "assistant",
                "content": f"[orchestrator lesson] {lesson}",
                "timestamp": now,
                "is_correction": False,
                "is_explicit_remember": False,
                "tool_call_chain_len": 0,
            },
        ]
        try:
            with get_client(timeout=20.0) as c:
                c.ingest(
                    project_id=self.project_id,
                    session_id=session_id,
                    transcript_segments=segments,
                    priority="immediate",
                )
            self._log("memory", "lesson saved")
        except Exception as e:
            self._log("memory", f"lesson save failed: {e}")

    # ── Planning ──────────────────────────────────────────────────────────────

    _PLAN_SYSTEM = (
        "You are an orchestrator. Break the user's goal into focused subagent tasks.\n\n"
        "Rules:\n"
        "- Each task must be self-contained: its brief includes everything the subagent needs.\n"
        "- Mark parallel=true for tasks with no inter-dependency.\n"
        "- Use depends_on (list of task ids) for tasks that need prior results.\n"
        f"- Maximum {_MAX_SUBAGENTS} tasks. Fewer is better -- don't over-fragment.\n"
        "- Each brief must end with a Return Format instruction.\n"
        "- Never tell a subagent to retrieve or save memory -- that is handled externally.\n\n"
        "Output JSON only:\n"
        "{\n"
        '  "tasks": [\n'
        '    {"id": 0, "brief": "...", "parallel": true,  "depends_on": []},\n'
        '    {"id": 1, "brief": "...", "parallel": false, "depends_on": [0]}\n'
        "  ],\n"
        '  "reasoning": "one sentence on why this decomposition"\n'
        "}"
    )

    def _plan(self, goal: str, fragments: list[dict]) -> list[SubagentTask]:
        context_block = _format_fragments(fragments)
        user_msg = (
            f"<memory_context>\n{context_block}\n</memory_context>\n\n"
            f"Goal: {goal}"
        )
        data = self._call_cli_json(user_msg, system=self._PLAN_SYSTEM, timeout=120)
        if not data:
            self._log("plan", "planning failed -- single-task fallback")
            return [SubagentTask(id=0, brief=goal)]

        reasoning = data.get("reasoning", "")
        if reasoning:
            self._log("plan", f"reasoning: {reasoning}")

        tasks = []
        for t in data.get("tasks", []):
            tasks.append(SubagentTask(
                id=int(t["id"]),
                brief=str(t["brief"]),
                parallel=bool(t.get("parallel", True)),
                depends_on=[int(d) for d in t.get("depends_on", [])],
            ))
        return tasks

    # ── Execution ─────────────────────────────────────────────────────────────

    def _execute_tasks(
        self,
        tasks: list[SubagentTask],
        fragments: list[dict],
    ) -> list[SubagentResult]:
        results:   dict[int, SubagentResult] = {}
        remaining: list[SubagentTask]         = list(tasks)

        while remaining:
            ready = [t for t in remaining if all(d in results for d in t.depends_on)]

            if not ready:
                self._log("warn", "dependency cycle detected -- running remaining sequentially")
                for t in remaining:
                    r = self._run_subagent(t, fragments, results)
                    results[t.id] = r
                break

            parallel_ready   = [t for t in ready if t.parallel]
            sequential_ready = [t for t in ready if not t.parallel]

            if parallel_ready:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures = {
                        pool.submit(self._run_subagent, t, fragments, results): t
                        for t in parallel_ready
                    }
                    for future in as_completed(futures):
                        r = future.result()
                        results[r.task_id] = r

            for t in sequential_ready:
                r = self._run_subagent(t, fragments, results)
                results[t.id] = r

            remaining = [t for t in remaining if t.id not in results]

        return [results[t.id] for t in tasks if t.id in results]

    def _run_subagent(
        self,
        task: SubagentTask,
        fragments: list[dict],
        prior: dict[int, SubagentResult],
    ) -> SubagentResult:
        deps_str = (
            f" [needs: {','.join(str(d) for d in task.depends_on)}]"
            if task.depends_on else ""
        )
        self._log(f"task-{task.id}", f">> {task.brief[:70]}{deps_str}")
        t_start = time.monotonic()

        prior_block = ""
        if task.depends_on:
            parts = []
            for dep_id in task.depends_on:
                if dep_id in prior and prior[dep_id].ok:
                    parts.append(
                        f"<result_of_task_{dep_id}>\n{prior[dep_id].output}\n</result_of_task_{dep_id}>"
                    )
            if parts:
                prior_block = "\n<prior_results>\n" + "\n".join(parts) + "\n</prior_results>"

        context_block = _format_fragments(fragments)
        full_prompt = (
            f"<memory_context>\n{context_block}\n</memory_context>"
            + prior_block
            + f"\n\n## Task\n{task.brief}"
        )

        try:
            output = self._call_cli(full_prompt, timeout=180)
            elapsed = time.monotonic() - t_start
            self._log(f"task-{task.id}", f"done ({elapsed:.1f}s)")
            return SubagentResult(task_id=task.id, output=output, elapsed=elapsed)
        except Exception as e:
            elapsed = time.monotonic() - t_start
            self._log(f"task-{task.id}", f"FAILED ({elapsed:.1f}s): {e}")
            return SubagentResult(task_id=task.id, output="", elapsed=elapsed, error=str(e))

    # ── Synthesis ─────────────────────────────────────────────────────────────

    def _synthesize(self, goal: str, results: list[SubagentResult]) -> tuple[str, str]:
        if not results:
            return "No results.", ""

        if len(results) == 1 and results[0].ok:
            lesson = self._extract_lesson(goal, results[0].output)
            return results[0].output, lesson

        results_block = "\n\n".join(
            f"Task {r.task_id} result:\n{r.output}" if r.ok
            else f"Task {r.task_id}: ERROR -- {r.error}"
            for r in results
        )
        prompt = (
            f"Goal: {goal}\n\n"
            f"Subagent results:\n{results_block}\n\n"
            "Synthesize a final answer.\n"
            "Also extract one LESSON: a durable, timeless pattern for future runs of similar goals.\n"
            "The lesson must be a reusable principle, not a description of this specific run.\n\n"
            'Return JSON only:\n{"answer": "...", "lesson": "..."}'
        )
        data = self._call_cli_json(prompt, timeout=60)
        if data:
            return data.get("answer", ""), data.get("lesson", "")

        combined = "\n\n---\n\n".join(r.output for r in results if r.ok)
        return combined, ""

    def _extract_lesson(self, goal: str, output: str) -> str:
        prompt = (
            f"Goal: {goal}\n\nResult:\n{output[:1000]}\n\n"
            "Extract one LESSON: a durable, timeless principle useful for future similar goals.\n"
            'Return JSON only: {"lesson": "..."}'
        )
        data = self._call_cli_json(prompt, timeout=30)
        return data.get("lesson", "")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, prefix: str, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        with _PRINT_LOCK:
            print(f"[{ts}] {prefix:<10} {msg}", flush=True)

    def _log_header(self, goal: str) -> None:
        with _PRINT_LOCK:
            print("-" * 56, flush=True)
            print(" ORCHESTRATOR", flush=True)
            print("-" * 56, flush=True)
        self._log("goal", goal)

    def _log_footer(self) -> None:
        with _PRINT_LOCK:
            print("-" * 56, flush=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_fragments(fragments: list[dict]) -> str:
    if not fragments:
        return "(no memory context)"
    lines = []
    for f in fragments:
        crs     = f.get("crs", 0)
        content = f.get("content", "")
        lines.append(f'<memory crs="{crs:.2f}">{content}</memory>')
    return "\n".join(lines)
