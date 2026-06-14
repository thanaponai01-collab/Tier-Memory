"""
memory_system.cli — human-friendly command interface for memoryd.

Usage:
    python -m memory_system.cli <subcommand> [--json] [args]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from memory_system.daemon import MemoryClientError, get_client, is_running
from memory_system.daemon.state import DEFAULT_PORT, read_state
from memory_system.project import new_session_id, resolve_project_id


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fail(msg: str, json_out: bool = False) -> None:
    if json_out:
        print(json.dumps({"status": "error", "message": msg}))
    else:
        print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


def _require_daemon(args) -> None:
    if not is_running():
        _fail(
            "memoryd is not running. Start it with:\n"
            "  python -m memory_system.cli daemon start",
            getattr(args, "json", False),
        )


def _ensure_daemon() -> None:
    """Start memoryd in the background if it is not already running."""
    if is_running():
        return

    print("memoryd not running — starting automatically...")
    cmd = [sys.executable, "-m", "memory_system.daemon.server", "--port", str(DEFAULT_PORT)]
    if platform.system() == "Windows":
        subprocess.Popen(
            cmd,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # Poll until the daemon accepts connections (up to 8 seconds)
    deadline = time.monotonic() + 8.0
    while time.monotonic() < deadline:
        if is_running():
            print("memoryd started.")
            return
        time.sleep(0.25)

    _fail("memoryd did not start in time. Run 'python -m memory_system.cli daemon start --foreground' to debug.")


def parse_time_window(text: Optional[str]) -> tuple[Optional[str], int, str]:
    """Turn a free-form 'when' expression into (since_iso, limit, label).

    No LLM — a small deterministic grammar over the time-shaped questions a user
    actually asks the memory: 'last week', 'past 3 days', 'since 2026-06-01',
    'last 5 sessions', a bare count, or nothing (defaults to the last 7 days).

    Returns:
      since_iso — inclusive ISO lower bound on session start (None = no date floor)
      limit     — max sessions to return
      label     — human description of the window, for the header line
    """
    now = datetime.now(timezone.utc)
    DEFAULT_LIMIT = 10
    WINDOW_CAP = 50  # when a date floor is set, cap how many sessions we list
    text = (text or "").strip().lower()

    if not text:
        return (now - timedelta(days=7)).isoformat(), DEFAULT_LIMIT, "the last 7 days"

    # Bare integer, or "last/past N sessions" => last N sessions (no date floor)
    m = re.fullmatch(r"(?:(?:last|past)\s+)?(\d+)(?:\s+sessions?)?", text)
    if m and ("session" in text or text.isdigit()):
        n = max(1, int(m.group(1)))
        return None, n, f"the last {n} session(s)"

    # ISO date, optionally prefixed with since/after/from
    m = re.fullmatch(r"(?:since\s+|after\s+|from\s+)?(\d{4}-\d{2}-\d{2})", text)
    if m:
        return f"{m.group(1)}T00:00:00+00:00", WINDOW_CAP, f"since {m.group(1)}"

    if text == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), WINDOW_CAP, "today"
    if text == "yesterday":
        start = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return start.isoformat(), WINDOW_CAP, "yesterday"

    if text in ("this week", "last week", "past week", "week", "1w"):
        return (now - timedelta(days=7)).isoformat(), WINDOW_CAP, "the last 7 days"
    if text in ("this month", "last month", "past month", "month", "1m"):
        return (now - timedelta(days=30)).isoformat(), WINDOW_CAP, "the last 30 days"

    # "last/past N days|hours|weeks" and compact forms "7d" "12h" "2w"
    m = (re.fullmatch(r"(?:last|past)\s+(\d+)\s+(hour|hours|day|days|week|weeks)", text)
         or re.fullmatch(r"(\d+)\s*(h|d|w)", text))
    if m:
        n, unit = int(m.group(1)), m.group(2)
        if unit.startswith("h"):
            return (now - timedelta(hours=n)).isoformat(), WINDOW_CAP, f"the last {n} hour(s)"
        if unit.startswith("w"):
            return (now - timedelta(weeks=n)).isoformat(), WINDOW_CAP, f"the last {n} week(s)"
        return (now - timedelta(days=n)).isoformat(), WINDOW_CAP, f"the last {n} day(s)"

    # Unrecognised — fall back to the safe default rather than erroring.
    return (now - timedelta(days=7)).isoformat(), DEFAULT_LIMIT, "the last 7 days"


# ── Subcommand handlers ───────────────────────────────────────────────────────

def cmd_status(args) -> None:
    state = read_state()
    if not state or not is_running():
        if args.json:
            print(json.dumps({"status": "not_running"}))
        else:
            print("memoryd: not running")
        return

    try:
        with get_client() as c:
            ping = c.ping()
            stats_resp = c.stats()
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return  # unreachable; silences type checkers

    pid = ping.get("pid", state.get("pid", "?"))
    port = state.get("port", DEFAULT_PORT)
    active = sum(stats_resp.get("stats", {}).values())
    idx_size = stats_resp.get("vector_index_size", "?")
    queue = stats_resp.get("ingest_queue_depth", 0)
    store_path = stats_resp.get("store_path", "?")
    embedder_ok = stats_resp.get("embedder_ok")
    embedder_detail = stats_resp.get("embedder_detail", "")
    embedder_health = stats_resp.get("embedder_health", {})
    health = stats_resp.get("health", {})
    newest = health.get("newest_created_at")
    added_recently = health.get("added_recently")
    total = health.get("fragments_total")
    cache = stats_resp.get("cache", {})
    citations = stats_resp.get("citations", {})

    if args.json:
        print(json.dumps({
            "status": "running",
            "pid": pid,
            "port": port,
            "fragments_active": active,
            "vector_index_size": idx_size,
            "ingest_queue_depth": queue,
            "store_path": store_path,
            "embedder_ok": embedder_ok,
            "embedder_detail": embedder_detail,
            "embedder_health": embedder_health,
            "health": health,
            "cache": cache,
            "citations": citations,
        }))
    else:
        print(f"memoryd: running (pid {pid}, port {port})")
        print(f"store: {store_path}")
        frag_line = f"fragments: {active} active"
        if total is not None:
            frag_line += f" ({total} total)"
        print(frag_line)
        print(f"vector index: {idx_size} entries")
        # Embedder liveness — a dead embedder is the silent failure that makes
        # everything else look fine while nothing new can actually be stored.
        if embedder_ok is True:
            print(f"embedder: alive  [{embedder_detail}]")
        elif embedder_ok is False:
            line = f"embedder: DOWN  [{embedder_detail}]  <- new memories cannot embed"
            since = embedder_health.get("down_since")
            last_ok = embedder_health.get("last_ok_at")
            extra = []
            if since:
                extra.append(f"down since {since[:19]}")
            if last_ok:
                extra.append(f"last ok {last_ok[:19]}")
            if extra:
                line += "  (" + ", ".join(extra) + ")"
            print(line)
        # Freshness — is the flywheel actually still being fed?
        if newest:
            recent_note = ""
            if added_recently is not None:
                recent_note = f"  ({added_recently} added in last 24h)"
            print(f"freshness: newest memory {newest[:19]}{recent_note}")
        print(f"ingest queue: {queue} pending")
        # Redrive — sessions whose distillation failed in an outage, waiting to be
        # re-distilled from cold storage. Non-zero = recoverable holes, not loss.
        redrive_pending = stats_resp.get("redrive_pending", 0)
        if redrive_pending:
            print(f"redrive: {redrive_pending} session(s) awaiting re-distillation "
                  f"(outage holes — run 'mem redrive run' to heal)")
        # Prompt cache — a warm stable prefix means cheaper turns AND a stable
        # durable memory. A falling hit rate = the prefix is churning call-to-call.
        cr = cache.get("cache_read_tok", 0)
        cw = cache.get("cache_creation_tok", 0)
        if cr or cw:
            rate = cache.get("cache_hit_rate", 0.0)
            print(f"prompt cache: {rate*100:.0f}% hit rate "
                  f"({cr:,} read / {cw:,} write tokens)")
        else:
            print("prompt cache: no cache activity yet "
                  "(prefix below min size, or no turns since enabling)")
        # Outcome loop — is surfaced memory actually being used in real answers?
        cited_frags = citations.get("cited_fragments", 0)
        total_cites = citations.get("total_citations", 0)
        if cited_frags:
            print(f"memory used: {cited_frags} fragments cited in real answers "
                  f"({total_cites} times) <- read->use loop is live")
        else:
            print("memory used: no citations yet "
                  "(read reflex active from the next session onward)")
        # System issues — the watchdog grown up: surface any failure the system
        # caught (failed ingest, crashed audit, etc.), not just embedder death.
        issues = stats_resp.get("issues", {})
        err_n = issues.get("errors", 0)
        warn_n = issues.get("warnings", 0)
        if err_n or warn_n:
            print(f"issues: {err_n} error(s), {warn_n} warning(s) in last 24h "
                  "<- see the dashboard Issues panel")
        else:
            print("issues: none in last 24h")
        # Auto-detect (best-effort): nudge if a smarter model is now configured.
        try:
            with get_client() as c:
                up = c.upgrade_status()
            if up.get("upgrade_available"):
                improvable = up.get("improvable_facts", up.get("fragments_behind", 0))
                print(f"upgrade: a smarter model is configured - {improvable} stored "
                      f"facts could level up (run 'mem upgrade')")
        except MemoryClientError:
            pass  # never let the nudge break status


def cmd_search(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.search(args.query, project_id=args.project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
        return

    fragments = resp.get("fragments", [])
    if not fragments:
        print("No results.")
        return

    for i, f in enumerate(fragments, 1):
        crs = f.get("crs", 0)
        fid = f.get("id", "?")
        content = f.get("content", "")
        scope = f.get("scope", "")
        created = f.get("created_at", "")
        extras = "  ".join(x for x in [
            f"scope: {scope}" if scope else "",
            f"created: {created}" if created else "",
        ] if x)
        print(f"[{i:02d}] (crs={crs:.2f}) {content}")
        suffix = f"  {extras}" if extras else ""
        print(f"     id: {fid}{suffix}")


def cmd_recent(args) -> None:
    _require_daemon(args)
    since_iso, limit, label = parse_time_window(getattr(args, "when", None))
    project = args.project or resolve_project_id(Path.cwd())
    try:
        with get_client() as c:
            resp = c.recent(since_iso=since_iso, limit=limit, project_id=project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
        return

    sessions = resp.get("sessions", [])
    if not sessions:
        print(f"No sessions found in {label}.")
        return

    print(f"What you worked on - {label} (newest first):\n")
    for s in sessions:
        started = (s.get("started_at") or "")[:16].replace("T", " ")
        turns = s.get("turn_count") or 0
        frags = s.get("frag_count") or 0
        print(f"* {started}  ({turns} turns, {frags} memories)")
        summary = (s.get("summary") or "").strip()
        if summary:
            for line in summary.splitlines():
                print(f"    {line.rstrip()}")
        else:
            highlights = s.get("highlights") or []
            if highlights:
                for snip in highlights:
                    one_line = " ".join(snip.split())
                    if len(one_line) > 160:
                        one_line = one_line[:157] + "..."
                    print(f"    - {one_line}")
            else:
                print("    (no distilled summary)")
        print()


def cmd_pin(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.pin(args.fragment_id, pinned=not args.unpin)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
    else:
        action = "unpinned" if args.unpin else "pinned"
        print(f"Fragment {args.fragment_id} {action}.")


def cmd_forget(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.forget(args.fragment_id)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
    else:
        print(f"Fragment {args.fragment_id} forgotten.")


def cmd_feedback(args) -> None:
    _require_daemon(args)
    value = 1.0 if args.good else (-1.0 if args.bad else 0.0)
    try:
        with get_client() as c:
            resp = c.feedback(args.fragment_id, value)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
    else:
        print(f"Fragment {args.fragment_id} feedback set to {value:+.1f}.")


def cmd_audit(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.audit(project_id=args.project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
    else:
        print("Audit complete:")
        print(f"  contradictions resolved : {resp.get('contradictions_found', 0)}")
        print(f"  facts decayed           : {resp.get('fragments_decayed', 0)}")
        print(f"  orphans pruned          : {resp.get('orphans_pruned', 0)}")
        print(f"  triples removed         : {resp.get('triples_removed', 0)}")
        print(f"  centrality updated      : {resp.get('centrality_updated', 0)}")
        print(f"  fragments evicted       : {resp.get('evicted', 0)}")


def cmd_redrive(args) -> None:
    _require_daemon(args)
    action = getattr(args, "action", None) or "status"
    try:
        with get_client() as c:
            resp = c.redrive(action=action, project_id=args.project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
        return

    if action == "status":
        n = resp.get("pending", 0)
        if n:
            print(f"Redrive: {n} session(s) lost to a distillation outage are waiting to be re-distilled.")
            print("Run 'mem redrive run' to heal them now (needs the embedder up); the daemon also retries on its own.")
        else:
            print("Redrive: no failed sessions waiting.")
            print("(Run 'mem redrive scan' to find older holes that predate this safety net.)")
    elif action == "scan":
        marked = resp.get("marked", 0)
        pending = resp.get("pending", 0)
        print(f"Redrive scan: marked {marked} zero-fragment session(s) as failed.")
        print(f"{pending} session(s) now awaiting redrive — run 'mem redrive run' to heal them.")
    elif action == "run":
        att = resp.get("attempted", 0)
        rec = resp.get("recovered", 0)
        deferred = resp.get("deferred")
        if deferred:
            print(f"Redrive deferred: the embedder is DOWN, {deferred} session(s) still waiting. Start Ollama and retry.")
        elif att == 0:
            print("Redrive: nothing to do — no failed sessions are due.")
        else:
            print(f"Redrive: re-distilled {rec} of {att} session(s) from cold storage.")
            if rec < att:
                print(f"  {att - rec} still failed (distillation unreachable or nothing to extract) — attempts bumped, will retry.")


def cmd_stats(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.stats(project_id=args.project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
        return

    stats = resp.get("stats", {})
    active = sum(stats.values()) if stats else 0
    idx_size = resp.get("vector_index_size", "?")
    queue = resp.get("ingest_queue_depth", 0)
    print(f"fragments: {active} active")
    print(f"vector index: {idx_size} entries")
    print(f"ingest queue: {queue} pending")
    if stats:
        print("breakdown:")
        for key, count in sorted(stats.items()):
            print(f"  {key}: {count}")


def cmd_profile_show(args) -> None:
    from memory_system.project import PROFILE_PATH, load_global_profile
    profile = load_global_profile()
    if profile:
        print(profile)
    else:
        print(f"No profile set. Create one at:\n  {PROFILE_PATH}")


def cmd_profile_set(args) -> None:
    from memory_system.project import PROFILE_PATH, save_global_profile
    save_global_profile(args.text)
    print(f"Profile saved to {PROFILE_PATH}")


def cmd_profile_edit(args) -> None:
    from memory_system.project import PROFILE_PATH
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not PROFILE_PATH.exists():
        PROFILE_PATH.write_text("", encoding="utf-8")
    editor = (
        os.environ.get("EDITOR")
        or os.environ.get("VISUAL")
        or ("notepad" if platform.system() == "Windows" else "nano")
    )
    subprocess.run([editor, str(PROFILE_PATH)])


def cmd_orchestrate(args) -> None:
    _ensure_daemon()
    from memory_system.orchestrator import Orchestrator
    project_id = args.project or resolve_project_id(Path.cwd())
    orch = Orchestrator(
        project_id=project_id,
        cwd=Path.cwd(),
        max_workers=args.max_workers,
        dry_run=args.dry_run,
    )
    answer = orch.run(args.goal)
    if not args.dry_run:
        print("\n" + answer)


def cmd_run(args) -> None:
    _ensure_daemon()
    from memory_system.agent import AgentRunner
    project_id = args.project or resolve_project_id(Path.cwd())
    session_id = new_session_id()
    runner = AgentRunner(project_id, session_id, Path.cwd(), model=args.model)

    if args.query:
        runner.run(args.query)
        print("[memory: completing background consolidation...]", end=" ", flush=True)
        runner.wait_for_ingest()
        print("done.")
    else:
        try:
            while True:
                try:
                    query = input("> ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if not query:
                    continue
                runner.run(query)
        finally:
            runner.wait_for_ingest()


def cmd_dashboard(args) -> None:
    _ensure_daemon()
    import webbrowser
    import time

    state = read_state()
    if not state:
        _fail("No running memoryd found.", args.json)
        return

    host = state.get("host", "127.0.0.1")
    port = state.get("port", DEFAULT_PORT)
    web_port = port + 1
    url = f"http://{host}:{web_port}"

    if args.json:
        print(json.dumps({"status": "ok", "url": url}))
        return

    print(f"Opening Web Dashboard at {url} ...")
    time.sleep(0.5)
    webbrowser.open(url)


def cmd_savings(args) -> None:
    _require_daemon(args)
    try:
        with get_client() as c:
            resp = c.savings()
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
        return

    print("=" * 60)
    print("  MEMORY SYSTEM - TOKEN & COST LEDGER")
    print("=" * 60)
    print(f"  Total Coding Sessions Run : {resp.get('total_sessions', 0)}")
    print(f"  Total Turns Executed       : {resp.get('total_turns', 0)}")
    print(f"  Actual Input Tokens        : {resp.get('total_input_tokens', 0):,}")
    print(f"  Actual Output Tokens       : {resp.get('total_output_tokens', 0):,}")
    print("-" * 60)
    basis = resp.get("basis", "estimated")
    if basis == "measured":
        disc = resp.get("cache_read_discount", 0.9)
        print(f"  HARNESS PROMPT-CACHE (not memory-attributable)")
        print(f"    cache reads reused       : {resp.get('cache_read_tok', 0):,} tokens")
        print(f"    cache hit rate           : {resp.get('cache_hit_rate', 0.0) * 100:.1f}%")
        print(f"    savings (USD)            : ${resp.get('cache_savings_usd', 0.0):.2f}"
              f"  ({disc:g}x discount applied)")
        print("-" * 60)
        print(f"  MEMORY-ATTRIBUTABLE")
        print(f"    read-reflex injections   : {resp.get('injection_events', 0):,}")
        print(f"    injected tokens (cost)   : {resp.get('injected_tokens', 0):,} tokens")
        print(f"    injection cost (USD)     : ${resp.get('injection_cost_usd', 0.0):.2f}")
        print("-" * 60)
        print(f"  NET (cache savings - injection cost) : ${resp.get('net_usd', 0.0):.2f}")
    else:
        ratio = resp.get("assumed_compression_ratio") or 8.0
        print(f"  ESTIMATED TOKENS SAVED     : {resp.get('tokens_saved', 0):,}")
        print(f"    (projection — no API token data yet; assumes {ratio:g}x compression)")
        print(f"  ESTIMATED COST SAVED (USD) : ${resp.get('cost_saved_usd', 0.0):.2f}")
    print("=" * 60)


def _print_reindex_report(resp: dict) -> None:
    print("Reindex complete:")
    print(f"  fragments reindexed    : {resp.get('fragments_reindexed', 0)}")
    print(f"  facts resynthesized    : {resp.get('facts_resynthesized', 0)}")
    print(f"  cold sessions replayed : {resp.get('cold_sessions_reprocessed', 0)}")
    print(f"  errors                 : {resp.get('errors', 0)}")
    _print_reindex_scorecard(resp.get("before"), resp.get("after"))


def _poll_reindex(json_mode: bool = False) -> None:
    """Poll a background reindex until it finishes, printing live progress.
    Ctrl-C only stops watching — the daemon keeps working; re-check with
    'mem reindex --status'."""
    last_line = ""
    try:
        while True:
            with get_client() as c:
                st = c.reindex_status()
            status = st.get("status")
            if status == "running":
                phase = st.get("phase", "working")
                done, total = st.get("done", 0), st.get("total", 0)
                bar = f"{done}/{total}" if total else f"{done}"
                line = f"  ...{phase}: {bar}"
                if line != last_line and not json_mode:
                    print(line)
                    last_line = line
                time.sleep(4)
                continue
            if status == "done":
                if json_mode:
                    print(json.dumps(st.get("report", {})))
                else:
                    print()
                    _print_reindex_report(st.get("report", {}))
                return
            if status == "error":
                _fail(f"reindex failed: {st.get('message', 'unknown error')}", json_mode)
                return
            # idle / unknown — nothing running
            if not json_mode:
                print("No reindex is running.")
            return
    except KeyboardInterrupt:
        print("\nStopped watching. The reindex is still running in the background.")
        print("Check on it any time with:  mem reindex --status")


def cmd_reindex(args) -> None:
    _require_daemon(args)
    project = getattr(args, "project", None)

    # --status: just report on any in-flight (or finished) background reindex.
    if getattr(args, "status", False):
        try:
            with get_client() as c:
                st = c.reindex_status()
        except MemoryClientError as e:
            _fail(str(e), args.json)
            return
        if args.json:
            print(json.dumps(st))
            return
        status = st.get("status", "idle")
        if status == "running":
            print(f"reindex running - {st.get('phase','working')}: "
                  f"{st.get('done',0)}/{st.get('total',0)}")
        elif status == "done":
            _print_reindex_report(st.get("report", {}))
        elif status == "error":
            print(f"reindex errored: {st.get('message','unknown')}")
        else:
            print("No reindex is running.")
        return

    resynthesize = getattr(args, "resynthesize", False)
    reprocess_cold = getattr(args, "reprocess_cold", False)
    try:
        with get_client() as c:
            resp = c.reindex(project_id=project, resynthesize=resynthesize,
                             reprocess_cold=reprocess_cold, background=True)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if resp.get("status") != "started":
        _fail(f"could not start reindex: {resp}", args.json)
        return
    if not args.json:
        print("Reindex started in the background. Watching progress "
              "(Ctrl-C to detach; it keeps running)...")
    _poll_reindex(json_mode=args.json)


def _print_reindex_scorecard(before: Optional[dict], after: Optional[dict]) -> None:
    """Show whether the crank actually improved the memory, in plain language."""
    if not before or not after:
        return

    # (label, key, higher_is_better, formatter)
    rows = [
        ("memory's confidence in what it knows", "avg_confidence", True,
         lambda v: f"{v * 100:.1f}%"),
        ("vague facts still needing work",       "low_confidence_facts", False,
         lambda v: f"{int(v)}"),
        ("open goals",                           "open_goals", False,
         lambda v: f"{int(v)}"),
        ("duplicates merged away (deprecated)",  "deprecated_fragments", True,
         lambda v: f"{int(v)}"),
        ("active things remembered",             "active_fragments", None,
         lambda v: f"{int(v)}"),
    ]

    print("\n  Did it help? (before -> after)")
    moved = False
    for label, key, higher_better, fmt in rows:
        b = before.get(key, 0)
        a = after.get(key, 0)
        diff = a - b
        if abs(diff) > 1e-9:
            moved = True
        if higher_better is None or abs(diff) < 1e-9:
            verdict = ""
        elif (diff > 0) == higher_better:
            verdict = "  [+] better"
        else:
            verdict = "  [-] worse"
        print(f"    {label:<38}: {fmt(b)} -> {fmt(a)}{verdict}")
    if not moved:
        print("    (nothing changed - this turn of the crank was pure churn)")


def cmd_upgrade(args) -> None:
    _require_daemon(args)
    project = getattr(args, "project", None)
    confirmed = getattr(args, "yes", False)
    try:
        with get_client() as c:
            status = c.upgrade_status(project_id=project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json and not confirmed:
        print(json.dumps(status))
        return

    cur = status.get("current_model", "?")
    stored = status.get("stored_model") or "none recorded (pre-provenance)"
    available = status.get("upgrade_available", False)

    print("Model-upgrade check:")
    print(f"  memory was built by   : {stored}")
    print(f"  you are now running   : {cur}")
    print(f"  {status.get('note', '')}")

    if not available:
        print("\nNothing to do - your memory is already current.")
        return

    # An upgrade IS available. Show the plan + cost (the 'safe mix' gate).
    total = status.get("fragments_total", 0)
    improvable = status.get("improvable_facts", status.get("low_confidence_facts", 0))
    resynth_model = status.get("resynthesis_model") or cur
    cold = status.get("cold_sessions", 0)
    print("\nIf you turn the crank, it would:")
    print(f"  - re-embed all {total} active fragments (keeps search vectors in sync)")
    print(f"  - re-synthesize {improvable} stored facts with {resynth_model} (LLM calls)")
    print(f"  - replay {cold} cold sessions back through consolidation (LLM calls)")
    print("  This costs time and model calls; it does not delete the raw record.")

    if not confirmed:
        scope = f"--project {project} " if project else ""
        print(f"\nThis is a plan only. To actually run it:\n  mem upgrade {scope}--yes")
        return

    # Confirmed — launch the reprocess in the background and watch progress.
    print("\nConfirmed. Leveling memory up to the current model...")
    try:
        with get_client() as c:
            resp = c.reindex(project_id=project, resynthesize=True,
                             reprocess_cold=True, background=True)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return
    if resp.get("status") != "started":
        _fail(f"could not start reprocess: {resp}", args.json)
        return
    if not args.json:
        print("Running in the background. Watching progress "
              "(Ctrl-C to detach; it keeps running)...")
    _poll_reindex(json_mode=args.json)


def cmd_export(args) -> None:
    _require_daemon(args)
    project = args.project or resolve_project_id(Path.cwd())
    try:
        with get_client() as c:
            resp = c.export(project_id=project)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    payload = {
        "format_version": "1",
        "exported_at": resp.get("exported_at", ""),
        "project_id": resp.get("project_id", project),
        "fragment_count": resp.get("fragment_count", 0),
        "fragments": resp.get("fragments", []),
    }

    out = json.dumps(payload, ensure_ascii=False, indent=2)

    if args.output:
        Path(args.output).write_text(out, encoding="utf-8")
        if not args.json:
            print(f"Exported {payload['fragment_count']} fragments to {args.output}")
    else:
        print(out)


def cmd_import(args) -> None:
    _require_daemon(args)
    try:
        data = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        _fail(f"Could not read import file: {e}", args.json)
        return

    fragments = data.get("fragments") if isinstance(data, dict) else data
    if not isinstance(fragments, list):
        _fail("Import file must contain a 'fragments' list.", args.json)
        return

    project_override = getattr(args, "project", None)
    print(f"Importing {len(fragments)} fragments…")
    try:
        with get_client(timeout=300.0) as c:
            resp = c.import_memory(fragments, project_id=project_override)
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps(resp))
    else:
        print("Import complete:")
        print(f"  imported  : {resp.get('imported', 0)}")
        print(f"  skipped   : {resp.get('skipped', 0)}  (already existed)")
        print(f"  errors    : {resp.get('errors', 0)}")


def cmd_absorb(args) -> None:
    """Absorb hand-curated markdown notes into the searchable store as
    high-confidence fragments, so natural questions about identity/goals match
    the user's own crisp words instead of groping through conversation summaries.
    Idempotent: re-running skips notes already absorbed (stable cur_<hash> ids)."""
    from memory_system.curate import build_fragments_from_dir
    from memory_system.config import load_config

    notes_dir = Path(args.source)
    if not notes_dir.is_dir():
        _fail(f"Not a directory: {notes_dir}", args.json)
        return

    cfg = load_config()
    project = args.project or resolve_project_id(Path.cwd())
    frags = build_fragments_from_dir(notes_dir, project)
    if not frags:
        _fail(f"No curated notes (*.md) found in {notes_dir}", args.json)
        return

    _require_daemon(args)
    if not args.json:
        print(f"Absorbing {len(frags)} curated notes from {notes_dir}")
        print(f"  into project {project} as high-confidence fragments…")

    # Send in small batches: the daemon reads each request as one line, and
    # asyncio's readline caps a line at 64 KiB — a single request with all the
    # notes (curated bodies are long) overruns that and drops the connection.
    _BATCH = 5
    totals = {"imported": 0, "skipped": 0, "errors": 0}
    try:
        with get_client(timeout=300.0) as c:
            for i in range(0, len(frags), _BATCH):
                resp = c.import_memory(frags[i : i + _BATCH], project_id=project)
                for k in totals:
                    totals[k] += int(resp.get(k, 0))
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    if args.json:
        print(json.dumps({"status": "ok", **totals}))
    else:
        print("Absorb complete:")
        print(f"  absorbed  : {totals['imported']}")
        print(f"  skipped   : {totals['skipped']}  (already absorbed)")
        print(f"  errors    : {totals['errors']}")
        print("Re-run 'mem eval' to see whether identity/goal questions now hit.")


def cmd_obsidian_export(args) -> None:
    from memory_system.obsidian_bridge.exporter import ObsidianExporter
    from memory_system.config import load_config
    project = args.project or None  # None = all projects
    cfg = load_config()
    scope = f"project {project}" if project else "all projects"
    print(f"Exporting memory ({scope}) to {args.vault} ...")
    exporter = ObsidianExporter(args.vault, cfg=cfg, project_id=project)
    result = exporter.export()
    print(f"Export complete:")
    print(f"  fragments : {result['fragments']}")
    print(f"  entities  : {result['entities']}")
    print(f"  patterns  : {result['patterns']}")
    print(f"  index     : {Path(args.vault) / '_index.md'}")


def cmd_obsidian_watch(args) -> None:
    _require_daemon(args)
    from memory_system.obsidian_bridge.watcher import VaultWatcher
    from memory_system.daemon.state import read_state, DEFAULT_PORT
    project = args.project or resolve_project_id(Path.cwd())  # watcher always needs a target project
    state = read_state()
    host = state.get("host", "127.0.0.1") if state else "127.0.0.1"
    port = state.get("port", DEFAULT_PORT) if state else DEFAULT_PORT
    watcher = VaultWatcher(args.vault, project_id=project, host=host, port=port)
    watcher.run(interval=args.interval)


def cmd_obsidian_sync(args) -> None:
    from memory_system.obsidian_bridge.exporter import ObsidianExporter
    from memory_system.obsidian_bridge.watcher import VaultWatcher
    from memory_system.config import load_config
    from memory_system.daemon.state import read_state, DEFAULT_PORT
    _require_daemon(args)
    project = args.project or None  # None = all projects
    cfg = load_config()

    scope = f"project {project}" if project else "all projects"
    print(f"Exporting memory ({scope}) to {args.vault} ...")
    exporter = ObsidianExporter(args.vault, cfg=cfg, project_id=project)
    result = exporter.export()
    print(f"  fragments={result['fragments']} entities={result['entities']} patterns={result['patterns']}")

    state = read_state()
    host = state.get("host", "127.0.0.1") if state else "127.0.0.1"
    port = state.get("port", DEFAULT_PORT) if state else DEFAULT_PORT
    watcher = VaultWatcher(args.vault, project_id=project, host=host, port=port)
    watcher.run(interval=args.interval)


def cmd_obsidian_init(args) -> None:
    from memory_system.obsidian_bridge.exporter import ObsidianExporter
    from memory_system.config import load_config
    vault = Path(args.vault)
    exporter = ObsidianExporter(vault, cfg=load_config())
    exporter._init_vault_skeleton()
    print(f"Vault initialized at {vault}")
    print(f"  Created: notes/  corrections/  memories/  entities/  patterns/  _README.md")


def cmd_goal(args) -> None:
    from memory_system import mirror
    project = args.project or resolve_project_id(Path.cwd())

    if args.goal_command == "add":
        goal = mirror.add_goal(project, args.statement)
        if args.json:
            print(json.dumps({"status": "ok", "id": goal.id, "statement": goal.statement}))
        else:
            print(f"Goal saved [{goal.id}]: {goal.statement}")
            print("Look in the mirror anytime with:  mem mirror")
        return

    if args.goal_command == "done":
        ok = mirror.close_goal(project, args.id)
        if args.json:
            print(json.dumps({"status": "ok" if ok else "not_found", "id": args.id}))
        elif ok:
            print(f"Goal {args.id} marked done.")
        else:
            print(f"No open goal with id {args.id}.")
        return

    if args.goal_command == "confirm":
        ok = mirror.confirm_proposal(project, args.id)
        if args.json:
            print(json.dumps({"status": "ok" if ok else "not_found", "id": args.id}))
        elif ok:
            print(f"Proposal {args.id} confirmed — it's now one of your goals.")
        else:
            print(f"No pending proposal with id {args.id}.")
        return

    if args.goal_command == "dismiss":
        ok = mirror.dismiss_proposal(project, args.id)
        if args.json:
            print(json.dumps({"status": "ok" if ok else "not_found", "id": args.id}))
        elif ok:
            print(f"Proposal {args.id} waved off.")
        else:
            print(f"No pending proposal with id {args.id}.")
        return

    # default: list
    goals = mirror.list_goals(project, status="open")
    proposals = mirror.list_proposals(project)
    if args.json:
        print(json.dumps({
            "goals": [
                {"id": g.id, "statement": g.statement, "created_at": g.created_at}
                for g in goals
            ],
            "proposals": [
                {"id": g.id, "statement": g.statement} for g in proposals
            ],
        }))
        return
    if not goals:
        print("No open goals. Add one with:  mem goal add \"what you're trying to do\"")
    else:
        print("Open goals:")
        for g in goals:
            print(f"  [{g.id}]  {g.statement}")
    if proposals:
        print("\nAwaiting your nod (the mirror noticed these — not yours until you confirm):")
        for g in proposals:
            print(f"  [{g.id}]  {g.statement}")
        print("  confirm with:  mem goal confirm <id>     wave off with:  mem goal dismiss <id>")


def cmd_propose(args) -> None:
    from memory_system import mirror
    project = args.project or resolve_project_id(Path.cwd())
    try:
        proposals = mirror.propose_goals(project)
    except Exception as e:  # noqa: BLE001 — surface any LLM/DB issue plainly
        _fail(f"Could not look for patterns: {e}", getattr(args, "json", False))
        return
    if args.json:
        print(json.dumps({"status": "ok", "proposals": proposals}))
        return
    if not proposals:
        print("The mirror looked, but found no clear recurring pattern worth proposing.")
        print("That's the honest answer when nothing stands out — try again after more work.")
        return
    print("The mirror noticed these recurring intentions you haven't named:")
    for p in proposals:
        print(f"\n  [{p['id']}]  {p['statement']}")
        if p.get("reason"):
            print(f"        why: {p['reason']}")
    print("\nThese are guesses, not your goals. Make one yours:  mem goal confirm <id>")
    print("Wave one off:  mem goal dismiss <id>")


def cmd_mirror(args) -> None:
    from memory_system import mirror
    project = args.project or resolve_project_id(Path.cwd())
    try:
        reflection = mirror.reflect(project)
    except Exception as e:  # noqa: BLE001 — surface any LLM/DB issue plainly
        _fail(f"Could not produce a reflection: {e}", getattr(args, "json", False))
        return
    if args.json:
        print(json.dumps({"status": "ok", "reflection": reflection}))
        return
    print("=" * 60)
    print("  THE MIRROR — what you said vs. what you did")
    print("=" * 60)
    print(reflection)
    print("=" * 60)


def cmd_eval(args) -> None:
    """The tasting spoon: replay a held-out set of (query -> expected fragment)
    pairs through the real read path and report recall@k + MRR. Read-only — it
    never touches access counts or the v4 training signal."""
    from memory_system import eval as memeval
    from memory_system.config import load_config

    cfg = load_config()
    eval_path = Path(args.file) if args.file else (
        Path(cfg.storage.db_path).parent / memeval.DEFAULT_EVAL_FILENAME
    )

    # --grow: mine the Stop hook's citation log into new cases and append them.
    if getattr(args, "grow", False):
        records = memeval.load_citation_records(memeval.CITATION_LOG)
        if not records:
            msg = (f"No citations logged yet at {memeval.CITATION_LOG}.\n"
                   "The Stop hook writes one each time an answer actually uses an "
                   "injected memory — use Claude Code a while, then re-run.")
            if args.json:
                print(json.dumps({"status": "ok", "added": 0, "note": msg}))
            else:
                print(msg)
            return

        existing = memeval.load_eval_set(eval_path) if eval_path.exists() else []
        candidates = memeval.mine_candidates(records, existing)
        if not candidates:
            if args.json:
                print(json.dumps({"status": "ok", "added": 0,
                                  "note": "every logged citation already has a case"}))
            else:
                print("Nothing new to add — every logged citation already has a case.")
            return

        # Append the mined candidates to the eval set, preserving existing cases.
        if eval_path.exists():
            doc = json.loads(eval_path.read_text(encoding="utf-8"))
            if not isinstance(doc, dict):
                doc = {"cases": doc if isinstance(doc, list) else []}
        else:
            doc = {"cases": []}
        doc.setdefault("cases", [])
        doc["cases"].extend(memeval.case_to_dict(c) for c in candidates)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

        if args.json:
            print(json.dumps({"status": "ok", "added": len(candidates),
                              "eval_set": str(eval_path)}))
        else:
            print(f"Added {len(candidates)} new case(s) mined from real citations to:")
            print(f"  {eval_path}")
            print("They're tagged \"_source\": \"auto-cited\" — review and prune any that")
            print("look wrong, then run:  mem eval")
        return

    # --init: drop a starter set the user can edit, then stop.
    if getattr(args, "init", False):
        if eval_path.exists() and not args.force:
            _fail(f"{eval_path} already exists. Pass --force to overwrite.", args.json)
            return
        memeval.write_starter_template(eval_path)
        if args.json:
            print(json.dumps({"status": "ok", "wrote": str(eval_path)}))
        else:
            print(f"Wrote a starter eval set to:\n  {eval_path}")
            print("Edit it with queries that matter to you, then run:  mem eval")
        return

    if not eval_path.exists():
        _fail(
            f"No eval set found at {eval_path}.\n"
            "Create a starter you can edit with:  mem eval --init",
            args.json,
        )
        return

    try:
        cases = memeval.load_eval_set(eval_path)
    except (ValueError, json.JSONDecodeError) as e:
        _fail(f"Could not read eval set: {e}", args.json)
        return

    _require_daemon(args)

    default_project = args.project or resolve_project_id(Path.cwd())
    k_values = memeval.DEFAULT_K_VALUES

    case_outputs = []
    try:
        with get_client() as c:
            for case in cases:
                project = case.project or default_project
                resp = c.retrieve(
                    project_id=project,
                    query_text=case.query,
                    scopes=["project", "global"],
                    read_only=True,
                )
                case_outputs.append((case, resp.get("fragments", [])))
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    report = memeval.score(case_outputs, k_values=k_values)

    if args.json:
        print(json.dumps({"status": "ok", "eval_set": str(eval_path), **report.as_dict()}))
        return

    # ── Human-readable scorecard ──────────────────────────────────────────────
    print("=" * 64)
    print("  MEMORY EVAL - does the right thing surface? (read-only)")
    print("=" * 64)
    print(f"  eval set : {eval_path}")
    print(f"  cases    : {report.total}   (project: {default_project})")
    print("-" * 64)
    for r in report.results:
        if r.rank is None:
            mark = "[MISS]"
            where = f"not in top {r.returned}" if r.returned else "nothing returned"
        else:
            mark = "[ hit]"
            where = f"rank {r.rank}/{r.returned}"
        q = r.case.query if len(r.case.query) <= 40 else r.case.query[:37] + "..."
        print(f"  {mark}  {q:<42} {where}")
    print("-" * 64)
    print(f"  matched : {report.matched}/{report.total}")
    for k in k_values:
        print(f"  recall@{k:<2}: {report.recall_at(k) * 100:5.1f}%")
    print(f"  MRR     : {report.mrr:.3f}   (1.000 = correct hit always ranked #1)")
    print("=" * 64)
    print("  Watch recall@5 and MRR. Re-run after any threshold/v4 change;")
    print("  if the number drops, the change hurt retrieval.")


def cmd_why(args) -> None:
    """The taster's instrument: show WHAT the read reflex would inject for a
    query and WHY each fragment earned its place — its CRS, confidence, age, and
    whether the answer has ever actually cited it before. Read-only; never
    touches access counts or v4's signal. With --good/--bad N you reward or
    punish the fragment at rank N, feeding the one truly non-proxy signal (you)
    straight into the column v4's label reads."""
    from memory_system.config import load_config
    from memory_system.schema import Database

    _require_daemon(args)
    project = args.project or resolve_project_id(Path.cwd())

    try:
        with get_client() as c:
            resp = c.retrieve(
                project_id=project,
                query_text=args.query,
                scopes=["project", "global"],
                read_only=True,
            )
    except MemoryClientError as e:
        _fail(str(e), args.json)
        return

    fragments = resp.get("fragments", [])

    # Direct DB read for the "did it earn its place?" signal the retrieve
    # response doesn't carry. Same direct-Database pattern as `mem goal`/absorb.
    cite_map: dict[str, dict] = {}
    if fragments:
        cfg = load_config()
        db = Database(cfg.storage.db_path)
        db.connect()
        try:
            ids = [f.get("id") for f in fragments if f.get("id")]
            placeholders = ",".join("?" * len(ids))
            rows = db.fetchall(
                f"SELECT id, COALESCE(times_cited,0) AS tc, last_cited_at "
                f"FROM memory_fragments WHERE id IN ({placeholders})",
                tuple(ids),
            )
            cite_map = {r["id"]: {"tc": r["tc"], "last": r["last_cited_at"]} for r in rows}
        finally:
            db.close()

    # ── One-touch human signal: thumb the fragment at rank N (1-based) ──────────
    rank_n = args.good if args.good is not None else args.bad
    if rank_n is not None:
        if not fragments:
            _fail("nothing was retrieved for that query — nothing to rate.", args.json)
            return
        if rank_n < 1 or rank_n > len(fragments):
            _fail(f"rank {rank_n} out of range (1..{len(fragments)}).", args.json)
            return
        fid = fragments[rank_n - 1].get("id")
        value = 1.0 if args.good is not None else -1.0
        try:
            with get_client() as c:
                c.feedback(fid, value)
        except MemoryClientError as e:
            _fail(str(e), args.json)
            return
        if args.json:
            print(json.dumps({"status": "ok", "fragment_id": fid, "user_feedback": value}))
        else:
            verb = "rewarded" if value > 0 else "penalised"
            print(f"Rank {rank_n} ({fid}) {verb} ({value:+.0f}). v4 will weigh it accordingly on the next refit.")
        return

    if args.json:
        out = []
        for i, f in enumerate(fragments, start=1):
            cm = cite_map.get(f.get("id"), {})
            out.append({
                "rank": i, "id": f.get("id"), "crs": f.get("crs"),
                "confidence": f.get("confidence"), "scope": f.get("scope"),
                "created_at": f.get("created_at"), "times_cited": cm.get("tc", 0),
                "content": f.get("content", ""),
            })
        print(json.dumps({"status": "ok", "query": args.query,
                          "project": project, "fragments": out}))
        return

    # ── Human-readable trace ───────────────────────────────────────────────────
    print("=" * 72)
    print("  WHY THESE? - what the read reflex would inject, and why (read-only)")
    print("=" * 72)
    print(f"  query   : {args.query}")
    print(f"  project : {project}")
    print("-" * 72)
    if not fragments:
        print("  (nothing cleared the relevance gate — the reflex would stay silent)")
        print("=" * 72)
        return
    for i, f in enumerate(fragments, start=1):
        cm = cite_map.get(f.get("id"), {})
        tc = cm.get("tc", 0)
        cited = f"cited {tc}x" if tc else "never cited"
        crs = f.get("crs", 0.0)
        conf = f.get("confidence", 0.0)
        content = " ".join(str(f.get("content", "")).split())
        if len(content) > 96:
            content = content[:93] + "..."
        print(f"  #{i}  crs={crs:<6.3f} conf={conf:<5.2f} {cited:<12} {f.get('created_at','')}")
        print(f"      {content}")
    print("-" * 72)
    print("  Reward/penalise what you see:  mem why \"<query>\" --good N   (or --bad N)")
    print("=" * 72)


def cmd_daemon_start(args) -> None:
    if is_running():
        _fail(
            "memoryd is already running. Check status with:\n"
            "  python -m memory_system.cli status"
        )

    cmd = [
        sys.executable, "-m", "memory_system.daemon.server",
        "--port", str(args.port),
        "--log-level", args.log_level,
    ]

    if args.foreground:
        subprocess.run(cmd)
    elif platform.system() == "Windows":
        proc = subprocess.Popen(
            cmd,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"memoryd starting (pid {proc.pid}, port {args.port})")
    else:
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"memoryd starting (pid {proc.pid}, port {args.port})")


def cmd_daemon_stop(args) -> None:
    state = read_state()
    if not state:
        _fail("No running memoryd found (no state file).")

    pid = state.get("pid")
    if not pid:
        _fail("State file is missing a PID.")

    try:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                check=True,
                capture_output=True,
            )
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
        print(f"memoryd (pid {pid}) stopped.")
    except subprocess.CalledProcessError as e:
        _fail(f"taskkill failed: {e.stderr.decode().strip()}")
    except (ProcessLookupError, PermissionError) as e:
        _fail(f"Could not stop memoryd: {e}")


# ── Argument parser ───────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m memory_system.cli",
        description="CLI for the memory daemon (memoryd)",
    )
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="Show daemon status and fragment counts")
    sub.add_parser("dashboard", help="Launch the Visual Web Dashboard in your browser")
    sub.add_parser("savings", help="Show weekly and all-time token savings analytics")

    p = sub.add_parser("search", help="Search memory fragments")
    p.add_argument("query")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p = sub.add_parser(
        "recent",
        help="Temporal recall — what you worked on recently, by date "
             "(e.g. 'last week', 'past 3 days', 'since 2026-06-01', '5')",
    )
    p.add_argument("when", nargs="?", default=None, metavar="WHEN",
                   help="Time window: 'last week', '3d', 'since 2026-06-01', "
                        "a bare count for last-N-sessions (default: last 7 days)")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p = sub.add_parser("pin", help="Pin (or unpin) a fragment")
    p.add_argument("fragment_id")
    p.add_argument("--unpin", action="store_true")

    p = sub.add_parser("forget", help="Mark a fragment as deprecated")
    p.add_argument("fragment_id")

    p = sub.add_parser("audit", help="Trigger an immediate memory audit")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p = sub.add_parser("stats", help="Show fragment statistics")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p_profile = sub.add_parser("profile", help="Manage the global user profile")
    ps = p_profile.add_subparsers(dest="profile_command", required=True)
    ps.add_parser("show", help="Print the current profile")
    p = ps.add_parser("set", help="Replace the profile with a string")
    p.add_argument("text", help="Profile text (wrap in quotes)")
    ps.add_parser("edit", help="Open the profile in $EDITOR")

    p = sub.add_parser("feedback", help="Mark a fragment as helpful (+) or unhelpful (-)")
    p.add_argument("fragment_id")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--good",  action="store_true", help="Mark as helpful (+1.0)")
    group.add_argument("--bad",   action="store_true", help="Mark as unhelpful (-1.0)")
    group.add_argument("--reset", action="store_true", help="Clear feedback (0.0)")

    p = sub.add_parser("reindex", help="Re-embed all fragments with the current embedding model")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Reindex only this project (default: all projects)")
    p.add_argument("--resynthesize", action="store_true",
                   help="Also re-synthesize low-confidence semantic facts via LLM")
    p.add_argument("--reprocess-cold", action="store_true",
                   help="Re-run consolidation over recent cold sessions with the current model")
    p.add_argument("--status", action="store_true",
                   help="Show progress of a running/finished background reindex and exit")

    p = sub.add_parser("upgrade",
                       help="Check if a smarter model is configured and level memory up to it")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Check/upgrade only this project (default: all projects)")
    p.add_argument("--yes", action="store_true",
                   help="Confirm and run the reprocess (without this, only shows the plan)")

    p = sub.add_parser("export", help="Export project memory to a JSON backup file")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Project to export (default: derived from CWD)")
    p.add_argument("--output", "-o", default=None, metavar="FILE",
                   help="Write to FILE instead of stdout")

    p = sub.add_parser("import", help="Import memory from a JSON backup file")
    p.add_argument("file", metavar="FILE", help="Path to the JSON backup to import")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Override project_id for all imported fragments")

    p = sub.add_parser("orchestrate", help="Run an autonomous multi-agent orchestration")
    p.add_argument("goal", help="High-level goal to accomplish")
    p.add_argument("--project", default=None, metavar="PROJECT_ID")
    p.add_argument("--max-workers", type=int, default=4,
                   help="Max parallel subagents (default: 4)")
    p.add_argument("--dry-run", action="store_true",
                   help="Show the plan without executing subagents")

    p = sub.add_parser("run", help="Run the agent on a query (omit query for REPL)")
    p.add_argument("query", nargs="?", default=None,
                   help="Query to run (omit for interactive REPL mode)")
    p.add_argument("--model", default="claude-opus-4-7-20251101")
    p.add_argument("--project", default=None, metavar="PROJECT_ID",
                   help="Override project ID (default: derived from CWD)")

    p_goal = sub.add_parser("goal", help="Declare and manage your stated goals (the intent mirror)")
    p_goal.add_argument("--project", default=None, metavar="PROJECT")
    gs = p_goal.add_subparsers(dest="goal_command")
    pg = gs.add_parser("add", help="Declare a goal — what you're trying to do")
    pg.add_argument("statement", help="What you're trying to do (wrap in quotes)")
    pg.add_argument("--project", default=None, metavar="PROJECT")
    gs.add_parser("list", help="List your open goals (and any pending proposals)")
    pg = gs.add_parser("done", help="Mark a goal as done")
    pg.add_argument("id", help="Goal id (from 'goal list')")
    pg.add_argument("--project", default=None, metavar="PROJECT")
    pg = gs.add_parser("confirm", help="Accept a proposed goal as your own")
    pg.add_argument("id", help="Proposal id (from 'propose' or 'goal list')")
    pg.add_argument("--project", default=None, metavar="PROJECT")
    pg = gs.add_parser("dismiss", help="Wave off a proposed goal")
    pg.add_argument("id", help="Proposal id (from 'propose' or 'goal list')")
    pg.add_argument("--project", default=None, metavar="PROJECT")

    p = sub.add_parser("eval", help="Score retrieval quality (recall@k + MRR) against a held-out query set")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Default project scope for cases (default: derived from CWD)")
    p.add_argument("--file", default=None, metavar="FILE",
                   help="Path to the eval set JSON (default: <db dir>/eval_set.json)")
    p.add_argument("--init", action="store_true",
                   help="Write a starter eval set you can edit, then exit")
    p.add_argument("--force", action="store_true",
                   help="With --init, overwrite an existing eval set")
    p.add_argument("--grow", action="store_true",
                   help="Mine real citations into new eval cases and append them, then exit")

    p = sub.add_parser("why", help="Show what the read reflex would inject for a query and why — then reward/penalise it")
    p.add_argument("query", help="The query to trace retrieval for")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Project scope (default: derived from CWD)")
    p.add_argument("--good", type=int, default=None, metavar="N",
                   help="Reward the fragment shown at rank N (+1 feedback)")
    p.add_argument("--bad", type=int, default=None, metavar="N",
                   help="Penalise the fragment shown at rank N (-1 feedback)")

    p = sub.add_parser("absorb", help="Absorb hand-curated markdown notes into searchable memory as high-confidence fragments")
    p.add_argument("--source", required=True, metavar="DIR",
                   help="Directory of curated *.md notes to absorb")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Project scope to absorb into (default: derived from CWD)")

    p = sub.add_parser("redrive",
                       help="Re-distill sessions whose distillation failed during an outage (the drop-on-failure safety net)")
    p.add_argument("action", nargs="?", default="status", choices=["status", "scan", "run"],
                   help="status: how many are waiting (default) | scan: find old holes | run: heal them now")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Project scope (default: all projects)")

    p = sub.add_parser("mirror", help="Reflect the gap between your goals and what you actually did")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p = sub.add_parser("propose", help="Let the mirror notice recurring intentions and propose them as goals")
    p.add_argument("--project", default=None, metavar="PROJECT")

    p_obs = sub.add_parser("obsidian", help="Bidirectional Obsidian vault sync")
    obs_sub = p_obs.add_subparsers(dest="obsidian_command", required=True)

    p = obs_sub.add_parser("export", help="Export memory DB to vault markdown notes (one-shot)")
    p.add_argument("--vault", required=True, metavar="VAULT_PATH",
                   help="Path to Obsidian vault directory")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Export only this project (default: derived from CWD)")

    p = obs_sub.add_parser("watch", help="Watch vault/notes/ and vault/corrections/, feed into daemon")
    p.add_argument("--vault", required=True, metavar="VAULT_PATH")
    p.add_argument("--project", default=None, metavar="PROJECT")
    p.add_argument("--interval", type=float, default=5.0, metavar="SECONDS",
                   help="Poll interval in seconds (default: 5)")

    p = obs_sub.add_parser("sync", help="Export once, then watch (the usual workflow)")
    p.add_argument("--vault", required=True, metavar="VAULT_PATH")
    p.add_argument("--project", default=None, metavar="PROJECT")
    p.add_argument("--interval", type=float, default=5.0, metavar="SECONDS")

    p = obs_sub.add_parser("init", help="Initialize vault skeleton without exporting")
    p.add_argument("--vault", required=True, metavar="VAULT_PATH")

    p_daemon = sub.add_parser("daemon", help="Manage the memoryd process")
    ds = p_daemon.add_subparsers(dest="daemon_command", required=True)

    p = ds.add_parser("start", help="Start memoryd in the background")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--foreground", action="store_true",
                   help="Block instead of detaching (useful for debugging)")

    ds.add_parser("stop", help="Stop a running memoryd")

    return parser


def main() -> None:
    # Memory content can contain Unicode (arrows, em-dashes, accents) that a
    # legacy console codepage (Windows cp1252) cannot encode, which would crash
    # any command that prints fragment text. Make stdout/stderr lossy-but-safe.
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # not a reconfigurable stream (e.g. piped/captured) — leave as-is

    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "status":   cmd_status,
        "search":   cmd_search,
        "recent":   cmd_recent,
        "pin":      cmd_pin,
        "forget":   cmd_forget,
        "feedback": cmd_feedback,
        "audit":    cmd_audit,
        "stats":    cmd_stats,
        "redrive":  cmd_redrive,
        "reindex":  cmd_reindex,
        "upgrade":  cmd_upgrade,
        "export":   cmd_export,
        "import":   cmd_import,
        "orchestrate": cmd_orchestrate,
        "run":      cmd_run,
        "dashboard": cmd_dashboard,
        "savings":   cmd_savings,
        "goal":      cmd_goal,
        "mirror":    cmd_mirror,
        "propose":   cmd_propose,
        "eval":      cmd_eval,
        "why":       cmd_why,
        "absorb":    cmd_absorb,
    }

    if args.command == "daemon":
        {"start": cmd_daemon_start, "stop": cmd_daemon_stop}[args.daemon_command](args)
    elif args.command == "obsidian":
        {
            "export": cmd_obsidian_export,
            "watch":  cmd_obsidian_watch,
            "sync":   cmd_obsidian_sync,
            "init":   cmd_obsidian_init,
        }[args.obsidian_command](args)
    elif args.command == "profile":
        {
            "show": cmd_profile_show,
            "set":  cmd_profile_set,
            "edit": cmd_profile_edit,
        }[args.profile_command](args)
    else:
        handlers[args.command](args)


if __name__ == "__main__":
    main()
