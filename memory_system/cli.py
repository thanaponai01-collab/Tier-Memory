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
import subprocess
import sys
import time
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
            print(f"embedder: DOWN  [{embedder_detail}]  <- new memories cannot embed")
        # Freshness — is the flywheel actually still being fed?
        if newest:
            recent_note = ""
            if added_recently is not None:
                recent_note = f"  ({added_recently} added in last 24h)"
            print(f"freshness: newest memory {newest[:19]}{recent_note}")
        print(f"ingest queue: {queue} pending")
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
        # Auto-detect (best-effort): nudge if a smarter model is now configured.
        try:
            with get_client() as c:
                up = c.upgrade_status()
            if up.get("upgrade_available"):
                behind = up.get("fragments_behind", 0)
                print(f"upgrade: a smarter model is configured - {behind} fragments "
                      f"could level up (run 'mem upgrade')")
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
    print("  MEMORY SYSTEM — TOKENS & COST SAVINGS REPORT")
    print("=" * 60)
    print(f"  Total Coding Sessions Run : {resp.get('total_sessions', 0)}")
    print(f"  Total Turns Executed       : {resp.get('total_turns', 0)}")
    print(f"  Actual Input Tokens        : {resp.get('total_input_tokens', 0):,}")
    print(f"  Actual Output Tokens       : {resp.get('total_output_tokens', 0):,}")
    print("-" * 60)
    basis = resp.get("basis", "estimated")
    if basis == "measured":
        print(f"  MEASURED TOKENS SAVED      : {resp.get('tokens_saved', 0):,}")
        print(f"    (prompt-cache reads — input tokens not re-charged)")
        print(f"  MEASURED COST SAVED (USD)  : ${resp.get('cost_saved_usd', 0.0):.2f}")
        print(f"  Cache hit rate             : {resp.get('cache_hit_rate', 0.0) * 100:.1f}%")
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
    behind = status.get("fragments_behind", 0)
    total = status.get("fragments_total", 0)
    facts = status.get("low_confidence_facts", 0)
    cold = status.get("cold_sessions", 0)
    print("\nIf you turn the crank, it would:")
    print(f"  - re-embed all {total} active fragments ({behind} are behind the new model)")
    print(f"  - re-synthesize {facts} vague facts with the current model (LLM calls)")
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

    p = sub.add_parser("absorb", help="Absorb hand-curated markdown notes into searchable memory as high-confidence fragments")
    p.add_argument("--source", required=True, metavar="DIR",
                   help="Directory of curated *.md notes to absorb")
    p.add_argument("--project", default=None, metavar="PROJECT",
                   help="Project scope to absorb into (default: derived from CWD)")

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
    parser = _build_parser()
    args = parser.parse_args()

    handlers = {
        "status":   cmd_status,
        "search":   cmd_search,
        "pin":      cmd_pin,
        "forget":   cmd_forget,
        "feedback": cmd_feedback,
        "audit":    cmd_audit,
        "stats":    cmd_stats,
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
