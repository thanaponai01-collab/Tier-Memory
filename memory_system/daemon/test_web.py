"""
Integration tests for the Daemon's Web HTTP Server & API endpoints.
Runs daemon on a custom test port (7999 / web 8000) using a local temp DB.
"""

from __future__ import annotations
import asyncio
import json
import platform
import socket
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
from pathlib import Path

from ..config import load_config
from .server import MemoryDaemon
from ..models import MemoryFragment
from ..ids import new_id

TEST_HOST = "127.0.0.1"
TEST_PORT = 7999
WEB_PORT  = 8000
API_BASE  = f"http://{TEST_HOST}:{WEB_PORT}/api"


def _free_port(port: int) -> None:
    """Kill any process listening on the given port so the test can bind it."""
    if platform.system() != "Windows":
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"],
                capture_output=True, text=True, timeout=5,
            )
            for pid in result.stdout.strip().splitlines():
                if pid.isdigit():
                    subprocess.run(["kill", "-9", pid], capture_output=True)
        except Exception:
            pass
        return

    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port} " in line and "LISTENING" in line:
                parts = line.split()
                pid = parts[-1]
                if pid.isdigit() and int(pid) > 0:
                    subprocess.run(
                        ["taskkill", "/F", "/PID", pid],
                        capture_output=True, timeout=5,
                    )
    except Exception:
        pass


async def _wait_for_port(port: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((TEST_HOST, port), timeout=0.1)
            s.close()
            return True
        except OSError:
            await asyncio.sleep(0.1)
    return False


def _send_request(url: str, method: str = "GET", body: dict | None = None) -> tuple[int, dict | str]:
    req = urllib.request.Request(url, method=method)
    if body is not None:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(body).encode("utf-8")

    try:
        with urllib.request.urlopen(req, timeout=5.0) as response:
            content_type = response.headers.get("Content-Type", "")
            resp_body = response.read()
            if "application/json" in content_type:
                return response.status, json.loads(resp_body.decode("utf-8"))
            return response.status, resp_body.decode("utf-8")
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode("utf-8"))
        except Exception:
            return e.code, str(e)
    except Exception as e:
        return 500, str(e)


async def _stop_daemon(daemon: MemoryDaemon, task: asyncio.Task) -> None:
    daemon.stop()
    try:
        await asyncio.wait_for(task, timeout=5.0)
    except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
        pass


async def main():
    print("=" * 60)
    print("  RUNNING DAEMON WEB API INTEGRATION TESTS")
    print("=" * 60)

    # Free any zombie processes from a previous failed run
    _free_port(TEST_PORT)
    _free_port(WEB_PORT)
    await asyncio.sleep(0.3)

    # 1. Setup Config with Temp DB
    temp_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
    temp_path = Path(temp_dir.name)

    cfg = load_config(None)
    cfg.storage.db_path = str(temp_path / "test_memory.db")
    cfg.storage.cold_storage_path = str(temp_path / "cold")
    cfg.embedding.model = "random"
    cfg.embedding.dimensions = 64

    # 2. Start Daemon
    daemon = MemoryDaemon(cfg)
    daemon_task = asyncio.create_task(daemon.run(host=TEST_HOST, port=TEST_PORT))

    print("Starting test daemon...")
    # Python 3.14 eager task factory runs _init_storage() before create_task returns,
    # but asyncio.start_server is async; wait until both ports are reachable.
    if not await _wait_for_port(TEST_PORT) or not await _wait_for_port(WEB_PORT):
        print("[FAIL] Daemon did not start in time.")
        await _stop_daemon(daemon, daemon_task)
        temp_dir.cleanup()
        return

    if daemon._db is None:
        print("[FAIL] daemon._db not initialized.")
        await _stop_daemon(daemon, daemon_task)
        temp_dir.cleanup()
        return

    print("Daemon online. Running test suites...")

    # Insert a test fragment directly into the daemon's DB
    db = daemon._db
    frag = MemoryFragment(
        id=new_id(),
        project_id="test_proj",
        scope="project",
        category="fact",
        content="Testing the web api",
        token_count=10,
        embedding_model="random",
        embedding_dim=64,
        embedding=[0.1] * 64,
    )
    db.upsert_fragment(frag)
    daemon._idx.add(frag.id, frag.embedding)

    passes = 0
    failures = 0

    def assert_test(name: str, condition: bool, info: str = ""):
        nonlocal passes, failures
        if condition:
            print(f"  [PASS] {name} {info}")
            passes += 1
        else:
            print(f"  [FAIL] {name} {info}")
            failures += 1

    async def req(url: str, method: str = "GET", body: dict | None = None):
        # Run the blocking HTTP call in a thread so the asyncio event loop stays
        # free to handle asyncio.run_coroutine_threadsafe() calls from the HTTP
        # handler threads (avoids a deadlock on /api/status, /api/stats, etc.)
        return await asyncio.to_thread(_send_request, url, method, body)

    # Test 1: GET Static Dashboard
    status, body = await req(f"http://{TEST_HOST}:{WEB_PORT}/index.html")
    assert_test("GET /index.html", status == 200 and "Tiered Memory" in body, f"(status={status})")

    # Test 2: GET /api/status
    status, json_data = await req(f"{API_BASE}/status")
    assert_test("GET /api/status", status == 200 and isinstance(json_data, dict) and json_data.get("status") == "ok", f"(body={json_data})")

    # Test 3: GET /api/stats
    status, json_data = await req(f"{API_BASE}/stats")
    assert_test("GET /api/stats", status == 200 and isinstance(json_data, dict) and json_data.get("status") == "ok", f"(body={json_data})")

    # Test 4: GET /api/fragments — must contain our inserted fragment
    status, json_data = await req(f"{API_BASE}/fragments")
    frags = json_data.get("fragments", []) if isinstance(json_data, dict) else []
    frag_ids = [f["id"] for f in frags]
    assert_test("GET /api/fragments", status == 200 and frag.id in frag_ids, f"(count={len(frags)}, target_in={frag.id in frag_ids})")

    # Test 5: GET /api/graph
    status, json_data = await req(f"{API_BASE}/graph")
    assert_test("GET /api/graph", status == 200 and isinstance(json_data, dict) and "nodes" in json_data and "links" in json_data, f"(nodes={len(json_data.get('nodes', [])) if isinstance(json_data, dict) else '?'})")

    # Test 6: GET /api/savings
    status, json_data = await req(f"{API_BASE}/savings")
    assert_test("GET /api/savings", status == 200 and isinstance(json_data, dict) and "summary" in json_data and "sessions" in json_data, f"(summary={json_data.get('summary') if isinstance(json_data, dict) else json_data})")

    # Test 7: POST /api/pin toggle on
    status, json_data = await req(f"{API_BASE}/pin", "POST", {"fragment_id": frag.id, "pinned": True})
    assert_test("POST /api/pin (True)", status == 200 and isinstance(json_data, dict) and json_data.get("pinned") is True)
    updated_frag = db.get_fragment(frag.id)
    assert_test("DB Verification: Pinned == 1", updated_frag is not None and updated_frag.is_pinned == 1)

    # Test 8: POST /api/pin toggle off
    status, json_data = await req(f"{API_BASE}/pin", "POST", {"fragment_id": frag.id, "pinned": False})
    assert_test("POST /api/pin (False)", status == 200 and isinstance(json_data, dict) and json_data.get("pinned") is False)
    updated_frag = db.get_fragment(frag.id)
    assert_test("DB Verification: Pinned == 0", updated_frag is not None and updated_frag.is_pinned == 0)

    # Test 9: POST /api/feedback
    status, json_data = await req(f"{API_BASE}/feedback", "POST", {"fragment_id": frag.id, "value": 1.0})
    assert_test("POST /api/feedback (+1.0)", status == 200 and isinstance(json_data, dict) and json_data.get("user_feedback") == 1.0)
    updated_frag = db.get_fragment(frag.id)
    assert_test("DB Verification: Feedback == 1.0", updated_frag is not None and updated_frag.user_feedback == 1.0)

    # Test 10: POST /api/audit
    status, json_data = await req(f"{API_BASE}/audit", "POST", {"project_id": "test_proj"})
    assert_test("POST /api/audit", status == 200 and isinstance(json_data, dict) and "fragments_decayed" in json_data, f"(evicted={json_data.get('evicted', 0) if isinstance(json_data, dict) else json_data})")

    # Test 11: POST /api/forget
    status, json_data = await req(f"{API_BASE}/forget", "POST", {"fragment_id": frag.id})
    assert_test("POST /api/forget", status == 200 and isinstance(json_data, dict) and json_data.get("forgotten") is True)
    forgotten_frag = db.get_fragment(frag.id)
    assert_test("DB Verification: Deprecated == 1", forgotten_frag is not None and forgotten_frag.is_deprecated == 1)

    # 3. Cleanup
    print("Stopping test daemon...")
    await _stop_daemon(daemon, daemon_task)
    temp_dir.cleanup()

    print("=" * 60)
    print(f"  RESULTS: {passes} PASSED, {failures} FAILED")
    print("=" * 60)

    if failures > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    asyncio.run(main())
