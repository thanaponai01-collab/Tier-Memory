"""
memory_system.daemon — local TCP server + client for concurrent CLI sessions.

Quickstart:
    # Start the daemon (blocks):
    python -m memory_system.daemon.server

    # In any CLI session:
    from memory_system.daemon import get_client
    with get_client() as client:
        result = client.retrieve(project_id="my-app", query_text="docker build")
        client.ingest(project_id="my-app", session_id="sess_123", transcript_segments=[...])
"""

from .client import MemoryClient, MemoryClientError
from .state import read_state, DEFAULT_PORT


def get_client(timeout: float = 60.0) -> MemoryClient:
    """
    Returns a connected MemoryClient pointed at the running daemon.
    Raises MemoryClientError if no daemon is running.
    """
    return MemoryClient.from_state(timeout=timeout)


def is_running() -> bool:
    """True if a daemon state file exists (daemon may or may not be reachable)."""
    state = read_state()
    if not state:
        return False
    try:
        client = MemoryClient(
            host=state["host"], port=state["port"], timeout=1.0
        )
        client.connect()
        client.ping()
        client.close()
        return True
    except Exception:
        return False


def _get_daemon_class():
    from .server import MemoryDaemon
    return MemoryDaemon


__all__ = [
    "MemoryClient", "MemoryClientError",
    "get_client", "is_running",
    "DEFAULT_PORT",
]
