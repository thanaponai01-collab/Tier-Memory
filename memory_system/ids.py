"""
ULID generation — time-ordered, lexicographically sortable unique IDs.
Falls back to UUID if the `python-ulid` package is not installed.
"""

import time
import os
import uuid


def new_id() -> str:
    try:
        from ulid import ULID
        return str(ULID())
    except ImportError:
        # Fallback: timestamp prefix + random suffix, same lexicographic ordering
        ts = format(int(time.time() * 1000), "013x")
        rand = os.urandom(10).hex()
        return f"{ts}{rand}"
