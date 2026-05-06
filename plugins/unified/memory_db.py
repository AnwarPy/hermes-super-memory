"""MemoryDB interface and global model cache."""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# P4: MemoryDB SQLite reference (lazy init — only when needed)
_memory_db = None


def _get_memory_db():
    """Lazy-load MemoryDB singleton for graph search."""
    global _memory_db
    if _memory_db is None:
        try:
            _scripts_dir = os.path.expanduser("~/hermes-super-memory/scripts")
            if os.path.isdir(_scripts_dir) and _scripts_dir not in sys.path:
                sys.path.insert(0, _scripts_dir)
            from db import MemoryDB
            _memory_db = MemoryDB()
            _memory_db.init()
        except Exception:
            return None
    return _memory_db
