"""Test configuration — handles path setup for both local and CI environments."""

import sys
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ============================================================
# Path setup: works on local machine AND GitHub Actions CI
# ============================================================

# CI environment: project root is the checkout directory
# Local environment: plugin lives in ~/.hermes/plugins/
_plugin_dir = Path(__file__).resolve().parent.parent
if str(_plugin_dir) not in sys.path:
    sys.path.insert(0, str(_plugin_dir))

# For imports like 'from unified import ...' we need the parent of 'unified/'
_parent_dir = _plugin_dir.parent
if str(_parent_dir) not in sys.path:
    sys.path.insert(0, str(_parent_dir))


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def clear_model_singletons():
    """Clear embedding model singleton cache before each test."""
    try:
        from unified.embedding_model import _MODEL_SINGLETON
        _MODEL_SINGLETON.clear()
    except ImportError:
        pass
    yield
    try:
        from unified.embedding_model import _MODEL_SINGLETON
        _MODEL_SINGLETON.clear()
    except ImportError:
        pass


@pytest.fixture
def tmp_memory_db(tmp_path):
    """Create a temporary MemoryDB for integration tests."""
    from unified.memory_db import _get_memory_db
    # Reset singleton so tests use the temp DB
    import unified.memory_db as mem_db_module
    mem_db_module._memory_db = None

    # Create a mock db path — tests that need real SQLite can override
    yield tmp_path / "test_memory.db"

    # Cleanup
    mem_db_module._memory_db = None
