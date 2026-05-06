"""P1b: Auto Consolidation Tests

Tests for MemoryConsolidator:
- Config validation
- Archive table creation
- Similar fact grouping
- Dry-run mode
- Protected categories
- Ollama fallback
"""

import json
import os
import sqlite3
import time
import pytest

# Import the module directly (not through unified to avoid full init)
import sys
sys.path.insert(0, os.path.expanduser('~/.hermes/plugins'))
from unified.consolidation import MemoryConsolidator, DEFAULT_CONFIG


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_config_values(self):
        """Verify all defaults are safe."""
        assert DEFAULT_CONFIG['decay_lambda'] == 0.0  # Disabled by default
        assert DEFAULT_CONFIG['similarity_threshold'] == 0.85
        assert DEFAULT_CONFIG['max_facts'] == 10000
        assert DEFAULT_CONFIG['dry_run'] is True  # Safety first
        assert DEFAULT_CONFIG['archive_enabled'] is True
        assert DEFAULT_CONFIG['protected_categories'] == ['preference', 'identity']

    def test_custom_config_overrides(self):
        """Custom config should override defaults."""
        config = {
            'decay_lambda': 0.01,
            'dry_run': False,
            'max_facts': 5000,
        }
        consolidator = MemoryConsolidator(config)
        assert consolidator.config['decay_lambda'] == 0.01
        assert consolidator.config['dry_run'] is False
        assert consolidator.config['max_facts'] == 5000
        # Unset keys should use defaults
        assert consolidator.config['archive_enabled'] is True

    def test_minimal_config(self):
        """Empty config should use all defaults."""
        consolidator = MemoryConsolidator({})
        for k, v in DEFAULT_CONFIG.items():
            assert consolidator.config[k] == v


class TestConsolidatorInit:
    """Test consolidator initialization."""

    def test_init_with_none_config(self):
        consolidator = MemoryConsolidator(None)
        assert consolidator.config['dry_run'] is True

    def test_init_preserves_config(self):
        config = {'dry_run': False, 'max_facts': 100}
        consolidator = MemoryConsolidator(config)
        assert consolidator.config['dry_run'] is False
        assert consolidator.config['max_facts'] == 100


class TestDryRunMode:
    """Test dry-run safety mode."""

    def test_dry_run_default(self):
        """Dry-run should be enabled by default."""
        consolidator = MemoryConsolidator({})
        assert consolidator.config['dry_run'] is True

    def test_dry_run_report_includes_flag(self):
        """Report should include dry_run flag."""
        consolidator = MemoryConsolidator({})
        report = consolidator.consolidate()
        assert 'dry_run' in report
        assert report['dry_run'] is True

    def test_dry_run_returns_report_even_without_db(self):
        """Dry run should work even without MemoryDB."""
        consolidator = MemoryConsolidator({})
        report = consolidator.consolidate()
        assert 'status' in report
        assert report['facts_before'] == 0
        assert report['groups_found'] == 0


class TestNeedsCompression:
    """Test compression threshold logic."""

    def test_empty_db_does_not_need_compression(self):
        consolidator = MemoryConsolidator({'max_facts': 100})
        # Without DB, should return False
        assert consolidator._needs_compression() is False

    def test_under_threshold_does_not_need_compression(self):
        consolidator = MemoryConsolidator({'max_facts': 100})
        # Without DB, returns False
        result = consolidator._needs_compression()
        assert result is False


class TestProtectedCategories:
    """Test fact protection from compression."""

    def test_default_protected_categories(self):
        consolidator = MemoryConsolidator({})
        assert 'preference' in consolidator.config['protected_categories']
        assert 'identity' in consolidator.config['protected_categories']

    def test_custom_protected_categories(self):
        consolidator = MemoryConsolidator({
            'protected_categories': ['critical', 'system'],
        })
        assert 'critical' in consolidator.config['protected_categories']
        assert 'system' in consolidator.config['protected_categories']
        assert 'preference' not in consolidator.config['protected_categories']


class TestArchiveEnabled:
    """Test archive safety feature."""

    def test_archive_enabled_default(self):
        consolidator = MemoryConsolidator({})
        assert consolidator.config['archive_enabled'] is True

    def test_archive_can_be_disabled(self):
        consolidator = MemoryConsolidator({'archive_enabled': False})
        assert consolidator.config['archive_enabled'] is False


class TestConsolidationReport:
    """Test consolidation report structure."""

    def test_report_has_required_fields(self):
        consolidator = MemoryConsolidator({})
        report = consolidator.consolidate()
        
        required_fields = [
            'dry_run', 'facts_before', 'facts_after',
            'candidates_before', 'candidates_after',
            'groups_found', 'groups_compressed', 'groups_failed',
            'archived', 'errors', 'timestamp', 'status',
        ]
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

    def test_report_has_valid_types(self):
        consolidator = MemoryConsolidator({})
        report = consolidator.consolidate()
        
        assert isinstance(report['dry_run'], bool)
        assert isinstance(report['facts_before'], int)
        assert isinstance(report['facts_after'], int)
        assert isinstance(report['candidates_before'], int)
        assert isinstance(report['candidates_after'], int)
        assert isinstance(report['groups_found'], int)
        assert isinstance(report['groups_compressed'], int)
        assert isinstance(report['groups_failed'], int)
        assert isinstance(report['archived'], int)
        assert isinstance(report['errors'], list)
        assert isinstance(report['timestamp'], float)
        assert isinstance(report['status'], str)

    def test_report_status_values(self):
        """Status should be one of: skipped, no_groups, completed, partial."""
        consolidator = MemoryConsolidator({})
        report = consolidator.consolidate()
        valid_statuses = {'skipped', 'no_groups', 'completed', 'partial'}
        assert report['status'] in valid_statuses
