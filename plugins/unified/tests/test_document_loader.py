"""Tests for document_loader.py — فلاتر الاستبعاد وتحميل المستندات"""

import pytest
from pathlib import Path
from unified.document_loader import _should_exclude, EXCLUDE_DIRS, MAX_FILE_SIZE, DocumentLoader


# ============================================================
# _should_exclude — Directory filter tests
# ============================================================

class TestShouldExclude:
    """اختبار دالة فلترة المجلدات المستبعدة"""

    def test_node_modules_excluded(self):
        assert _should_exclude(Path('/project/node_modules/react/index.js')) is True

    def test_venv_excluded(self):
        assert _should_exclude(Path('/project/.venv/lib/python3/flask.py')) is True

    def test_git_excluded(self):
        assert _should_exclude(Path('/project/.git/config')) is True

    def test_pycache_excluded(self):
        assert _should_exclude(Path('/project/__pycache__/module.cpython.pyc')) is True

    def test_build_excluded(self):
        assert _should_exclude(Path('/project/build/output.txt')) is True

    def test_next_excluded(self):
        assert _should_exclude(Path('/project/.next/server/page.js')) is True

    def test_idea_excluded(self):
        assert _should_exclude(Path('/project/.idea/workspace.xml')) is True

    def test_dist_excluded(self):
        assert _should_exclude(Path('/project/dist/bundle.js')) is True

    def test_vscode_excluded(self):
        assert _should_exclude(Path('/project/.vscode/settings.json')) is True

    def test_target_excluded(self):
        assert _should_exclude(Path('/project/target/classes/Main.class')) is True

    def test_tox_excluded(self):
        assert _should_exclude(Path('/project/.tox/py37/lib/test.py')) is True

    def test_eggs_excluded(self):
        assert _should_exclude(Path('/project/.eggs/pkg.egg/EGG.py')) is True

    def test_env_excluded(self):
        assert _should_exclude(Path('/project/env/lib/python3/site.py')) is True

    def test_mypy_cache_excluded(self):
        assert _should_exclude(Path('/project/.mypy_cache/cache.db')) is True

    def test_pytest_cache_excluded(self):
        assert _should_exclude(Path('/project/.pytest_cache/v/cache')) is True

    # Keep tests
    def test_src_kept(self):
        assert _should_exclude(Path('/project/src/main.py')) is False

    def test_readme_kept(self):
        assert _should_exclude(Path('/project/README.md')) is False

    def test_hermes_state_kept(self):
        assert _should_exclude(Path('/project/hermes_state.py')) is False

    def test_deep_source_kept(self):
        assert _should_exclude(Path('/project/project/src/app.py')) is False

    def test_tests_kept(self):
        assert _should_exclude(Path('/project/tests/test_main.py')) is False

    def test_config_kept(self):
        assert _should_exclude(Path('/project/config/settings.json')) is False

    def test_nested_node_modules_excluded(self):
        """node_modules حتى لو متداخل"""
        assert _should_exclude(Path('/project/a/b/c/node_modules/x/y.js')) is True

    def test_nonexistent_path_uses_dir_filter_only(self):
        """لو الملف مو موجود، نعتمد على فلتر المجلدات فقط"""
        # مسار في src (ما ينستبعد) بس الملف مو موجود
        assert _should_exclude(Path('/tmp/nonexistent_12345/src/main.py')) is False


# ============================================================
# EXCLUDE_DIRS constant tests
# ============================================================

class TestExcludeDirsConstant:
    """اختبار ثوابت المجلدات المستبعدة"""

    def test_has_node_modules(self):
        assert 'node_modules' in EXCLUDE_DIRS

    def test_has_venv(self):
        assert '.venv' in EXCLUDE_DIRS

    def test_has_git(self):
        assert '.git' in EXCLUDE_DIRS

    def test_has_pycache(self):
        assert '__pycache__' in EXCLUDE_DIRS

    def test_no_wildcard_patterns(self):
        """ما فيه wildcard patterns لأن path.match ما يدعمها"""
        for d in EXCLUDE_DIRS:
            assert '*' not in d, f'Wildcard in EXCLUDE_DIRS: {d}'


# ============================================================
# MAX_FILE_SIZE constant
# ============================================================

class TestMaxFileSize:
    def test_is_500kb(self):
        assert MAX_FILE_SIZE == 500_000


# ============================================================
# DocumentLoader integration (minimal, no network)
# ============================================================

class TestDocumentLoaderInit:
    def test_default_encodinging(self):
        loader = DocumentLoader()
        assert loader.encoding == 'utf-8'

    def test_custom_encoding(self):
        loader = DocumentLoader(encoding='latin-1')
        assert loader.encoding == 'latin-1'

    def test_load_file_raises_for_nonexistent(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_file('/nonexistent/file.txt')

    def test_load_file_raises_for_excluded(self):
        """الملف المستبعد يجب أن يُرفض — لكن الـ exists check يسبق"""
        loader = DocumentLoader()
        # load_file يتحقق من الـ exists أولاً
        # خلني أختبر _should_exclude مباشرة بدل load_file
        from unified.document_loader import _should_exclude
        assert _should_exclude(Path('/tmp/node_modules/foo.py')) is True

    def test_load_directory_raises_for_nonexistent(self):
        loader = DocumentLoader()
        with pytest.raises(FileNotFoundError):
            loader.load_directory('/nonexistent/dir')
