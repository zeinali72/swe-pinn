"""Unit tests for optimisation.utils storage helpers."""
import os
import tempfile
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from optimisation.utils import setup_study_storage, _is_remote_storage, create_storage


def _has_psycopg2():
    try:
        import psycopg2  # noqa: F401
        return True
    except ImportError:
        return False


class TestSetupStudyStorage(unittest.TestCase):
    """Tests for setup_study_storage()."""

    def test_default_sqlite_when_no_args(self):
        """Returns a sqlite:/// URL under optimisation/database/ by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = setup_study_storage(None, tmpdir)
            self.assertTrue(url.startswith("sqlite:///"))
            self.assertIn("optimisation", url)
            self.assertIn("all_my_studies.db", url)
            # Directory should have been created
            db_dir = os.path.join(tmpdir, "optimisation", "database")
            self.assertTrue(os.path.isdir(db_dir))

    def test_explicit_sqlite(self):
        """Explicit sqlite:/// path is resolved relative to project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = setup_study_storage("sqlite:///optimisation/database/test.db", tmpdir)
            self.assertTrue(url.startswith("sqlite:///"))
            self.assertIn(tmpdir, url)
            self.assertTrue(url.endswith("test.db"))

    def test_absolute_sqlite_unchanged(self):
        """Absolute sqlite:/// paths are returned as-is."""
        url = setup_study_storage("sqlite:////tmp/abs_test.db", "/repo")
        self.assertEqual(url, "sqlite:////tmp/abs_test.db")

    def test_postgresql_passthrough(self):
        """PostgreSQL URLs are returned unchanged."""
        pg_url = "postgresql://user:pass@host/db"
        url = setup_study_storage(pg_url, "/repo")
        self.assertEqual(url, pg_url)

    def test_postgres_shorthand_passthrough(self):
        """postgres:// (shorthand) URLs are returned unchanged."""
        pg_url = "postgres://user:pass@host/db"
        url = setup_study_storage(pg_url, "/repo")
        self.assertEqual(url, pg_url)

    def test_unsupported_url_raises(self):
        """Unsupported storage URLs raise ValueError."""
        with self.assertRaises(ValueError):
            setup_study_storage("mysql://user:pass@host/db", "/repo")

    def test_env_var_fallback(self):
        """OPTUNA_STORAGE env var is used when args_storage is None."""
        pg_url = "postgresql://envuser:pass@host/db"
        old = os.environ.get("OPTUNA_STORAGE")
        try:
            os.environ["OPTUNA_STORAGE"] = pg_url
            url = setup_study_storage(None, "/repo")
            self.assertEqual(url, pg_url)
        finally:
            if old is None:
                os.environ.pop("OPTUNA_STORAGE", None)
            else:
                os.environ["OPTUNA_STORAGE"] = old

    def test_cli_arg_overrides_env_var(self):
        """--storage CLI arg takes precedence over OPTUNA_STORAGE env var."""
        old = os.environ.get("OPTUNA_STORAGE")
        try:
            os.environ["OPTUNA_STORAGE"] = "postgresql://env@host/db"
            url = setup_study_storage("postgresql://cli@host/db", "/repo")
            self.assertEqual(url, "postgresql://cli@host/db")
        finally:
            if old is None:
                os.environ.pop("OPTUNA_STORAGE", None)
            else:
                os.environ["OPTUNA_STORAGE"] = old


class TestIsRemoteStorage(unittest.TestCase):
    """Tests for _is_remote_storage()."""

    def test_sqlite_is_not_remote(self):
        self.assertFalse(_is_remote_storage("sqlite:///path/to/db.db"))

    def test_postgresql_is_remote(self):
        self.assertTrue(_is_remote_storage("postgresql://user:pass@host/db"))

    def test_postgres_shorthand_is_remote(self):
        self.assertTrue(_is_remote_storage("postgres://user:pass@host/db"))

    def test_arbitrary_string_is_not_remote(self):
        """Non-postgresql strings are not treated as remote."""
        self.assertFalse(_is_remote_storage("mysql://user:pass@host/db"))
        self.assertFalse(_is_remote_storage("some_typo://url"))


class TestCreateStorage(unittest.TestCase):
    """Tests for create_storage()."""

    def test_sqlite_returns_string(self):
        """SQLite URLs are returned as plain strings (Optuna handles them)."""
        result = create_storage("sqlite:///path/to/db.db")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "sqlite:///path/to/db.db")

    @unittest.skipUnless(
        _has_psycopg2(), "psycopg2 not installed"
    )
    def test_postgresql_returns_rdb_storage(self):
        """PostgreSQL URLs produce an optuna.storages.RDBStorage instance."""
        import optuna
        result = create_storage("postgresql://user:pass@localhost/db")
        self.assertIsInstance(result, optuna.storages.RDBStorage)


if __name__ == "__main__":
    unittest.main()
