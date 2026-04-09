"""Unit tests for optimisation.utils storage helpers."""
import os
import tempfile
import unittest

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from optimisation.utils import (
    setup_study_storage,
    _is_remote_storage,
    _load_env_file,
    _build_remote_url,
    create_storage,
)


def _has_psycopg2():
    try:
        import psycopg2  # noqa: F401
        return True
    except ImportError:
        return False


class TestLoadEnvFile(unittest.TestCase):
    """Tests for _load_env_file()."""

    def test_missing_file_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _load_env_file(tmpdir)
            self.assertEqual(result, {})

    def test_parses_key_value_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("OPTUNA_DB_USER=alice\n")
                f.write("OPTUNA_DB_PASSWORD=secret\n")
            result = _load_env_file(tmpdir)
            self.assertEqual(result["OPTUNA_DB_USER"], "alice")
            self.assertEqual(result["OPTUNA_DB_PASSWORD"], "secret")

    def test_strips_quotes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write('OPTUNA_DB_USER="alice"\n')
                f.write("OPTUNA_DB_PASSWORD='secret'\n")
            result = _load_env_file(tmpdir)
            self.assertEqual(result["OPTUNA_DB_USER"], "alice")
            self.assertEqual(result["OPTUNA_DB_PASSWORD"], "secret")

    def test_skips_comments_and_blank_lines(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("# comment\n\nKEY=val\n")
            result = _load_env_file(tmpdir)
            self.assertEqual(result, {"KEY": "val"})


class TestBuildRemoteUrl(unittest.TestCase):
    """Tests for _build_remote_url()."""

    def test_missing_env_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                _build_remote_url(tmpdir)

    def test_missing_keys_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("OPTUNA_DB_USER=alice\n")
            with self.assertRaises(KeyError):
                _build_remote_url(tmpdir)

    def test_builds_url_with_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("OPTUNA_DB_USER=alice\n")
                f.write("OPTUNA_DB_PASSWORD=secret\n")
                f.write("OPTUNA_DB_HOST=db.example.com\n")
                f.write("OPTUNA_DB_NAME=mydb\n")
            url = _build_remote_url(tmpdir)
            self.assertEqual(
                url,
                "postgresql://alice:secret@db.example.com:5432/mydb?sslmode=require",
            )

    def test_custom_port_and_sslmode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("OPTUNA_DB_USER=alice\n")
                f.write("OPTUNA_DB_PASSWORD=secret\n")
                f.write("OPTUNA_DB_HOST=db.example.com\n")
                f.write("OPTUNA_DB_NAME=mydb\n")
                f.write("OPTUNA_DB_PORT=5433\n")
                f.write("OPTUNA_DB_SSLMODE=disable\n")
            url = _build_remote_url(tmpdir)
            self.assertIn(":5433/", url)
            self.assertIn("sslmode=disable", url)


class TestSetupStudyStorage(unittest.TestCase):
    """Tests for setup_study_storage()."""

    def test_local_backend_creates_sqlite(self):
        """storage_backend='local' returns a sqlite:/// URL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = setup_study_storage("local", tmpdir)
            self.assertTrue(url.startswith("sqlite:///"))
            self.assertIn("all_my_studies.db", url)
            db_dir = os.path.join(tmpdir, "optimisation", "database")
            self.assertTrue(os.path.isdir(db_dir))

    def test_default_is_local(self):
        """Omitted backend defaults to local SQLite."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = setup_study_storage("local", tmpdir)
            self.assertTrue(url.startswith("sqlite:///"))

    def test_remote_backend_reads_env_file(self):
        """storage_backend='remote' builds URL from .env credentials."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = os.path.join(tmpdir, ".env")
            with open(env_path, "w") as f:
                f.write("OPTUNA_DB_USER=alice\n")
                f.write("OPTUNA_DB_PASSWORD=secret\n")
                f.write("OPTUNA_DB_HOST=db.example.com\n")
                f.write("OPTUNA_DB_NAME=mydb\n")
            url = setup_study_storage("remote", tmpdir)
            self.assertTrue(url.startswith("postgresql://"))
            self.assertIn("alice", url)
            self.assertIn("db.example.com", url)

    def test_remote_backend_missing_env_raises(self):
        """storage_backend='remote' without .env raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                setup_study_storage("remote", tmpdir)

    def test_cli_storage_overrides_backend(self):
        """--storage CLI arg overrides the config backend."""
        pg_url = "postgresql://cli@host/db"
        url = setup_study_storage("local", "/repo", cli_storage=pg_url)
        self.assertEqual(url, pg_url)

    def test_explicit_sqlite_via_cli(self):
        """Explicit sqlite:/// path via CLI is resolved relative to project root."""
        with tempfile.TemporaryDirectory() as tmpdir:
            url = setup_study_storage(
                "remote", tmpdir,
                cli_storage="sqlite:///optimisation/database/test.db",
            )
            self.assertTrue(url.startswith("sqlite:///"))
            self.assertIn(tmpdir, url)
            self.assertTrue(url.endswith("test.db"))

    def test_absolute_sqlite_via_cli_unchanged(self):
        """Absolute sqlite:/// paths via CLI are returned as-is."""
        url = setup_study_storage(
            "local", "/repo",
            cli_storage="sqlite:////tmp/abs_test.db",
        )
        self.assertEqual(url, "sqlite:////tmp/abs_test.db")

    def test_unsupported_cli_url_raises(self):
        """Unsupported storage URLs via CLI raise ValueError."""
        with self.assertRaises(ValueError):
            setup_study_storage("local", "/repo", cli_storage="mysql://user:pass@host/db")


class TestIsRemoteStorage(unittest.TestCase):
    """Tests for _is_remote_storage()."""

    def test_sqlite_is_not_remote(self):
        self.assertFalse(_is_remote_storage("sqlite:///path/to/db.db"))

    def test_postgresql_is_remote(self):
        self.assertTrue(_is_remote_storage("postgresql://user:pass@host/db"))

    def test_postgres_shorthand_is_remote(self):
        self.assertTrue(_is_remote_storage("postgres://user:pass@host/db"))

    def test_arbitrary_string_is_not_remote(self):
        self.assertFalse(_is_remote_storage("mysql://user:pass@host/db"))
        self.assertFalse(_is_remote_storage("some_typo://url"))


class TestCreateStorage(unittest.TestCase):
    """Tests for create_storage()."""

    def test_sqlite_returns_string(self):
        result = create_storage("sqlite:///path/to/db.db")
        self.assertIsInstance(result, str)
        self.assertEqual(result, "sqlite:///path/to/db.db")

    @unittest.skipUnless(_has_psycopg2(), "psycopg2 not installed")
    def test_postgresql_returns_rdb_storage(self):
        import optuna
        result = create_storage("postgresql://user:pass@localhost/db")
        self.assertIsInstance(result, optuna.storages.RDBStorage)


if __name__ == "__main__":
    unittest.main()
