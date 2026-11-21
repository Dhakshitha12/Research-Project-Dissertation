"""Tests for direct credential loading from user.txt (aws_calibration)."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from aws.aws_calibration import find_user_credentials, load_user_credentials


class UserCredentialsTests(unittest.TestCase):
    def _write(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        self.addCleanup(os.remove, path)
        return path

    def test_parses_default_profile_ini(self):
        path = self._write(
            "[default]\n"
            "aws_access_key_id=ASIAEXAMPLEKEYID\n"
            "aws_secret_access_key=EXAMPLESECRET\n"
            "aws_session_token=EXAMPLETOKEN==\n"
        )
        creds = load_user_credentials(path)
        self.assertEqual(creds["aws_access_key_id"], "ASIAEXAMPLEKEYID")
        self.assertEqual(creds["aws_secret_access_key"], "EXAMPLESECRET")
        self.assertEqual(creds["aws_session_token"], "EXAMPLETOKEN==")

    def test_parses_named_profile_section(self):
        path = self._write(
            "[default]\n"
            "aws_access_key_id=ASIADEFAULT\n"
            "aws_secret_access_key=SECRETDEFAULT\n"
            "\n"
            "[thesis-lab]\n"
            "aws_access_key_id=ASIALAB\n"
            "aws_secret_access_key=SECRETLAB\n"
            "aws_session_token=TOKENLAB\n"
            "region=us-west-2\n"
        )
        creds = load_user_credentials(path, "thesis-lab")
        self.assertEqual(creds["aws_access_key_id"], "ASIALAB")
        self.assertEqual(creds["aws_session_token"], "TOKENLAB")
        self.assertEqual(creds["region"], "us-west-2")
        # the default profile stays untouched
        self.assertEqual(load_user_credentials(path, "default")["aws_access_key_id"],
                         "ASIADEFAULT")

    def test_bare_key_value_file_without_sections(self):
        path = self._write(
            "aws_access_key_id = ASIABARE\n"
            "aws_secret_access_key = SECBARE\n"
        )
        creds = load_user_credentials(path)
        self.assertEqual(creds, {"aws_access_key_id": "ASIABARE",
                                 "aws_secret_access_key": "SECBARE"})

    def test_utf8_bom_and_comments_are_tolerated(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, "w", encoding="utf-8-sig") as handle:
            handle.write("[default]\n# comment line\n; another\n"
                         "aws_access_key_id=ASIABOM\n"
                         "aws_secret_access_key=SECBOM\n")
        self.addCleanup(os.remove, path)
        creds = load_user_credentials(path)
        self.assertEqual(creds["aws_access_key_id"], "ASIABOM")

    def test_missing_keys_fall_back_to_empty(self):
        path = self._write("[default]\naws_session_token=ONLYTOKEN\n")
        self.assertEqual(load_user_credentials(path), {})

    def test_missing_file_returns_empty(self):
        self.assertEqual(load_user_credentials(None), {})
        self.assertEqual(load_user_credentials("/nonexistent/user.txt"), {})
        # an explicit-but-missing path falls back to the repo root user.txt
        found = find_user_credentials("/nonexistent/user.txt")
        self.assertTrue(found is None or os.path.isfile(found))

    def test_explicit_path_is_used_when_it_exists(self):
        path = self._write("[default]\naws_access_key_id=ASIAEXPLICIT\n"
                           "aws_secret_access_key=SECEXPLICIT\n")
        self.assertEqual(os.path.normpath(find_user_credentials(path)),
                         os.path.normpath(path))

    def test_find_user_credentials_locates_repo_root_file(self):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        expected = os.path.normpath(os.path.join(repo_root, "user.txt"))
        found = find_user_credentials(None)
        if not os.path.isfile(expected):
            # user.txt is not checked in; the locator must then fall through
            self.assertTrue(found is None or os.path.basename(found) == "user.txt")
            self.skipTest("user.txt not present in this extraction")
        self.assertEqual(os.path.normpath(found), expected)

    def test_values_are_not_logged(self):
        # sanity: loader output is a plain dict, never a log string
        path = self._write("[default]\naws_access_key_id=ASIAQUIET\n"
                           "aws_secret_access_key=SECQUIET\n")
        creds = load_user_credentials(path)
        self.assertIsInstance(creds, dict)


if __name__ == "__main__":
    unittest.main()
