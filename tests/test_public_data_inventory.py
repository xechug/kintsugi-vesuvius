import json
import os
import tempfile
import unittest
from pathlib import Path


class PublicDataInventoryTests(unittest.TestCase):
    def test_aws_command_is_unsigned_and_has_no_profile(self):
        from scripts.public_data_inventory import aws_list_command, unsigned_aws_environment

        command = aws_list_command("PHercParis4/segments/")
        environment = unsigned_aws_environment(
            {
                "PATH": os.environ.get("PATH", ""),
                "AWS_PROFILE": "corporate",
                "AWS_DEFAULT_PROFILE": "corporate",
                "AWS_ACCESS_KEY_ID": "secret",
            }
        )

        self.assertIn("--no-sign-request", command)
        self.assertNotIn("--profile", command)
        self.assertFalse(any(key.startswith("AWS_") for key in environment))

    def test_prefix_and_keys_reject_traversal_or_full_s3_urls(self):
        from scripts.public_data_inventory import validate_object_key

        for invalid in ("../secret", "a/../../secret", "/absolute", "s3://other/key"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                validate_object_key(invalid)
        self.assertEqual(validate_object_key("PHercParis4/segments/file.json"), "PHercParis4/segments/file.json")

    def test_download_plan_enforces_byte_budget_and_free_space(self):
        from scripts.public_data_inventory import ObjectInfo, plan_downloads

        objects = [
            ObjectInfo("a.bin", 60, "etag-a", "2026-01-01"),
            ObjectInfo("b.bin", 50, "etag-b", "2026-01-02"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "budget"):
                plan_downloads(objects, Path(tmp), max_bytes=100, free_bytes=1_000)
            with self.assertRaisesRegex(ValueError, "free disk"):
                plan_downloads(objects, Path(tmp), max_bytes=200, free_bytes=100)

            plan = plan_downloads(objects, Path(tmp), max_bytes=200, free_bytes=1_000)

            self.assertEqual(plan.total_bytes, 110)
            self.assertEqual(len(plan.items), 2)

    def test_s3_listing_parser_preserves_provenance_fields(self):
        from scripts.public_data_inventory import parse_s3_listing

        payload = {
            "Contents": [
                {
                    "Key": "PHercParis4/segments/a/meta.json",
                    "Size": 123,
                    "ETag": '"abc123"',
                    "LastModified": "2026-07-01T00:00:00+00:00",
                }
            ]
        }

        objects = parse_s3_listing(json.dumps(payload))

        self.assertEqual(objects[0].key, "PHercParis4/segments/a/meta.json")
        self.assertEqual(objects[0].size, 123)
        self.assertEqual(objects[0].etag, "abc123")


if __name__ == "__main__":
    unittest.main()
