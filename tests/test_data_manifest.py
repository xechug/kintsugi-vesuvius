import json
import tempfile
import unittest
from pathlib import Path

from vesuvius_code.data_manifest import DataManifest, DataManifestError


class DataManifestTests(unittest.TestCase):
    def _write(self, root, sources):
        path = root / "manifest.json"
        path.write_text(json.dumps({"schema_version": 1, "sources": sources}))
        return path

    def test_manifest_exposes_train_and_external_validation_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = self._write(
                root,
                [
                    {
                        "name": "fragment1",
                        "zarr": "f1.zarr",
                        "mask": "f1-mask.png",
                        "ink": "f1-ink.png",
                        "split_manifest": "f1-split.json",
                        "splits": {"train": "train"},
                    },
                    {
                        "name": "fragment3",
                        "zarr": "f3.zarr",
                        "mask": "f3-mask.png",
                        "ink": "f3-ink.png",
                        "split_manifest": "f3-split.json",
                        "splits": {"validation": "validation", "test": "test"},
                    },
                ],
            )

            manifest = DataManifest.load(path)

            self.assertEqual([source.name for source in manifest.for_role("train")], ["fragment1"])
            self.assertEqual(
                [source.name for source in manifest.for_role("validation")],
                ["fragment3"],
            )
            self.assertEqual(manifest.sources[1].split_for("test"), "test")

    def test_manifest_rejects_duplicate_names_and_missing_validation(self):
        source = {
            "name": "duplicate",
            "zarr": "volume.zarr",
            "mask": "mask.png",
            "ink": "ink.png",
            "split_manifest": "split.json",
            "splits": {"train": "train"},
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            duplicate_path = self._write(root, [source, source])
            with self.assertRaises(DataManifestError):
                DataManifest.load(duplicate_path)

            missing_validation_path = self._write(root, [source])
            with self.assertRaisesRegex(DataManifestError, "validation"):
                DataManifest.load(missing_validation_path)


if __name__ == "__main__":
    unittest.main()
