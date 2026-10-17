"""Cached neural data sets are keyed by everything they are built from."""

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from experiments.neural import common


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        patcher = mock.patch.object(common, "CACHE", Path(self.folder.name))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.builds = 0

    def build(self):
        self.builds += 1
        return dict(value=torch.tensor(float(self.builds)))

    def test_same_identity_loads_the_saved_set(self):
        first = common._cached("data_8_1", dict(nodes=32), self.build)
        second = common._cached("data_8_1", dict(nodes=32), self.build)
        self.assertEqual(self.builds, 1)
        self.assertEqual(float(second["value"]), float(first["value"]))

    def test_changed_settings_build_a_new_set(self):
        common._cached("data_8_1", dict(nodes=32), self.build)
        changed = common._cached("data_8_1", dict(nodes=64), self.build)
        self.assertEqual(self.builds, 2)
        self.assertEqual(float(changed["value"]), 2.0)

    def test_changed_reference_code_builds_a_new_set(self):
        common._cached("data_8_1", dict(nodes=32), self.build)
        sources = dict(common.label_sources(), reference="0"*64)
        with mock.patch.object(common, "label_sources", return_value=sources):
            common._cached("data_8_1", dict(nodes=32), self.build)
        self.assertEqual(self.builds, 2)

    def test_a_file_with_another_identity_is_refused(self):
        common._cached("data_8_1", dict(nodes=32), self.build)
        (path,) = Path(self.folder.name).glob("data_8_1_*.pt")
        value = torch.load(path, weights_only=False)
        value["identity"]["nodes"] = 64
        torch.save(value, path)
        with self.assertRaises(RuntimeError):
            common._cached("data_8_1", dict(nodes=32), self.build)


if __name__ == "__main__":
    unittest.main()
