import json
import unittest
from pathlib import Path

import numpy as np

from src.common_benchmark import build_cases
from src.neural.deeponet_data import build_dataset


ROOT = Path(__file__).resolve().parents[1]


class DeepONetDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        settings = json.loads(
            (ROOT / "experiments/config/neural/deeponet_settings.json").read_text()
        )
        benchmark_settings = json.loads(
            (ROOT / "experiments/config/classical/common_benchmark_settings.json").read_text()
        )
        cls.settings = settings
        cls.cases = build_cases(benchmark_settings)
        cls.dataset = build_dataset(settings, cls.cases)

    def test_shapes_and_finite_targets(self):
        expected = {"train": 8192, "validation": 2048, "test": 68}
        for split, count in expected.items():
            values = self.dataset[split]
            self.assertEqual(values["branch"].shape, (count, 6))
            self.assertEqual(values["trunk"].shape, (count, 2))
            self.assertEqual(values["targets"].shape, (count, 6))
            self.assertTrue(np.isfinite(values["targets"]).all())

    def test_contract_splits_are_disjoint(self):
        train = {tuple(row) for row in self.dataset["train"]["branch"]}
        validation = {tuple(row) for row in self.dataset["validation"]["branch"]}
        test = {tuple(row) for row in self.dataset["test"]["branch"]}
        self.assertTrue(train.isdisjoint(validation))
        self.assertTrue(train.isdisjoint(test))
        self.assertTrue(validation.isdisjoint(test))

    def test_test_split_matches_frozen_benchmark(self):
        test = self.dataset["test"]
        self.assertEqual(test["case_id"].tolist(), [c["case_id"] for c in self.cases])
        self.assertEqual(sum(test["branch"][:, 5] == 1.0), 34)
        self.assertEqual(sum(test["branch"][:, 5] == -1.0), 34)

    def test_time_coordinates_respect_contract_horizon(self):
        for split in ("train", "validation", "test"):
            branch = self.dataset[split]["branch"]
            trunk = self.dataset[split]["trunk"]
            self.assertTrue(np.all(trunk[:, 1] > 0.0))
            self.assertTrue(np.all(trunk[:, 1] <= branch[:, 1]))

    def test_generation_is_deterministic(self):
        benchmark_settings = json.loads(
            (ROOT / "experiments/config/classical/common_benchmark_settings.json").read_text()
        )
        repeated = build_dataset(self.settings, build_cases(benchmark_settings))
        for split in self.dataset:
            for name in self.dataset[split]:
                np.testing.assert_array_equal(
                    self.dataset[split][name], repeated[split][name]
                )


if __name__ == "__main__":
    unittest.main()
