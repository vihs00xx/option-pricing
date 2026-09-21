import json
import unittest
from pathlib import Path

import numpy as np

from src.common_benchmark import PARAMETERS, build_cases, reference_values


ROOT = Path(__file__).resolve().parents[1]


class CommonBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings = json.loads(
            (ROOT / "experiments" / "config" / "classical"
             / "common_benchmark_settings.json").read_text()
        )
        cls.cases = build_cases(cls.settings)

    def test_case_count_identity_and_balance(self):
        self.assertEqual(len(self.cases), 68)
        self.assertEqual(len({case["case_id"] for case in self.cases}), 68)
        self.assertEqual(sum(case["option"] == "call" for case in self.cases), 34)
        self.assertEqual(self.cases, build_cases(self.settings))

    def test_interpolation_and_extrapolation_splits(self):
        domain = self.settings["training_domain"]
        for case in self.cases:
            outside = sum(
                not domain[name][0] <= float(case[name]) <= domain[name][1]
                for name in PARAMETERS
            )
            if case["split"] == "interpolation":
                self.assertEqual(outside, 0)
            elif case["split"] == "extrapolation":
                self.assertEqual(outside, 1)
            self.assertEqual(case["excluded_from_training"], 1)

    def test_references_are_finite(self):
        for case in self.cases:
            values = reference_values(case)
            self.assertTrue(all(np.isfinite(value) for value in values.values()))


if __name__ == "__main__":
    unittest.main()
