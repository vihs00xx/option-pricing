"""DeepONet pricer: Greeks, derivative labels and a short training run."""

import unittest

import torch

from src.neural.coordinates import reduced_coordinates
from src.neural.deeponet import DeepONet
from src.neural.reference import derivative_labels, price_and_greeks
from src.neural.training import Trainer, labelled_data, score
from tests.neural_helpers import DOMAIN, SHORT_RUN, STRIKE, contracts, finite_difference_check, monitor_panel, \
    small_model


class DeepONetTests(unittest.TestCase):
    def test_autograd_greeks_match_finite_differences(self):
        x, put = contracts(32, seed=9)
        for layout, output, similarity in (("raw", "plain", None), ("reduced", "plain", None),
                                           ("reduced", "levy", None), ("reduced", "levy", 0.35),
                                           ("reduced", "levy_envelope", 0.35)):
            model = small_model("deeponet", layout, output, similarity=similarity)
            self.assertIsInstance(model, DeepONet)
            finite_difference_check(self, model, x, put)

    def test_branch_sees_only_parameters_and_trunk_only_coordinates(self):
        model = small_model("deeponet", "reduced", "plain")
        self.assertEqual(model.branch_index.tolist(), [3, 4, 5])
        self.assertEqual(model.trunk_index.tolist(), [0, 1, 2])
        model = small_model("deeponet", "reduced", "levy", similarity=0.35)
        self.assertEqual(model.trunk_index.tolist(), [0, 1, 2, 6])
        with self.assertRaises(ValueError):
            small_model("deeponet", "reduced", "levy", similarity="levy")

    def test_model_derivative_labels_match_finite_differences(self):
        model = small_model("deeponet")
        x, _ = contracts(32, seed=9)
        features, _ = reduced_coordinates(torch.as_tensor(x))
        labels = derivative_labels(model.call, features)
        h = 1e-6
        for column, label in ((2, 6), (3, 7), (5, 9)):
            up, down = features.clone(), features.clone()
            up[:, column] += h
            down[:, column] -= h
            fd = (model.call(up)-model.call(down))/(2*h)
            torch.testing.assert_close(labels[:, label], fd.detach(), rtol=1e-5, atol=1e-8)

    def test_short_runs_reduce_monitor_score(self):
        monitor = monitor_panel()
        data = labelled_data(DOMAIN, STRIKE, 256, seed=22)
        for layout, output, loss in (("reduced", "levy", {"price": 1.0, "derivatives": 1.0, "pde": 1.0}),
                                     ("raw", "plain", {"price": 1.0})):
            model = small_model("deeponet", layout, output)
            before = score(price_and_greeks(model.price, *monitor["inputs"]), monitor["reference"])[0]
            trainer = Trainer(model, dict(SHORT_RUN, loss=loss), DOMAIN, STRIKE, monitor, data, log=lambda *_: None)
            record = trainer.run(seed=0)
            self.assertLess(record["best_monitor"], before)


if __name__ == "__main__":
    unittest.main()
