"""PINN pricer: Greeks, PDE losses and a short training run."""

import unittest

import torch

from src.neural.pinn import PINN
from src.neural.reference import price_and_greeks
from src.neural.training import Trainer, score
from tests.neural_helpers import DOMAIN, SHORT_RUN, STRIKE, contracts, finite_difference_check, monitor_panel, \
    small_model


class PINNTests(unittest.TestCase):
    def test_autograd_greeks_match_finite_differences(self):
        x, put = contracts(32, seed=9)
        for layout, output, similarity in (("raw", "plain", None), ("reduced", "plain", None),
                                           ("reduced", "levy", None), ("reduced", "levy", "levy"),
                                           ("reduced", "levy_envelope", "levy")):
            model = small_model("pinn", layout, output, similarity=similarity)
            self.assertIsInstance(model, PINN)
            finite_difference_check(self, model, x, put)

    def test_similarity_residual_stays_finite_at_expiry(self):
        from src.neural.coordinates import reduced_coordinates
        from src.neural.pde import reduced_residual
        model = small_model("pinn", similarity="levy")
        x, _ = contracts(16, seed=5)
        features, _ = reduced_coordinates(torch.as_tensor(x))
        features[:, 2] = 1e-6
        features[:8, 0] = 0.0
        residual, _ = reduced_residual(model.call, features)
        self.assertTrue(torch.isfinite(residual).all())

    def test_short_runs_reduce_monitor_score(self):
        monitor = monitor_panel()
        for layout, output, loss in (("reduced", "levy", {"pde": 1.0, "boundary": 1.0}),
                                     ("raw", "plain", {"pde": 1.0, "terminal": 1.0})):
            model = small_model("pinn", layout, output)
            before = score(price_and_greeks(model.price, *monitor["inputs"]), monitor["reference"])[0]
            trainer = Trainer(model, dict(SHORT_RUN, loss=loss), DOMAIN, STRIKE, monitor, None, log=lambda *_: None)
            record = trainer.run(seed=0)
            self.assertLess(record["best_monitor"], before)
            stages = {entry["stage"] for entry in record["history"]}
            self.assertTrue({"adam", "lbfgs", "nncg"} <= stages, stages)

    def test_interrupted_training_resumes_identically(self):
        import tempfile
        from pathlib import Path
        monitor = monitor_panel(16)
        settings = dict(SHORT_RUN, loss={"pde": 1.0, "boundary": 1.0},
                        optimizer=dict(SHORT_RUN["optimizer"], adam_steps=40, check_every=20, lbfgs_iterations=0,
                                       nncg_steps=0))
        straight = small_model("pinn")
        Trainer(straight, settings, DOMAIN, STRIKE, monitor, None, log=lambda *_: None).run(seed=3)

        def interrupt(message):
            if message.startswith("adam 20:"):
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"resume.pt"
            resumed = small_model("pinn")
            with self.assertRaises(KeyboardInterrupt):
                Trainer(resumed, settings, DOMAIN, STRIKE, monitor, None, log=interrupt).run(seed=3, resume_path=path)
            resumed = small_model("pinn")
            Trainer(resumed, settings, DOMAIN, STRIKE, monitor, None, log=lambda *_: None).run(seed=3, resume_path=path)
        for a, b in zip(straight.state_dict().values(), resumed.state_dict().values()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
