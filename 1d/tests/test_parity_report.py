"""Regression tests for the corrected parity report generator.

The generated conclusions must come from metrics: counts from selection
records, improvements and regressions measured, variability computed, and no
unconditional claims that could survive a results change.
"""

import unittest

from experiments.deeponet.compare_supervision import NAMES
from experiments.deeponet.generate_corrected_reports import build_corrected_parity_report


def entry(architecture, seed, selected, validation, stress_theta, gamma_freq):
    shaped = {constraint: {'proportion_above_1e_8': 0.05, 'maximum_violation': 0.1}
              for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma')}
    shaped['negative_gamma']['proportion_above_1e_8'] = gamma_freq
    metrics = {name: {'rmse': value} for name, value in validation.items()}
    fresh = {name: {'rmse': value} for name, value in validation.items()}
    fresh['theta'] = {'rmse': stress_theta}
    return {'architecture': architecture, 'seed': seed, 'selected': selected,
            'selection_recorded': selected, 'selection_recomputed': selected,
            'selection_confirmed': True, 'eligible_recomputed': ['baseline', selected],
            'validation': {'baseline': {**metrics, 'price': {'rmse': 1.0}},
                           selected: metrics},
            'fresh': {'in_domain': fresh, 'stress': fresh},
            'shape': {'validation': shaped, 'in_domain': shaped, 'stress': shaped},
            'training': {selected: {'optimizer_steps': 12800, 'training_wall_seconds': 10.,
                                    'stop_reason': 'epochs_completed'}},
            'verification': {'parent': {'status': 'verified', 'checkpoint_verified': True,
                                        'dataset_verified': ['file_sha256'], 'scope': 'train_'},
                             selected: {'status': 'verified', 'checkpoint_verified': True,
                                        'dataset_verified': ['trusted_file_sha256'], 'scope': 'train_'}}}


class CorrectedParityReportTests(unittest.TestCase):
    def test_mixed_improvements_and_regressions_are_reported(self):
        base = {name: 1.0 for name in NAMES}
        runs = []
        for seed, prices in zip((22092026, 22092027, 22092028),
                                ((0.10, 0.08), (0.11, 0.081), (0.12, 0.079))):
            control_validation = {**base, 'price': prices[0]}
            parity_validation = {**base, 'price': prices[1]}
            runs.append(entry('control', seed, 'lbfgs', control_validation, 1.0, 0.05))
            runs.append(entry('parity', seed, 'adam_small_steps', parity_validation, 1.4, 0.09))
        text = build_corrected_parity_report({'runs': runs, 'family_counts':
                                              {'adam_small_steps': 2, 'lbfgs': 1}})
        self.assertIn('Adam 3, L-BFGS 3', text)  # counts generated from selection records
        self.assertIn('regresses', text)          # stress theta and gamma frequency regressions
        self.assertIn('theta RMSE on stress: regresses', text)
        self.assertIn('negative_gamma frequency on stress: regresses', text)
        self.assertIn('lower sample standard deviation', text)
        self.assertIn('does not guarantee price bounds', text)
        for forbidden in ('unambiguous', 'every quantity improves', 'results unchanged',
                          'remains unchanged', 'establishes general seed robustness'):
            self.assertNotIn(forbidden, text)

    def test_all_improvements_report_no_regressions(self):
        base = {name: 1.0 for name in NAMES}
        runs = []
        for seed in (22092026, 22092027, 22092028):
            runs.append(entry('control', seed, 'lbfgs', {**base, 'price': 0.1}, 1.0, 0.05))
            runs.append(entry('parity', seed, 'lbfgs', {**base, 'price': 0.05}, 0.5, 0.02))
        text = build_corrected_parity_report({'runs': runs})
        self.assertIn('none measured at the 1% threshold', text)


if __name__ == '__main__':
    unittest.main()
