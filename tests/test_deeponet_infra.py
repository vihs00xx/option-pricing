"""Regression tests for DeepONet experimental infrastructure.

Covers the shared sampling normalization, the metadata schema and evaluator
compatibility, strict resumption with atomic completion manifests, recovery of
finished training with missing evaluation outputs, and gate-driven report
generation. All artifacts live in temporary directories.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

import experiments.deeponet.run_training_ablation as ablation
from experiments.deeponet.evaluate import evaluate_checkpoint
from experiments.deeponet.train import ROOT
from experiments.deeponet.run_training_ablation import reports
from src.neural import run_metadata as runs
from src.neural.deeponet import BlackScholesDeepONet, ParityDeepONet


def make_dataset(directory, name, offset=0.):
    rng = np.random.default_rng(int(abs(offset) * 1000) + 7)
    arrays = {}
    for split, count in (('train', 8), ('validation', 4)):
        branch = np.column_stack((
            rng.uniform(80., 120., count), rng.uniform(.05, 2., count),
            rng.uniform(-.02, .1, count), rng.uniform(0., .06, count),
            rng.uniform(.1, .6, count), np.where(np.arange(count) % 2, 1., -1.)))
        trunk = np.column_stack((branch[:, 0] * rng.uniform(.8, 1.2, count),
                                 rng.uniform(.01, 1., count)))
        targets = rng.normal(offset, 1., (count, 6))
        arrays[f'{split}_branch'] = branch
        arrays[f'{split}_trunk'] = trunk
        arrays[f'{split}_targets'] = targets
        arrays[f'{split}_contract_id'] = np.arange(count)
    path = Path(directory) / name
    np.savez(path, **arrays)
    return path


def make_training_run(folder, dataset_record_, architecture, seed=1):
    """Simulate a finished training job that has not yet written evaluation outputs."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = (ParityDeepONet(width=4, depth=1, rank=4) if architecture.get('model') == 'parity'
             else BlackScholesDeepONet(width=4, depth=1, rank=4))
    record = runs.build_run_metadata(
        arm='fixture', seed=seed, stage='stage2', architecture=model.architecture,
        effective_settings={'seed': seed, 'epochs': 1}, settings_sha256='fixture',
        dataset=dataset_record_,
        validation_split={'arrays': ['validation_branch', 'validation_trunk', 'validation_targets'],
                          'rows': 4},
        normalization=None, parent=None, checkpoints={}, training={
            'optimizer_steps': 3, 'selected_epoch': 1, 'selected_validation_mse': 0.5,
            'training_wall_seconds': 0.25, 'stop_reason': 'epochs_completed'},
        label_columns_used=['price'], final_test_accessed=False)
    for kind in ('selected', 'last'):
        torch.save(dict(architecture=model.architecture, state_dict=model.state_dict(),
                        metadata={k: v for k, v in record.items() if k != 'checkpoints'}),
                   folder / f'{kind}.pt')
    record['checkpoints'] = {kind: runs.sha256(folder / f'{kind}.pt') for kind in ('selected', 'last')}
    (folder / 'history.csv').write_text('epoch,steps\n1,3\n')
    runs.save_json_atomic(folder / 'training_metadata.json', record)
    return record


def fixture_spec(folder, dataset_record_, architecture):
    return dict(name='fixture', stage='stage2', seed=1, folder=str(folder),
                settings={'seed': 1, 'epochs': 1}, settings_sha256='fixture',
                architecture=architecture, dataset=dataset_record_, normalization=None,
                penalty_types=[], penalty_weight=0., parent=None, source_sha256={'x.py': 'ab'})


class NormalizationTests(unittest.TestCase):
    def test_sampling_arms_share_scales_and_coefficients(self):
        with tempfile.TemporaryDirectory() as directory:
            uniform = make_dataset(directory, 'uniform.npz', offset=0.)
            mixed = make_dataset(directory, 'mixed.npz', offset=5.)
            records = ablation.compute_normalization({
                'uniform': runs.dataset_record(uniform, ROOT),
                'mixed': runs.dataset_record(mixed, ROOT),
                'frozen': runs.dataset_record(uniform, ROOT)})
            self.assertEqual(records['uniform']['greek_scales'], records['mixed']['greek_scales'])
            self.assertEqual(records['uniform']['effective_greek_loss_coefficients'],
                             records['mixed']['effective_greek_loss_coefficients'])
            self.assertEqual(records['uniform']['target_ordering'], list(runs.TARGET_ORDERING))
            self.assertIn('root mean square', records['mixed']['method'])
            self.assertEqual(records['mixed']['source_dataset']['file_sha256'],
                             runs.sha256(uniform))
            with np.load(mixed, allow_pickle=False) as data:
                own = (torch.tensor(data['train_targets'][:, 1:]).square().mean(0)
                       .sqrt().clamp_min(1e-12)).tolist()
            self.assertNotEqual(list(own), records['mixed']['greek_scales'])

    def test_job_specs_carry_identical_normalization(self):
        with tempfile.TemporaryDirectory() as directory:
            uniform = make_dataset(directory, 'uniform.npz', offset=0.)
            mixed = make_dataset(directory, 'mixed.npz', offset=5.)
            arrays = {key: runs.dataset_record(path, ROOT)
                      for key, path in (('uniform', uniform), ('mixed', mixed))}
            normal = ablation.compute_normalization({**arrays, 'frozen': arrays['uniform']})
            arm_spec = {'arch': 'base', 'data': 'uniform', 'penalties': ()}
            specs = [ablation.job_spec(f'{key}/22092026', 'stage2', {**arm_spec, 'data': key}, 22092026,
                                       f'arms/{key}', arrays, normal, {'x.py': 'ab'})
                     for key in ('uniform', 'mixed')]
            self.assertEqual(specs[0]['normalization'], specs[1]['normalization'])
            self.assertEqual(specs[0]['normalization']['effective_greek_loss_coefficients'],
                             specs[1]['normalization']['effective_greek_loss_coefficients'])


class MetadataAndEvaluatorTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.dataset = make_dataset(self.directory, 'data.npz')

    def tearDown(self):
        self.temporary.cleanup()

    def test_evaluator_runs_ordinary_and_parity_checkpoints(self):
        for architecture in ({'width': 4, 'depth': 1, 'rank': 4},
                             {'width': 4, 'depth': 1, 'rank': 4, 'model': 'parity'}):
            folder = self.directory / ('parity' if 'model' in architecture else 'plain')
            record = make_training_run(folder, runs.dataset_record(self.dataset, ROOT),
                                       architecture)
            result = evaluate_checkpoint(folder / 'selected.pt')
            self.assertEqual(result['split'], 'validation')
            self.assertIn('content_sha256', result['dataset']['verified_hash_kinds'])
            self.assertIn('file_sha256', result['dataset']['verified_hash_kinds'])
            self.assertIn('price', result['metrics']['all'])
            self.assertEqual(result['checkpoint_sha256'], record['checkpoints']['selected'])

    def test_wrong_dataset_hash_is_rejected(self):
        folder = self.directory / 'run'
        record = make_training_run(folder, runs.dataset_record(self.dataset, ROOT), {})
        record['dataset']['file_sha256'] = '0' * 64
        runs.save_json_atomic(folder / 'training_metadata.json', record)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            evaluate_checkpoint(folder / 'selected.pt')
        self.assertIn('dataset file hash mismatch', str(caught.exception))
        record = make_training_run(self.directory / 'run2',
                                   runs.dataset_record(self.dataset, ROOT), {})
        record['dataset'] = {'path': record['dataset']['path']}
        runs.save_json_atomic((self.directory / 'run2' / 'training_metadata.json'), record)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            evaluate_checkpoint(self.directory / 'run2' / 'selected.pt')
        self.assertIn('no hash of any kind', str(caught.exception))

    def test_wrong_checkpoint_hash_is_rejected(self):
        folder = self.directory / 'run'
        record = make_training_run(folder, runs.dataset_record(self.dataset, ROOT), {})
        record['checkpoints']['selected'] = '1' * 64
        runs.save_json_atomic(folder / 'training_metadata.json', record)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            evaluate_checkpoint(folder / 'selected.pt')
        self.assertIn('checkpoint hash mismatch', str(caught.exception))

    def test_train_scope_hash_never_covers_tampered_validation(self):
        """A legacy training-array hash must not validate tampered validation
        targets; the evaluated split needs its own trusted coverage."""
        folder = self.directory / 'run'
        record = make_training_run(folder, runs.dataset_record(self.dataset, ROOT), {})
        # degrade to a legacy-style record: training-array content hash only
        record['dataset'] = {'path': record['dataset']['path'],
                             'content_sha256': record['dataset']['content_sha256'],
                             'content_hash_scope': runs.CONTENT_HASH_SCOPE,
                             'content_hash_scope_prefix': 'train_'}
        runs.save_json_atomic(folder / 'training_metadata.json', record)
        with np.load(self.dataset, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays['validation_targets'] = arrays['validation_targets'] + 123.
        np.savez(self.dataset, **arrays)
        # training arrays untouched: the recorded content hash still matches,
        # yet the evaluated validation split is no longer the trained-on one
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            evaluate_checkpoint(folder / 'selected.pt')
        self.assertIn('remains unverified', str(caught.exception))
        # a wrong trusted hash is a mismatch, not a bypass
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, record['dataset'], evaluated_split='validation',
                                trusted_file_sha256='0' * 64)
        self.assertIn('trusted full-file hash mismatch', str(caught.exception))

    def test_trusted_file_hash_covers_evaluated_split(self):
        record = runs.dataset_record(self.dataset, ROOT, include_file_hash=False)
        checked = runs.verify_dataset(self.dataset, record, evaluated_split='validation',
                                      trusted_file_sha256=runs.sha256(self.dataset))
        self.assertIn('trusted_file_sha256', checked)
        with np.load(self.dataset, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        arrays['validation_targets'] = arrays['validation_targets'] + 123.
        np.savez(self.dataset, **arrays)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, record, evaluated_split='validation',
                                trusted_file_sha256=record.get('file_sha256') or '1' * 64)
        self.assertIn('trusted full-file hash mismatch', str(caught.exception))

    def test_recorded_file_hash_covers_evaluated_split(self):
        record = runs.dataset_record(self.dataset, ROOT)
        checked = runs.verify_dataset(self.dataset, record, evaluated_split='validation')
        self.assertIn('file_sha256', checked)

    def test_metadata_without_dataset_identity_fails_actionably(self):
        folder = self.directory / 'legacy'
        folder.mkdir()
        (folder / 'training_metadata.json').write_text(json.dumps(
            {'settings': {'seed': 1}, 'checkpoint_sha256': {'selected': '2' * 64}}))
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.load_run_metadata(folder)
        message = str(caught.exception)
        self.assertIn('cannot verify', message.replace('verification cannot be performed', 'cannot verify'))
        self.assertIn('start a new run directory', message)


class ResumptionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.dataset = make_dataset(self.directory, 'data.npz')
        self.dataset_record = runs.dataset_record(self.dataset, ROOT)
        self.folder = self.directory / 'run'
        self.record = make_training_run(self.folder, self.dataset_record, {})
        self.spec = fixture_spec('run', self.dataset_record, {})

    def tearDown(self):
        self.temporary.cleanup()

    def complete(self, spec=None):
        spec = spec or self.spec
        for name in runs.REQUIRED_ARTIFACTS:
            if not (self.folder / name).exists():
                (self.folder / name).write_text('placeholder\n')
        record = runs.load_run_metadata(self.folder)
        ablation.write_manifest(self.folder, spec, record)

    def test_manifest_written_last_and_partial_metadata_cannot_complete(self):
        self.assertIsNone(runs.verify_run_manifest(self.folder))
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            (self.folder / 'validation_metrics.json').unlink(missing_ok=True)
            runs.write_run_manifest(self.folder, {'status': 'complete', 'artifacts':
                                                  {'validation_metrics.json': '0' * 64}})
        self.assertIn('cannot mark', str(caught.exception))
        self.assertFalse((self.folder / 'run_manifest.json').exists())

    def test_atomic_write_leaves_no_partial_file_on_failure(self):
        target = self.directory / 'atomic.json'
        with self.assertRaises(TypeError):
            runs.save_json_atomic(target, {'bad': object()})
        self.assertFalse(target.exists())

    def test_resume_rejects_changed_settings_scales_source_and_parent(self):
        self.complete()
        changes = {
            'settings': {**self.spec['settings'], 'epochs': 2},
            'normalization': {'greek_scales': [1., 2., 3., 4., 5.]},
            'source_sha256': {'x.py': 'cd'},
            'parent': {'folder': 'other_parent'},
        }
        for key, value in changes.items():
            expected = ablation.job_identity(self.spec)
            expected['normalization'] = {'greek_scales': [9.]}
            expected[key] = value
            with self.assertRaises(runs.MetadataVerificationError) as caught:
                runs.verify_run_manifest(self.folder, expected)
            self.assertIn('start a new run directory', str(caught.exception))
        runs.verify_run_manifest(self.folder, ablation.job_identity(self.spec))

    def test_recovery_regenerates_missing_evaluation_outputs(self):
        self.complete()
        for name in ('validation_metrics.json', 'shape_metrics.json', 'validation_regions.csv',
                     'validation_worst_points.csv', 'run_manifest.json'):
            (self.folder / name).unlink()
        self.assertEqual(ablation.classify_job(self.spec, self.directory), 'recover')
        ablation.finalize_job(self.spec, self.directory)
        for name in ('validation_metrics.json', 'shape_metrics.json', 'validation_regions.csv',
                     'validation_worst_points.csv', 'run_manifest.json'):
            self.assertTrue((self.folder / name).exists(), name)
        self.assertIsNotNone(runs.verify_run_manifest(self.folder, ablation.job_identity(self.spec)))

    def test_changed_source_identity_is_rejected(self):
        self.complete()
        expected = ablation.job_identity(self.spec)
        expected['source_sha256'] = {'x.py': 'ff'}
        with self.assertRaises(runs.MetadataVerificationError):
            runs.verify_run_manifest(self.folder, expected)


class ReportTests(unittest.TestCase):
    def test_report_with_regressions_lists_exceptions(self):
        rng = np.random.default_rng(3)
        results, runs_info = {}, {}
        for arm in ('control', 'parity', 'penalty_price', 'penalty_delta', 'penalty_gamma',
                    'penalty_all', 'uniform', 'mixed'):
            for seed in ablation.SEEDS:
                metrics = {name: {'rmse': float(1. + rng.random()), 'max_absolute_error': 1.}
                           for name in ablation.NAMES}
                shape = {constraint: {'proportion_above_1e_8': .05, 'maximum_violation': .1}
                         for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma')}
                regions = {region: {name: 1. for name in ablation.NAMES}
                           for region in ('all', 'near_strike', 'near_expiry',
                                          'near_strike_and_expiry')}
                if arm == 'mixed':
                    regions['all']['rho'] = 2.
                    metrics['price'] = {'rmse': 2., 'max_absolute_error': 1.}
                results[f'{arm}_{seed}'] = {
                    name: dict(metrics=dict(metrics), shape=dict(shape), regions=dict(regions))
                    for name in ('validation', 'in_domain', 'stress', 'focus')}
                results[f'{arm}_{seed}']['parity'] = {
                    name: {'mean': .01, 'maximum': .02} for name in ablation.NAMES}
                runs_info[f'{arm}_{seed}'] = {'optimizer_steps': 12800,
                                              'training_wall_seconds': 10.,
                                              'stop_reason': 'epochs_completed'}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            reports(output, results, runs_info, {'weight': .1, 'gate_failed': False})
            text = (output / 'report.md').read_text()
            exceptions = json.loads((output / 'report_exceptions.json').read_text())
        self.assertTrue(any('mixed vs uniform rho RMSE on focus all' in line for line in exceptions))
        self.assertTrue(any('regresses' in line for line in exceptions))
        self.assertIn('regresses', text)
        self.assertIn('matched update counts', text)
        for forbidden in ('unambiguous', 'every quantity improves in every region', 'unchanged'):
            self.assertNotIn(forbidden, text)


if __name__ == '__main__':
    unittest.main()
