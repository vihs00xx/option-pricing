"""Regression tests for content-hash scope verification.

Pins the scope-masquerade fix: hashes are computed over the arrays of their
declared scope, a declared scope never covers a different evaluated split, and
legacy training-scope hashes cannot authorize validation or test evaluation.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from experiments.deeponet.evaluate import evaluate_checkpoint
from experiments.deeponet.train import ROOT
from src.neural import run_metadata as runs
from src.neural.deeponet import BlackScholesDeepONet


def make_dataset(directory, name='data.npz'):
    rng = np.random.default_rng(11)
    arrays = {}
    for split, count in (('train', 8), ('validation', 4)):
        branch = np.column_stack((
            rng.uniform(80., 120., count), rng.uniform(.05, 2., count),
            rng.uniform(-.02, .1, count), rng.uniform(0., .06, count),
            rng.uniform(.1, .6, count), np.where(np.arange(count) % 2, 1., -1.)))
        trunk = np.column_stack((branch[:, 0] * rng.uniform(.8, 1.2, count), rng.uniform(.01, 1., count)))
        arrays[f'{split}_branch'] = branch
        arrays[f'{split}_trunk'] = trunk
        arrays[f'{split}_targets'] = rng.normal(0., 1., (count, 6))
    path = Path(directory) / name
    np.savez(path, **arrays)
    return path


def tamper(path, key):
    with np.load(path, allow_pickle=False) as data:
        arrays = {name: data[name].copy() for name in data.files}
    arrays[key] = arrays[key] + 123.
    np.savez(path, **arrays)


def make_run(folder, dataset, dataset_record_, legacy=False):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(2)
    model = BlackScholesDeepONet(width=4, depth=1, rank=4)
    for kind in ('selected', 'last'):
        torch.save(dict(architecture=model.architecture, state_dict=model.state_dict(), metadata={}),
                   folder / f'{kind}.pt')
    checkpoints = {kind: runs.sha256(folder / f'{kind}.pt') for kind in ('selected', 'last')}
    if legacy:
        record = {'settings': {'seed': 2, 'dataset': str(dataset)}, 'dataset': str(dataset),
                  'dataset_content_sha256': dataset_record_.get('content_sha256'),
                  'stage': 'stage2', 'seed': 2, 'architecture': model.architecture,
                  'label_columns_used': ['price'], 'checkpoint_sha256': checkpoints}
    else:
        record = runs.build_run_metadata(
            arm='fixture', seed=2, stage='stage2', architecture=model.architecture,
            effective_settings={'seed': 2}, settings_sha256='fixture', dataset=dataset_record_,
            validation_split={'arrays': ['validation_branch', 'validation_targets'], 'rows': 4},
            normalization=None, parent=None, checkpoints=checkpoints,
            training={'optimizer_steps': 3, 'training_wall_seconds': .1, 'stop_reason': 'epochs_completed'},
            label_columns_used=['price'])
    runs.save_json_atomic(folder / 'training_metadata.json', record)
    return record


class ScopeVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.dataset = make_dataset(self.directory)

    def tearDown(self):
        self.temporary.cleanup()

    def test_validation_scoped_hash_verifies_unchanged_arrays(self):
        record = runs.dataset_record(self.dataset, self.directory, include_file_hash=False,
                                     content_scope_prefix='validation_')
        checked = runs.verify_dataset(self.dataset, record, evaluated_split='validation')
        self.assertEqual(checked, ['content_sha256'])

    def test_validation_scoped_hash_rejects_changed_validation_targets(self):
        record = runs.dataset_record(self.dataset, self.directory, include_file_hash=False,
                                     content_scope_prefix='validation_')
        tamper(self.dataset, 'validation_targets')
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, record, evaluated_split='validation')
        self.assertIn('content hash mismatch', str(caught.exception))

    def test_relabeling_training_hash_as_validation_is_rejected(self):
        train_hash = runs.dataset_record(self.dataset, self.directory, include_file_hash=False)['content_sha256']
        masquerade = {'path': str(self.dataset), 'content_sha256': train_hash,
                      'content_hash_scope_prefix': 'validation_',
                      'content_hash_scope': "all npz arrays whose name starts with 'validation_'"}
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, masquerade, evaluated_split='validation')
        self.assertIn('content hash mismatch', str(caught.exception))

    def test_training_scope_hash_cannot_authorize_validation(self):
        record = runs.dataset_record(self.dataset, self.directory, include_file_hash=False)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, record, evaluated_split='validation')
        message = str(caught.exception)
        self.assertIn("content hash of scope 'train_'", message)
        self.assertIn('remains unverified', message)

    def test_out_of_scope_changes_keep_scope_but_break_file_hash(self):
        validation_record = runs.dataset_record(self.dataset, self.directory, include_file_hash=False,
                                                content_scope_prefix='validation_')
        file_record = runs.dataset_record(self.dataset, self.directory)
        tamper(self.dataset, 'train_targets')
        runs.verify_dataset(self.dataset, validation_record, evaluated_split='validation')
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, file_record)
        self.assertIn('dataset file hash mismatch', str(caught.exception))

    def test_unsupported_contradictory_and_empty_scopes_are_rejected(self):
        base = runs.dataset_record(self.dataset, self.directory, include_file_hash=False)
        unsupported = {**base, 'content_hash_scope_prefix': 'bogus_'}
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, unsupported)
        self.assertIn('unsupported content hash scope', str(caught.exception))
        contradictory = {**base, 'content_hash_scope_prefix': 'validation_'}
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, contradictory)
        self.assertIn('contradictory scope fields', str(caught.exception))
        missing = {key: value for key, value in base.items() if key != 'content_hash_scope_prefix'}
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(self.dataset, missing)
        self.assertIn('declares no scope prefix', str(caught.exception))
        empty = runs.dataset_record(self.dataset, self.directory, include_file_hash=False,
                                    content_scope_prefix='validation_')
        stripped = self.directory / 'stripped.npz'
        with np.load(self.dataset, allow_pickle=False) as data:
            np.savez(stripped, **{k: data[k] for k in data.files if k.startswith('train_')})
        empty['path'] = str(stripped)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            runs.verify_dataset(stripped, empty)
        self.assertIn('empty selection', str(caught.exception))

    def test_trusted_full_file_hash_permits_legacy_evaluation_when_it_matches(self):
        record = runs.dataset_record(self.dataset, self.directory, include_file_hash=False)
        make_run(self.directory / 'run', self.dataset, record, legacy=True)
        trusted = runs.sha256(self.dataset)
        with mock.patch('experiments.deeponet.evaluate.trusted_file_sha256', return_value=trusted):
            result = evaluate_checkpoint(self.directory / 'run' / 'selected.pt')
        self.assertIn('trusted_file_sha256', result['dataset']['verified_hash_kinds'])
        with mock.patch('experiments.deeponet.evaluate.trusted_file_sha256', return_value='0' * 64):
            with self.assertRaises(runs.MetadataVerificationError) as caught:
                evaluate_checkpoint(self.directory / 'run' / 'selected.pt')
        self.assertIn('trusted full-file hash mismatch', str(caught.exception))

    def test_evaluator_rejects_scope_masquerade(self):
        train_hash = runs.dataset_record(self.dataset, self.directory, include_file_hash=False)['content_sha256']
        masquerade = {'path': str(self.dataset), 'content_sha256': train_hash,
                      'content_hash_scope_prefix': 'validation_',
                      'content_hash_scope': "all npz arrays whose name starts with 'validation_'"}
        make_run(self.directory / 'run', self.dataset, masquerade)
        with self.assertRaises(runs.MetadataVerificationError) as caught:
            evaluate_checkpoint(self.directory / 'run' / 'selected.pt')
        self.assertIn('content hash mismatch', str(caught.exception))


if __name__ == '__main__':
    unittest.main()
