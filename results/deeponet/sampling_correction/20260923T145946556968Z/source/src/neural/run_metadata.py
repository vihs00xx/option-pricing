"""Run metadata schema, hashing and verification for DeepONet experiments.

Schema v2 records the dataset path with BOTH a file SHA-256 and an array-content
SHA-256 (never compared interchangeably), the validation split, architecture,
effective settings, parent checkpoint hash, checkpoint hashes and the Greek
normalization record. Legacy runs are read through an explicit compatibility
adapter; when a legacy record cannot verify its dataset, verification fails
with an actionable error instead of being bypassed.
"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np


SCHEMA = 'deeponet_run_metadata_v2'
MANIFEST_SCHEMA = 'deeponet_run_manifest_v2'
LEGACY_ADAPTER = 'deeponet_legacy_adapter_v1'
TARGET_ORDERING = ('delta', 'gamma', 'vega', 'theta', 'rho')
CONTENT_HASH_SCOPE = "all npz arrays whose name starts with 'train_'"
CONTENT_HASH_METHOD = ('sha256 over the train-scope arrays sorted by name, each contributing '
                       'its name, dtype.str, shape and C-order bytes')
SCALE_METHOD = ('per-column root mean square of the training label rows '
                '(columns 1..5 of the six target columns: price, delta, gamma, vega, theta, rho)')
REQUIRED_ARTIFACTS = ('selected.pt', 'last.pt', 'history.csv', 'training_metadata.json',
                      'validation_metrics.json', 'shape_metrics.json',
                      'validation_regions.csv', 'validation_worst_points.csv')


class MetadataVerificationError(ValueError):
    """Metadata cannot verify a required artifact; the message is actionable."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def content_sha256(arrays):
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode())
        hasher.update(value.dtype.str.encode())
        hasher.update(str(value.shape).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


def dataset_record(npz_path, root, include_file_hash=True, include_content_hash=True):
    """Record dataset identity with BOTH file and array-content hashes.

    `root` is the project root consumers resolve the stored path against; files
    outside it are stored as absolute paths.
    """
    npz_path = Path(npz_path)
    resolved = (npz_path if npz_path.is_absolute() else root / npz_path).resolve()
    try:
        stored = str(resolved.relative_to(Path(root).resolve()))
    except ValueError:
        stored = str(resolved)
    record = dict(path=stored)
    if include_file_hash:
        record['file_sha256'] = sha256(npz_path)
    if include_content_hash:
        with np.load(npz_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files if key.startswith('train_')}
        record['content_sha256'] = content_sha256(arrays)
        record['content_hash_scope'] = CONTENT_HASH_SCOPE
        record['content_hash_method'] = CONTENT_HASH_METHOD
    return record


def normalization_record(scales, source_dataset, root):
    return dict(greek_scales=[float(value) for value in scales],
                target_ordering=list(TARGET_ORDERING),
                method=SCALE_METHOD,
                source_dataset=dict(source_dataset),
                source_dataset_path=str((Path(root) / source_dataset['path'])),
                effective_greek_loss_coefficients=[
                    float(0.01 / (float(value) ** 2)) for value in scales])


def build_run_metadata(**fields):
    for key in ('arm', 'seed', 'stage', 'architecture', 'effective_settings', 'dataset',
                'checkpoints', 'training'):
        if key not in fields:
            raise MetadataVerificationError(f'run metadata missing required field: {key}')
    return {'schema': SCHEMA, **fields}


def load_run_metadata(folder):
    """Load a run's training metadata through the v2 schema or the legacy adapter."""
    folder = Path(folder)
    path = folder / 'training_metadata.json'
    if not path.exists():
        raise MetadataVerificationError(
            f'no training_metadata.json in {folder}; cannot identify the run. '
            'Recover the partial attempt or start a new run directory.')
    record = json.loads(path.read_text())
    if record.get('schema') == SCHEMA:
        for key in ('arm', 'seed', 'stage', 'architecture', 'effective_settings', 'dataset',
                    'checkpoints', 'training'):
            if key not in record:
                raise MetadataVerificationError(
                    f'{path} declares schema {SCHEMA} but lacks required field {key!r}; '
                    'repair the metadata or start a new run directory.')
        return {**record, 'legacy': False, 'adapter': SCHEMA}
    return _adapt_legacy(folder, record)


def _adapt_legacy(folder, record):
    settings = record.get('settings') or {}
    dataset_path = record.get('dataset') or settings.get('dataset')
    file_sha = record.get('dataset_sha256')
    content_sha = record.get('dataset_content_sha256')
    if dataset_path is None or (file_sha is None and content_sha is None):
        raise MetadataVerificationError(
            f'{folder / "training_metadata.json"} predates schema {SCHEMA} and records no '
            'verifiable dataset identity (no dataset path or no dataset hash of any kind); '
            'verification cannot be performed. Record a dataset hash for this run or start a '
            'new run directory; evaluation must not bypass dataset verification.')
    dataset = {'path': dataset_path}
    if file_sha is not None:
        dataset['file_sha256'] = file_sha
    if content_sha is not None:
        dataset['content_sha256'] = content_sha
        dataset['content_hash_scope'] = CONTENT_HASH_SCOPE
        dataset['content_hash_method'] = CONTENT_HASH_METHOD
    checkpoints = record.get('checkpoints') or record.get('checkpoint_sha256')
    if not checkpoints:
        raise MetadataVerificationError(
            f'{folder / "training_metadata.json"} records no checkpoint hashes; '
            'verification cannot be performed. Record checkpoint hashes or start a new run directory.')
    normalization = record.get('normalization')
    if normalization is None and 'greek_scales' in record:
        normalization = {'greek_scales': record['greek_scales'],
                         'target_ordering': list(TARGET_ORDERING),
                         'method': SCALE_METHOD,
                         'source_dataset': {'path': dataset_path},
                         'legacy_note': 'legacy run: scales computed from this run\'s own training labels'}
    return {'schema': LEGACY_ADAPTER, 'legacy': True, 'adapter': LEGACY_ADAPTER,
            'arm': record.get('arm'), 'seed': record.get('seed') or settings.get('seed'),
            'stage': record.get('stage'),
            'label_columns_used': record.get('label_columns_used'),
            'architecture': record.get('architecture') or settings.get('architecture'),
            'effective_settings': record.get('effective_settings') or settings,
            'dataset': dataset, 'checkpoints': checkpoints, 'normalization': normalization,
            'parent': {'path': record.get('parent_checkpoint'),
                       'file_sha256': record.get('parent_sha256')},
            'training': {key: record.get(key) for key in
                         ('optimizer_steps', 'selected_epoch', 'selected_validation_mse',
                          'training_wall_seconds', 'stop_reason', 'peak_sampled_rss_mb')}}


def verify_dataset(dataset_path, dataset_record_):
    """Verify the recorded dataset identity; hash kinds are never interchangeable."""
    dataset_path = Path(dataset_path)
    checked = []
    expected_file = dataset_record_.get('file_sha256')
    expected_content = dataset_record_.get('content_sha256')
    if expected_file is None and expected_content is None:
        raise MetadataVerificationError(
            f'dataset record for {dataset_path} carries no hash of any kind; refusing to '
            'evaluate against an unverified dataset.')
    if expected_file is not None:
        actual = sha256(dataset_path)
        if actual != expected_file:
            raise MetadataVerificationError(
                f'dataset file hash mismatch for {dataset_path}: recorded file SHA-256 '
                f'{expected_file[:16]}..., found {actual[:16]}... . The file changed since the '
                'run was recorded; restore it or start a new run directory.')
        checked.append('file_sha256')
    if expected_content is not None:
        with np.load(dataset_path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files if key.startswith('train_')}
        actual = content_sha256(arrays)
        if actual != expected_content:
            raise MetadataVerificationError(
                f'dataset content hash mismatch for {dataset_path} (scope: '
                f'{dataset_record_.get("content_hash_scope", CONTENT_HASH_SCOPE)}): recorded '
                f'{expected_content[:16]}..., found {actual[:16]}... . Training labels changed; '
                'restore the dataset or start a new run directory.')
        checked.append('content_sha256')
    return checked


def verify_checkpoint(checkpoint_path, view, stem='selected'):
    expected = (view.get('checkpoints') or {}).get(stem)
    if expected is None:
        raise MetadataVerificationError(
            f'metadata for {checkpoint_path} records no {stem!r} checkpoint hash; refusing to '
            'use an unverified checkpoint.')
    actual = sha256(checkpoint_path)
    if actual != expected:
        raise MetadataVerificationError(
            f'checkpoint hash mismatch for {checkpoint_path}: recorded {expected[:16]}..., '
            f'found {actual[:16]}... . The checkpoint changed after training; restore it or '
            'start a new run directory.')
    return actual


def save_json_atomic(path, value):
    path = Path(path)
    text = json.dumps(value, indent=2) + '\n'
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix='.tmp')
    try:
        with os.fdopen(descriptor, 'w') as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def write_run_manifest(folder, manifest):
    """Atomically mark a run complete; every listed artifact must exist."""
    folder = Path(folder)
    for name in manifest.get('artifacts', {}):
        if not (folder / name).exists():
            raise MetadataVerificationError(
                f'cannot mark {folder} complete: required artifact {name} is missing. '
                'Regenerate the missing outputs first.')
    save_json_atomic(folder / 'run_manifest.json', {'schema': MANIFEST_SCHEMA, **manifest})


def verify_run_manifest(folder, expected=None):
    """Return the manifest of a complete verified run, or None when incomplete.

    With `expected` (the requested job's frozen identity) any mismatch is
    rejected with instructions to start a new run directory.
    """
    folder = Path(folder)
    path = folder / 'run_manifest.json'
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    if manifest.get('schema') != MANIFEST_SCHEMA or manifest.get('status') != 'complete':
        return None
    for name, digest in manifest.get('artifacts', {}).items():
        artifact = folder / name
        if not artifact.exists() or sha256(artifact) != digest:
            raise MetadataVerificationError(
                f'completion manifest of {folder} lists {name} with SHA-256 {digest[:16]}..., '
                'but the artifact is missing or changed. Restore the artifact or start a new '
                'run directory.')
    if expected is not None:
        for key in ('settings', 'settings_sha256', 'dataset', 'normalization', 'parent',
                    'architecture', 'source_sha256'):
            if key not in expected:
                continue
            if key not in manifest:
                raise MetadataVerificationError(
                    f'completed run {folder} has no recorded {key!r} to validate against the '
                    'requested job. Start a new run directory.')
            if not _matches(expected[key], manifest[key]):
                raise MetadataVerificationError(
                    f'completed run {folder} does not match the requested job: field {key!r} '
                    'differs from the recorded value. Resume must not mix configurations; '
                    'start a new run directory.')
    return manifest


def _matches(expected, actual):
    """Recursive subset match: every key the request pins must agree."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and _matches(value, actual[key]) for key, value in expected.items())
    return expected == actual
