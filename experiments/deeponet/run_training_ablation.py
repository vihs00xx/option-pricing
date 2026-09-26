"""Controlled training ablations for the DeepONet surrogate.

One campaign, one factor per arm:
  - parity:      call-put parity enforced by construction (architecture change)
  - penalties:   bound-violation penalties (loss change), one penalty type each
                 plus all three combined
  - sampling:    coordinates concentrated near expiry and near the strike
                 versus an equal-sized uniformly sampled dataset (data change)

Budgets are matched update counts (fixed epochs with a large wall safety cap):
equal update counts do not imply equal computation or wall time, so actual
optimizer updates and runtime are recorded and reported separately. The
sampling comparison shares one fixed set of Greek normalization scales computed
from the uniform arm's training labels, so the coordinate distribution is the
only changed factor. Every job runs from a frozen job specification (effective
settings, dataset hashes, normalization record, parent identity, source hashes)
embedded in the study protocol before training; completion is marked by an
atomic run manifest written only after all artifacts are saved.
"""

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.refine_optimizer import shape_metrics
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural import run_metadata as runs
from src.neural.deeponet import BlackScholesDeepONet, ParityDeepONet, load_deeponet
from src.neural.deeponet_data import (_contract_samples, contract_coordinate_split,
                                      synthetic_split)


SETTINGS_PATH = ROOT / 'experiments/config/neural/deeponet_training.json'
SETTINGS_JSON = json.loads(SETTINGS_PATH.read_text())
DOMAIN = json.loads((ROOT / 'experiments/config/neural/deeponet_settings.json').read_text())['training_domain']
STRESS_DOMAIN = {**DOMAIN, 'K': [75., 125.], 'T': [.03, 2.5], 'r': [-.03, .12],
                 'q': [0., .08], 'sigma': [.08, .65], 'moneyness': [.6, 1.4], 'tau_minimum': .005}
GREEK_WEIGHT = .01
WALL_SAFETY_SECONDS = 36000
PENALTY_WEIGHTS = (.1, 1., 10.)
SEEDS = (22092026, 22092027, 22092028)
SAMPLING_CONTRACT_SEED = 22092030
SAMPLING_COORDINATE_SEEDS = dict(uniform=22092031, mixed=22092032)
EVAL_SEEDS = dict(in_domain=27092026, stress=27093026, focus_contracts=27094026, focus=27094027)
SCALES_SOURCE = {'frozen': 'frozen', 'uniform': 'uniform', 'mixed': 'uniform'}
ARMS = {
    'control': dict(arch='base', data='frozen', penalties=()),
    'parity': dict(arch='parity', data='frozen', penalties=()),
    'penalty_price': dict(arch='base', data='frozen', penalties=('price',)),
    'penalty_delta': dict(arch='base', data='frozen', penalties=('delta',)),
    'penalty_gamma': dict(arch='base', data='frozen', penalties=('gamma',)),
    'penalty_all': dict(arch='base', data='frozen', penalties=('price', 'delta', 'gamma')),
    'uniform': dict(arch='base', data='uniform', penalties=()),
    'mixed': dict(arch='base', data='mixed', penalties=()),
}
PENALTY_ARMS = ('penalty_price', 'penalty_delta', 'penalty_gamma', 'penalty_all')
LAMBDA_RULE = ('Among penalty weights, keep candidates whose validation price RMSE is at most 1.1 times '
               'the control and every Greek RMSE at most 1.15 times the control on the same seed; among '
               'those minimize the summed validation violation frequencies (price bounds, Delta bounds, '
               'negative Gamma), ties to the smaller weight. If none passes, minimize the summed '
               'frequencies and mark the gate failed.')
ACCEPT_RULE = ('Mean price RMSE at most 1.1 times control mean and every mean Greek RMSE at most 1.15 '
               'times control mean across the three seeds.')
BUDGET_NOTE = ('Budgets are matched update counts (identical epochs and batch size per run); equal update '
               'counts do not imply equal computation or equal wall time. Actual optimizer updates and '
               'runtime are reported separately.')

save_json = runs.save_json_atomic


def content_sha256(arrays):
    return runs.content_sha256(arrays)


def build_fresh_evaluation(train_contracts):
    """Evaluation-only package: in-domain, stress, near-expiry/strike focus and
    paired call/put rows for parity residuals."""
    in_domain = synthetic_split(256, 16, DOMAIN, EVAL_SEEDS['in_domain'])
    stress = synthetic_split(128, 16, STRESS_DOMAIN, EVAL_SEEDS['stress'])
    focus = contract_coordinate_split(
        _contract_samples(128, DOMAIN, EVAL_SEEDS['focus_contracts']), 16, DOMAIN,
        EVAL_SEEDS['focus'], mixed=True)
    train_rows = {tuple(row) for row in train_contracts}
    package = {}
    for name, split in (('in_domain', in_domain), ('stress', stress), ('focus', focus)):
        if train_rows & {tuple(row) for row in split['branch']}:
            raise ValueError(f'Evaluation contracts overlap training contracts: {name}')
        for key, value in split.items():
            package[f'{name}_{key}'] = value
    call = in_domain['branch'].copy()
    call[:, 5] = 1.
    put = in_domain['branch'].copy()
    put[:, 5] = -1.
    package['pairs_call_branch'] = call
    package['pairs_put_branch'] = put
    package['pairs_trunk'] = in_domain['trunk']
    return package


def parity_residuals(model, branch, trunk):
    call = branch.clone()
    call[:, 5] = 1.
    put = branch.clone()
    put[:, 5] = -1.
    with_call = model.price_and_greeks(call, trunk)
    with_put = model.price_and_greeks(put, trunk)
    discount_k = branch[:, 0] * torch.exp(-branch[:, 2] * trunk[:, 1])
    difference = trunk[:, 0] * torch.exp(-branch[:, 3] * trunk[:, 1]) - discount_k
    zero = torch.zeros(len(branch), dtype=torch.float64)
    expected = {
        'price': difference,
        'delta': torch.exp(-branch[:, 3] * trunk[:, 1]),
        'gamma': zero,
        'vega': zero,
        'theta': branch[:, 3] * trunk[:, 0] * torch.exp(-branch[:, 3] * trunk[:, 1])
                 - branch[:, 2] * discount_k,
        'rho': trunk[:, 1] * discount_k,
    }
    residual = {name: (with_call[name] - with_put[name] - expected[name]).abs() for name in NAMES}
    return {name: dict(mean=float(value.mean()), maximum=float(value.max()))
            for name, value in residual.items()}


def region_metrics(model, branch, trunk, targets):
    predicted = model.price_and_greeks(branch, trunk)
    errors = np.column_stack([predicted[name].numpy() for name in NAMES]) - targets
    ratio = (trunk[:, 0] / branch[:, 0]).numpy()
    tau = trunk[:, 1].numpy()
    masks = {'all': np.ones(len(branch), dtype=bool),
             'near_strike': np.abs(ratio - 1.) <= .05,
             'near_expiry': tau <= .05,
             'near_strike_and_expiry': (np.abs(ratio - 1.) <= .05) & (tau <= .05)}
    return {group: {name: float(np.sqrt(np.mean(errors[mask, i] ** 2))) if mask.any() else None
                    for i, name in enumerate(NAMES)}
            for group, mask in masks.items()}


def compute_normalization(arrays_by_key):
    """One normalization record per dataset key; sampling arms share the uniform
    arm's training-label scales so sampling is the only changed factor. The
    uniform record must be present whenever a sampling arm is."""
    records = {}
    for key, source_key in SCALES_SOURCE.items():
        if key not in arrays_by_key:
            continue
        if source_key not in arrays_by_key:
            raise ValueError(
                f'normalization for {key!r} requires the {source_key!r} dataset record '
                '(sampling arms share the uniform arm\'s training-label scales)')
        arrays = arrays_by_key[source_key]
        with np.load(ROOT / arrays['path'], allow_pickle=False) as data:
            targets = torch.tensor(data['train_targets'], dtype=torch.float64)
        scales = targets[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
        records[key] = runs.normalization_record(scales.tolist(), arrays, ROOT)
    return records


def effective_settings(stage, seed):
    settings = dict(SETTINGS_JSON)
    if stage == 'pilot':
        settings.update(seed=seed, epochs=500, wall_seconds=WALL_SAFETY_SECONDS,
                        protocol_id='deeponet_training_ablation_v2',
                        note='Ablation pilot: fixed 500 epochs (matched update counts); training and validation data only')
    else:
        settings.update(seed=seed, epochs=800, wall_seconds=WALL_SAFETY_SECONDS,
                        learning_rate=.0003, validation_every=10, progress_every_seconds=45,
                        protocol_id='deeponet_training_ablation_v2',
                        note='Ablation continuation: fixed 800 epochs (matched update counts); training and validation data only')
    return settings


def settings_sha256(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()


def job_spec(name, stage, arm_spec, seed, folder, arrays_by_key, normalization_by_key,
             source_sha256, penalty_types=(), penalty_weight=0., parent=None):
    settings = effective_settings(stage, seed)
    architecture = dict(settings['architecture'])
    if arm_spec['arch'] == 'parity':
        architecture['model'] = 'parity'
    spec = dict(name=name, stage=stage, arch=arm_spec['arch'], data=arm_spec['data'], seed=seed,
                folder=folder, settings=settings, settings_sha256=settings_sha256(settings),
                architecture=architecture,
                dataset=arrays_by_key[arm_spec['data']],
                normalization=normalization_by_key[arm_spec['data']] if stage == 'stage2' else None,
                penalty_types=list(penalty_types), penalty_weight=float(penalty_weight),
                parent=parent, source_sha256=source_sha256)
    return spec


def job_identity(spec):
    return {key: spec[key] for key in ('settings', 'settings_sha256', 'dataset', 'normalization',
                                       'parent', 'architecture', 'source_sha256')
            if key in spec and spec[key] is not None}


def run_job(spec, study):
    torch.set_num_threads(spec['settings']['threads'])
    folder = study / spec['folder']
    folder.mkdir(parents=True, exist_ok=False)
    dataset_path = ROOT / spec['dataset']['path']
    runs.verify_dataset(dataset_path, spec['dataset'], evaluated_split='validation')
    with np.load(dataset_path, allow_pickle=False) as data:
        b, t, y = (torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                   for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    penalty_types = tuple(spec['penalty_types'])
    penalty_weight = float(spec['penalty_weight'])
    parent_record = None
    if spec['stage'] == 'pilot':
        torch.manual_seed(spec['seed'])
        model = ParityDeepONet() if spec['arch'] == 'parity' else BlackScholesDeepONet()
        training = (b, t, y[:, 0])
        loss = None
    else:
        parent_folder = study / spec['parent']['folder']
        view = runs.load_run_metadata(parent_folder)
        parent_hash = runs.verify_checkpoint(parent_folder / 'selected.pt', view)
        parent_record = {'folder': spec['parent']['folder'], 'file_sha256': parent_hash}
        runs.verify_dataset(ROOT / view['dataset']['path'], view['dataset'])
        model = load_deeponet(parent_folder / 'selected.pt')
        torch.manual_seed(spec['seed'])
        training = (b, t, y)
        scales = torch.tensor(spec['normalization']['greek_scales'], dtype=torch.float64)
        gamma_scale = float(scales[1])
        loss = lambda m, cb, ct, tg: supervised_loss(
            m, cb, ct, tg, scales, GREEK_WEIGHT, penalty_weight, penalty_types, gamma_scale)
    print(json.dumps({'starting': spec['name']}), flush=True)
    history, last, summary = train_model(model, training, validation, spec['settings'], loss)
    record = runs.build_run_metadata(
        arm=spec['name'], seed=spec['seed'], stage=spec['stage'], architecture=model.architecture,
        effective_settings=spec['settings'], settings_sha256=spec['settings_sha256'],
        dataset=spec['dataset'],
        validation_split={'arrays': ['validation_branch', 'validation_trunk', 'validation_targets'],
                          'rows': int(len(vb))},
        normalization=spec['normalization'], parent=parent_record,
        checkpoints={}, training={**summary},
        label_columns_used=['price'] if spec['stage'] == 'pilot' else list(NAMES),
        greek_weight=None if spec['stage'] == 'pilot' else GREEK_WEIGHT,
        penalty_types=list(penalty_types), penalty_weight=penalty_weight,
        run_protocol_id='deeponet_training_ablation_v2', final_test_accessed=False)
    for kind, state in (('selected', model.state_dict()), ('last', last)):
        torch.save(dict(architecture=model.architecture, state_dict=state, metadata=record),
                   folder / f'{kind}.pt')
    record['checkpoints'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
    write_csv(folder / 'history.csv', history)
    save_json(folder / 'validation_metrics.json', diagnostics(model, vb, vt, vy, folder))
    save_json(folder / 'shape_metrics.json', shape_metrics(model, vb, vt))
    save_json(folder / 'training_metadata.json', record)
    write_manifest(folder, spec, record)
    print(json.dumps({'completed': spec['name'],
                      'optimizer_steps': record['training']['optimizer_steps']}), flush=True)


def write_manifest(folder, spec, record):
    artifacts = {name: digest(folder / name) for name in runs.REQUIRED_ARTIFACTS}
    manifest = dict(status='complete', job=spec['name'], completed_at=datetime.now(timezone.utc).isoformat(),
                    settings=spec['settings'], settings_sha256=spec['settings_sha256'],
                    dataset=spec['dataset'], normalization=spec['normalization'],
                    parent=record.get('parent'), architecture=record['architecture'],
                    source_sha256=spec['source_sha256'], artifacts=artifacts,
                    training=record['training'], metadata_sha256=digest(folder / 'training_metadata.json'))
    runs.write_run_manifest(folder, manifest)


def finalize_job(spec, study):
    """Recovery: training already finished and verified; regenerate evaluation
    outputs and the completion manifest without retraining."""
    folder = study / spec['folder']
    view = runs.load_run_metadata(folder)
    runs.verify_dataset(ROOT / view['dataset']['path'], view['dataset'], evaluated_split='validation')
    runs.verify_checkpoint(folder / 'selected.pt', view, 'selected')
    runs.verify_checkpoint(folder / 'last.pt', view, 'last')
    dataset_path = ROOT / view['dataset']['path']
    with np.load(dataset_path, allow_pickle=False) as data:
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    model = load_deeponet(folder / 'selected.pt')
    if not (folder / 'validation_metrics.json').exists() or not (folder / 'validation_regions.csv').exists():
        save_json(folder / 'validation_metrics.json', diagnostics(model, vb, vt, vy, folder))
    if not (folder / 'shape_metrics.json').exists():
        save_json(folder / 'shape_metrics.json', shape_metrics(model, vb, vt))
    print(json.dumps({'recovered': spec['name']}), flush=True)
    write_manifest(folder, spec, view)


def classify_job(spec, study):
    folder = study / spec['folder']
    expected = job_identity(spec)
    manifest = runs.verify_run_manifest(folder, expected)
    if manifest is not None:
        return 'complete'
    if not folder.exists():
        return 'pending'
    try:
        view = runs.load_run_metadata(folder)
        runs.verify_checkpoint(folder / 'selected.pt', view, 'selected')
        runs.verify_checkpoint(folder / 'last.pt', view, 'last')
        return 'recover'
    except runs.MetadataVerificationError:
        return 'retry'


def attempt_folder(study, folder_name):
    for attempt in range(2, 100):
        candidate = f'{folder_name}_attempt{attempt}'
        if not (study / candidate).exists():
            return candidate
    raise RuntimeError(f'no free attempt folder for {folder_name}')


def launch(specs, study, jobs_dir, concurrency):
    pending, recovering = [], []
    for spec in specs:
        status = classify_job(spec, study)
        if status == 'complete':
            print(json.dumps({'skipped_verified_complete': spec['name']}), flush=True)
        elif status == 'retry':
            preserved = spec['folder']
            spec = {**spec, 'folder': attempt_folder(study, preserved),
                    'parent': spec.get('parent'), 'preserved_partial_attempt': preserved}
            print(json.dumps({'preserved_partial_attempt': preserved,
                              'new_attempt': spec['folder']}), flush=True)
            pending.append(spec)
        elif status == 'recover':
            recovering.append(spec)
        else:
            pending.append(spec)

    def launch_one(entry):
        recovery, spec = entry
        job_path = jobs_dir / f"{spec['name'].replace('/', '_')}.json"
        save_json(job_path, spec)
        arguments = [sys.executable, '-m', 'experiments.deeponet.run_training_ablation',
                     '--job-file', str(job_path.relative_to(ROOT)), '--study', str(study.relative_to(ROOT))]
        if recovery:
            arguments.append('--finalize-only')
        with (jobs_dir / f"{job_path.stem}.log").open('w') as handle:
            result = subprocess.run(arguments, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Job failed: {spec['name']} (see jobs/{job_path.stem}.log)")
        return spec['name']

    entries = [(True, spec) for spec in recovering] + [(False, spec) for spec in pending]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for name in pool.map(launch_one, entries):
            print(json.dumps({'job_done': name}), flush=True)


def run_folders_for(output, chosen):
    return {(arm, seed): (ROOT / chosen['folder'] if arm == 'penalty_all' and seed == 22092026
                          else Path(output) / f'arms/{arm}_{seed}')
            for arm in ARMS for seed in SEEDS}


def select_penalty_weight(output, weights):
    control = json.loads((output / 'arms/control_22092026/validation_metrics.json').read_text())
    rows = []
    for weight in weights:
        folder = output / f'lambda_sweep/weight_{weight}_22092026'
        metrics = json.loads((folder / 'validation_metrics.json').read_text())
        shape = json.loads((folder / 'shape_metrics.json').read_text())
        frequency = sum(values['proportion_above_1e_8'] for values in shape.values())
        gate = (metrics['price']['rmse'] <= 1.1 * control['price']['rmse']
                and all(metrics[name]['rmse'] <= 1.15 * control[name]['rmse'] for name in NAMES[1:]))
        rows.append(dict(weight=weight, folder=str(folder.relative_to(ROOT)), gate_passed=gate,
                         summed_violation_frequency=frequency,
                         validation_price_rmse=metrics['price']['rmse'], shape=shape))
    eligible = [row for row in rows if row['gate_passed']]
    pool = eligible if eligible else rows
    selected = min(pool, key=lambda row: (row['summed_violation_frequency'], row['weight']))
    return dict(weight=selected['weight'], folder=selected['folder'], gate_failed=not eligible,
                rule=LAMBDA_RULE, candidates=rows)


def evaluate(output, run_folders, evaluation, chosen):
    tensors = {}
    for name in ('in_domain', 'stress', 'focus'):
        tensors[name] = tuple(torch.tensor(evaluation[f'{name}_{key}'], dtype=torch.float64)
                              for key in ('branch', 'trunk')) + (evaluation[f'{name}_targets'],)
    pairs = (torch.tensor(evaluation['pairs_call_branch'], dtype=torch.float64),
             torch.tensor(evaluation['pairs_trunk'], dtype=torch.float64))
    with np.load(ROOT / SETTINGS_JSON['dataset'], allow_pickle=False) as data:
        tensors['validation'] = (torch.tensor(data['validation_branch'], dtype=torch.float64),
                                 torch.tensor(data['validation_trunk'], dtype=torch.float64),
                                 data['validation_targets'].copy())
    results, runs_info = {}, {}
    for (arm, seed), folder in sorted(run_folders.items()):
        view = runs.load_run_metadata(folder)
        runs.verify_checkpoint(folder / 'selected.pt', view, 'selected')
        model = load_deeponet(folder / 'selected.pt')
        record = {}
        for name, (branch, trunk, targets) in tensors.items():
            if name == 'validation':
                record[name] = dict(metrics=json.loads((folder / 'validation_metrics.json').read_text()),
                                    shape=json.loads((folder / 'shape_metrics.json').read_text()),
                                    regions=region_metrics(model, branch, trunk, targets))
            else:
                destination = output / 'evaluation' / arm / f'seed_{seed}' / name
                destination.mkdir(parents=True, exist_ok=True)
                record[name] = dict(metrics=diagnostics(model, branch, trunk, targets, destination),
                                    shape=shape_metrics(model, branch, trunk),
                                    regions=region_metrics(model, branch, trunk, targets))
        record['parity'] = parity_residuals(model, *pairs)
        results[f'{arm}_{seed}'] = record
        runs_info[f'{arm}_{seed}'] = view['training']
        save_json(output / 'evaluation' / arm / f'seed_{seed}' / 'summary.json', record)
        print(json.dumps({'evaluated': f'{arm}_{seed}'}), flush=True)
    save_json(output / 'evaluation_results.json', results)
    save_json(output / 'training_accounting.json', runs_info)
    return results, runs_info


def aggregate(results, arm, path):
    values = []
    for seed in SEEDS:
        record = results[f'{arm}_{seed}']
        for key in path:
            record = record[key]
        values.append(record)
    return values


def mean_sd(values):
    array = np.asarray(values, dtype=float)
    return float(array.mean()), float(array.std(ddof=1))


def reports(output, results, runs_info, chosen, filename='report.md', caveats=()):
    exceptions = []

    def stat(arm, path):
        return mean_sd(aggregate(results, arm, path))

    def compare(reference, arm, path, label):
        old = stat(reference, path)[0]
        new = stat(arm, path)[0]
        change = new / old - 1. if old else 0.
        if change > .01:
            exceptions.append(f'{label}: regresses ({reference} {old:.6g} -> {arm} {new:.6g}, {change:+.1%})')
            verdict = 'worse'
        elif change < -.01:
            verdict = 'better'
        else:
            verdict = 'effectively flat'
        return f'{label} {change:+.1%} ({verdict})'

    def rmse_cells(path, arms):
        return [f'{stat(arm, path)[0]:.6g} ± {stat(arm, path)[1]:.4g}' for arm in arms]

    def frequency(arm, constraint, dataset):
        return stat(arm, [dataset, 'shape', constraint, 'proportion_above_1e_8'])[0]

    def severity(arm, constraint, dataset):
        return stat(arm, [dataset, 'shape', constraint, 'maximum_violation'])[0]

    def acceptable(arm):
        price = stat(arm, ['validation', 'metrics', 'price', 'rmse'])[0]
        control_price = stat('control', ['validation', 'metrics', 'price', 'rmse'])[0]
        greeks_ok = all(
            stat(arm, ['validation', 'metrics', name, 'rmse'])[0]
            <= 1.15 * stat('control', ['validation', 'metrics', name, 'rmse'])[0] for name in NAMES[1:])
        gate = price <= 1.1 * control_price and greeks_ok
        if not gate:
            exceptions.append(f'{arm}: fails the accuracy acceptance gate ({ACCEPT_RULE})')
        return gate

    lines = ['# DeepONet training ablations', '',
             'Three single-factor comparisons under one two-stage training procedure '
             '(price-only pilot of 500 epochs, then supervised continuation of 800 epochs at '
             'Greek-loss weight 0.01, batch size 512), three seeds per arm. ' + BUDGET_NOTE + ' '
             'Model selection used training and validation data only. The fresh evaluation package '
             '(in-domain, stress, near-expiry/near-strike focus, paired call/put rows) was opened once '
             'after all selection decisions and is an observed benchmark, not an untouched test; a '
             'later independent assessment needs fresh cases. The final benchmark was evaluated '
             'once in an earlier study (`results/deeponet/primary/final_test_metrics.json`) and is '
             'already observed; the latest work did not access it again.', '',
             f'Penalty weight selected on validation: {chosen["weight"]} '
             f'({"gate failed, fallback rule applied" if chosen["gate_failed"] else "gate passed"}). '
             'Acceptability rule: ' + ACCEPT_RULE, '']
    lines += ['## Budget accounting (matched update counts)', '',
              'Equal update counts do not imply equal computation or wall time.', '',
              '| Run | Optimizer updates | Runtime (s) | Stop reason |', '|---|---:|---:|---|']
    for run, training in sorted(runs_info.items()):
        lines.append(f"| {run} | {training.get('optimizer_steps')} | "
                     f"{training.get('training_wall_seconds'):.1f} | {training.get('stop_reason')} |")

    lines += ['', '## 1. Call-put parity by construction', '',
              'The parity arm predicts the shared midpoint (C+P)/2 and derives both prices from '
              'C-P = S exp(-q tau) - K exp(-r tau); the control arm predicts prices directly. '
              'Parity residuals are absolute call-put inconsistencies of each prediction relation on '
              'paired rows (mean over rows, mean over seeds):', '',
              '| Quantity | Control | Parity |', '|---|---:|---:|']
    for name in NAMES:
        lines.append(f"| {name} | {stat('control', ['parity', name, 'mean'])[0]:.6g} | "
                     f"{stat('parity', ['parity', name, 'mean'])[0]:.6g} |")
    for dataset, title in (('validation', 'Validation RMSE (mean ± SD over seeds)'),
                           ('in_domain', 'Fresh in-domain RMSE (observed benchmark)'),
                           ('stress', 'Fresh stress RMSE (observed benchmark)')):
        lines += ['', title + ':', '', '| Quantity | Control | Parity |', '|---|---:|---:|']
        for name in NAMES:
            cells = rmse_cells([dataset, 'metrics', name, 'rmse'], ('control', 'parity'))
            lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation shape violations, frequency and maximum severity reported separately:', '',
              '| Constraint | Control frequency | Parity frequency | Control max severity | Parity max severity |',
              '|---|---:|---:|---:|---:|']
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        lines.append(f'| {constraint} | {frequency("control", constraint, "validation"):.4f} | '
                     f'{frequency("parity", constraint, "validation"):.4f} | '
                     f'{severity("control", constraint, "validation"):.4g} | '
                     f'{severity("parity", constraint, "validation"):.4g} |')
    conclusions = []
    for dataset in ('validation', 'in_domain', 'stress'):
        better = [name for name in NAMES
                  if stat('parity', [dataset, 'metrics', name, 'rmse'])[0]
                  < stat('control', [dataset, 'metrics', name, 'rmse'])[0]]
        conclusions.append(f'{dataset}: parity improves {len(better)} of {len(NAMES)} RMSEs '
                           f"({', '.join(better) if better else 'none'})")
    for dataset in ('validation', 'stress'):
        for name in NAMES:
            compare('control', 'parity', [dataset, 'metrics', name, 'rmse'],
                    f'parity vs control {name} RMSE on {dataset}')
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        compare('control', 'parity', ['validation', 'shape', constraint, 'proportion_above_1e_8'],
                f'parity vs control {constraint} violation frequency on validation')
        compare('control', 'parity', ['validation', 'shape', constraint, 'maximum_violation'],
                f'parity vs control {constraint} maximum severity on validation')
    lines += ['', 'Conclusions from measured values:', '']
    lines += [f'- {text}.' for text in conclusions]
    lines += [f"- Parity residual relations: {compare('control', 'parity', ['parity', 'price', 'mean'], 'call-put price residual')} "
              '(the parity arm is at float round-off by construction).']
    lines += [f"- Price-bound violation frequency on validation: "
              f"{compare('control', 'parity', ['validation', 'shape', 'price_bounds', 'proportion_above_1e_8'], 'price-bound frequency')}, "
              f"maximum severity: {compare('control', 'parity', ['validation', 'shape', 'price_bounds', 'maximum_violation'], 'price-bound severity')}."
              ]
    gate = acceptable('parity')
    lines += [f"- Parity arm passes the accuracy acceptance gate: **{gate}**."]

    lines += ['', '## 2. Bound-violation penalties', '',
              'Each penalty arm adds one squared normalized violation term (all three for penalty_all) '
              f'with weight {chosen["weight"]} to the supervised loss during the continuation stage; '
              'stage 1 is identical for every arm. Frequency and maximum severity are separate measures:', '',
              '| Arm | Price frequency | Delta frequency | Gamma frequency | Price max severity | '
              'Delta max severity | Gamma max severity | Price RMSE | Acceptable |',
              '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    penalty_gates = {}
    for arm in ('control',) + PENALTY_ARMS:
        cells = [f'{frequency(arm, c, "validation"):.4f}' for c in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        cells += [f'{severity(arm, c, "validation"):.4g}' for c in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        price = stat(arm, ['validation', 'metrics', 'price', 'rmse'])[0]
        gate_value = True if arm == 'control' else acceptable(arm)
        penalty_gates[arm] = gate_value
        lines.append(f'| {arm} | ' + ' | '.join(cells) + f' | {price:.6g} | {gate_value} |')
    lines += ['', 'Fresh stress-set violations (observed benchmark):', '',
              '| Arm | Price frequency | Delta frequency | Gamma frequency | Price max severity | '
              'Delta max severity | Gamma max severity |', '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ('control',) + PENALTY_ARMS:
        cells = [f'{frequency(arm, c, "stress"):.4f}' for c in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        cells += [f'{severity(arm, c, "stress"):.4g}' for c in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation RMSE (mean ± SD):', '', '| Quantity | Control | '
              + ' | '.join(PENALTY_ARMS) + ' |', '|---|---:|' + '---:|' * len(PENALTY_ARMS)]
    for name in NAMES:
        cells = rmse_cells(['validation', 'metrics', name, 'rmse'], ('control',) + PENALTY_ARMS)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    penalty_notes = []
    for arm, target in (('penalty_price', 'price_bounds'), ('penalty_delta', 'delta_bounds'),
                        ('penalty_gamma', 'negative_gamma'), ('penalty_all', None)):
        targets = ('price_bounds', 'delta_bounds', 'negative_gamma') if target is None else (target,)
        for constraint in targets:
            penalty_notes.append(f'{arm} {constraint} frequency: '
                                 + compare('control', arm, ['validation', 'shape', constraint, 'proportion_above_1e_8'],
                                           f'{arm} {constraint} frequency')
                                 + f', maximum severity: '
                                 + compare('control', arm, ['validation', 'shape', constraint, 'maximum_violation'],
                                           f'{arm} {constraint} severity'))
        for name in NAMES[:2]:
            compare('control', arm, ['validation', 'metrics', name, 'rmse'], f'{arm} {name} RMSE')
    lines += ['', 'Conclusions from measured values and gates:', '']
    lines += [f'- {text}.' for text in penalty_notes]
    lines += ['- Gates: ' + ', '.join(f'{arm}={"pass" if value else "FAIL"}'
                                      for arm, value in penalty_gates.items() if arm != 'control') + '.']

    lines += ['', '## 3. Near-expiry and near-strike sampling', '',
              'The mixed arm replaces half of each contract\'s uniform coordinates with concentrations '
              'near the strike, near expiry and near both, at identical dataset size and identical '
              'contracts. Both arms are normalized by one fixed set of Greek scales computed from the '
              'uniform arm\'s training labels (recorded in both runs\' metadata), so the coordinate '
              'distribution is the only changed factor. Focus-set RMSE by region (fresh cases, mean ± SD):', '',
              '| Quantity | Region | Uniform | Mixed |', '|---|---|---:|---:|']
    for region in ('all', 'near_strike', 'near_expiry', 'near_strike_and_expiry'):
        for name in NAMES:
            cells = rmse_cells(['focus', 'regions', region, name], ('uniform', 'mixed'))
            lines.append(f'| {name} | {region} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation RMSE:', '', '| Quantity | Uniform | Mixed |', '|---|---:|---:|']
    for name in NAMES:
        cells = rmse_cells(['validation', 'metrics', name, 'rmse'], ('uniform', 'mixed'))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation shape violations, frequency and maximum severity:', '',
              '| Constraint | Uniform frequency | Mixed frequency | Uniform max severity | Mixed max severity |',
              '|---|---:|---:|---:|---:|']
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        lines.append(f'| {constraint} | {frequency("uniform", constraint, "validation"):.4f} | '
                     f'{frequency("mixed", constraint, "validation"):.4f} | '
                     f'{severity("uniform", constraint, "validation"):.4g} | '
                     f'{severity("mixed", constraint, "validation"):.4g} |')
    sampling_notes = []
    for region in ('all', 'near_strike', 'near_expiry', 'near_strike_and_expiry'):
        for name in NAMES:
            sampling_notes.append(compare('uniform', 'mixed', ['focus', 'regions', region, name],
                                          f'mixed vs uniform {name} RMSE on focus {region}'))
    for name in NAMES:
        sampling_notes.append(compare('uniform', 'mixed', ['validation', 'metrics', name, 'rmse'],
                                      f'mixed vs uniform {name} RMSE on validation'))
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        compare('uniform', 'mixed', ['validation', 'shape', constraint, 'proportion_above_1e_8'],
                f'mixed vs uniform {constraint} violation frequency on validation')
        compare('uniform', 'mixed', ['validation', 'shape', constraint, 'maximum_violation'],
                f'mixed vs uniform {constraint} maximum severity on validation')
    lines += ['', 'Conclusions from measured values:', '']
    lines += [f'- {text}.' for text in sampling_notes]

    lines += ['', '## Exceptions and gate results', '',
              'Every measured regression and failed gate in this report:', '']
    lines += [f'- {text}.' for text in exceptions] or ['- none measured.']
    for caveat in caveats:
        lines += ['', f'Caveat: {caveat}']
    (output / filename).write_text('\n'.join(lines) + '\n')
    save_json(output / (Path(filename).stem + '_exceptions.json'), exceptions)


def run_setup(concurrency):
    output = ROOT / 'results/deeponet/training_ablation' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    data_dir = output / 'data'
    data_dir.mkdir()
    (output / 'jobs').mkdir()
    with np.load(ROOT / SETTINGS_JSON['dataset'], allow_pickle=False) as data:
        frozen = {key: data[key].copy() for key in data.files}
    frozen_contracts = np.unique(frozen['train_branch'], axis=0)
    uniform_contracts = _contract_samples(256, DOMAIN, SAMPLING_CONTRACT_SEED)
    splits = {'uniform': contract_coordinate_split(
                  uniform_contracts, 32, DOMAIN, SAMPLING_COORDINATE_SEEDS['uniform']),
              'mixed': contract_coordinate_split(
                  uniform_contracts, 32, DOMAIN, SAMPLING_COORDINATE_SEEDS['mixed'], mixed=True)}
    for name, split in splits.items():
        arrays = {f'train_{key}': value for key, value in split.items()}
        for key in ('validation_branch', 'validation_trunk', 'validation_targets'):
            arrays[key] = frozen[key]
        np.savez_compressed(data_dir / f'train_{name}.npz', **arrays)
    evaluation = build_fresh_evaluation(np.concatenate((frozen_contracts, uniform_contracts)))
    np.savez_compressed(data_dir / 'evaluation.npz', **evaluation)
    arrays_by_key = {'frozen': runs.dataset_record(SETTINGS_JSON['dataset'], ROOT)}
    for name in splits:
        arrays_by_key[name] = runs.dataset_record(data_dir / f'train_{name}.npz', ROOT)
    normalization_by_key = compute_normalization(arrays_by_key)
    composition = {name: {region: int(np.sum(split['region'] == code))
                          for code, region in enumerate(('broad', 'near_strike', 'near_expiry',
                                                         'near_strike_and_expiry'))}
                   for name, split in splits.items()}
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/train.py',
                 ROOT / 'experiments/deeponet/compare_supervision.py',
                 ROOT / 'experiments/deeponet/refine_optimizer.py', ROOT / 'src/neural/deeponet.py',
                 ROOT / 'src/neural/deeponet_data.py', ROOT / 'src/neural/run_metadata.py',
                 ROOT / 'src/black_scholes.py', SETTINGS_PATH):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    pilots = {}
    for arm, spec in ARMS.items():
        pilots.setdefault((spec['arch'], spec['data']), []).append(arm)
    jobs = {}
    for (arch, data_key), _ in sorted(pilots.items()):
        for seed in SEEDS:
            name = f'pilot/{arch}_{data_key}_{seed}'
            jobs[name] = job_spec(name, 'pilot', dict(arch=arch, data=data_key, penalties=()), seed,
                                  f'pilot/{arch}_{data_key}_{seed}', arrays_by_key,
                                  normalization_by_key, sources)
    for arm in ('control', 'parity', 'uniform', 'mixed'):
        spec = ARMS[arm]
        for seed in SEEDS:
            name = f'{arm}/{seed}'
            parent = {'folder': f"pilot/{spec['arch']}_{spec['data']}_{seed}"}
            jobs[name] = job_spec(name, 'stage2', spec, seed, f'arms/{arm}_{seed}', arrays_by_key,
                                  normalization_by_key, sources, parent=parent)
    for weight in PENALTY_WEIGHTS:
        name = f'lambda_sweep/weight_{weight}/22092026'
        spec = ARMS['penalty_all']
        parent = {'folder': f"pilot/{spec['arch']}_{spec['data']}_22092026"}
        jobs[name] = job_spec(name, 'stage2', spec, 22092026, f'lambda_sweep/weight_{weight}_22092026',
                              arrays_by_key, normalization_by_key, sources,
                              penalty_types=('price', 'delta', 'gamma'), penalty_weight=weight,
                              parent=parent)
    protocol = dict(
        protocol_id='deeponet_training_ablation_v2',
        factors={'parity': 'architecture: shared midpoint with call-put parity by construction',
                 'penalties': 'loss: squared normalized bound violations for price, Delta and negative Gamma',
                 'sampling': 'data: equal-sized coordinate sets, uniform versus near-expiry/near-strike concentration'},
        arms={name: dict(arch=spec['arch'], data=spec['data'], penalties=list(spec['penalties']))
              for name, spec in ARMS.items()},
        seeds=SEEDS, greek_weight=GREEK_WEIGHT, schema=runs.SCHEMA,
        procedure=dict(stage1='price-only pilot: 500 epochs (matched update counts), Adam lr 0.001, batch 512',
                       stage2='supervised continuation: 800 epochs (matched update counts), Adam lr 0.0003, batch 512',
                       selection='minimum validation price MSE normalized by strike',
                       threads=2, concurrent_jobs=concurrency, budget=BUDGET_NOTE,
                       wall_safety_cap_seconds=WALL_SAFETY_SECONDS),
        penalty_weight_candidates=PENALTY_WEIGHTS, penalty_weight_rule=LAMBDA_RULE,
        acceptability_rule=ACCEPT_RULE,
        normalization={key: record for key, record in normalization_by_key.items()},
        sampling=dict(contract_seed=SAMPLING_CONTRACT_SEED, coordinate_seeds=SAMPLING_COORDINATE_SEEDS,
                      contracts_shared_between_arms=True, coordinates_per_contract=32,
                      train_composition=composition,
                      note='Equal dataset sizes; only the coordinate distribution differs. '
                           'Both sampling arms share the uniform arm\'s Greek normalization scales.'),
        evaluation=dict(status='fresh_cases_opened_once_after_all_selection_decisions; observed benchmark once read',
                        seeds=EVAL_SEEDS, sets=['in_domain', 'stress', 'focus', 'pairs'],
                        file=str((data_dir / 'evaluation.npz').relative_to(ROOT)),
                        content_sha256=content_sha256(evaluation),
                        note='A later independent assessment needs fresh cases; these are now observed.'),
        dataset_arrays=arrays_by_key, source_sha256=sources, jobs=jobs,
        final_test_accessed=False, python=platform.python_version(), torch=str(torch.__version__))
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output)}), flush=True)
    return output


def run_study(concurrency, resume=None):
    if resume is not None:
        output = resume if resume.is_absolute() else ROOT / resume
    else:
        output = run_setup(concurrency)
    protocol = json.loads((output / 'protocol.json').read_text())
    for relative, expected in protocol['source_sha256'].items():
        actual = digest(ROOT / relative)
        if actual != expected:
            raise runs.MetadataVerificationError(
                f'source {relative} changed since this study was frozen '
                f'(recorded {expected[:16]}..., found {actual[:16]}...). '
                'Resume validates source identity; start a new run directory.')
    jobs_dir = output / 'jobs'
    data_dir = output / 'data'
    jobs = protocol['jobs']
    if not jobs:
        raise runs.MetadataVerificationError(
            'study protocol has no frozen job specifications; this study predates strict '
            'resumption and cannot be resumed safely. Start a new run directory.')
    launch([spec for spec in jobs.values() if spec['stage'] == 'pilot'], output, jobs_dir, concurrency)
    launch([spec for spec in jobs.values() if spec['stage'] == 'stage2'], output, jobs_dir, concurrency)
    chosen = select_penalty_weight(output, PENALTY_WEIGHTS)
    save_json(output / 'penalty_weight_selection.json', chosen)
    print(json.dumps({'penalty_weight': {'weight': chosen['weight'], 'gate_failed': chosen['gate_failed']}}),
          flush=True)
    penalty_specs = []
    for arm in PENALTY_ARMS:
        for seed in SEEDS:
            if arm == 'penalty_all' and seed == 22092026:
                continue
            name = f'{arm}/{seed}'
            spec = job_spec(name, 'stage2', ARMS[arm], seed, f'arms/{arm}_{seed}',
                            protocol['dataset_arrays'], protocol['normalization'],
                            protocol['source_sha256'],
                            penalty_types=ARMS[arm]['penalties'], penalty_weight=chosen['weight'],
                            parent={'folder': f"pilot/{ARMS[arm]['arch']}_{ARMS[arm]['data']}_{seed}"})
            jobs[name] = spec
            penalty_specs.append(spec)
    save_json(output / 'protocol.json', {**protocol, 'jobs': jobs})
    launch(penalty_specs, output, jobs_dir, concurrency)
    run_folders = run_folders_for(output, chosen)
    with np.load(data_dir / 'evaluation.npz', allow_pickle=False) as data:
        evaluation = {key: data[key].copy() for key in data.files}
    results, runs_info = evaluate(output, run_folders, evaluation, chosen)
    reports(output, results, runs_info, chosen)
    print(json.dumps({'complete': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--finalize-only', action='store_true',
                        help='regenerate evaluation outputs and manifest without retraining')
    parser.add_argument('--jobs', type=int, default=3,
                        help='concurrent training jobs (each uses 2 torch threads)')
    parser.add_argument('--resume', type=Path,
                        help='continue an existing study folder, skipping verified complete jobs')
    args = parser.parse_args()
    if args.job_file:
        spec = json.loads(args.job_file.read_text())
        study = ROOT / args.study
        if args.finalize_only:
            finalize_job(spec, study)
        else:
            run_job(spec, study)
        return
    run_study(args.jobs, args.resume)


if __name__ == '__main__':
    main()
