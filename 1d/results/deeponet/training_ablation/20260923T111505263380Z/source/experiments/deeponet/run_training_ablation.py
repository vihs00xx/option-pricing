"""Controlled training ablations for the DeepONet surrogate.

One campaign, one factor per arm:
  - parity:      call-put parity enforced by construction (architecture change)
  - penalties:   bound-violation penalties (loss change), one penalty type each
                 plus all three combined
  - sampling:    coordinates concentrated near expiry and near the strike
                 versus an equal-sized uniformly sampled dataset (data change)

Every arm trains with the established two-stage family procedure (price-only
pilot, then supervised continuation at Greek-loss weight 0.01) for identical
wall budgets and seeds. Arms within a comparison differ in exactly one factor.
Model selection uses training and validation data only; the fresh evaluation
package is opened once after all selection decisions and is an observed
benchmark once read. The frozen final benchmark is never accessed.
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
from src.neural.deeponet import BlackScholesDeepONet, ParityDeepONet, load_deeponet
from src.neural.deeponet_data import (_contract_samples, contract_coordinate_split,
                                      synthetic_split)


SETTINGS_PATH = ROOT / 'experiments/config/neural/deeponet_training.json'
SETTINGS_JSON = json.loads(SETTINGS_PATH.read_text())
DOMAIN = json.loads((ROOT / 'experiments/config/neural/deeponet_settings.json').read_text())['training_domain']
STRESS_DOMAIN = {**DOMAIN, 'K': [75., 125.], 'T': [.03, 2.5], 'r': [-.03, .12],
                 'q': [0., .08], 'sigma': [.08, .65], 'moneyness': [.6, 1.4], 'tau_minimum': .005}
GREEK_WEIGHT = .01
STAGE2_SETTINGS = dict(epochs=800, wall_seconds=36000, learning_rate=.0003, validation_every=10)
PENALTY_WEIGHTS = (.1, 1., 10.)
SEEDS = (22092026, 22092027, 22092028)
SAMPLING_CONTRACT_SEED = 22092030
SAMPLING_COORDINATE_SEEDS = dict(uniform=22092031, mixed=22092032)
EVAL_SEEDS = dict(in_domain=27092026, stress=27093026, focus_contracts=27094026, focus=27094027)
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


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def content_sha256(arrays):
    hasher = hashlib.sha256()
    for name in sorted(arrays):
        value = np.ascontiguousarray(arrays[name])
        hasher.update(name.encode())
        hasher.update(value.dtype.str.encode())
        hasher.update(str(value.shape).encode())
        hasher.update(value.tobytes())
    return hasher.hexdigest()


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


def run_job(spec, study):
    torch.set_num_threads(spec['threads'])
    folder = study / spec['folder']
    folder.mkdir(parents=True, exist_ok=False)
    arrays = spec['arrays']
    with np.load(ROOT / arrays['file'], allow_pickle=False) as data:
        b, t, y = (torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                   for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    scales = y[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    penalty_types = tuple(spec.get('penalty_types', ()))
    penalty_weight = float(spec.get('penalty_weight', 0.))
    if spec['stage'] == 'pilot':
        torch.manual_seed(spec['seed'])
        model = ParityDeepONet() if spec['arch'] == 'parity' else BlackScholesDeepONet()
        settings = dict(SETTINGS_JSON)
        settings.update(seed=spec['seed'], wall_seconds=36000,
                        protocol_id='deeponet_training_ablation_v2',
                        note='Ablation pilot: fixed 500 epochs; training and validation data only')
        training = (b, t, y[:, 0])
        loss = None
        parent = None
    else:
        parent = study / spec['parent']
        metadata = json.loads((parent / 'training_metadata.json').read_text())
        if digest(parent / 'selected.pt') != metadata['checkpoint_sha256']['selected']:
            raise ValueError(f'Pilot checkpoint hash mismatch: {parent}')
        model = load_deeponet(parent / 'selected.pt')
        torch.manual_seed(spec['seed'])
        settings = dict(SETTINGS_JSON)
        settings.update(**STAGE2_SETTINGS, seed=spec['seed'], progress_every_seconds=45,
                        protocol_id='deeponet_training_ablation_v2',
                        note='Ablation continuation: fixed 800 epochs; training and validation data only')
        training = (b, t, y)
        gamma_scale = float(scales[1])
        loss = lambda m, cb, ct, tg: supervised_loss(
            m, cb, ct, tg, scales, GREEK_WEIGHT, penalty_weight, penalty_types, gamma_scale)
    print(json.dumps({'starting': spec['name']}), flush=True)
    history, last, summary = train_model(model, training, validation, settings, loss)
    record = {**summary, 'settings': settings, 'arm': spec['name'], 'seed': spec['seed'],
              'stage': spec['stage'], 'arch': spec['arch'], 'data': spec['data'],
              'dataset': arrays['file'], 'dataset_content_sha256': arrays['content_sha256'],
              'greek_scales': scales.tolist(),
              'parent_checkpoint': None if parent is None else str(parent.relative_to(ROOT)),
              'parent_sha256': None if parent is None else digest(parent / 'selected.pt'),
              'label_columns_used': ['price'] if spec['stage'] == 'pilot' else list(NAMES),
              'greek_weight': None if spec['stage'] == 'pilot' else GREEK_WEIGHT,
              'penalty_types': list(penalty_types), 'penalty_weight': penalty_weight,
              'final_test_accessed': False}
    for kind, state in (('selected', model.state_dict()), ('last', last)):
        torch.save(dict(architecture=model.architecture, state_dict=state, metadata=record),
                   folder / f'{kind}.pt')
    record['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
    save_json(folder / 'training_metadata.json', record)
    write_csv(folder / 'history.csv', history)
    save_json(folder / 'validation_metrics.json', diagnostics(model, vb, vt, vy, folder))
    save_json(folder / 'shape_metrics.json', shape_metrics(model, vb, vt))
    print(json.dumps({'completed': spec['name'],
                      'selected_validation_mse': record['selected_validation_mse']}), flush=True)


def launch(specs, study, jobs_dir, concurrency):
    pending = [spec for spec in specs if not (study / spec['folder'] / 'training_metadata.json').exists()]
    skipped = [spec['name'] for spec in specs if (study / spec['folder'] / 'training_metadata.json').exists()]
    if skipped:
        print(json.dumps({'skipped_completed_jobs': skipped}), flush=True)

    def launch_one(spec):
        job_path = jobs_dir / f"{spec['name'].replace('/', '_')}.json"
        save_json(job_path, spec)
        with (jobs_dir / f"{job_path.stem}.log").open('w') as handle:
            result = subprocess.run(
                [sys.executable, '-m', 'experiments.deeponet.run_training_ablation',
                 '--job-file', str(job_path.relative_to(ROOT)), '--study', str(study.relative_to(ROOT))],
                cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        if result.returncode != 0:
            raise RuntimeError(f"Job failed: {spec['name']} (see jobs/{job_path.stem}.log)")
        return spec['name']

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        for name in pool.map(launch_one, pending):
            print(json.dumps({'job_done': name}), flush=True)


def continuation_spec(arm, spec, seed, penalty_types, penalty_weight, arrays_by_key):
    arch, data_key = spec['arch'], spec['data']
    folder = f'{arm}_{seed}' if arm.startswith('lambda_sweep/') else f'arms/{arm}_{seed}'
    return dict(name=f'{arm}/{seed}', stage='stage2', arch=arch, data=data_key, seed=seed, threads=2,
                penalty_types=list(penalty_types), penalty_weight=penalty_weight,
                parent=f'pilot/{arch}_{data_key}_{seed}', folder=folder,
                arrays=arrays_by_key[data_key])


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
    results = {}
    for (arm, seed), folder in sorted(run_folders.items()):
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
        save_json(output / 'evaluation' / arm / f'seed_{seed}' / 'summary.json', record)
        print(json.dumps({'evaluated': f'{arm}_{seed}'}), flush=True)
    save_json(output / 'evaluation_results.json', results)
    reports(output, results, chosen)


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


def reports(output, results, chosen):
    def stat(arm, path):
        return mean_sd(aggregate(results, arm, path))

    def rmse_cells(path, arms):
        return [f'{stat(arm, path)[0]:.6g} ± {stat(arm, path)[1]:.4g}' for arm in arms]

    def violation_cells(arm, constraint, dataset):
        frequency = stat(arm, [dataset, 'shape', constraint, 'proportion_above_1e_8'])[0]
        severity = stat(arm, [dataset, 'shape', constraint, 'maximum_violation'])[0]
        return f'{frequency:.4f}', f'{severity:.4g}'

    def acceptable(arm):
        price = stat(arm, ['validation', 'metrics', 'price', 'rmse'])[0]
        control_price = stat('control', ['validation', 'metrics', 'price', 'rmse'])[0]
        greeks_ok = all(
            stat(arm, ['validation', 'metrics', name, 'rmse'])[0]
            <= 1.15 * stat('control', ['validation', 'metrics', name, 'rmse'])[0] for name in NAMES[1:])
        return price <= 1.1 * control_price and greeks_ok

    lines = ['# DeepONet training ablations', '',
             'Three single-factor comparisons under one compute-matched two-stage training procedure '
             '(price-only pilot of 8,000 optimizer steps, then supervised continuation of 12,800 steps '
             'at Greek-loss weight 0.01), three seeds each, identical optimizer-step budgets per run. '
             'Model selection used training '
             'and validation data only. The fresh evaluation package (in-domain, stress, near-expiry/'
             'near-strike focus, paired call/put rows) was opened once after all selection decisions and '
             'is now observed; a later independent assessment needs fresh cases. The frozen final '
             'benchmark was not accessed.', '',
             f"Penalty weight selected on validation: {chosen['weight']} "
             f"({'gate failed' if chosen['gate_failed'] else 'gate passed'}). Acceptability rule: "
             + ACCEPT_RULE, '',
             '## 1. Call-put parity by construction', '',
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
    lines += ['', 'Validation shape violations (frequency / severity):', '',
              '| Constraint | Control | Parity |', '|---|---|---|']
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        control_cells = violation_cells('control', constraint, 'validation')
        parity_cells = violation_cells('parity', constraint, 'validation')
        lines.append(f'| {constraint} | {" / ".join(control_cells)} | {" / ".join(parity_cells)} |')
    lines += ['', f"Parity arm acceptable under the rule: **{acceptable('parity')}**.", '']

    lines += ['## 2. Bound-violation penalties', '',
              'Each penalty arm adds one squared normalized violation term (all three for penalty_all) '
              f'with weight {chosen["weight"]} to the supervised loss during the continuation stage; '
              'stage 1 is unchanged for every arm. Validation violations (frequency / severity):', '',
              '| Arm | Price | Delta | Gamma | Price RMSE | Acceptable |', '|---|---|---|---|---:|---|']
    for arm in ('control',) + PENALTY_ARMS:
        cells = [' / '.join(violation_cells(arm, constraint, 'validation'))
                 for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        price = stat(arm, ['validation', 'metrics', 'price', 'rmse'])[0]
        lines.append(f'| {arm} | ' + ' | '.join(cells) + f' | {price:.6g} | {acceptable(arm)} |')
    lines += ['', 'Fresh stress-set violations (observed benchmark):', '',
              '| Arm | Price | Delta | Gamma |', '|---|---|---|']
    for arm in ('control',) + PENALTY_ARMS:
        cells = [' / '.join(violation_cells(arm, constraint, 'stress'))
                 for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma')]
        lines.append(f'| {arm} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation RMSE (mean ± SD):', '', '| Quantity | Control | '
              + ' | '.join(PENALTY_ARMS) + ' |', '|---|---:|' + '---:|' * len(PENALTY_ARMS)]
    for name in NAMES:
        cells = rmse_cells(['validation', 'metrics', name, 'rmse'], ('control',) + PENALTY_ARMS)
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')

    lines += ['', '## 3. Near-expiry and near-strike sampling', '',
              'The mixed arm replaces half of each contract\'s uniform coordinates with concentrations '
              'near the strike, near expiry and near both, at identical dataset size and identical '
              'contracts. Focus-set RMSE by region (fresh cases, mean ± SD):', '',
              '| Quantity | Region | Uniform | Mixed |', '|---|---|---:|---:|']
    for region in ('all', 'near_strike', 'near_expiry', 'near_strike_and_expiry'):
        for name in NAMES:
            cells = rmse_cells(['focus', 'regions', region, name], ('uniform', 'mixed'))
            lines.append(f'| {name} | {region} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation RMSE:', '', '| Quantity | Uniform | Mixed |', '|---|---:|---:|']
    for name in NAMES:
        cells = rmse_cells(['validation', 'metrics', name, 'rmse'], ('uniform', 'mixed'))
        lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Validation shape violations (frequency / severity):', '',
              '| Constraint | Uniform | Mixed |', '|---|---|---|']
    for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
        uniform_cells = violation_cells('uniform', constraint, 'validation')
        mixed_cells = violation_cells('mixed', constraint, 'validation')
        lines.append(f'| {constraint} | {" / ".join(uniform_cells)} | {" / ".join(mixed_cells)} |')
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    save_json(output / 'completion.json', dict(status='complete', final_test_accessed=False,
                                               penalty_weight=chosen,
                                               evaluation_status='observed_after_selection'))


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
    arrays_by_key = {'frozen': dict(file=SETTINGS_JSON['dataset'],
                                    content_sha256=content_sha256({key: value for key, value in frozen.items()
                                                                   if key.startswith('train_')}))}
    for name in splits:
        with np.load(data_dir / f'train_{name}.npz', allow_pickle=False) as data:
            arrays_by_key[name] = dict(file=str((data_dir / f'train_{name}.npz').relative_to(ROOT)),
                                       content_sha256=content_sha256({key: data[key].copy() for key in data.files
                                                                      if key.startswith('train_')}))
    composition = {name: {region: int(np.sum(split['region'] == code))
                          for code, region in enumerate(('broad', 'near_strike', 'near_expiry',
                                                         'near_strike_and_expiry'))}
                   for name, split in splits.items()}
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/train.py',
                 ROOT / 'experiments/deeponet/compare_supervision.py',
                 ROOT / 'experiments/deeponet/refine_optimizer.py', ROOT / 'src/neural/deeponet.py',
                 ROOT / 'src/neural/deeponet_data.py', ROOT / 'src/black_scholes.py', SETTINGS_PATH):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    protocol = dict(
        protocol_id='deeponet_training_ablation_v2',
        factors={'parity': 'architecture: shared midpoint with call-put parity by construction',
                 'penalties': 'loss: squared normalized bound violations for price, Delta and negative Gamma',
                 'sampling': 'data: equal-sized coordinate sets, uniform versus near-expiry/near-strike concentration'},
        arms={name: dict(arch=spec['arch'], data=spec['data'], penalties=list(spec['penalties']))
              for name, spec in ARMS.items()},
        seeds=SEEDS, greek_weight=GREEK_WEIGHT,
        procedure=dict(stage1='price-only pilot: 500 epochs (8,000 steps), Adam lr 0.001, batch 512',
                       stage2='supervised continuation: 800 epochs (12,800 steps), Adam lr 0.0003, batch 512',
                       selection='minimum validation price MSE normalized by strike',
                       threads=2, concurrent_jobs=concurrency,
                       budget='compute-matched: optimizer steps fixed per run',
                       note='Compute-matched budgets: optimizer steps are fixed per run. Wall-clock budgets '
                            'proved unstable on this host (a prior campaign measured a 4.8x step-count spread '
                            'between identical runs), so equality of compute is enforced by steps.'),
        penalty_weight_candidates=PENALTY_WEIGHTS, penalty_weight_rule=LAMBDA_RULE,
        acceptability_rule=ACCEPT_RULE,
        sampling=dict(contract_seed=SAMPLING_CONTRACT_SEED, coordinate_seeds=SAMPLING_COORDINATE_SEEDS,
                      contracts_shared_between_arms=True, coordinates_per_contract=32,
                      train_composition=composition,
                      note='Equal dataset sizes; only the coordinate distribution differs.'),
        evaluation=dict(status='fresh_cases_opened_once_after_all_selection_decisions; observed once read',
                        seeds=EVAL_SEEDS, sets=['in_domain', 'stress', 'focus', 'pairs'],
                        file=str((data_dir / 'evaluation.npz').relative_to(ROOT)),
                        content_sha256=content_sha256(evaluation),
                        note='A later independent assessment needs fresh cases; these are now observed.'),
        dataset_arrays=arrays_by_key, source_sha256=sources,
        final_test_accessed=False, python=platform.python_version(), torch=str(torch.__version__))
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output)}), flush=True)
    return output


def run_study(concurrency, resume=None):
    output = resume if resume is not None else run_setup(concurrency)
    arrays_by_key = json.loads((output / 'protocol.json').read_text())['dataset_arrays']
    data_dir = output / 'data'
    jobs_dir = output / 'jobs'
    pilots = {}
    for arm, spec in ARMS.items():
        pilots.setdefault((spec['arch'], spec['data']), []).append(arm)
    pilot_specs = [dict(name=f'pilot/{arch}_{data_key}_{seed}', stage='pilot', arch=arch,
                        data=data_key, seed=seed, threads=2,
                        folder=f'pilot/{arch}_{data_key}_{seed}', arrays=arrays_by_key[data_key])
                   for (arch, data_key), _ in sorted(pilots.items()) for seed in SEEDS]
    launch(pilot_specs, output, jobs_dir, concurrency)
    stage2_specs = [continuation_spec(arm, ARMS[arm], seed, (), 0., arrays_by_key)
                    for arm in ('control', 'parity', 'uniform', 'mixed') for seed in SEEDS]
    stage2_specs += [continuation_spec(f'lambda_sweep/weight_{weight}', ARMS['penalty_all'], 22092026,
                                       ('price', 'delta', 'gamma'), weight, arrays_by_key)
                     for weight in PENALTY_WEIGHTS]
    launch(stage2_specs, output, jobs_dir, concurrency)
    chosen = select_penalty_weight(output, PENALTY_WEIGHTS)
    save_json(output / 'penalty_weight_selection.json', chosen)
    print(json.dumps({'penalty_weight': chosen}), flush=True)
    penalty_specs = [continuation_spec(arm, ARMS[arm], seed, ARMS[arm]['penalties'], chosen['weight'],
                                       arrays_by_key)
                     for arm in PENALTY_ARMS for seed in SEEDS
                     if not (arm == 'penalty_all' and seed == 22092026)]
    launch(penalty_specs, output, jobs_dir, concurrency)
    run_folders = {(arm, seed): (ROOT / chosen['folder'] if arm == 'penalty_all' and seed == 22092026
                                 else output / f'arms/{arm}_{seed}')
                   for arm in ARMS for seed in SEEDS}
    with np.load(data_dir / 'evaluation.npz', allow_pickle=False) as data:
        evaluation = {key: data[key].copy() for key in data.files}
    evaluate(output, run_folders, evaluation, chosen)
    print(json.dumps({'complete': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job-file', type=Path)
    parser.add_argument('--study', type=Path)
    parser.add_argument('--jobs', type=int, default=3,
                        help='concurrent training jobs (each uses 2 torch threads)')
    parser.add_argument('--resume', type=Path,
                        help='continue an existing study folder, skipping completed jobs')
    args = parser.parse_args()
    if args.job_file:
        run_job(json.loads(args.job_file.read_text()), ROOT / args.study)
        return
    run_study(args.jobs, args.resume)


if __name__ == '__main__':
    main()
