import argparse
import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural.deeponet import BlackScholesDeepONet, load_deeponet


SETTINGS = ROOT / 'experiments/config/neural/deeponet_weight_sweep.json'


def choose_weight(candidates, price_limit, scales):
    rows = []
    for candidate in candidates:
        metrics = candidate['metrics']
        score = float(np.mean([(metrics[name]['rmse'] / scale) ** 2
                               for name, scale in zip(NAMES[1:], scales)]))
        rows.append({**candidate, 'greek_score': score,
                     'price_gate_passed': metrics['price']['rmse'] <= price_limit})
    eligible = [row for row in rows if row['price_gate_passed']]
    selected = min(eligible, key=lambda row: (row['greek_score'], row['weight'])) if eligible else min(
        rows, key=lambda row: (row['metrics']['price']['rmse'], row['weight']))
    return selected, rows


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--settings', type=Path, default=SETTINGS)
    args = parser.parse_args()
    config = json.loads(args.settings.read_text())
    pilot = ROOT / config['pilot']
    reference = ROOT / config['reference_comparison']
    pilot_meta = json.loads((pilot / 'training_metadata.json').read_text())
    base_settings = pilot_meta['settings']
    dataset = ROOT / base_settings['dataset']
    if digest(dataset) != pilot_meta['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    retained = {}
    for folder in (pilot, reference / 'price_only', reference / 'price_and_greeks'):
        meta = json.loads((folder / 'training_metadata.json').read_text())
        if meta['dataset_sha256'] != digest(dataset):
            raise ValueError('Reference dataset mismatch')
        for kind in ('selected', 'last'):
            path = folder / f'{kind}.pt'
            if digest(path) != meta['checkpoint_sha256'][kind]:
                raise ValueError(f'Reference checkpoint mismatch: {path}')
            retained[str(path.relative_to(ROOT))] = digest(path)
    output = ROOT / 'results/deeponet/weight_sweeps' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    source_hashes = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/train.py',
                 ROOT / 'experiments/deeponet/compare_supervision.py', ROOT / 'src/neural/deeponet.py', args.settings):
        relative = path.resolve().relative_to(ROOT)
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        source_hashes[str(relative)] = digest(path)
    torch.set_num_threads(base_settings['threads'])
    with np.load(dataset, allow_pickle=False) as data:
        b, t, y = [torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                   for key in ('branch', 'trunk', 'targets')]
        vb, vt = [torch.tensor(data[f'validation_{key}'], dtype=torch.float64) for key in ('branch', 'trunk')]
        vy = data['validation_targets'].copy()
    scales = y[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    control_metrics = json.loads((reference / 'price_only/validation_metrics.json').read_text())
    price_limit = config['price_rmse_limit_relative_to_price_only'] * control_metrics['price']['rmse']
    protocol = dict(config=config, price_rmse_limit=price_limit, greek_scales=scales.tolist(),
                    retained_checkpoints=retained, dataset_sha256=digest(dataset), source_sha256=source_hashes,
                    final_test_accessed=False, python=platform.python_version(), torch=str(torch.__version__),
                    device='cpu', dtype='float64')
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output), 'price_limit': price_limit}), flush=True)

    def run(folder, model, settings, weight, parent=None):
        folder.mkdir(parents=True, exist_ok=False)
        settings = {**settings, 'progress_every_seconds': 45,
                    'loss': 'price_mse_over_strike_squared' if weight is None else 'price_mse_over_strike_squared_plus_weighted_training_rms_greek_mse',
                    'greek_weight': weight, 'protocol_id': config['protocol_id'],
                    'note': 'Validation-only development; final test excluded'}
        torch.manual_seed(settings['seed'])
        training = (b, t, y[:, 0] if weight is None else y)
        loss = None if weight is None else lambda m, cb, ct, targets: supervised_loss(m, cb, ct, targets, scales, weight)
        print(json.dumps({'starting': str(folder.relative_to(output)), 'seed': settings['seed'], 'weight': weight}), flush=True)
        history, last, summary = train_model(model, training, validation, settings, loss)
        metadata = {**protocol, **summary, 'settings': settings,
                    'parent_checkpoint': None if parent is None else str(parent.relative_to(ROOT)),
                    'parent_sha256': None if parent is None else digest(parent),
                    'label_columns_used': ['price'] if weight is None else list(NAMES)}
        for kind, state in (('selected', model.state_dict()), ('last', last)):
            torch.save(dict(architecture=model.architecture, state_dict=state, metadata=metadata), folder / f'{kind}.pt')
        metadata['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
        save_json(folder / 'training_metadata.json', metadata)
        write_csv(folder / 'history.csv', history)
        metrics = diagnostics(model, vb, vt, vy, folder)
        save_json(folder / 'validation_metrics.json', metrics)
        print(json.dumps({'completed': str(folder.relative_to(output)), 'metrics': metrics}), flush=True)
        return metrics

    continuation = {**base_settings, 'epochs': 100000, 'wall_seconds': config['continuation_seconds'],
                    'learning_rate': config['learning_rate']}
    candidates = []
    for weight in config['weights']:
        if weight == 1 and config['reuse_weight_one_reference']:
            folder = reference / 'price_and_greeks'
            meta = json.loads((folder / 'training_metadata.json').read_text())
            if meta['parent_sha256'] != digest(pilot / 'selected.pt'):
                raise ValueError('Reference parent mismatch')
            for key in ('wall_seconds', 'learning_rate', 'seed', 'architecture', 'batch_size', 'validation_every', 'threads'):
                if meta['settings'][key] != continuation[key]:
                    raise ValueError(f'Reference settings mismatch: {key}')
            metrics = json.loads((folder / 'validation_metrics.json').read_text())
        else:
            folder = output / f'weight_{weight:g}'
            metrics = run(folder, load_deeponet(pilot / 'selected.pt'), continuation, weight, pilot / 'selected.pt')
        candidates.append(dict(weight=weight, metrics=metrics, checkpoint=str((folder / 'selected.pt').relative_to(ROOT))))
    selected, rows = choose_weight(candidates, price_limit, scales.numpy())
    save_json(output / 'selection.json', dict(selected=selected, candidates=rows, rule=config['selection']))
    print(json.dumps({'selected_weight': selected['weight'], 'price_gate_passed': selected['price_gate_passed']}), flush=True)
    repetitions = [dict(seed=base_settings['seed'], role='selection_seed', metrics=selected['metrics'], checkpoint=selected['checkpoint'])]
    for seed in config['replication_seeds']:
        torch.manual_seed(seed)
        pilot_folder = output / f'seed_{seed}/pilot'
        run(pilot_folder, BlackScholesDeepONet(**base_settings['architecture']),
            {**base_settings, 'seed': seed}, None)
        parent = pilot_folder / 'selected.pt'
        folder = output / f'seed_{seed}/weighted'
        metrics = run(folder, load_deeponet(parent), {**continuation, 'seed': seed}, selected['weight'], parent)
        repetitions.append(dict(seed=seed, role='fresh_seed', metrics=metrics,
                                checkpoint=str((folder / 'selected.pt').relative_to(ROOT))))
        save_json(output / 'seed_results.json', repetitions)
    aggregate = {}
    for name in NAMES:
        values = [row['metrics'][name]['rmse'] for row in repetitions]
        aggregate[name] = dict(mean=float(np.mean(values)), sample_std=float(np.std(values, ddof=1)), seeds=len(values))
    save_json(output / 'seed_aggregate.json', aggregate)
    if any(digest(ROOT / path) != value for path, value in retained.items()):
        raise ValueError('A retained checkpoint changed during the study')
    save_json(output / 'completion.json', dict(status='complete', retained_hashes_verified=True, final_test_accessed=False))
    lines = ['# DeepONet Greek-loss weight study', '',
             f"Selected weight: {selected['weight']:g}; price gate passed: {selected['price_gate_passed']}. "
             f'Price RMSE limit: {price_limit:.6g}. Each continuation received 300 CPU seconds.', '',
             '| Weight | Price RMSE | Greek score | Passes price gate |', '|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['weight']:g} | {row['metrics']['price']['rmse']:.6g} | {row['greek_score']:.6g} | {row['price_gate_passed']} |")
    lines += ['', '## Replication', '',
              'Three seeds: the selection seed and two new full-pipeline initializations. '
              'The dataset is fixed. This measures training variability, not uncertainty over datasets. '
              'The weight was fixed before new-seed runs. The final test was not accessed.', '',
              '| Quantity | RMSE mean | Sample standard deviation |', '|---|---:|---:|']
    for name, value in aggregate.items():
        lines.append(f"| {name} | {value['mean']:.6g} | {value['sample_std']:.6g} |")
    lines += ['', 'The original price-only and weight-one checkpoints were retained and their hashes rechecked. '
              'The weight-one arm was reused from the previous matched configuration; it was not rerun. '
              'There are no paired price-only continuations for the two new seeds, so superiority over '
              'price-only training across seeds is not established.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'complete': str(output), 'aggregate': aggregate}), flush=True)


if __name__ == '__main__':
    main()
