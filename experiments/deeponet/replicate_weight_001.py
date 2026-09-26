import json
import platform
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural.deeponet import load_deeponet


WEIGHT = 0.01
SEEDS = (22092027, 22092028)
PRIOR = ROOT / 'results/deeponet/weight_sweeps/20260922T142444840021Z'


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verified_checkpoint(path):
    metadata = json.loads((path.parent / 'training_metadata.json').read_text())
    if digest(path) != metadata['checkpoint_sha256'][path.stem]:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    return metadata


def main():
    protocol = json.loads((PRIOR / 'protocol.json').read_text())
    reference = json.loads((PRIOR / 'seed_results.json').read_text())
    old_candidate = json.loads((PRIOR / 'selection.json').read_text())['candidates'][0]
    if old_candidate['weight'] != WEIGHT:
        raise ValueError('Expected 0.01 reference weight')
    dataset = ROOT / 'results/deeponet/data/black_scholes_operator_v1.npz'
    if digest(dataset) != protocol['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    output = ROOT / 'results/deeponet/weight_001_replication' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/train.py',
                 ROOT / 'experiments/deeponet/compare_supervision.py', ROOT / 'src/neural/deeponet.py'):
        relative = path.resolve().relative_to(ROOT)
        destination = output / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[str(relative)] = digest(path)
    torch.set_num_threads(2)
    with np.load(dataset, allow_pickle=False) as data:
        b, t, y = (torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                   for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    scales = y[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    if not np.allclose(scales.numpy(), protocol['greek_scales'], rtol=0, atol=1e-14):
        raise ValueError('Greek scales differ from prior study')
    retained = {}
    for row in reference:
        checkpoint = ROOT / row['checkpoint']
        verified_checkpoint(checkpoint)
        retained[str(checkpoint.relative_to(ROOT))] = digest(checkpoint)
    original = ROOT / old_candidate['checkpoint']
    verified_checkpoint(original)
    retained[str(original.relative_to(ROOT))] = digest(original)
    parents = {}
    for seed in SEEDS:
        parent = PRIOR / f'seed_{seed}/pilot/selected.pt'
        metadata = verified_checkpoint(parent)
        if metadata['settings']['seed'] != seed or metadata['dataset_sha256'] != digest(dataset):
            raise ValueError('Parent pilot mismatch')
        parents[seed] = parent
        retained[str(parent.relative_to(ROOT))] = digest(parent)
    settings = dict(verified_checkpoint(original)['settings'])
    if settings['greek_weight'] != WEIGHT:
        raise ValueError('Reference objective mismatch')
    for seed in SEEDS:
        prior_settings = verified_checkpoint(ROOT / reference[1 if seed == SEEDS[0] else 2]['checkpoint'])['settings']
        for key in ('wall_seconds', 'learning_rate', 'batch_size', 'validation_every', 'architecture', 'threads'):
            if settings[key] != prior_settings[key]:
                raise ValueError(f'Pairing mismatch for {key}')
    run_protocol = dict(weight=WEIGHT, seeds=SEEDS, dataset_sha256=digest(dataset),
                        source_sha256=sources, greek_scales=scales.tolist(),
                        retained_checkpoint_sha256=retained, parent_study=str(PRIOR.relative_to(ROOT)),
                        selection_rule='Pair each seed with weight 1 at equal budget. Prefer weight 0.01 if mean price RMSE is lower and each of Gamma, Theta and Rho mean RMSE is within 20% of weight 1; otherwise retain weight 1.',
                        final_test_accessed=False, python=platform.python_version(), torch=str(torch.__version__))
    save_json(output / 'protocol.json', run_protocol)
    results = [dict(seed=reference[0]['seed'], metrics=old_candidate['metrics'],
                    checkpoint=str(original.relative_to(ROOT)), role='selection_seed')]
    print(json.dumps({'output': str(output), 'weight': WEIGHT, 'seeds': SEEDS}), flush=True)
    for seed in SEEDS:
        parent = parents[seed]
        folder = output / f'seed_{seed}'
        folder.mkdir()
        config = {**settings, 'seed': seed, 'progress_every_seconds': 45}
        model = load_deeponet(parent)
        torch.manual_seed(seed)
        loss = lambda m, cb, ct, targets: supervised_loss(m, cb, ct, targets, scales, WEIGHT)
        print(json.dumps({'starting_seed': seed, 'parent': str(parent)}), flush=True)
        history, last, summary = train_model(model, (b, t, y),
                                             (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64)),
                                             config, loss)
        metadata = dict(run_protocol=run_protocol, settings=config, **summary,
                        dataset_sha256=digest(dataset), parent_checkpoint=str(parent.relative_to(ROOT)),
                        parent_sha256=digest(parent), label_columns_used=list(NAMES))
        for kind, state in (('selected', model.state_dict()), ('last', last)):
            torch.save(dict(architecture=model.architecture, state_dict=state, metadata=metadata),
                       folder / f'{kind}.pt')
        metadata['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
        save_json(folder / 'training_metadata.json', metadata)
        write_csv(folder / 'history.csv', history)
        metrics = diagnostics(model, vb, vt, vy, folder)
        save_json(folder / 'validation_metrics.json', metrics)
        results.append(dict(seed=seed, metrics=metrics, checkpoint=str((folder / 'selected.pt').relative_to(ROOT)),
                            role='fresh_seed'))
        save_json(output / 'seed_results.json', results)
        print(json.dumps({'completed_seed': seed, 'metrics': metrics}), flush=True)
    for path, expected in retained.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'Retained checkpoint changed: {path}')
    prior_by_seed = {row['seed']: row for row in reference}
    summary = {}
    for name in NAMES:
        new = np.array([row['metrics'][name]['rmse'] for row in results])
        old = np.array([prior_by_seed[row['seed']]['metrics'][name]['rmse'] for row in results])
        summary[name] = dict(weight_001_mean=float(new.mean()), weight_001_std=float(new.std(ddof=1)),
                             weight_1_mean=float(old.mean()), weight_1_std=float(old.std(ddof=1)),
                             paired_differences=(new-old).tolist())
    choose_001 = (summary['price']['weight_001_mean'] < summary['price']['weight_1_mean']
                  and all(summary[name]['weight_001_mean'] <= 1.2*summary[name]['weight_1_mean']
                          for name in ('gamma', 'theta', 'rho')))
    selected = 'weight_0.01' if choose_001 else 'weight_1'
    save_json(output / 'paired_summary.json', dict(metrics=summary, selected_family=selected,
                                                   selection_rule=run_protocol['selection_rule']))
    lines = ['# Paired DeepONet Greek weight replication', '',
             f'Selected model family: **{selected}** by the prespecified validation rule. '
             'This is a three-seed exploratory result; one seed participated in weight selection. '
             'The final test set was not accessed.', '',
             '| Quantity | Weight 0.01 mean ± SD | Weight 1 mean ± SD |', '|---|---:|---:|']
    for name, item in summary.items():
        lines.append(f"| {name} | {item['weight_001_mean']:.6g} ± {item['weight_001_std']:.6g} | "
                     f"{item['weight_1_mean']:.6g} ± {item['weight_1_std']:.6g} |")
    lines += ['', '| Seed | Weight 0.01 price RMSE | Weight 1 price RMSE |', '|---|---:|---:|']
    for row in results:
        old = prior_by_seed[row['seed']]
        lines.append(f"| {row['seed']} | {row['metrics']['price']['rmse']:.6g} | {old['metrics']['price']['rmse']:.6g} |")
    lines += ['', 'Both weights used the same pilot checkpoint, seed, learning rate, batch size and '
              '300-second continuation budget within each pair. The original seed reused its previously '
              'completed runs; the two fresh seeds were newly continued at weight 0.01. '
              'Training data and analytical Greek scales are identical. '
              'The selected model family is a validation decision, not a final test result.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    save_json(output / 'completion.json', dict(status='complete', retained_hashes_verified=True,
                                               final_test_accessed=False, selected_family=selected))
    print(json.dumps({'complete': str(output), 'selected_family': selected, 'summary': summary}), flush=True)


if __name__ == '__main__':
    main()
