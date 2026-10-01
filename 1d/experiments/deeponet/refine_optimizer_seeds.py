"""Cross-seed replication of the DeepONet optimizer refinement procedure.

Repeats the two-arm refinement (small-step Adam vs full-batch L-BFGS, identical
budgets and selection rule) from results/deeponet/optimizer_refinement for the
remaining saved weight-0.01 seed checkpoints, then aggregates all three seeds
for a dependability verdict. Selection uses validation data only. The fresh
audit opened by the first refinement is reused as an observed benchmark for
cross-seed comparability; it is not an untouched test.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.refine_optimizer import lbfgs_refine, shape_metrics
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural.deeponet import load_deeponet


PRIOR = ROOT / 'results/deeponet/optimizer_refinement/20260923T090451734503Z'
PRIOR_SEED = 22092027
PARENTS = {
    22092026: ROOT / 'results/deeponet/weight_sweeps/20260922T142444840021Z/weight_0.01/selected.pt',
    22092028: ROOT / 'results/deeponet/weight_001_replication/20260923T084432489626Z/seed_22092028/selected.pt',
}
ARMS = ('adam_small_steps', 'lbfgs')
SELECTION_RULE = ('Each Greek RMSE must remain within 10% of the parent model, '
                  'then minimize price RMSE among baseline and refined arms.')
VERDICT_RULE = {
    'refinement_improves_price': 'Every seed selects a refined arm with lower price RMSE than its parent.',
    'lbfgs_dependable': ('L-BFGS is the selected arm on at least two of three seeds and lowers '
                         'price RMSE versus the parent on all three seeds.'),
}


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def verified_parent(path, seed, dataset_sha):
    metadata = json.loads((path.parent / 'training_metadata.json').read_text())
    if digest(path) != metadata['checkpoint_sha256']['selected']:
        raise ValueError(f'Parent checkpoint hash mismatch: {path}')
    if metadata['settings']['seed'] != seed or metadata['dataset_sha256'] != dataset_sha:
        raise ValueError(f'Parent metadata mismatch for seed {seed}')
    return metadata


def load_audit(path):
    arrays = {}
    with np.load(path, allow_pickle=False) as data:
        for split in ('in_domain', 'stress'):
            arrays[split] = {key: data[f'{split}_{key}'].copy()
                            for key in ('branch', 'trunk', 'targets')}
    return arrays


def prior_record():
    summary = json.loads((PRIOR / 'summary.json').read_text())
    selection = json.loads((PRIOR / 'selection.json').read_text())
    record = dict(seed=PRIOR_SEED, selection=selection['selected'], eligible=selection['eligible'],
                  validation=selection['validation'],
                  shape={arm: json.loads((PRIOR / arm / 'shape_metrics.json').read_text())
                         for arm in ('baseline', 'adam_small_steps', 'lbfgs')},
                  observed_audit=summary['fresh_audit'])
    record['audit_shape'] = {
        split: {name: json.loads((PRIOR / 'audit' / name / split / 'shape_metrics.json').read_text())
                for name in ('baseline', 'selected')} for split in ('in_domain', 'stress')}
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--finalize', type=Path,
                        help='rebuild aggregate, verdict and report for an existing study folder')
    args = parser.parse_args()
    if args.finalize:
        output = args.finalize
        protocol = json.loads((output / 'protocol.json').read_text())
        prior_protocol = json.loads((PRIOR / 'protocol.json').read_text())
        records = json.loads((output / 'seed_results.json').read_text())
        parents = {int(seed): (ROOT / entry['checkpoint'], None)
                   for seed, entry in protocol['parents'].items()}
        finalize(output, records, parents, protocol, ROOT / protocol['observed_audit']['file'], prior_protocol)
        return
    prior_protocol = json.loads((PRIOR / 'protocol.json').read_text())
    audit_path = PRIOR / 'fresh_audit.npz'
    if digest(audit_path) != prior_protocol['fresh_audit_sha256']:
        raise ValueError('Observed audit hash mismatch')
    registry = json.loads((ROOT / 'experiments/config/neural/deeponet_model_registry.json').read_text())
    if registry['primary']['checkpoint'].replace('\\', '/') != prior_protocol['parent_checkpoint'].replace('\\', '/'):
        raise ValueError('Prior refinement parent differs from registry primary')
    dataset = ROOT / json.loads((PARENTS[22092026].parent / 'training_metadata.json').read_text())['settings']['dataset']
    if digest(dataset) != prior_protocol['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    parents = {seed: (path, verified_parent(path, seed, prior_protocol['dataset_sha256']))
               for seed, path in PARENTS.items()}
    scales_reference = prior_protocol['greek_scales']
    output = ROOT / 'results/deeponet/optimizer_refinement_seeds' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with np.load(dataset, allow_pickle=False) as data:
        train = tuple(torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                      for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    scales = train[2][:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    if not np.allclose(scales.numpy(), scales_reference, rtol=0, atol=1e-14):
        raise ValueError('Greek scales differ from prior refinement')
    audit = load_audit(audit_path)
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/refine_optimizer.py',
                 ROOT / 'experiments/deeponet/train.py', ROOT / 'experiments/deeponet/compare_supervision.py',
                 ROOT / 'src/neural/deeponet.py', ROOT / 'src/neural/deeponet_data.py',
                 ROOT / 'src/black_scholes.py'):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    protocol = dict(
        protocol_id='deeponet_optimizer_refinement_seeds_v1',
        purpose='Repeat the two-arm optimizer refinement on the other saved weight-0.01 seeds and judge dependability across seeds.',
        prior_refinement=str(PRIOR.relative_to(ROOT)), prior_refinement_summary_sha256=digest(PRIOR / 'summary.json'),
        prior_seed=PRIOR_SEED,
        parents={str(seed): dict(checkpoint=str(path.relative_to(ROOT)), sha256=digest(path))
                 for seed, (path, _) in parents.items()},
        dataset_sha256=digest(dataset), source_sha256=sources, greek_scales=scales.tolist(),
        wall_seconds_per_arm=180, refinement_seed=23092026, greek_weight=.01,
        lbfgs_loss_multiplier=10000, lbfgs_chunk_size=1024,
        selection_rule=SELECTION_RULE, verdict_rule=VERDICT_RULE,
        observed_audit=dict(file=str(audit_path.relative_to(ROOT)), sha256=digest(audit_path),
                            status='observed_benchmark_reused_for_cross_seed_comparability',
                            selection_uses='validation_only',
                            note='This audit was opened by the prior refinement; it is not an untouched test.'),
        final_test_accessed=False)
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output)}), flush=True)
    records = []
    for seed, (parent, metadata) in parents.items():
        seed_dir = output / f'seed_{seed}'
        seed_dir.mkdir()
        base_dir = seed_dir / 'baseline'
        base_dir.mkdir()
        baseline = load_deeponet(parent)
        results = {'baseline': diagnostics(baseline, vb, vt, vy, base_dir)}
        save_json(base_dir / 'shape_metrics.json', shape_metrics(baseline, vb, vt))
        for arm in ARMS:
            folder = seed_dir / arm
            folder.mkdir()
            model = load_deeponet(parent)
            torch.manual_seed(protocol['refinement_seed'])
            settings = {**metadata['settings'], 'seed': protocol['refinement_seed'], 'epochs': 100000,
                        'wall_seconds': 180, 'learning_rate': .00002, 'progress_every_seconds': 45,
                        'optimizer': arm, 'protocol_id': 'deeponet_optimizer_refinement_seeds_v1'}
            print(json.dumps({'starting_seed': seed, 'arm': arm}), flush=True)
            if arm == 'lbfgs':
                settings.update(learning_rate=1., batch_size=len(train[0]), loss_multiplier=1e4,
                                chunk_size=1024, history_size=50, max_iter=1, line_search_fn='strong_wolfe')
                history, last, summary = lbfgs_refine(model, train, validation, scales, 180)
            else:
                loss = lambda m, b, t, y: supervised_loss(m, b, t, y, scales, .01)
                history, last, summary = train_model(model, train, validation, settings, loss)
            record = {**protocol, **summary, 'settings': settings, 'seed': seed,
                      'parent_checkpoint': str(parent.relative_to(ROOT)), 'parent_sha256': digest(parent),
                      'label_columns_used': list(NAMES), 'final_test_accessed': False}
            for kind, state in (('selected', model.state_dict()), ('last', last)):
                torch.save(dict(architecture=model.architecture, state_dict=state, metadata=record),
                           folder / f'{kind}.pt')
            record['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
            save_json(folder / 'training_metadata.json', record)
            write_csv(folder / 'history.csv', history)
            results[arm] = diagnostics(model, vb, vt, vy, folder)
            save_json(folder / 'validation_metrics.json', results[arm])
            save_json(folder / 'shape_metrics.json', shape_metrics(model, vb, vt))
            print(json.dumps({'completed_seed': seed, 'arm': arm, 'price_rmse': results[arm]['price']['rmse']}), flush=True)
        eligible = [name for name, metrics in results.items() if all(
            metrics[greek]['rmse'] <= 1.1 * results['baseline'][greek]['rmse'] for greek in NAMES[1:])]
        selected = min(eligible, key=lambda name: results[name]['price']['rmse'])
        save_json(seed_dir / 'selection.json', dict(seed=seed, selected=selected, eligible=eligible,
                                                    validation=results, rule=SELECTION_RULE))
        checkpoint = parent if selected == 'baseline' else seed_dir / selected / 'selected.pt'
        selected_model = load_deeponet(checkpoint)
        audit_shape = {}
        audit_results = {}
        for name, path in (('baseline', parent), ('selected', checkpoint)):
            model = baseline if path == parent else selected_model
            audit_results[name] = {}
            audit_shape[name] = {}
            for split in ('in_domain', 'stress'):
                values = audit[split]
                folder = seed_dir / 'audit' / name / split
                folder.mkdir(parents=True)
                audit_results[name][split] = diagnostics(
                    model, torch.tensor(values['branch'], dtype=torch.float64),
                    torch.tensor(values['trunk'], dtype=torch.float64), values['targets'], folder)
                audit_shape[name][split] = shape_metrics(
                    model, torch.tensor(values['branch'], dtype=torch.float64),
                    torch.tensor(values['trunk'], dtype=torch.float64))
                save_json(folder / 'shape_metrics.json', audit_shape[name][split])
        records.append(dict(seed=seed, selection=selected, eligible=eligible, validation=results,
                            shape={arm: json.loads((seed_dir / arm / 'shape_metrics.json').read_text())
                                   for arm in ('baseline',) + ARMS},
                            observed_audit=audit_results,
                            audit_shape={split: {name: audit_shape[name][split] for name in audit_shape}
                                         for split in ('in_domain', 'stress')},
                            selected_checkpoint=str(checkpoint.relative_to(ROOT))))
        save_json(output / 'seed_results.json', records)
    finalize(output, records, parents, protocol, audit_path, prior_protocol)


def finalize(output, records, parents, protocol, audit_path, prior_protocol):
    records = [*records, prior_record()]
    records.sort(key=lambda row: row['seed'])
    for seed, (parent, _) in parents.items():
        if digest(parent) != protocol['parents'][str(seed)]['sha256']:
            raise ValueError(f'Parent checkpoint changed during refinement: seed {seed}')
    if digest(audit_path) != prior_protocol['fresh_audit_sha256']:
        raise ValueError('Observed audit changed during refinement')
    aggregate = {}
    for name in NAMES:
        aggregate[name] = dict(
            baseline=[row['validation']['baseline'][name]['rmse'] for row in records],
            selected=[row['validation'][row['selection']][name]['rmse'] for row in records])
    relative_price = [(row['validation'][row['selection']]['price']['rmse']
                       / row['validation']['baseline']['price']['rmse'] - 1.) for row in records]
    refined_everywhere = all(row['selection'] != 'baseline' for row in records)
    lbfgs_verdict = (sum(row['selection'] == 'lbfgs' for row in records) >= 2 and all(
        row['validation']['lbfgs']['price']['rmse'] < row['validation']['baseline']['price']['rmse']
        for row in records))
    verdict = dict(refinement_improves_price=refined_everywhere and all(value < 0 for value in relative_price),
                   lbfgs_dependable=lbfgs_verdict, relative_price_rmse_change=relative_price,
                   rule=VERDICT_RULE)
    save_json(output / 'seed_aggregate.json', dict(per_seed=records, aggregate=aggregate, verdict=verdict))
    lines = ['# DeepONet optimizer refinement across seeds', '',
             'The two-arm refinement (small-step Adam vs full-batch L-BFGS, 180 seconds per arm) '
             'was repeated on the other saved weight-0.01 seeds with the procedure from '
             '`experiments/deeponet/refine_optimizer.py`. Selection used validation data only. The audit from the first '
             'refinement is reused here as an observed benchmark for cross-seed comparability, not as an '
             'untouched test. The final benchmark was evaluated once in an earlier study and is already '
             'observed; this study did not access it again.', '',
             '| Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Selected |',
             '|---|---:|---:|---:|---|']
    for row in records:
        lines.append(f"| {row['seed']} | {row['validation']['baseline']['price']['rmse']:.6g} | "
                     f"{row['validation']['adam_small_steps']['price']['rmse']:.6g} | "
                     f"{row['validation']['lbfgs']['price']['rmse']:.6g} | {row['selection']} |")
    lines += ['', '| Seed | Quantity | Parent | Selected |', '|---|---|---:|---:|']
    for row in records:
        for name in NAMES:
            lines.append(f"| {row['seed']} | {name} | {row['validation']['baseline'][name]['rmse']:.6g} | "
                         f"{row['validation'][row['selection']][name]['rmse']:.6g} |")
    lines += ['', 'Validation shape violations (proportion above 1e-8 / maximum):', '',
             '| Seed | Constraint | Parent | Selected |', '|---|---|---|---|']
    for row in records:
        for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
            parent_shape = row['shape']['baseline'][constraint]
            selected_shape = row['shape'][row['selection']][constraint]
            lines.append(f"| {row['seed']} | {constraint} | "
                         f"{parent_shape['proportion_above_1e_8']:.4f} / {parent_shape['maximum_violation']:.4f} | "
                         f"{selected_shape['proportion_above_1e_8']:.4f} / {selected_shape['maximum_violation']:.4f} |")
    lines += ['', 'Observed audit price RMSE (observed benchmark, evaluated after validation selection):', '',
             '| Seed | Model | In training range | Broader stress range |', '|---|---|---:|---:|']
    for row in records:
        for name in ('baseline', 'selected'):
            lines.append(f"| {row['seed']} | {name} | {row['observed_audit'][name]['in_domain']['price']['rmse']:.6g} | "
                         f"{row['observed_audit'][name]['stress']['price']['rmse']:.6g} |")
    lines += ['', '## Verdict', '',
             f"- Refinement improves price accuracy on every seed: **{verdict['refinement_improves_price']}**.",
             f"- L-BFGS is a dependable improvement: **{verdict['lbfgs_dependable']}**.",
             f"- Relative price RMSE change of the selected arm per seed: "
             + ', '.join(f'{value:+.1%}' for value in relative_price) + '.', '',
             'Verdict rule: ' + '; '.join(f'{key}: {text}' for key, text in VERDICT_RULE.items()), '']
    (output / 'report.md').write_text('\n'.join(lines))
    save_json(output / 'completion.json', dict(status='complete', seeds=[row['seed'] for row in records],
                                               observed_audit_reused=True, final_test_accessed=False,
                                               verdict=verdict))
    print(json.dumps({'complete': str(output), 'verdict': verdict}), flush=True)


if __name__ == '__main__':
    main()
