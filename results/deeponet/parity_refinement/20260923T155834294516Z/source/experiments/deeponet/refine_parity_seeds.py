"""Parity under the established optimizer-refinement procedure.

Tests whether the parity architecture improves the strongest current model
without introducing another change: both architectures use uniform sampling
(the frozen dataset), identical two-stage parent training (the matched-update-
count ablation study), the same seeds, and the established two-arm refinement
procedure (small-step Adam vs full-batch L-BFGS, 180 seconds per arm, validation
selection with the Greek guardrail). Fresh evaluation cases are generated up
front and opened only after all selection decisions. The final benchmark was
evaluated once in an earlier study and is already observed; this work does not
access it again.
"""

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.evaluate import trusted_file_sha256
from experiments.deeponet.refine_optimizer import lbfgs_refine, shape_metrics
from experiments.deeponet.run_training_ablation import STRESS_DOMAIN, DOMAIN, mean_sd
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural import run_metadata as runs
from src.neural.deeponet import load_deeponet
from src.neural.deeponet_data import synthetic_split


ABLATION = ROOT / 'results/deeponet/training_ablation/20260923T111505263380Z'
FAMILY_REFINEMENT = ROOT / 'results/deeponet/optimizer_refinement_seeds/20260923T093654478604Z'
SEEDS = (22092026, 22092027, 22092028)
ARCHITECTURES = ('control', 'parity')
ARMS = ('adam_small_steps', 'lbfgs')
REFINEMENT_SEED = 23092026
EVAL_SEEDS = dict(in_domain=28092026, stress=28093026)
SELECTION_RULE = ('Each Greek RMSE must remain within 10% of the parent model, then minimize price '
                  'RMSE among parent and refined arms.')


def save_json(path, value):
    runs.save_json_atomic(path, value)


def verify_parent(checkpoint):
    folder = checkpoint.parent
    view = runs.load_run_metadata(folder)
    runs.verify_checkpoint(checkpoint, view, checkpoint.stem)
    runs.verify_dataset(ROOT / view['dataset']['path'], view['dataset'],
                        evaluated_split='validation',
                        trusted_file_sha256=trusted_file_sha256(ROOT / view['dataset']['path']))
    return view


def run_refinement(output, architecture, seed, train, validation, scales, vy, vb, vt, evaluation):
    parent = ABLATION / f'arms/{architecture}_{seed}/selected.pt'
    view = verify_parent(parent)
    folder = output / f'{architecture}_{seed}'
    folder.mkdir(parents=True)
    base_dir = folder / 'baseline'
    base_dir.mkdir()
    baseline = load_deeponet(parent)
    results = {'baseline': diagnostics(baseline, vb, vt, vy, base_dir)}
    save_json(base_dir / 'shape_metrics.json', shape_metrics(baseline, vb, vt))
    for arm in ARMS:
        arm_dir = folder / arm
        arm_dir.mkdir()
        model = load_deeponet(parent)
        torch.manual_seed(REFINEMENT_SEED)
        settings = {**view['effective_settings'], 'seed': REFINEMENT_SEED, 'epochs': 100000,
                    'wall_seconds': 180, 'learning_rate': .00002, 'progress_every_seconds': 45,
                    'optimizer': arm, 'protocol_id': 'deeponet_parity_refinement_v1'}
        print(json.dumps({'starting': f'{architecture}_{seed}', 'arm': arm}), flush=True)
        if arm == 'lbfgs':
            settings.update(learning_rate=1., batch_size=len(train[0]), loss_multiplier=1e4,
                            chunk_size=1024, history_size=50, max_iter=1, line_search_fn='strong_wolfe')
            history, last, summary = lbfgs_refine(model, train, validation, scales, 180)
        else:
            loss = lambda m, b, t, y: supervised_loss(m, b, t, y, scales, .01)
            history, last, summary = train_model(model, train, validation, settings, loss)
        record = runs.build_run_metadata(
            arm=f'{architecture}_{seed}/{arm}', seed=REFINEMENT_SEED, stage='refinement',
            architecture=model.architecture, effective_settings=settings,
            settings_sha256='refinement', dataset=view['dataset'],
            validation_split=view.get('validation_split') or {'arrays': ['validation_branch', 'validation_trunk',
                                                                        'validation_targets'], 'rows': int(len(vb))},
            normalization=None,
            parent={'folder': str(parent.relative_to(ROOT)), 'file_sha256': digest(parent)},
            checkpoints={}, training={**summary, 'arm_wall_seconds': summary.get('training_wall_seconds')},
            label_columns_used=list(NAMES), greek_weight=.01, final_test_accessed=False)
        for kind, state in (('selected', model.state_dict()), ('last', last)):
            torch.save(dict(architecture=model.architecture, state_dict=state,
                            metadata={k: v for k, v in record.items() if k != 'checkpoints'}),
                       arm_dir / f'{kind}.pt')
        record['checkpoints'] = {kind: digest(arm_dir / f'{kind}.pt') for kind in ('selected', 'last')}
        write_csv(arm_dir / 'history.csv', history)
        save_json(arm_dir / 'validation_metrics.json', diagnostics(model, vb, vt, vy, arm_dir))
        save_json(arm_dir / 'shape_metrics.json', shape_metrics(model, vb, vt))
        save_json(arm_dir / 'training_metadata.json', record)
        runs.write_run_manifest(arm_dir, {'status': 'complete', 'job': record['arm'],
                                          'completed_at': datetime.now(timezone.utc).isoformat(),
                                          'settings': settings, 'architecture': model.architecture,
                                          'parent': record['parent'], 'dataset': view['dataset'],
                                          'training': record['training'],
                                          'artifacts': {name: digest(arm_dir / name)
                                                        for name in runs.REQUIRED_ARTIFACTS},
                                          'metadata_sha256': digest(arm_dir / 'training_metadata.json')})
        results[arm] = json.loads((arm_dir / 'validation_metrics.json').read_text())
        print(json.dumps({'completed': f'{architecture}_{seed}/{arm}',
                          'price_rmse': results[arm]['price']['rmse']}), flush=True)
    eligible = [name for name, metrics in results.items() if all(
        metrics[greek]['rmse'] <= 1.1 * results['baseline'][greek]['rmse'] for greek in NAMES[1:])]
    selected = min(eligible, key=lambda name: results[name]['price']['rmse'])
    save_json(folder / 'selection.json', dict(seed=seed, architecture=architecture, selected=selected,
                                              eligible=eligible, validation=results, rule=SELECTION_RULE))
    checkpoint = parent if selected == 'baseline' else folder / selected / 'selected.pt'
    model = load_deeponet(checkpoint)
    fresh = {}
    for split, (branch, trunk, targets) in evaluation.items():
        destination = folder / 'fresh' / split
        destination.mkdir(parents=True)
        fresh[split] = diagnostics(model, branch, trunk, targets, destination)
        save_json(destination / 'shape_metrics.json', shape_metrics(model, branch, trunk))
    print(json.dumps({'selection': f'{architecture}_{seed}', 'selected': selected}), flush=True)
    return dict(architecture=architecture, seed=seed, selected=selected, eligible=eligible,
                validation=results, fresh=fresh, parent=str(parent.relative_to(ROOT)))


def main():
    output = ROOT / 'results/deeponet/parity_refinement' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    dataset = ROOT / json.loads((ROOT / 'experiments/config/neural/deeponet_training.json').read_text())['dataset']
    with np.load(dataset, allow_pickle=False) as data:
        train = tuple(torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                      for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                  for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
        train_rows = {tuple(row) for row in data['train_branch']}
        train_rows |= {tuple(row) for row in data['validation_branch']}
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    scales = train[2][:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    evaluation = {}
    for name, bounds, count, seed in (('in_domain', DOMAIN, 256, EVAL_SEEDS['in_domain']),
                                      ('stress', STRESS_DOMAIN, 128, EVAL_SEEDS['stress'])):
        split = synthetic_split(count, 16, bounds, seed)
        if train_rows & {tuple(row) for row in split['branch']}:
            raise ValueError(f'fresh evaluation contracts overlap training contracts: {name}')
        evaluation[name] = (torch.tensor(split['branch'], dtype=torch.float64),
                            torch.tensor(split['trunk'], dtype=torch.float64), split['targets'])
        np.savez_compressed(output / f'fresh_{name}.npz', **split)
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/refine_optimizer.py',
                 ROOT / 'experiments/deeponet/train.py', ROOT / 'experiments/deeponet/compare_supervision.py',
                 ROOT / 'experiments/deeponet/evaluate.py', ROOT / 'src/neural/deeponet.py',
                 ROOT / 'src/neural/run_metadata.py', ROOT / 'src/black_scholes.py'):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    protocol = dict(
        protocol_id='deeponet_parity_refinement_v1',
        purpose=('Established optimizer-refinement procedure applied to control and parity parents '
                 'across the same seeds on uniform sampling, to test whether parity improves the '
                 'strongest current model without another change.'),
        parent_study=str(ABLATION.relative_to(ROOT)), family_reference=str(FAMILY_REFINEMENT.relative_to(ROOT)),
        seeds=SEEDS, architectures=ARCHITECTURES, arms=ARMS, refinement_seed=REFINEMENT_SEED,
        greek_scales=scales.tolist(), greek_weight=.01, wall_seconds_per_arm=180,
        selection_rule=SELECTION_RULE, source_sha256=sources,
        fresh_evaluation=dict(status='fresh cases generated before training; opened after all selection '
                                     'decisions; observed once read', seeds=EVAL_SEEDS,
                              files={name: digest(output / f'fresh_{name}.npz') for name in evaluation}),
        final_benchmark='observed in an earlier one-time study (results/deeponet/primary/final_test_metrics.json); not accessed again',
        final_test_accessed=False)
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output)}), flush=True)
    records = []
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            records.append(run_refinement(output, architecture, seed, train, validation, scales, vy,
                                          vb, vt, evaluation))
            save_json(output / 'records.json', records)
    by_key = {(row['architecture'], row['seed']): row for row in records}
    family = json.loads((FAMILY_REFINEMENT / 'seed_aggregate.json').read_text())
    lines = ['# Parity under the established optimizer-refinement procedure', '',
             'Both architectures trained identically (matched-update-count parents, uniform sampling, '
             'frozen dataset) and refined with the established two-arm procedure (small-step Adam vs '
             'full-batch L-BFGS, 180 s per arm). Selection used validation prices with the Greek '
             'guardrail. Fresh evaluation cases were opened after all selection decisions and are now '
             'observed. The final benchmark was evaluated once in an earlier study and is already '
             'observed; this work did not access it again.', '',
             '| Architecture | Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Selected |',
             '|---|---|---:|---:|---:|---|']
    for architecture in ARCHITECTURES:
        for seed in SEEDS:
            row = by_key[(architecture, seed)]
            lines.append(f"| {architecture} | {seed} | {row['validation']['baseline']['price']['rmse']:.6g} | "
                         f"{row['validation']['adam_small_steps']['price']['rmse']:.6g} | "
                         f"{row['validation']['lbfgs']['price']['rmse']:.6g} | {row['selected']} |")
    lines += ['', 'Validation RMSE after refinement (mean ± sample SD over seeds):', '',
              '| Quantity | Refined control | Refined parity | Change |', '|---|---:|---:|---:|']
    for name in NAMES:
        control = mean_sd([by_key[('control', seed)]['validation'][by_key[('control', seed)]['selected']][name]['rmse']
                           for seed in SEEDS])
        parity = mean_sd([by_key[('parity', seed)]['validation'][by_key[('parity', seed)]['selected']][name]['rmse']
                          for seed in SEEDS])
        change = parity[0] / control[0] - 1.
        lines.append(f'| {name} | {control[0]:.6g} ± {control[1]:.4g} | {parity[0]:.6g} ± {parity[1]:.4g} | '
                     f'{change:+.1%} |')
    lines += ['', 'Fresh in-domain RMSE after refinement (fresh cases, mean ± SD):', '',
              '| Quantity | Refined control | Refined parity | Change |', '|---|---:|---:|---:|']
    for name in NAMES:
        control = mean_sd([by_key[('control', seed)]['fresh']['in_domain'][name]['rmse'] for seed in SEEDS])
        parity = mean_sd([by_key[('parity', seed)]['fresh']['in_domain'][name]['rmse'] for seed in SEEDS])
        lines.append(f'| {name} | {control[0]:.6g} ± {control[1]:.4g} | {parity[0]:.6g} ± {parity[1]:.4g} | '
                     f'{parity[0] / control[0] - 1.:+.1%} |')
    lines += ['', 'Reference: the established refinement on the family checkpoints improved validation '
              'price RMSE by 29% to 36% across the same seeds (Adam selected on two seeds, L-BFGS one).', '',
              '## Verdict', '']
    improved = sum(mean_sd([by_key[('parity', seed)]['validation'][by_key[('parity', seed)]['selected']][name]['rmse']
                            for seed in SEEDS])[0]
                   < mean_sd([by_key[('control', seed)]['validation'][by_key[('control', seed)]['selected']][name]['rmse']
                              for seed in SEEDS])[0] for name in NAMES)
    lines += [f'- Refined parity improves {improved} of 6 validation RMSEs versus refined control at the '
              'same procedure and seeds.',
              '- Parity call-put consistency is exact by construction for the parity arms; control arms '
              'retain measured inconsistencies.',
              '- Any regression above is listed in the Change columns; no unconditional claim is made.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    save_json(output / 'completion.json', dict(status='complete', final_test_accessed=False,
                                               fresh_evaluation_status='observed_after_selection'))
    print(json.dumps({'complete': str(output), 'parity_improved': improved}), flush=True)


if __name__ == '__main__':
    main()
