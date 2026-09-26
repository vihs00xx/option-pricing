"""Corrected sampling comparison: uniform vs mixed with shared Greek normalization.

The previous sampling runs changed sampling AND normalization jointly (each arm
normalized Greek losses by its own training labels). This rerun isolates the
sampling factor: both arms use one fixed set of Greek scales computed from the
uniform arm's training labels and frozen in the protocol before training.
Budgets are matched update counts (500 pilot epochs, 800 supervised
continuation epochs, batch 512); actual updates and runtime are recorded
separately. Compatible pilots are reused after verification (pilots train
prices only, so the Greek normalization change does not require retraining).
Evaluation uses the frozen validation split and the source study's already
observed evaluation package, labeled as observed; it is not an untouched test
and the frozen final benchmark is never accessed.
"""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics
from experiments.deeponet.refine_optimizer import shape_metrics
from experiments.deeponet.run_training_ablation import (SETTINGS_JSON, compute_normalization,
                                                        job_spec, mean_sd, region_metrics)
from experiments.deeponet.train import ROOT, digest
from src.neural import run_metadata as runs
from src.neural.deeponet import load_deeponet


SEEDS = (22092026, 22092027, 22092028)
ARMS = {'uniform': 'uniform', 'mixed': 'mixed'}
EXPECTED_UPDATES = 800 * 16
PREVIOUS_STUDY_NOTE = ('The previous sampling results changed sampling and normalization jointly; '
                       'this study isolates sampling with shared normalization scales.')


def save_json(path, value):
    runs.save_json_atomic(path, value)


def load_source_datasets(source, source_protocol):
    """Fresh dual-hash dataset records, verified against the previous study's
    recorded content hash so dataset identity is proven, not assumed."""
    datasets = {}
    for key in ('uniform', 'mixed'):
        old = source_protocol['dataset_arrays'][key]
        path = ROOT / old['file']
        record = runs.dataset_record(path, ROOT)
        runs.verify_dataset(path, {'path': record['path'], 'content_sha256': old['content_sha256'],
                                   'content_hash_scope': runs.CONTENT_HASH_SCOPE})
        datasets[key] = record
    return datasets


def verify_pilot(folder, seed, data_key, source_dataset, expected_architecture):
    """Verify a pilot is compatible for reuse: dataset, architecture, seed,
    price-only procedure and provenance all verified; never assumed."""
    view = runs.load_run_metadata(folder)
    runs.verify_dataset(ROOT / view['dataset']['path'], view['dataset'])
    runs.verify_checkpoint(folder / 'selected.pt', view)
    checks = {
        'dataset_path_matches_source': view['dataset']['path'] == source_dataset['path'],
        'architecture_matches': view['architecture'] == expected_architecture,
        'seed_matches': view['seed'] == seed,
        'price_only_labels': view.get('label_columns_used') == ['price'],
        'pilot_stage': view['stage'] == 'pilot',
        'epochs_500': view['effective_settings'].get('epochs') == 500,
    }
    if not all(checks.values()):
        raise runs.MetadataVerificationError(
            f'pilot {folder} is not compatible for reuse: checks {checks}. '
            'Retrain the pilot in a new run directory.')
    return dict(folder=str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder),
                file_sha256=digest(folder / 'selected.pt'),
                training_metadata_sha256=digest(folder / 'training_metadata.json'),
                dataset=view['dataset'], checks=checks)


def build_report(output, results, previous, runs_info, normalization, premature):
    exceptions = []

    def values(arm, path):
        out = []
        for seed in SEEDS:
            record = results[f'{arm}_{seed}']
            for key in path:
                record = record[key]
            out.append(record)
        return out

    def stat(arm, path):
        return mean_sd(values(arm, path))

    def compare(reference, arm, path, label):
        old = stat(reference, path)[0]
        new = stat(arm, path)[0]
        change = new / old - 1. if old else 0.
        verdict = 'better' if change < -.01 else ('worse' if change > .01 else 'effectively flat')
        if verdict == 'worse':
            exceptions.append(f'{label}: regresses ({reference} {old:.6g} -> {arm} {new:.6g}, {change:+.1%})')
        return f'{label} {change:+.1%} ({verdict})'

    lines = ['# Corrected sampling comparison (uniform vs mixed)', '',
             'Equal-sized datasets, identical contracts; only the coordinate distribution differs. '
             'Both arms use one fixed set of Greek normalization scales computed from the uniform '
             'arm\'s training labels and frozen in the protocol before training, so sampling is the '
             'only changed factor. Budgets are matched update counts (500 price-only pilot epochs, '
             '800 supervised continuation epochs, batch 512); equal update counts do not imply equal '
             'computation or wall time. Checkpoint selection used validation prices only. The '
             'evaluation package from the previous study is reused here for direct comparison and is '
             'explicitly an observed benchmark, not an untouched test. The frozen final benchmark was '
             'not accessed.', '',
             '## Shared normalization (the correction)', '',
             '| Target | Scale | Effective Greek-loss coefficient |', '|---|---:|---:|']
    for name, scale, coefficient in zip(normalization['target_ordering'],
                                        normalization['greek_scales'],
                                        normalization['effective_greek_loss_coefficients']):
        lines.append(f'| {name} | {scale:.6g} | {coefficient:.6g} |')
    lines += ['', f"Source dataset: `{normalization['source_dataset']['path']}` "
              f"(file SHA-256 {normalization['source_dataset']['file_sha256'][:16]}...). "
              f"Method: {normalization['method']}.", '', PREVIOUS_STUDY_NOTE, '',
              '## Budget accounting (matched update counts)', '',
              '| Run | Optimizer updates | Expected | Runtime (s) | Stop reason | Premature |',
              '|---|---:|---:|---:|---|---|']
    for run, info in sorted(runs_info.items()):
        premature_flag = premature.get(run, False)
        lines.append(f"| {run} | {info['optimizer_steps']} | {EXPECTED_UPDATES} | "
                     f"{info['training_wall_seconds']:.1f} | {info['stop_reason']} | {premature_flag} |")

    lines += ['', '## Validation RMSE per seed (mean ± sample SD across seeds)', '',
              '| Quantity | Uniform per seed | Mixed per seed | Uniform mean ± SD | Mixed mean ± SD |',
              '|---|---|---|---:|---:|']
    for name in NAMES:
        uniform = values('uniform', ['validation', 'metrics', name, 'rmse'])
        mixed = values('mixed', ['validation', 'metrics', name, 'rmse'])
        u, m = stat('uniform', ['validation', 'metrics', name, 'rmse']), \
            stat('mixed', ['validation', 'metrics', name, 'rmse'])
        lines.append(f'| {name} | {", ".join(f"{v:.6g}" for v in uniform)} | '
                     f'{", ".join(f"{v:.6g}" for v in mixed)} | {u[0]:.6g} ± {u[1]:.4g} | '
                     f'{m[0]:.6g} ± {m[1]:.4g} |')

    lines += ['', '## Broad and near-expiry/near-strike errors (focus set, observed benchmark)', '',
              '| Quantity | Region | Uniform mean ± SD | Mixed mean ± SD | Change |',
              '|---|---|---:|---:|---:|']
    for region in ('all', 'near_strike', 'near_expiry', 'near_strike_and_expiry'):
        for name in NAMES:
            u, m = stat('uniform', ['focus', 'regions', region, name]), \
                stat('mixed', ['focus', 'regions', region, name])
            change = compare('uniform', 'mixed', ['focus', 'regions', region, name],
                             f'mixed vs uniform {name} RMSE on focus {region}')
            lines.append(f'| {name} | {region} | {u[0]:.6g} ± {u[1]:.4g} | {m[0]:.6g} ± {m[1]:.4g} | '
                         f'{change.split(" (")[0]} |')
    for name in NAMES:
        compare('uniform', 'mixed', ['validation', 'metrics', name, 'rmse'],
                f'mixed vs uniform {name} RMSE on validation')

    lines += ['', '## Constraint violations (frequency and maximum severity reported separately)', '',
              '| Dataset | Constraint | Uniform frequency | Mixed frequency | Uniform max severity | Mixed max severity |',
              '|---|---|---:|---:|---:|---:|']
    for dataset in ('validation', 'stress'):
        for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
            row = [stat(arm, [dataset, 'shape', constraint, key])[0]
                   for arm in ('uniform', 'mixed') for key in ('proportion_above_1e_8', 'maximum_violation')]
            lines.append(f'| {dataset} | {constraint} | {row[0]:.4f} | {row[2]:.4f} | '
                         f'{row[1]:.4g} | {row[3]:.4g} |')
            compare('uniform', 'mixed', [dataset, 'shape', constraint, 'proportion_above_1e_8'],
                    f'mixed vs uniform {constraint} violation frequency on {dataset}')
            compare('uniform', 'mixed', [dataset, 'shape', constraint, 'maximum_violation'],
                    f'mixed vs uniform {constraint} maximum severity on {dataset}')

    lines += ['', '## Comparison with the previous sampling results', '',
              PREVIOUS_STUDY_NOTE, '']
    for region in ('near_strike_and_expiry', 'near_expiry', 'near_strike'):
        for name in ('theta', 'gamma', 'delta'):
            path = ['focus', 'regions', region, name]
            old_mixed = mean_sd([previous[f'mixed_{seed}']['focus']['regions'][region][name]
                                 for seed in SEEDS])
            old_uniform = mean_sd([previous[f'uniform_{seed}']['focus']['regions'][region][name]
                                   for seed in SEEDS])
            previous_gain = old_mixed[0] / old_uniform[0] - 1.
            corrected_gain = stat('mixed', path)[0] / stat('uniform', path)[0] - 1.
            lines.append(f'- {name} on focus {region}: previous mixed-vs-uniform {previous_gain:+.1%}, '
                         f'corrected (shared scales) {corrected_gain:+.1%}.')
    lines += ['', '## Exceptions (measured regressions under shared scales)', '']
    lines += [f'- {text}.' for text in exceptions] or ['- none measured.']
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    save_json(output / 'report_exceptions.json', exceptions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-study', type=Path,
                        default=ROOT / 'results/deeponet/training_ablation/20260923T111505263380Z')
    parser.add_argument('--jobs', type=int, default=3)
    args = parser.parse_args()
    source = args.source_study if args.source_study.is_absolute() else ROOT / args.source_study
    source_protocol = json.loads((source / 'protocol.json').read_text())
    source_datasets = load_source_datasets(source, source_protocol)
    normalization = compute_normalization(source_datasets)
    if normalization['uniform']['greek_scales'] != normalization['mixed']['greek_scales']:
        raise RuntimeError('shared normalization construction failed')
    output = ROOT / 'results/deeponet/sampling_correction' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'jobs').mkdir()
    pilots = {}
    expected_architecture = dict(SETTINGS_JSON['architecture'])
    for data_key in ARMS.values():
        for seed in SEEDS:
            folder = source / f'pilot/base_{data_key}_{seed}'
            pilots[f'{data_key}_{seed}'] = verify_pilot(folder, seed, data_key,
                                                        source_datasets[data_key], expected_architecture)
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/run_training_ablation.py',
                 ROOT / 'experiments/deeponet/train.py', ROOT / 'experiments/deeponet/compare_supervision.py',
                 ROOT / 'experiments/deeponet/evaluate.py', ROOT / 'src/neural/deeponet.py',
                 ROOT / 'src/neural/deeponet_data.py', ROOT / 'src/neural/run_metadata.py'):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    jobs = {}
    for data_key in ARMS.values():
        for seed in SEEDS:
            name = f'{data_key}/{seed}'
            jobs[name] = job_spec(name, 'stage2', {'arch': 'base', 'data': data_key, 'penalties': ()},
                                  seed, f'arms/{data_key}_{seed}', source_datasets, 
                                  {'uniform': normalization['uniform'], 'mixed': normalization['mixed']},
                                  sources,
                                  parent={'folder': str((source / f'pilot/base_{data_key}_{seed}').resolve())})
    evaluation_file = source / 'data' / 'evaluation.npz'
    protocol = dict(
        protocol_id='deeponet_sampling_correction_v1',
        purpose='Isolate the sampling factor: uniform vs mixed coordinates with shared Greek normalization scales.',
        previous_joint_change=PREVIOUS_STUDY_NOTE,
        source_study=dict(path=str(source.relative_to(ROOT)), protocol_sha256=digest(source / 'protocol.json'),
                          evaluation_results_sha256=digest(source / 'evaluation_results.json')),
        datasets={key: source_datasets[key] for key in ('uniform', 'mixed')},
        normalization=normalization, pilot_reuse=pilots, jobs=jobs, source_sha256=sources,
        procedure=dict(stage2='supervised continuation: 800 epochs (12,800 matched update counts), '
                              'Adam lr 0.0003, batch 512, Greek-loss weight 0.01',
                       expected_optimizer_updates=EXPECTED_UPDATES,
                       selection='minimum validation price MSE normalized by strike',
                       budgets=('matched update counts; equal update counts do not imply equal '
                                'computation or wall time')),
        evaluation=dict(status='observed benchmark reused from the source study for direct comparison; '
                               'not an untouched test',
                        file=str(evaluation_file.relative_to(ROOT)), file_sha256=digest(evaluation_file),
                        sets=['in_domain', 'stress', 'focus', 'validation']),
        final_test_accessed=False)
    save_json(output / 'protocol.json', protocol)
    print(json.dumps({'output': str(output)}), flush=True)
    from experiments.deeponet.run_training_ablation import launch
    launch([jobs[name] for name in ('uniform/22092026', 'uniform/22092027', 'uniform/22092028',
                                    'mixed/22092026', 'mixed/22092027', 'mixed/22092028')],
           output, output / 'jobs', args.jobs)
    runs_info, premature = {}, {}
    for name, spec in jobs.items():
        view = runs.load_run_metadata(output / spec['folder'])
        runs_info[name] = view['training']
        premature[name] = (view['training'].get('optimizer_steps') != EXPECTED_UPDATES
                           or view['training'].get('stop_reason') != 'epochs_completed')
    save_json(output / 'update_count_verification.json',
              dict(expected=EXPECTED_UPDATES, runs=runs_info, premature_termination=premature))
    with np.load(evaluation_file, allow_pickle=False) as data:
        evaluation = {key: data[key].copy() for key in data.files}
    tensors = {}
    for name in ('in_domain', 'stress', 'focus'):
        tensors[name] = tuple(torch.tensor(evaluation[f'{name}_{key}'], dtype=torch.float64)
                              for key in ('branch', 'trunk')) + (evaluation[f'{name}_targets'],)
    with np.load(ROOT / SETTINGS_JSON['dataset'], allow_pickle=False) as data:
        tensors['validation'] = (torch.tensor(data['validation_branch'], dtype=torch.float64),
                                 torch.tensor(data['validation_trunk'], dtype=torch.float64),
                                 data['validation_targets'].copy())
    results = {}
    for name, spec in sorted(jobs.items()):
        folder = output / spec['folder']
        model = load_deeponet(folder / 'selected.pt')
        record = {}
        for set_name, (branch, trunk, targets) in tensors.items():
            if set_name == 'validation':
                record[set_name] = dict(
                    metrics=json.loads((folder / 'validation_metrics.json').read_text()),
                    shape=json.loads((folder / 'shape_metrics.json').read_text()),
                    regions=region_metrics(model, branch, trunk, targets))
            else:
                destination = output / 'evaluation' / name / set_name
                destination.mkdir(parents=True, exist_ok=True)
                record[set_name] = dict(metrics=diagnostics(model, branch, trunk, targets, destination),
                                        shape=shape_metrics(model, branch, trunk),
                                        regions=region_metrics(model, branch, trunk, targets))
        results[name] = record
        print(json.dumps({'evaluated': name}), flush=True)
    previous = json.loads((source / 'evaluation_results.json').read_text())
    keyed = {f'{key}_{seed}': results[f'{key}/{seed}'] for key in ARMS for seed in SEEDS}
    build_report(output, keyed, previous, {f'{k.replace("/", "_")}': v for k, v in runs_info.items()},
                 normalization['uniform'], {f'{k.replace("/", "_")}': v for k, v in premature.items()})
    save_json(output / 'evaluation_results.json', keyed)
    save_json(output / 'completion.json', dict(status='complete', final_test_accessed=False,
                                               evaluation_status='observed reused benchmark',
                                               premature_termination=premature))
    print(json.dumps({'complete': str(output)}), flush=True)


if __name__ == '__main__':
    main()
