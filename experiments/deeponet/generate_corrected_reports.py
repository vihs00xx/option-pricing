"""Generate clearly identified corrected reports; originals are preserved.

Two modes:
  --ablation-study: regenerate the ablation report from measured values with
    gate-driven conclusions, explicit exceptions, matched-update-count
    terminology, actual steps/runtime, and the caveat that the previous
    sampling results changed sampling and normalization jointly.
  --cross-seed-study: regenerate the optimizer replication report with the
    corrected description of the replication outcome.

Neither mode modifies the original report.md files.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.deeponet.compare_supervision import NAMES
from experiments.deeponet.run_training_ablation import mean_sd, reports, run_folders_for
from experiments.deeponet.train import ROOT
from src.neural import run_metadata as runs


CORRECTED_HEADER = ('Corrected report (generated {stamp}). The original report.md in this folder is '
                    'preserved unmodified; corrections cover terminology (matched update counts), '
                    'gate-driven conclusions, and explicit exceptions.')


def corrected_ablation(study):
    study = study if study.is_absolute() else ROOT / study
    results = json.loads((study / 'evaluation_results.json').read_text())
    chosen = json.loads((study / 'penalty_weight_selection.json').read_text())
    runs_info = {}
    for (arm, seed), folder in sorted(run_folders_for(study, chosen).items()):
        runs_info[f'{arm}_{seed}'] = runs.load_run_metadata(folder)['training']
    caveats = [
        'The previous sampling results in this study changed sampling AND normalization jointly: '
        'each arm used Greek scales computed from its own training labels. The isolated sampling '
        'effect is measured by the corrected sampling study, where both arms share the uniform '
        'arm\'s training-label scales.',
    ]
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    reports(study, results, runs_info, chosen, filename='report_corrected.md',
            caveats=[CORRECTED_HEADER.format(stamp=stamp), *caveats])
    print(json.dumps({'corrected_report': str(study / 'report_corrected.md')}))


def corrected_cross_seed(study):
    study = study if study.is_absolute() else ROOT / study
    records = json.loads((study / 'seed_results.json').read_text())
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [CORRECTED_HEADER.format(stamp=stamp), '',
             '# Optimizer refinement replication (corrected description)', '',
             'Refinement helped validation price accuracy on all three seeds; Adam won two seeds '
             'and L-BFGS one under the recorded selection rule. Both refined arms (small-step Adam, '
             'full-batch L-BFGS) used 180-second wall budgets per arm; actual optimizer steps and '
             'runtime are listed below because equal wall time did not produce equal update counts.', '',
             '| Seed | Parent price RMSE | Adam price RMSE | L-BFGS price RMSE | Arm selected by the rule |',
             '|---|---:|---:|---:|---|']
    for row in sorted(records, key=lambda item: item['seed']):
        lines.append(f"| {row['seed']} | {row['validation']['baseline']['price']['rmse']:.6g} | "
                     f"{row['validation']['adam_small_steps']['price']['rmse']:.6g} | "
                     f"{row['validation']['lbfgs']['price']['rmse']:.6g} | {row['selection']} |")
    lines += ['', 'Measured validation price RMSE changes of the selected arm: '
              + ', '.join(f"{(row['validation'][row['selection']]['price']['rmse'] / row['validation']['baseline']['price']['rmse'] - 1.):+.1%}"
                          for row in sorted(records, key=lambda item: item['seed'])) + '.', '',
              'Budget accounting (recorded values):', '',
              '| Seed | Arm | Optimizer updates / closure calls | Runtime (s) | Stop reason |',
              '|---|---|---:|---:|---|']
    for row in sorted(records, key=lambda item: item['seed']):
        for arm in ('adam_small_steps', 'lbfgs'):
            folder = study / f"seed_{row['seed']}" / arm
            metadata = json.loads((folder / 'training_metadata.json').read_text())
            updates = metadata.get('optimizer_steps', metadata.get('closure_calls'))
            lines.append(f"| {row['seed']} | {arm} | {updates} | "
                         f"{metadata.get('training_wall_seconds', float('nan')):.1f} | "
                         f"{metadata.get('stop_reason', 'unknown')} |")
    lines += ['', 'The verdict wording "L-BFGS is a dependable improvement: False" in the original '
              'report stands as measured; the corrected description above states the replication '
              'outcome without implying either optimizer is uniformly superior.']
    (study / 'report_corrected.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'corrected_report': str(study / 'report_corrected.md')}))


REFINEMENT_ARMS = ('baseline', 'adam_small_steps', 'lbfgs')
STRESS_DESCRIPTION = ('The stress evaluation set samples a broader domain than training (wider K, T, r, '
                      'q, sigma and moneyness ranges with a lower time-to-expiry floor). It mixes '
                      'coordinates inside and outside the training domain, so it is not a pure '
                      'extrapolation sample.')
STABILITY_NOTE = ('This ratio measures variability of aggregate validation RMSE across these three '
                  'seeds under this procedure. It does not establish general seed robustness or '
                  'statistical significance, it is not uncertainty in individual predictions, and '
                  'the price-stability result does not automatically extend to the Greeks.')
GUARANTEES_NOTE = ('Exact call-put parity does not guarantee price bounds, a valid Delta, or a '
                   'nonnegative Gamma; those constraints are measured separately below.')


def collect_parity_data(study):
    """Audit saved parity-refinement artifacts with the corrected verifier and
    recompute every aggregate from saved per-run metrics. No retraining; no
    re-evaluation."""
    from experiments.deeponet.evaluate import trusted_file_sha256
    study = Path(study)
    data = {'runs': [], 'study': str(study)}
    for row in json.loads((study / 'records.json').read_text()):
        arch, seed = row['architecture'], row['seed']
        selection = json.loads((study / f'{arch}_{seed}' / 'selection.json').read_text())
        validation = selection['validation']
        eligible = [name for name in REFINEMENT_ARMS if all(
            validation[name][greek]['rmse'] <= 1.1 * validation['baseline'][greek]['rmse']
            for greek in NAMES[1:])]
        recomputed = min(eligible, key=lambda name: validation[name]['price']['rmse'])
        entry = {'architecture': arch, 'seed': seed, 'selected': row['selected'],
                 'selection_recorded': selection['selected'], 'selection_recomputed': recomputed,
                 'selection_confirmed': recomputed == row['selected'] == selection['selected'],
                 'eligible_recomputed': eligible, 'validation': validation,
                 'fresh': row['fresh'], 'shape': {}, 'training': {}, 'verification': {}}
        shape_folder = study / f'{arch}_{seed}'
        selected_shape = shape_folder / row['selected']
        entry['shape']['validation'] = json.loads((selected_shape / 'shape_metrics.json').read_text())
        for split in ('in_domain', 'stress'):
            entry['shape'][split] = json.loads(
                (shape_folder / 'fresh' / split / 'shape_metrics.json').read_text())
        parent_folder = (ROOT / row['parent']).parent
        for label, folder in [('parent', parent_folder)] + [
                (arm, study / f'{arch}_{seed}' / arm) for arm in ('adam_small_steps', 'lbfgs')]:
            status = {'folder': str(folder.relative_to(ROOT)) if folder.is_relative_to(ROOT) else str(folder)}
            try:
                view = runs.load_run_metadata(folder)
                runs.verify_checkpoint(folder / 'selected.pt', view, 'selected')
                dataset_path = ROOT / view['dataset']['path']
                kinds = runs.verify_dataset(dataset_path, view['dataset'], evaluated_split='validation',
                                            trusted_file_sha256=trusted_file_sha256(dataset_path))
                status.update(status='verified', checkpoint_verified=True,
                              dataset_verified=kinds, scope=view['dataset'].get('content_hash_scope_prefix'))
            except runs.MetadataVerificationError as error:
                status.update(status='unverified', checkpoint_verified=False, detail=str(error))
            entry['verification'][label] = status
            if label != 'parent':
                record = json.loads((folder / 'training_metadata.json').read_text())
                entry['training'][label] = record['training']
        data['runs'].append(entry)
    family_path = ROOT / 'results/deeponet/optimizer_refinement_seeds/20260923T093654478604Z' / 'seed_results.json'
    if family_path.exists():
        selections = [item['selection'] for item in json.loads(family_path.read_text())]
        data['family_counts'] = {name: selections.count(name) for name in ('adam_small_steps', 'lbfgs')}
    return data


def build_corrected_parity_report(data):
    runs_ = data['runs']
    exceptions = []

    def values(architecture, path):
        out = []
        for entry in runs_:
            if entry['architecture'] != architecture:
                continue
            record = entry
            for key in path:
                record = record[key]
            out.append(record)
        return out

    def stat(architecture, path):
        return mean_sd(values(architecture, path))

    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    lines = [CORRECTED_HEADER.format(stamp=stamp), '',
             '# Parity under the established optimizer-refinement procedure (corrected)', '',
             'Aggregates below are recomputed from the saved per-run metrics and selection records; '
             'identities are verified with the corrected scope-aware verifier. The original '
             'report.md in this folder is preserved unchanged. The final benchmark was evaluated '
             'once in an earlier study (`results/deeponet/primary/final_test_metrics.json`) and is '
             'already observed; this work did not access it again. Reused evaluation sets are '
             'observed benchmarks, not untouched tests.', '',
             '## Identity verification (corrected verifier)', '',
             '| Run | Role | Checkpoint | Dataset hashes | Scope | Status |',
             '|---|---|---|---|---|---|']
    unverified = []
    for entry in runs_:
        for label, status in entry['verification'].items():
            kinds = ', '.join(status.get('dataset_verified') or []) or 'none'
            lines.append(f"| {entry['architecture']}_{entry['seed']} | {label} | "
                         f"{'ok' if status.get('checkpoint_verified') else 'FAILED'} | {kinds} | "
                         f"{status.get('scope') or '-'} | {status['status']} |")
            if status['status'] != 'verified':
                unverified.append(f"{entry['architecture']}_{entry['seed']}/{label}: {status.get('detail')}")
    lines += ['']
    if unverified:
        lines += ['Some identities could not be established; the affected results are reported as '
                  'unverified and were not repaired:', '']
        lines += [f'- {text}' for text in unverified]
    else:
        lines += ['All recorded checkpoint and dataset identities verify under the corrected '
                  'verifier (legacy training-scope content hashes verified for training arrays; '
                  'validation coverage established by the independently recorded frozen-manifest '
                  'full-file hash). No provenance was repaired or regenerated.']
    lines += ['', '## Selections and optimizer counts (generated from selection records)', '',
              '| Architecture | Seed | Recorded | Recomputed | Eligible | Confirmed |',
              '|---|---|---|---|---|---|']
    counts = {}
    per_arch = {}
    for entry in runs_:
        counts[entry['selected']] = counts.get(entry['selected'], 0) + 1
        per_arch.setdefault(entry['architecture'], {})
        per_arch[entry['architecture']][entry['selected']] = \
            per_arch[entry['architecture']].get(entry['selected'], 0) + 1
        lines.append(f"| {entry['architecture']} | {entry['seed']} | {entry['selection_recorded']} | "
                     f"{entry['selection_recomputed']} | {', '.join(entry['eligible_recomputed'])} | "
                     f"{entry['selection_confirmed']} |")
    lines += ['', 'Optimizer selection across the six architecture-seed combinations: '
              f"Adam {counts.get('adam_small_steps', 0)}, L-BFGS {counts.get('lbfgs', 0)}.",
              '']
    for arch in ('control', 'parity'):
        lines.append(f"- {arch}: Adam {per_arch.get(arch, {}).get('adam_small_steps', 0)}, "
                     f"L-BFGS {per_arch.get(arch, {}).get('lbfgs', 0)}.")
    if data.get('family_counts'):
        family = data['family_counts']
        lines += ['', 'These counts are separate from the earlier three-seed refinement study on '
                  f"family checkpoints, where Adam won {family.get('adam_small_steps', 0)} and "
                  f"L-BFGS {family.get('lbfgs', 0)}."]

    for dataset, title in (('validation', 'Validation RMSE (mean ± sample SD over seeds)'),
                           ('fresh', 'Fresh in-domain RMSE (observed benchmark, mean ± SD)')):
        lines += ['', title + ':', '', '| Quantity | Refined control | Refined parity | Change |',
                  '|---|---:|---:|---:|']
        for name in NAMES:
            control = mean_sd([entry['validation'][entry['selected']][name]['rmse'] for entry in runs_
                               if entry['architecture'] == 'control'])
            parity = mean_sd([entry['validation'][entry['selected']][name]['rmse'] for entry in runs_
                              if entry['architecture'] == 'parity'])
            if dataset == 'fresh':
                control = mean_sd([entry['fresh']['in_domain'][name]['rmse'] for entry in runs_
                                   if entry['architecture'] == 'control'])
                parity = mean_sd([entry['fresh']['in_domain'][name]['rmse'] for entry in runs_
                                  if entry['architecture'] == 'parity'])
            change = parity[0] / control[0] - 1.
            if change > .01:
                exceptions.append(f'{name} RMSE on {dataset}: regresses ({control[0]:.6g} -> '
                                  f'{parity[0]:.6g}, {change:+.1%})')
            lines.append(f'| {name} | {control[0]:.6g} ± {control[1]:.4g} | '
                         f'{parity[0]:.6g} ± {parity[1]:.4g} | {change:+.1%} |')
    lines += ['', 'Fresh stress RMSE (observed benchmark, mean ± sample SD):', '',
              '| Quantity | Refined control | Refined parity | Change |', '|---|---:|---:|---:|']
    for name in NAMES:
        control = mean_sd([entry['fresh']['stress'][name]['rmse'] for entry in runs_
                           if entry['architecture'] == 'control'])
        parity = mean_sd([entry['fresh']['stress'][name]['rmse'] for entry in runs_
                          if entry['architecture'] == 'parity'])
        change = parity[0] / control[0] - 1.
        if change > .01:
            exceptions.append(f'{name} RMSE on stress: regresses ({control[0]:.6g} -> {parity[0]:.6g}, '
                              f'{change:+.1%})')
        lines.append(f'| {name} | {control[0]:.6g} ± {control[1]:.4g} | '
                     f'{parity[0]:.6g} ± {parity[1]:.4g} | {change:+.1%} |')
    lines += ['', STRESS_DESCRIPTION, '',
              '## Financial constraints (frequency and maximum severity reported separately)', '',
              'Severity aggregates are labeled precisely: "mean of per-seed maxima" averages the '
              'per-seed maximum violation; "worst maximum" is the largest maximum across all three '
              'seeds.', '']
    for dataset in ('validation', 'in_domain', 'stress'):
        lines += [f'| {dataset} | Constraint | Metric | Refined control | Refined parity | Change |',
                  '|---|---|---|---:|---:|---:|']
        for constraint in ('price_bounds', 'delta_bounds', 'negative_gamma'):
            for metric, key in (('frequency', 'proportion_above_1e_8'),
                                ('mean of per-seed maxima', 'maximum_violation'),
                                ('worst maximum', 'maximum_violation')):
                if metric == 'worst maximum':
                    control = max(values('control', ['shape', dataset, constraint, key]))
                    parity = max(values('parity', ['shape', dataset, constraint, key]))
                else:
                    control = stat('control', ['shape', dataset, constraint, key])[0]
                    parity = stat('parity', ['shape', dataset, constraint, key])[0]
                change = parity / control - 1. if control else 0.
                if change > .01:
                    exceptions.append(f'{constraint} {metric} on {dataset}: regresses '
                                      f'({control:.6g} -> {parity:.6g}, {change:+.1%})')
                lines.append(f'| {dataset} | {constraint} | {metric} | {control:.6g} | {parity:.6g} | '
                             f'{change:+.1%} |')
    lines += ['', GUARANTEES_NOTE]

    def sample_sd(architecture, name):
        samples = [entry['validation'][entry['selected']][name]['rmse'] for entry in runs_
                   if entry['architecture'] == architecture]
        return float(np.std(samples, ddof=1))

    lines += ['', '## Seed stability (qualified)', '']
    price = {}
    for arch in ('control', 'parity'):
        samples = [entry['validation'][entry['selected']]['price']['rmse'] for entry in runs_
                   if entry['architecture'] == arch]
        price[arch] = (float(np.mean(samples)), float(np.std(samples, ddof=1)))
    ratio = price['control'][1] / price['parity'][1]
    improved = sum(1 for name in NAMES
                   if mean_sd([entry['validation'][entry['selected']][name]['rmse'] for entry in runs_
                               if entry['architecture'] == 'parity'])[0]
                   < mean_sd([entry['validation'][entry['selected']][name]['rmse'] for entry in runs_
                              if entry['architecture'] == 'control'])[0])
    lines += [f"- Measured validation price RMSE sample standard deviation: control "
              f"{price['control'][1]:.6g}, parity {price['parity'][1]:.6g}.",
              f"- Parity showed approximately {ratio:.0f}x lower sample standard deviation across "
              'these three seeds.', '- ' + STABILITY_NOTE,
              '- Greek variabilities differ from the price result and are not covered by this claim: '
              + ', '.join(f'{name} SD control {sample_sd("control", name):.6g} vs '
                          f'parity {sample_sd("parity", name):.6g}' for name in NAMES[1:]) + '.', '',
              '## Verdict (computed)', '',
              f'- Refined parity improves {improved} of {len(NAMES)} validation RMSEs versus refined '
              'control at matched parents, procedure and seeds, and enforces exact call-put '
              'consistency by construction.',
              '- Financial constraint violations and limited seed coverage remain material '
              'limitations (see the constraint tables and the stability qualification above).',
              '', '## Measured regressions', '']
    lines += [f'- {text}.' for text in exceptions] or ['- none measured at the 1% threshold.']
    return '\n'.join(lines) + '\n'


def corrected_parity(study):
    study = study if study.is_absolute() else ROOT / study
    data = collect_parity_data(study)
    runs.save_json_atomic(study / 'audit.json', data)
    (study / 'report_corrected.md').write_text(build_corrected_parity_report(data))
    print(json.dumps({'corrected_report': str(study / 'report_corrected.md'),
                      'audit': str(study / 'audit.json')}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablation-study', type=Path)
    parser.add_argument('--cross-seed-study', type=Path)
    parser.add_argument('--parity-study', type=Path)
    args = parser.parse_args()
    if args.ablation_study:
        corrected_ablation(args.ablation_study)
    if args.cross_seed_study:
        corrected_cross_seed(args.cross_seed_study)
    if args.parity_study:
        corrected_parity(args.parity_study)


if __name__ == '__main__':
    main()
