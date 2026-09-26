import argparse
import csv
import json

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics
from experiments.deeponet.train import ROOT, digest
from src.neural.deeponet import load_deeponet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('comparison_directory')
    args = parser.parse_args()
    output = ROOT / args.comparison_directory
    protocol = json.loads((output / 'protocol.json').read_text())
    metrics = json.loads((output / 'comparison.json').read_text())
    dataset = ROOT / protocol['settings']['dataset']
    if digest(dataset) != protocol['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    torch.set_num_threads(protocol['settings']['threads'])
    with np.load(dataset, allow_pickle=False) as data:
        branch = torch.tensor(data['validation_branch'], dtype=torch.float64)
        trunk = torch.tensor(data['validation_trunk'], dtype=torch.float64)
        targets = data['validation_targets'].copy()
    lines = ['# DeepONet validation comparison', '',
             'Exploratory, single-seed comparison. The final benchmark was not accessed.', '',
             'Both arms start from the same selected pilot checkpoint with fresh Adam, '
             'learning rate 0.0003, identical shuffle seed and 300 additional CPU seconds. '
             'Checkpoint selection minimizes validation price MSE normalized by strike. '
             'The shared pilot cost was approximately 143 seconds.', '',
             'The price-only loss is mean squared price error divided by strike squared. '
             'The supervised arm adds the mean squared error across five Greeks, each '
             'divided by its training-label RMS squared, with coefficient 1. '
             'This is a fixed weighting choice, not a tuned optimum. '
             'Theta is per year; Vega and Rho are per unit decimal parameter change.', '',
             '| RMSE | Pilot | Longer price-only | Price + Greeks |',
             '|---|---:|---:|---:|']
    for name in NAMES:
        numbers = [metrics[arm][name]['rmse'] for arm in ('pilot', 'price_only', 'price_and_greeks')]
        lines.append(f'| {name} | ' + ' | '.join(f'{value:.6g}' for value in numbers) + ' |')
    lines.extend(['', '## Regional diagnostics', '',
                  'Bins are marginal groups, not causal effects; moneyness is S/K for both option types. '
                  'Worst individual points and every regional metric are saved alongside this report.', '',
                  '| Arm | Region | Price RMSE | Gamma RMSE | Theta RMSE |',
                  '|---|---|---:|---:|---:|'])
    for arm in ('pilot', 'price_only', 'price_and_greeks'):
        with (output / arm / 'validation_regions.csv').open() as handle:
            rows = list(csv.DictReader(handle))
        for dimension, group in (('remaining_maturity', '<0.1'), ('moneyness', '0.9-1.1'),
                                 ('option_type', 'call'), ('option_type', 'put')):
            values = {row['quantity']: float(row['rmse']) for row in rows
                      if row['dimension'] == dimension and row['group'] == group}
            lines.append(f'| {arm} | {dimension}: {group} | ' +
                         ' | '.join(f'{values[name]:.6g}' for name in ('price', 'gamma', 'theta')) + ' |')
    lines.extend(['', '## Last-checkpoint diagnostic', '',
                  'These are diagnostic outputs, not replacements for the price-selected checkpoints.', '',
                  '| Arm | Steps | Seconds | Selected epoch | Last price RMSE | Last Theta RMSE |',
                  '|---|---:|---:|---:|---:|---:|'])
    for arm in ('price_only', 'price_and_greeks'):
        folder = output / arm
        metadata = json.loads((folder / 'training_metadata.json').read_text())
        checkpoint = folder / 'last.pt'
        if digest(checkpoint) != metadata['checkpoint_sha256']['last']:
            raise ValueError('Checkpoint hash mismatch')
        last_dir = folder / 'last_diagnostics'
        last_dir.mkdir(exist_ok=True)
        last = diagnostics(load_deeponet(checkpoint), branch, trunk, targets, last_dir)
        (last_dir / 'validation_metrics.json').write_text(json.dumps(last, indent=2) + '\n')
        lines.append(f"| {arm} | {metadata['optimizer_steps']} | {metadata['training_wall_seconds']:.2f} | "
                     f"{metadata['selected_epoch']} | {last['price']['rmse']:.6g} | {last['theta']['rmse']:.6g} |")
    lines.extend(['', 'The protocol retains the inherited pilot configuration fields `loss` and `note`; '
                  'the actual per-arm objective is specified above and by `label_columns_used`, '
                  '`greek_scales`, and the archived source hash. '
                  'No confidence intervals or multi-seed superiority claims are supported by this run.', ''])
    (output / 'report.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
