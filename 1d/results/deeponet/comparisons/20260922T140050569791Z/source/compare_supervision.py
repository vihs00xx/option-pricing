import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from experiments.deeponet.train import ROOT, digest, train_model
from src.neural.deeponet import load_deeponet


NAMES = ('price', 'delta', 'gamma', 'vega', 'theta', 'rho')
PILOT = ROOT / 'results/deeponet/training/pilot_20260922T135135187961Z'


def supervised_loss(model, branch, trunk, targets, greek_scales):
    values = model.price_and_greeks(branch, trunk, create_graph=True)
    price_loss = ((values['price'] - targets[:, 0]) / branch[:, 0]).square().mean()
    errors = torch.stack([values[name] - targets[:, i] for i, name in enumerate(NAMES) if i], 1)
    return price_loss + (errors / greek_scales).square().mean()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostics(model, branch, trunk, targets, output):
    predicted = model.price_and_greeks(branch, trunk)
    errors = np.column_stack([predicted[name].numpy() for name in NAMES]) - targets
    b, t = branch.numpy(), trunk.numpy()
    ratio = t[:, 0] / b[:, 0]
    groups = [('all', 'all', np.ones(len(b), dtype=bool))]
    for dimension, values, edges, labels in (
        ('remaining_maturity', t[:, 1], [.1, .5, 1.], ['<0.1', '0.1-0.5', '0.5-1', '>=1']),
        ('contract_maturity', b[:, 1], [.5, 1.], ['<0.5', '0.5-1', '>=1']),
        ('moneyness', ratio, [.9, 1.1], ['<0.9', '0.9-1.1', '>=1.1']),
    ):
        indices = np.digitize(values, edges)
        groups.extend((dimension, label, indices == i) for i, label in enumerate(labels))
    groups.extend(('option_type', label, b[:, 5] == sign) for sign, label in ((1, 'call'), (-1, 'put')))
    rows = []
    for dimension, group, mask in groups:
        if not mask.any():
            continue
        for i, name in enumerate(NAMES):
            rows.append(dict(dimension=dimension, group=group, quantity=name, count=int(mask.sum()),
                             rmse=float(np.sqrt(np.mean(errors[mask, i] ** 2))),
                             max_absolute_error=float(np.max(np.abs(errors[mask, i])))))
    write_csv(output / 'validation_regions.csv', rows)
    worst = []
    for i, name in enumerate(NAMES):
        for index in np.argsort(np.abs(errors[:, i]))[-10:][::-1]:
            worst.append(dict(quantity=name, row=int(index), absolute_error=float(abs(errors[index, i])),
                              reference=float(targets[index, i]), predicted=float(predicted[name][index]),
                              K=b[index, 0], T=b[index, 1], r=b[index, 2], q=b[index, 3],
                              sigma=b[index, 4], option_type='call' if b[index, 5] == 1 else 'put',
                              S=t[index, 0], tau=t[index, 1], moneyness=ratio[index]))
    write_csv(output / 'validation_worst_points.csv', worst)
    return {name: {'rmse': rows[i]['rmse'], 'max_absolute_error': rows[i]['max_absolute_error']}
            for i, name in enumerate(NAMES)}


def main():
    pilot_metadata = json.loads((PILOT / 'training_metadata.json').read_text())
    checkpoint = PILOT / 'selected.pt'
    if digest(checkpoint) != pilot_metadata['checkpoint_sha256']['selected']:
        raise ValueError('Pilot checkpoint hash mismatch')
    settings = dict(pilot_metadata['settings'])
    dataset = ROOT / settings['dataset']
    if digest(dataset) != pilot_metadata['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    settings.update(epochs=100000, wall_seconds=300, learning_rate=.0003, validation_every=10)
    output = ROOT / 'results/deeponet/comparisons' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(settings['threads'])
    with np.load(dataset, allow_pickle=False) as data:
        train_b, train_t, train_y = [torch.tensor(data[f'train_{key}'], dtype=torch.float64)
                                    for key in ('branch', 'trunk', 'targets')]
        val_b, val_t = [torch.tensor(data[f'validation_{key}'], dtype=torch.float64)
                        for key in ('branch', 'trunk')]
        val_y = data['validation_targets'].copy()
    # Scales use training labels only; all six objectives have fixed weights.
    scales = train_y[:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    protocol = dict(settings=settings, parent_checkpoint=str(checkpoint.relative_to(ROOT)),
                    parent_sha256=digest(checkpoint), dataset_sha256=digest(dataset),
                    greek_scales=scales.tolist(), greek_weight=1., final_test_accessed=False,
                    design='Same pilot initialization, fresh Adam, same shuffle seed, 300 seconds per arm; price validation selection',
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                                   (Path(__file__), ROOT / 'experiments/deeponet/train.py', ROOT / 'src/neural/deeponet.py')})
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    pilot_dir = output / 'pilot'
    pilot_dir.mkdir()
    results = {'pilot': diagnostics(load_deeponet(checkpoint), val_b, val_t, val_y, pilot_dir)}
    print(json.dumps({'output': str(output), 'pilot': results['pilot']}, indent=2), flush=True)
    for arm in ('price_only', 'price_and_greeks'):
        folder = output / arm
        folder.mkdir()
        torch.manual_seed(settings['seed'])
        model = load_deeponet(checkpoint)
        training = (train_b, train_t, train_y[:, 0] if arm == 'price_only' else train_y)
        validation = (val_b, val_t, torch.tensor(val_y[:, 0], dtype=torch.float64))
        loss_fn = None if arm == 'price_only' else lambda m, b, t, y: supervised_loss(m, b, t, y, scales)
        history, last, summary = train_model(model, training, validation, settings, loss_fn)
        metadata = {**protocol, **summary, 'arm': arm,
                    'label_columns_used': ['price'] if arm == 'price_only' else list(NAMES)}
        for kind, state in (('selected', model.state_dict()), ('last', last)):
            torch.save(dict(architecture=model.architecture, state_dict=state, metadata=metadata), folder / f'{kind}.pt')
        metadata['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
        (folder / 'training_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        write_csv(folder / 'history.csv', history)
        results[arm] = diagnostics(model, val_b, val_t, val_y, folder)
        (folder / 'validation_metrics.json').write_text(json.dumps(results[arm], indent=2) + '\n')
        print(json.dumps({'arm': arm, **summary, 'metrics': results[arm]}, indent=2), flush=True)
    (output / 'comparison.json').write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
