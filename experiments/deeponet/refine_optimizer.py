import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.deeponet.compare_supervision import NAMES, diagnostics, supervised_loss, write_csv
from experiments.deeponet.train import ROOT, digest, train_model
from src.neural.deeponet import load_deeponet
from src.neural.deeponet_data import synthetic_split


def full_batch_closure(model, train, scales, chunk_size=1024):
    model.zero_grad(set_to_none=True)
    total = torch.zeros((), dtype=torch.float64)
    for start in range(0, len(train[0]), chunk_size):
        batch = tuple(value[start:start+chunk_size] for value in train)
        loss = supervised_loss(model, *batch, scales, .01) * (len(batch[0]) / len(train[0])) * 1e4
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite L-BFGS objective')
        loss.backward()
        total += loss.detach()
    return total


def lbfgs_refine(model, train, validation, scales, seconds):
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1., max_iter=1, max_eval=25,
                                 history_size=50, line_search_fn='strong_wolfe',
                                 tolerance_grad=1e-10, tolerance_change=1e-12)
    started = perf_counter()
    best = float('inf')
    selected = None
    best_step = 0
    history = []
    closures = 0
    step = 0
    last_progress = 0

    def validate():
        nonlocal best, selected, best_step
        with torch.no_grad():
            error = model(*validation[:2]) - validation[2]
            score = float((error / validation[0][:, 0]).square().mean())
        if not np.isfinite(score):
            raise FloatingPointError('Non-finite validation score')
        if score < best:
            best, best_step = score, step
            selected = {key: value.detach().clone() for key, value in model.state_dict().items()}
        history.append(dict(step=step, closure_calls=closures, validation_mse=score,
                            validation_price_rmse=float(error.square().mean().sqrt()),
                            wall_seconds=perf_counter()-started))

    def closure():
        nonlocal closures
        closures += 1
        return full_batch_closure(model, train, scales)

    validate()
    while perf_counter()-started < seconds:
        optimizer.step(closure)
        step += 1
        validate()
        if perf_counter()-started-last_progress >= 45:
            print(json.dumps(history[-1]), flush=True)
            last_progress = perf_counter()-started
    last = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(selected)
    return history, last, dict(selected_step=best_step, selected_validation_mse=best,
                               optimizer_steps=step, closure_calls=closures,
                               training_wall_seconds=perf_counter()-started,
                               stop_reason='wall_budget_checked_between_lbfgs_steps')


def shape_metrics(model, branch, trunk):
    values = {key: value.numpy() for key, value in model.price_and_greeks(branch, trunk).items()}
    b, t = branch.numpy(), trunk.numpy()
    disc_s = t[:, 0]*np.exp(-b[:, 3]*t[:, 1])
    disc_k = b[:, 0]*np.exp(-b[:, 2]*t[:, 1])
    call = b[:, 5] == 1
    lower = np.maximum(np.where(call, disc_s-disc_k, disc_k-disc_s), 0)
    upper = np.where(call, disc_s, disc_k)
    discount = np.exp(-b[:, 3]*t[:, 1])
    violations = {
        'price_bounds': np.maximum.reduce([lower-values['price'], values['price']-upper, np.zeros(len(b))]),
        'delta_bounds': np.maximum.reduce([np.where(call, 0, -discount)-values['delta'],
                                         values['delta']-np.where(call, discount, 0), np.zeros(len(b))]),
        'negative_gamma': np.maximum(-values['gamma'], 0),
    }
    return {name: dict(proportion_above_1e_8=float(np.mean(value > 1e-8)),
                       maximum_violation=float(value.max())) for name, value in violations.items()}


def main():
    registry_path = ROOT / 'experiments/config/neural/deeponet_model_registry.json'
    registry = json.loads(registry_path.read_text())
    parent = ROOT / registry['primary']['checkpoint']
    if digest(parent) != registry['primary']['checkpoint_sha256']:
        raise ValueError('Primary checkpoint hash mismatch')
    metadata = json.loads((parent.parent / 'training_metadata.json').read_text())
    dataset = ROOT / metadata['settings']['dataset']
    if digest(dataset) != metadata['dataset_sha256']:
        raise ValueError('Dataset hash mismatch')
    output = ROOT / 'results/deeponet/optimizer_refinement' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    with np.load(dataset, allow_pickle=False) as data:
        train = tuple(torch.tensor(data[f'train_{key}'], dtype=torch.float64) for key in ('branch', 'trunk', 'targets'))
        vb, vt = (torch.tensor(data[f'validation_{key}'], dtype=torch.float64) for key in ('branch', 'trunk'))
        vy = data['validation_targets'].copy()
    validation = (vb, vt, torch.tensor(vy[:, 0], dtype=torch.float64))
    scales = train[2][:, 1:].square().mean(0).sqrt().clamp_min(1e-12)
    domain = json.loads((ROOT / 'experiments/config/neural/deeponet_settings.json').read_text())['training_domain']
    stress_domain = {**domain, 'K': [75., 125.], 'T': [.03, 2.5], 'r': [-.03, .12],
                     'q': [0., .08], 'sigma': [.08, .65], 'moneyness': [.6, 1.4], 'tau_minimum': .005}
    arrays = {}
    for name, count, bounds, seed in (('in_domain', 256, domain, 25092026),
                                     ('stress', 128, stress_domain, 25093026)):
        split = synthetic_split(count, 16, bounds, seed)
        existing = {tuple(row) for row in np.concatenate((train[0].numpy(), vb.numpy()))}
        if any(tuple(row) in existing for row in split['branch']):
            raise ValueError('New audit contract overlaps training or validation')
        arrays.update({f'{name}_{key}': value for key, value in split.items()})
    audit_path = output / 'fresh_audit.npz'
    np.savez_compressed(audit_path, **arrays)
    del arrays, split
    sources = {}
    for path in (Path(__file__), ROOT / 'experiments/deeponet/train.py',
                 ROOT / 'experiments/deeponet/compare_supervision.py', ROOT / 'src/neural/deeponet.py',
                 ROOT / 'src/neural/deeponet_data.py', ROOT / 'src/black_scholes.py'):
        relative = path.relative_to(ROOT)
        saved = output / 'source' / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, saved)
        sources[str(relative)] = digest(path)
    protocol = dict(parent_checkpoint=str(parent.relative_to(ROOT)), parent_sha256=digest(parent),
                    dataset_sha256=digest(dataset), source_sha256=sources, greek_scales=scales.tolist(),
                    wall_seconds_per_arm=180, seed=23092026, greek_weight=.01,
                    fresh_audit_sha256=digest(audit_path), audit_seeds=[25092026, 25093026],
                    in_domain=domain, stress_domain=stress_domain, old_test_accessed=False,
                    selection='Lowest validation price RMSE among baseline and candidates with every Greek RMSE <= 1.1 times baseline. Fresh audit evaluated only after selection.',
                    lbfgs_loss_multiplier=10000, lbfgs_chunk_size=1024)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    print(json.dumps({'output': str(output)}), flush=True)
    base_dir = output / 'baseline'
    base_dir.mkdir()
    baseline = load_deeponet(parent)
    results = {'baseline': diagnostics(baseline, vb, vt, vy, base_dir)}
    (base_dir / 'shape_metrics.json').write_text(json.dumps(shape_metrics(baseline, vb, vt), indent=2)+'\n')
    for arm in ('adam_small_steps', 'lbfgs'):
        folder = output / arm
        folder.mkdir()
        model = load_deeponet(parent)
        torch.manual_seed(protocol['seed'])
        settings = {**metadata['settings'], 'seed': protocol['seed'], 'epochs': 100000,
                    'wall_seconds': 180, 'learning_rate': .00002, 'progress_every_seconds': 45,
                    'optimizer': arm, 'protocol_id': 'deeponet_optimizer_refinement_v1'}
        print(json.dumps({'starting': arm}), flush=True)
        if arm == 'lbfgs':
            settings.update(learning_rate=1., batch_size=len(train[0]), loss_multiplier=1e4,
                            chunk_size=1024, history_size=50, max_iter=1, line_search_fn='strong_wolfe')
            history, last, summary = lbfgs_refine(model, train, validation, scales, 180)
        else:
            loss = lambda m, b, t, y: supervised_loss(m, b, t, y, scales, .01)
            history, last, summary = train_model(model, train, validation, settings, loss)
        record = {**protocol, **summary, 'settings': settings, 'label_columns_used': list(NAMES),
                  'final_test_accessed': False}
        for kind, state in (('selected', model.state_dict()), ('last', last)):
            torch.save(dict(architecture=model.architecture, state_dict=state, metadata=record), folder / f'{kind}.pt')
        record['checkpoint_sha256'] = {kind: digest(folder / f'{kind}.pt') for kind in ('selected', 'last')}
        (folder / 'training_metadata.json').write_text(json.dumps(record, indent=2)+'\n')
        write_csv(folder / 'history.csv', history)
        results[arm] = diagnostics(model, vb, vt, vy, folder)
        (folder / 'validation_metrics.json').write_text(json.dumps(results[arm], indent=2)+'\n')
        (folder / 'shape_metrics.json').write_text(json.dumps(shape_metrics(model, vb, vt), indent=2)+'\n')
        print(json.dumps({'completed': arm, 'metrics': results[arm]}), flush=True)
    eligible = [name for name, metrics in results.items() if all(
        metrics[greek]['rmse'] <= 1.1*results['baseline'][greek]['rmse'] for greek in NAMES[1:])]
    selected = min(eligible, key=lambda name: results[name]['price']['rmse'])
    (output / 'selection.json').write_text(json.dumps(dict(selected=selected, eligible=eligible,
                                                         validation=results, rule=protocol['selection']), indent=2)+'\n')
    audit_results = {}
    selected_checkpoint = parent if selected == 'baseline' else output / selected / 'selected.pt'
    for name, checkpoint in (('baseline', parent), ('selected', selected_checkpoint)):
        model = load_deeponet(checkpoint)
        audit_results[name] = {}
        with np.load(audit_path, allow_pickle=False) as data:
            for split in ('in_domain', 'stress'):
                b, t = (torch.tensor(data[f'{split}_{key}'], dtype=torch.float64) for key in ('branch', 'trunk'))
                folder = output / 'audit' / name / split
                folder.mkdir(parents=True)
                audit_results[name][split] = diagnostics(model, b, t, data[f'{split}_targets'], folder)
                (folder / 'shape_metrics.json').write_text(json.dumps(shape_metrics(model, b, t), indent=2)+'\n')
    if digest(parent) != protocol['parent_sha256'] or digest(audit_path) != protocol['fresh_audit_sha256']:
        raise ValueError('Protected artifact changed')
    final = dict(selected=selected, selected_checkpoint=str(selected_checkpoint.relative_to(ROOT)),
                 selected_sha256=digest(selected_checkpoint), validation=results, fresh_audit=audit_results,
                 previous_primary_preserved=True, old_final_test_reused=False)
    (output / 'summary.json').write_text(json.dumps(final, indent=2)+'\n')
    print(json.dumps(final), flush=True)


if __name__ == '__main__':
    main()
