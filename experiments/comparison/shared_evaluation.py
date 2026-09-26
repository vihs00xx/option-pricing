"""Freeze and evaluate fresh 1D comparison cases without retraining."""

import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '2'

import argparse
import csv
import hashlib
import json
import platform
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import scipy
import torch
from scipy.stats import qmc

from src.black_scholes import black_scholes_greeks, black_scholes_price
from src.classical_benchmark import evaluate_batch, evaluate_case
from src.common_benchmark import inputs_from_case
from src.crank_nicolson import _clustered_s_max, _surface
from src.neural.deeponet import load_deeponet
from src.neural.fixed_pinn import load_fixed_pinn

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/shared_comparison/v1'
STUDY = ROOT / 'results/deeponet/parity_refinement/20260923T155834294516Z'
CORE = ('price', 'delta', 'gamma', 'theta')
ALL = ('price', 'delta', 'gamma', 'vega', 'theta', 'rho')
SEEDS = (22092026, 22092027, 22092028)
torch.set_num_threads(2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_cases(seed=2609202607):
    cases = []

    def sample(n, d):
        nonlocal seed
        seed += 1
        return qmc.LatinHypercube(d, seed=seed).random(n)

    def add(panel, region, K, T, m, tau, r, q, sigma, option='call', pair_id=''):
        cases.append(dict(case_id=f'{panel}_{len(cases):04d}', panel=panel, region=region,
                          pair_id=pair_id, S=float(K*m), K=float(K), T=float(T),
                          moneyness=float(m), tau=float(tau), r=float(r), q=float(q),
                          sigma=float(sigma), option=option, mc_index=len(cases)))

    for region, n in [('shared_domain', 128), ('pinn_core_tails', 64), ('short_maturity', 64)]:
        for i, (a, b) in enumerate(sample(n, 2)):
            m = .7 + .6*a if region == 'shared_domain' else (
                (.5 + .2*a if i % 2 == 0 else 1.3 + .2*a)
                if region == 'pinn_core_tails' else .9 + .2*a)
            tau = .01 + .04*b if region == 'short_maturity' else .05 + .95*b
            add('fixed', region, 100, 1, m, tau, .05, 0, .2)

    groups = [('interpolation', 64), ('short_maturity', 16)]
    groups += [(f'extrapolation_{name}', 8) for name in ('moneyness', 'maturity', 'sigma', 'r', 'q', 'K')]
    for region, n in groups:
        for i, u in enumerate(sample(n, 7)):
            K, T, m = 80+40*u[0], .1+1.9*u[1], .7+.6*u[2]
            tau, r, q, sigma = .05+(T-.05)*u[3], -.02+.12*u[4], .06*u[5], .1+.5*u[6]
            if region == 'short_maturity':
                tau, m = .01+.04*u[3], .95+.1*u[2]
            elif region == 'extrapolation_moneyness':
                m = .55+.1*u[2] if i % 2 == 0 else 1.35+.1*u[2]
            elif region == 'extrapolation_maturity':
                tau = 2.1+.4*u[3]
                T = tau
            elif region == 'extrapolation_sigma':
                sigma = .65+.1*u[6]
            elif region == 'extrapolation_r':
                r = .12+.04*u[4]
            elif region == 'extrapolation_q':
                q = .07+.03*u[5]
            elif region == 'extrapolation_K':
                K = 65+10*u[0] if i % 2 == 0 else 125+10*u[0]
            pair = f'{region}_{i}'
            for option in ('call', 'put'):
                add('varying', region, K, T, m, tau, r, q, sigma, option, pair)
    return cases


def arrays(cases):
    branch = np.array([[c['K'], c['T'], c['r'], c['q'], c['sigma'],
                        1 if c['option'] == 'call' else -1] for c in cases])
    trunk = np.array([[c['S'], c['tau']] for c in cases])
    return branch, trunk


def financial_keys(branch, trunk):
    joined = np.column_stack((branch[:, [0, 2, 3, 4, 5]], trunk))
    return {tuple(row) for row in np.round(joined, 10)}


def collision_audit(cases):
    proposed = financial_keys(*arrays(cases))
    entries = []
    for path in sorted((ROOT/'results/deeponet').rglob('*.npz')):
        with np.load(path, allow_pickle=False) as data:
            for name in data.files:
                if not name.endswith('branch'):
                    continue
                partner = name[:-6] + 'trunk'
                if partner not in data.files and name.startswith('pairs_'):
                    partner = 'pairs_trunk'
                if partner not in data.files:
                    continue
                hits = proposed & financial_keys(data[name], data[partner])
                if hits:
                    raise RuntimeError(f'Fresh cases overlap {path}:{name}')
                entries.append(dict(path=str(path.relative_to(ROOT)), split=name, rows=len(data[name]),
                                    file_sha256=digest(path), overlaps=0))
    return entries


def verify_pins(manifest, folder=OUT):
    if digest(folder/'cases.json') != manifest['cases_sha256']:
        raise RuntimeError('Frozen cases changed')
    for path, expected in manifest['source_sha256'].items():
        if digest(ROOT/path) != expected:
            raise RuntimeError(f'Frozen implementation changed: {path}')
    for model in manifest['models'].values():
        if digest(ROOT/model['checkpoint']) != model['sha256']:
            raise RuntimeError('Frozen checkpoint changed')


def freeze():
    if OUT.exists():
        raise RuntimeError('Comparison already exists; refusing to replace its protocol')
    registry = read(ROOT/'experiments/config/neural/pinn_model_registry.json')['primary_model']
    if digest(ROOT/registry['checkpoint']) != registry['checkpoint_sha256']:
        raise RuntimeError('Anchored PINN differs from its frozen registry')
    models = {'anchored_pinn': dict(checkpoint=registry['checkpoint'], sha256=registry['checkpoint_sha256'])}
    for seed in SEEDS:
        folder = STUDY/f'parity_{seed}'
        selection = read(folder/'selection.json')
        arm = selection['selected']
        if arm == 'baseline':
            checkpoint = ROOT/f'results/deeponet/training_ablation/20260923T111505263380Z/arms/parity_{seed}/selected.pt'
        else:
            checkpoint = folder/arm/'selected.pt'
        metadata = read(checkpoint.parent/'training_metadata.json')
        if digest(checkpoint) != metadata['checkpoints']['selected']:
            raise RuntimeError('Parity checkpoint metadata mismatch')
        models[f'parity_{seed}'] = dict(checkpoint=str(checkpoint.relative_to(ROOT)), sha256=digest(checkpoint),
                                      seed=seed, selected_arm=arm,
                                      validation_price_rmse=selection['validation'][arm]['price']['rmse'])
    primary = min((k for k in models if k.startswith('parity_')),
                  key=lambda k: (models[k]['validation_price_rmse'], models[k]['seed']))
    cases = make_cases()
    audit = collision_audit(cases)
    sources = list((ROOT/'src').rglob('*.py')) + [Path(__file__)]
    OUT.mkdir(parents=True)
    save(OUT/'cases.json', cases)
    write_csv(OUT/'cases.csv', cases)
    settings = read(ROOT/'experiments/config/classical/common_benchmark_settings.json')
    settings['monte_carlo']['seed_root'] = 926260000
    manifest = dict(protocol='fresh_shared_comparison_v1', frozen_at=datetime.now(timezone.utc).isoformat(),
                    status='frozen_before_prediction', cases_sha256=digest(OUT/'cases.json'),
                    models=models, primary_parity=primary, settings=settings,
                    source_sha256={str(p.relative_to(ROOT)):digest(p) for p in sources},
                    selection='Minimum existing validation price RMSE across the three already-selected parity seeds; no fresh labels used.',
                    panels={'fixed': '256 calls, K=100,T=1,r=.05,q=0,sigma=.2; four outputs. Regions reported separately.',
                            'varying': '256 calls/puts in 128 paired states; six outputs; PINN excluded. Maturity extrapolation moves tau and T together.'},
                    scope='Independent fresh query assessment of already-developed models, not a claim that the whole research programme was prospectively specified.',
                    timing=dict(batch_sizes=[1,8,32], warmups=1, repetitions=5, threads=2, device='cpu',
                                population='first 32 shared-domain fixed cases or interpolation varying cases',
                                includes='input conversion, solver, Greeks and host output conversion; excludes checkpoint loading',
                                workload='independent queries; no cached CN surface. Native RQMC internally computes six outputs even for the four-output panel.',
                                memory='Separate fresh subprocess per panel/method; sampled RSS at 1ms during each measured call, with pre-call RSS and peak increment.'),
                    metrics=dict(absolute=['rmse','mae','maximum_absolute_error','bias'],
                                 relative_floors=settings['relative_error_floors'], shape_tolerance=1e-5,
                                 groups=['panel','region','option','moneyness_band','maturity_band'],
                                 uncertainty='RQMC scramble standard errors retained; one randomized estimate per case does not measure repeated-trial coverage.'),
                    collision_audit=audit,
                    freshness='Novel deterministic LHS seeds; zero rounded financial-query overlap with available DeepONet NPZ splits. No claim of statistical independence of paired calls and puts.',
                    environment=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
                                     platform=platform.platform(), processor=platform.processor(), cpu_count=os.cpu_count()))
    save(OUT/'protocol.json', manifest)
    (OUT/'protocol.sha256').write_text(digest(OUT/'protocol.json'))
    for p in sources:
        target=OUT/'source'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(p.read_bytes())
    print(json.dumps(dict(frozen=str(OUT), primary=primary, cases=len(cases), audit_arrays=len(audit))), flush=True)


def get_protocol():
    if digest(OUT/'protocol.json') != (OUT/'protocol.sha256').read_text().strip():
        raise RuntimeError('Protocol identity changed')
    manifest=read(OUT/'protocol.json')
    verify_pins(manifest)
    return manifest, read(OUT/'cases.json')


def cn_core(case, settings):
    c = settings['crank_nicolson']
    p = inputs_from_case(case)
    upper = _clustered_s_max(p['S'],p['K'],p['tau'],p['r'],p['sigma'],p['q'])
    results=[]
    for factor in (1,2):
        _, spline = _surface(**p, s_max=upper, n_space=factor*c['base_grid_size'],
                             n_time=factor*c['base_grid_size'], rannacher_steps=c['rannacher_steps'],
                             mesh=c['mesh'], concentration=c['mesh_concentration'])
        v,d,g = (float(spline(p['S'], k)) for k in (0,1,2))
        theta = -.5*p['sigma']**2*p['S']**2*g-(p['r']-p['q'])*p['S']*d+p['r']*v
        results.append(dict(price=v,delta=d,gamma=g,theta=theta))
    return {q:(4*results[1][q]-results[0][q])/3 for q in CORE}


class Predictor:
    def __init__(self, method, manifest):
        self.method,self.manifest = method,manifest
        self.model=None
        if method in manifest['models']:
            loader=load_fixed_pinn if method=='anchored_pinn' else load_deeponet
            self.model=loader(ROOT/manifest['models'][method]['checkpoint']).eval()

    def __call__(self, cases):
        panel=cases[0]['panel']
        if any(c['panel']!=panel for c in cases):
            raise ValueError('Do not mix panel workloads')
        quantities=CORE if panel=='fixed' else ALL
        if self.method=='anchored_pinn':
            p=self.model.problem
            if any(c['option']!='call' or any(c[k]!=getattr(p,k) for k in ('K','T','r','q','sigma'))
                   or not 0<c['tau']<=p.T or not p.x_min<=c['S']/p.K<=p.x_max for c in cases):
                raise ValueError('PINN query outside fixed-contract scope')
            y=torch.tensor([[np.log(c['S']/p.K)] for c in cases],dtype=torch.float64,requires_grad=True)
            t=torch.tensor([[c['tau']] for c in cases],dtype=torch.float64,requires_grad=True)
            values=self.model.price_and_greeks(y,t)
            return {k:v.detach().numpy().reshape(-1) for k,v in values.items()},{}
        if self.model is not None:
            b,t=arrays(cases)
            b=torch.tensor(b,dtype=torch.float64)
            t=torch.tensor(t,dtype=torch.float64,requires_grad=True)
            if panel=='varying':
                values=self.model.price_and_greeks(b,t)
            else:
                v=self.model(b,t)
                dt=torch.autograd.grad(v.sum(),t,create_graph=True)[0]
                gamma=torch.autograd.grad(dt[:,0].sum(),t)[0][:,0]
                values=dict(price=v,delta=dt[:,0],gamma=gamma,theta=-dt[:,1])
            return {k:v.detach().numpy() for k,v in values.items()},{}
        if self.method=='black_scholes':
            values=evaluate_batch(self.method,cases,self.manifest['settings'])
            return {q:np.array([v[q] for v in values]) for q in quantities},{}
        values,errors=[],[]
        for c in cases:
            if self.method=='crank_nicolson' and panel=='fixed':
                v,e=cn_core(c,self.manifest['settings']),{}
            else:
                v,e=evaluate_case(self.method,c,self.manifest['settings'],c['mc_index'])
            values.append(v);errors.append(e)
        return ({q:np.array([v[q] for v in values]) for q in quantities},
                {q:np.array([e.get(q,0) for e in errors]) for q in quantities})


def methods(manifest,panel,secondary=False):
    result=['black_scholes','conditional_rqmc','crank_nicolson']
    if panel=='fixed':result.append('anchored_pinn')
    result.append(manifest['primary_parity'])
    if secondary:result += [f'parity_{s}' for s in SEEDS if f'parity_{s}'!=manifest['primary_parity']]
    return result


def evaluate():
    manifest,cases=get_protocol()
    output=OUT/'predictions'
    output.mkdir(exist_ok=True)
    for panel in ('fixed','varying'):
        selected=[c for c in cases if c['panel']==panel]
        for method in methods(manifest,panel,True):
            target=output/f'{panel}_{method}.json'
            if target.exists():
                stored=read(target)
                if stored['protocol_sha256']!=digest(OUT/'protocol.json'):
                    raise RuntimeError('Existing prediction protocol mismatch')
                continue
            predictor=Predictor(method,manifest)
            values={q:[] for q in (CORE if panel=='fixed' else ALL)}
            errors={q:[] for q in values}
            started=perf_counter()
            for start in range(0,len(selected),32):
                v,e=predictor(selected[start:start+32])
                for q in values:
                    if not np.isfinite(v[q]).all():raise RuntimeError('Non-finite prediction')
                    values[q].extend(v[q].tolist())
                    errors[q].extend(e.get(q,np.zeros(len(v[q]))).tolist())
                print(f'{panel} {method}: {min(start+32,len(selected))}/{len(selected)}',flush=True)
            save(target,dict(protocol_sha256=digest(OUT/'protocol.json'),case_ids=[c['case_id'] for c in selected],
                             values=values,standard_errors=errors,evaluation_seconds=perf_counter()-started))
    verify_pins(manifest)
    save(OUT/'accuracy_complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                         predictions_sha256={p.name:digest(p) for p in output.glob('*.json')}))


def time_worker(panel,method):
    manifest,cases=get_protocol()
    region='shared_domain' if panel=='fixed' else 'interpolation'
    selected=[c for c in cases if c['panel']==panel and c['region']==region]
    predictor=Predictor(method,manifest)
    rows=[]
    proc=psutil.Process()
    for size in manifest['timing']['batch_sizes']:
        batch=selected[:size]
        for _ in range(manifest['timing']['warmups']):predictor(batch)
        for repetition in range(manifest['timing']['repetitions']):
            baseline=proc.memory_info().rss
            samples=[baseline]
            stop=threading.Event()
            def sample():
                while not stop.wait(.001):samples.append(proc.memory_info().rss)
            monitor=threading.Thread(target=sample,daemon=True)
            monitor.start()
            start=perf_counter()
            values,_=predictor(batch)
            elapsed=perf_counter()-start
            samples.append(proc.memory_info().rss)
            stop.set();monitor.join()
            rows.append(dict(panel=panel,method=method,batch_size=size,repetition=repetition,
                             seconds=elapsed,per_option_seconds=elapsed/size,
                             outputs=len(values),pre_call_rss_mb=baseline/2**20,
                             peak_rss_mb=max(samples)/2**20,peak_increment_mb=(max(samples)-baseline)/2**20))
        print(f'timing {panel} {method} batch={size}',flush=True)
    (OUT/'timing').mkdir(exist_ok=True)
    save(OUT/'timing'/f'{panel}_{method}.json',rows)


def timing():
    manifest,_=get_protocol()
    for panel in ('fixed','varying'):
        for method in methods(manifest,panel):
            target=OUT/'timing'/f'{panel}_{method}.json'
            if target.exists():continue
            subprocess.run([sys.executable,'-m','experiments.comparison.shared_evaluation',
                            'time-worker','--panel',panel,'--method',method],cwd=ROOT,check=True)
    verify_pins(manifest)
    save(OUT/'timing_complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                       files_sha256={p.name:digest(p) for p in (OUT/'timing').glob('*.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','evaluate','timing','time-worker'])
    parser.add_argument('--panel');parser.add_argument('--method')
    args=parser.parse_args()
    if args.command=='time-worker':time_worker(args.panel,args.method)
    else:globals()[args.command]()
