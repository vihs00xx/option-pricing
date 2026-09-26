"""Summarize saved fresh-comparison outputs without evaluating models."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from experiments.comparison.shared_evaluation import ROOT, OUT, CORE, ALL, SEEDS, STUDY, digest, get_protocol, read, save, write_csv


def metric(predicted,reference,floor):
    error=predicted-reference
    eligible=np.abs(reference)>=floor
    return dict(cases=len(error),rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(np.abs(error))),maximum_absolute_error=float(np.max(np.abs(error))),
                bias=float(np.mean(error)),relative_error_cases=int(eligible.sum()),
                relative_rmse=float(np.sqrt(np.mean((error[eligible]/reference[eligible])**2))) if eligible.any() else None)


def violations(cases,values,tol):
    S,K,t,r,q=(np.array([c[k] for c in cases]) for k in ('S','K','tau','r','q'))
    A,B=S*np.exp(-q*t),K*np.exp(-r*t)
    call=np.array([c['option']=='call' for c in cases])
    lower=np.maximum(np.where(call,A-B,B-A),0)
    upper=np.where(call,A,B)
    dl=np.where(call,0,-np.exp(-q*t));du=np.where(call,np.exp(-q*t),0)
    severity=dict(price_bounds=np.maximum.reduce([lower-values['price'],values['price']-upper,np.zeros(len(S))]),
                  delta_bounds=np.maximum.reduce([dl-values['delta'],values['delta']-du,np.zeros(len(S))]),
                  negative_gamma=np.maximum(-values['gamma'],0))
    return {k:dict(count=int((v>tol).sum()),fraction=float(np.mean(v>tol)),maximum=float(v.max()))
            for k,v in severity.items()}


def groups(cases):
    yield 'all','all',np.ones(len(cases),dtype=bool)
    for field in ('region','option'):
        for value in sorted({c[field] for c in cases}):
            yield field,value,np.array([c[field]==value for c in cases])
    m=np.array([c['moneyness'] for c in cases]);t=np.array([c['tau'] for c in cases])
    for label,mask in [('below_.9',m<.9),('.9_to_1.1',(m>=.9)&(m<=1.1)),('above_1.1',m>1.1)]:
        if mask.any():yield 'moneyness_band',label,mask
    for label,mask in [('below_.05',t<.05),('.05_to_.5',(t>=.05)&(t<.5)),('at_least_.5',t>=.5)]:
        if mask.any():yield 'maturity_band',label,mask


def main():
    manifest,cases=get_protocol()
    if not (OUT/'accuracy_complete.json').exists():raise RuntimeError('Accuracy evaluation incomplete')
    for name,expected in read(OUT/'accuracy_complete.json')['predictions_sha256'].items():
        if digest(OUT/'predictions'/name)!=expected:raise RuntimeError('Saved prediction hash mismatch')
    metrics,shape,worst,parity=[],[],[],[]
    loaded={}
    for panel in ('fixed','varying'):
        selected=[c for c in cases if c['panel']==panel]
        ref=read(OUT/'predictions'/f'{panel}_black_scholes.json')
        refs={k:np.array(v) for k,v in ref['values'].items()}
        for path in sorted((OUT/'predictions').glob(f'{panel}_*.json')):
            model=path.stem[len(panel)+1:]
            data=read(path)
            if data['case_ids']!=[c['case_id'] for c in selected]:raise RuntimeError('Prediction ordering mismatch')
            values={k:np.array(v) for k,v in data['values'].items()}
            loaded[panel,model]=values
            for dim,group,mask in groups(selected):
                for q,predicted in values.items():
                    metrics.append(dict(panel=panel,method=model,dimension=dim,group=group,quantity=q,
                                        **metric(predicted[mask],refs[q][mask],manifest['metrics']['relative_floors'][q])))
                if dim in ('all','region'):
                    v=violations([c for c,m in zip(selected,mask) if m],{q:v[mask] for q,v in values.items()},
                                 manifest['metrics']['shape_tolerance'])
                    for constraint,result in v.items():
                        shape.append(dict(panel=panel,method=model,dimension=dim,group=group,constraint=constraint,**result))
            for q,predicted in values.items():
                for i in np.argsort(-np.abs(predicted-refs[q]))[:5]:
                    worst.append(dict(method=model,quantity=q,**selected[i],predicted=float(predicted[i]),
                                      reference=float(refs[q][i]),error=float(predicted[i]-refs[q][i])))
            if panel=='varying':
                differences=[]
                for i in range(0,len(selected),2):
                    c=selected[i]
                    d=c['S']*np.exp(-c['q']*c['tau'])-c['K']*np.exp(-c['r']*c['tau'])
                    differences.append(values['price'][i]-values['price'][i+1]-d)
                parity.append(dict(method=model,pairs=len(differences),maximum_absolute_parity_error=float(np.max(np.abs(differences)))))
    write_csv(OUT/'metrics.csv',metrics)
    write_csv(OUT/'shape_metrics.csv',shape)
    write_csv(OUT/'worst_cases.csv',worst)
    write_csv(OUT/'put_call_parity.csv',parity)
    timing=[]
    if (OUT/'timing_complete.json').exists():
        for name,expected in read(OUT/'timing_complete.json')['files_sha256'].items():
            if digest(OUT/'timing'/name)!=expected:raise RuntimeError('Saved timing hash mismatch')
        for path in sorted((OUT/'timing').glob('*.json')):
            if path.name.endswith('.environment.json'):continue
            records=read(path)
            for size in sorted({r['batch_size'] for r in records}):
                rs=[r for r in records if r['batch_size']==size]
                seconds=np.array([r['seconds'] for r in rs])
                timing.append(dict(panel=rs[0]['panel'],method=rs[0]['method'],batch_size=size,
                                   outputs=rs[0]['outputs'],repetitions=len(rs),median_seconds=float(np.median(seconds)),
                                   q25_seconds=float(np.quantile(seconds,.25)),q75_seconds=float(np.quantile(seconds,.75)),
                                   per_option_seconds=float(np.median(seconds)/size),throughput=float(size/np.median(seconds)),
                                   peak_rss_mb=max(r['peak_rss_mb'] for r in rs),
                                   peak_increment_mb=max(r['peak_increment_mb'] for r in rs)))
        write_csv(OUT/'timing_summary.csv',timing)
    seed_metrics=[]
    for panel in ('fixed','varying'):
        for q in (CORE if panel=='fixed' else ALL):
            rows=[r for r in metrics if r['panel']==panel and r['dimension']=='all' and r['quantity']==q
                  and r['method'].startswith('parity_')]
            a=np.array([r['rmse'] for r in rows])
            seed_metrics.append(dict(panel=panel,quantity=q,seeds=len(a),mean_rmse=float(a.mean()),
                                     sample_sd=float(a.std(ddof=1)),minimum=float(a.min()),maximum=float(a.max())))
    write_csv(OUT/'parity_seed_summary.csv',seed_metrics)
    primary=manifest['primary_parity']
    cost=[]
    source=ROOT/'results/pinn/primary/anchored/tables/training_metadata.json'
    record=read(source)
    for stage,seconds in [('initial',record['wall_seconds']),('refinement',record['refinement_wall_seconds']),
                          ('anchored refinement',record['anchored_refinement']['wall_seconds'])]:
        cost.append(dict(model='anchored_pinn',stage=stage,seconds=seconds,source=str(source.relative_to(ROOT)),sha256=digest(source)))
    for seed in SEEDS:
        parent=ROOT/'results/deeponet/training_ablation/20260923T111505263380Z'
        paths=[('pilot',parent/f'pilot/parity_frozen_{seed}/training_metadata.json'),
               ('Greek-supervised continuation',parent/f'arms/parity_{seed}/training_metadata.json')]
        paths += [(arm,STUDY/f'parity_{seed}/{arm}/training_metadata.json') for arm in ('adam_small_steps','lbfgs')]
        for stage,source in paths:
            record=read(source)
            seconds=(record.get('training') or record)['training_wall_seconds']
            cost.append(dict(model=f'parity_{seed}',stage=stage,seconds=seconds,source=str(source.relative_to(ROOT)),sha256=digest(source)))
    write_csv(OUT/'historical_training_costs.csv',cost)
    names={'black_scholes':'Analytical BS','conditional_rqmc':'Conditional RQMC','crank_nicolson':'CN + Richardson',
           'anchored_pinn':'Anchored PINN',primary:'Refined parity DeepONet'}
    def rows_for(panel,region='all'):
        return [r for r in metrics if r['panel']==panel and r['method'] in names
                and r['dimension']==('all' if region=='all' else 'region') and r['group']==region]
    def table(panel,region='all'):
        quantities=CORE if panel=='fixed' else ALL
        out='| Method | '+' | '.join(quantities)+' |\n|---|'+'---:|'*len(quantities)+'\n'
        rows=rows_for(panel,region)
        for model,name in names.items():
            values={r['quantity']:r['rmse'] for r in rows if r['method']==model}
            if values:out+='| '+name+' | '+' | '.join(f'{values[q]:.5g}' for q in quantities)+' |\n'
        return out
    report=f'''# Fresh shared comparison of 1D option-pricing methods

Protocol frozen: {manifest['frozen_at']}. The cases, numerical budgets, checkpoint hashes, source snapshot, and selection rule were saved before predictions. No model was retrained or selected using these results. These cases are now observed and must not be reused as untouched tests for subsequent tuning.

The primary parity checkpoint is **{primary}**, chosen by the lowest existing validation price RMSE among the three previously refined seeds. This does not promote or replace the older registered primary. The other two parity checkpoints are evaluated as secondary results, not as a test-set selection contest.

## Fixed-parameter comparison

All five methods use 256 fresh calls at K=100, T=1, r=0.05, q=0, and sigma=0.2. The four reported outputs are price, Delta, Gamma, and calendar Theta. No Vega or Rho claim is made for the fixed PINN.

### Shared training-domain region (128 queries)

Moneyness is 0.7–1.3 and remaining maturity is 0.05–1 year. These points lie within both neural models' supported domains. RMSEs:

{table('fixed','shared_domain')}
### All fixed-parameter cases (256 queries)

The pooled panel additionally contains 64 moneyness-tail cases (0.5–0.7 or 1.3–1.5, outside the DeepONet moneyness training range) and 64 short-maturity cases (0.01–0.05 year). Region-level metrics are retained separately; this mixture is a designed stress assessment rather than a market-frequency average.

{table('fixed')}
## Cross-contract comparison

The PINN is excluded. There are 128 market states, each queried as a call and a put: 128 interpolation queries, 32 short-maturity queries, and 96 extrapolation queries. Extrapolation moves one of moneyness, volatility, rate, dividend yield, or strike; the maturity category moves both remaining time and contractual maturity together. All quantities use decimal rate/volatility inputs and time in years.

### Interpolation (128 queries)

{table('varying','interpolation')}
### Full cross-contract set (256 queries)

{table('varying')}
## Financial consistency

All models use the same absolute tolerance of 1e-5 for the price bounds, option-specific Delta bounds, and nonnegative Gamma. Counts and worst violation magnitudes are reported separately. These finite sampled checks do not establish global no-arbitrage guarantees. Exact parity is a narrower property than admissible prices or Greeks.

| Panel | Model | Constraint | Violating fraction | Maximum magnitude |\n|---|---|---|---:|---:|\n'''
    for r in shape:
        if r['dimension']=='all' and r['method'] in names:
            report+=f"| {r['panel']} | {names[r['method']]} | {r['constraint']} | {r['fraction']:.2%} | {r['maximum']:.5g} |\n"
    report+='\n## Matched timing workloads\n\n'
    report+='CPU float64, two configured numerical-library threads, one warm-up and five measured repetitions per batch size. Each panel/method uses a fresh subprocess. Loading is excluded; query conversion, computation, and output conversion are included. Four outputs are requested in the fixed panel, six in the varying panel. The fixed CN adapter avoids unused volatility/rate bump solves and is tested against the existing full solver. Native conditional RQMC still calculates its six outputs internally. Timings describe these implementations, not an optimally cached grid or a theoretical lower bound.\n\n'
    if timing:
        report+='| Panel | Method | Batch | Median ms/query | Batch IQR ms | Queries/s | Peak RSS MB | Peak increment MB |\n|---|---|---:|---:|---:|---:|---:|---:|\n'
        for r in timing:
            report+=f"| {r['panel']} | {names[r['method']]} | {r['batch_size']} | {1000*r['per_option_seconds']:.5g} | {1000*(r['q75_seconds']-r['q25_seconds']):.5g} | {r['throughput']:.5g} | {r['peak_rss_mb']:.1f} | {r['peak_increment_mb']:.1f} |\n"
    else:report+='Timing measurements are not yet complete.\n'
    report+='\n## Previously recorded training costs\n\n| Model | Stage | Recorded seconds |\n|---|---|---:|\n'
    for r in cost:
        if r['model'] in ('anchored_pinn',primary):
            report+=f"| {r['model']} | {r['stage']} | {r['seconds']:.2f} |\n"
    report+='\nThe parity table includes both refinement arms because validation selection required both. Historical stage timings are not matched to the current inference hardware/thread protocol. They exclude unrecorded label-generation costs and the wider architecture/weight search. All three seeds\' recorded stages are retained in historical_training_costs.csv; these are not complete end-to-end project costs.\n'
    report+='''
RSS is process-wide and sampled at 1 ms; it includes the common imported runtime and is not isolated model storage or an exact allocator peak. The timing cases are predefined subsets of the shared-domain/interpolation cases. They are independent-query workloads; CN surface reuse and large neural batches beyond 32 are not assessed here.

## Interpretation and limits

- Analytical Black–Scholes supplies the labels, so its reported error is zero by definition. It remains an available direct solution for this problem.
- Conditional RQMC uses analytical continuation inside the estimator. Its strong performance here should not be attributed to an entirely label-free simulation method.
- Compare errors down a quantity's column on the same cases. Different Greeks have different units; their raw RMSE sizes are not interchangeable.
- Faster neural queries alone do not establish a practical advantage: accuracy requirements, financial violations, training, label generation, and architecture search all matter. No accuracy-matched or end-to-end break-even claim is made.
- The three parity seeds share training data and architecture. Their spread is descriptive, not evidence from three independently designed studies; the smallest fresh-test error must not be used to change checkpoint selection.
- RQMC standard errors are retained with each prediction. There is one randomized estimate per case, so this experiment does not measure repeated-trial interval coverage.
- New cases were checked for rounded financial-query overlaps against available DeepONet NPZ splits, including earlier evaluations. Excluding contractual T from this overlap check makes it stricter for the time-homogeneous analytical problem. This audit does not recover missing historical training samples.

## Reproduction

Use the project-local `.venv-benchmark` Python with NumPy 2.2.6. Run `python -m experiments.comparison.shared_evaluation evaluate`, then `python -m experiments.comparison.run_pinned_timing`, then `python -m experiments.comparison.summarize_shared`. The timing wrapper checks the frozen Python, NumPy, SciPy and PyTorch versions in each worker. Existing completed prediction files are reused under the frozen protocol; neither `freeze` nor model training should be repeated for this result. `protocol.json` and its digest pin the cases, checkpoints, settings, and implementation. Predictions, timing repetitions, worst cases, regional errors, and shape metrics remain in this result folder. The original partial timing series was archived after a global NumPy change; final timing measurements were restarted consistently under the restored versions, without changing cases, models, budgets or output workloads.
'''
    (OUT/'report.md').write_text(report,encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    colors={'conditional_rqmc':'#7259BA','crank_nicolson':'#3573C4','anchored_pinn':'#DB6B40',primary:'#04A8A5'}
    fig,axes=plt.subplots(2,3,figsize=(11,6.5),layout='constrained')
    for ax,q in zip(axes.flat,ALL):
        for panel,marker in [('fixed','o'),('varying','s')]:
            if panel=='fixed' and q not in CORE:continue
            for i,(method,color) in enumerate(colors.items()):
                values=[r['rmse'] for r in rows_for(panel) if r['quantity']==q and r['method']==method]
                if values:ax.scatter(i+(-.1 if panel=='fixed' else .1),values[0],color=color,marker=marker,s=48)
        ax.set_yscale('log');ax.set_title(q.capitalize(),loc='left',fontweight='bold')
        ax.set_xticks(range(4),['RQMC','CN','PINN','Parity']);ax.grid(axis='y',alpha=.2)
        ax.set_ylabel('RMSE (quantity-specific units)')
    fig.suptitle('Fresh shared comparisons: circles = fixed calls; squares = varying contracts\nCompare methods within each marker group; panels use different query sets.',fontsize=12)
    fig.savefig(OUT/'rmse_comparison.png',dpi=220);fig.savefig(OUT/'rmse_comparison.pdf');plt.close(fig)
    if timing:
        fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
        for ax,panel in zip(axes,('fixed','varying')):
            for method,name in names.items():
                rs=sorted([r for r in timing if r['panel']==panel and r['method']==method],key=lambda r:r['batch_size'])
                if rs:ax.plot([r['batch_size'] for r in rs],[r['per_option_seconds'] for r in rs],'-o',label=name,color=colors.get(method,'#14243A'))
            ax.set_xscale('log',base=2);ax.set_yscale('log');ax.set_xlabel('Batch size');ax.set_ylabel('Median seconds/query')
            ax.set_title(f"{panel.capitalize()}: {'four' if panel=='fixed' else 'six'} outputs",loc='left');ax.grid(alpha=.2)
        axes[0].legend(fontsize=8)
        fig.savefig(OUT/'batch_timing.png',dpi=220);fig.savefig(OUT/'batch_timing.pdf');plt.close(fig)
    print('Wrote metrics, regional diagnostics, figures, and report to',OUT)


if __name__=='__main__':main()
