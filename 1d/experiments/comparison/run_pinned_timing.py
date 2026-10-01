"""Run the frozen timing workload with an explicit software-version check."""

import argparse
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
import torch

from experiments.comparison.shared_evaluation import (
    ROOT, OUT, digest, get_protocol, methods, save, time_worker, verify_pins,
)


def assert_versions(expected, actual):
    mismatches={k:(expected[k],actual[k]) for k in expected if expected[k]!=actual[k]}
    if mismatches:
        raise RuntimeError(f'Frozen runtime mismatch: {mismatches}')


def environment(manifest):
    expected={k:manifest['environment'][k] for k in ('python','numpy','scipy','torch')}
    actual=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__)
    assert_versions(expected,actual)
    return {**actual,'executable':sys.executable,'numpy_file':np.__file__,
            'scipy_file':scipy.__file__,'torch_file':torch.__file__,
            'recorded_at':datetime.now(timezone.utc).isoformat()}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--panel');parser.add_argument('--method')
    args=parser.parse_args()
    manifest,_=get_protocol()
    current=environment(manifest)
    if args.worker:
        time_worker(args.panel,args.method)
        save(OUT/'timing'/f'{args.panel}_{args.method}.environment.json',current)
        return
    for panel in ('fixed','varying'):
        for method in methods(manifest,panel):
            target=OUT/'timing'/f'{panel}_{method}.json'
            env=target.with_suffix('.environment.json')
            if target.exists() and env.exists():continue
            subprocess.run([sys.executable,'-m','experiments.comparison.run_pinned_timing',
                            '--worker','--panel',panel,'--method',method],cwd=ROOT,check=True)
    verify_pins(manifest)
    save(OUT/'timing_complete.json',dict(completed_at=datetime.now(timezone.utc).isoformat(),
                                       environment=current,wrapper_sha256=digest(__file__),
                                       files_sha256={p.name:digest(p) for p in (OUT/'timing').glob('*.json')}))


if __name__=='__main__':main()
