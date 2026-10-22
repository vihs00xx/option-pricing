"""Run this folder's whole study on one machine, unattended: training and the references side by side.

Usage (from this folder): python -m experiments.unattended_run --levels D0_vanilla,D1_reduced,... --stop-at 2026-10-19T10:00
    [--threads 6] [--workers 3] [--phase0-limit 10]

Two chains run at once and log to artifacts/unattended_run.log:

1. the training ladders (run_ladders with --only in the given priority order, one level at a time,
   no new level after --stop-at);
2. the Phase 0 reference check on the first --phase0-limit development contracts and its
   report, then the test-set build and its verification.

While it runs, the process asks Windows not to idle-sleep (SetThreadExecutionState; the
request ends with the process and changes no setting).
"""

import argparse
import subprocess
import sys
import threading
from datetime import datetime

from experiments.records import ROOT

LOG = ROOT/"artifacts"/"unattended_run.log"
_lock = threading.Lock()


def log(message):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    with _lock:
        print(line, flush=True)
        with LOG.open("a", encoding="utf-8") as handle:
            handle.write(line+"\n")


def stay_awake():
    if sys.platform == "win32":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED: no idle sleep while this process lives.
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def chain(name, commands):
    for args in commands:
        log(f"{name}: start {' '.join(args)}")
        out = (ROOT/"artifacts"/f"unattended_{name}.out").open("a", encoding="utf-8")
        code = subprocess.call([sys.executable, *args], cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        log(f"{name}: exit {code} {' '.join(args)}")
        if code != 0:
            log(f"{name}: stopped")
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", required=True)
    parser.add_argument("--stop-at", required=True)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--phase0-limit", type=int, default=10)
    args = parser.parse_args()
    stay_awake()
    from experiments.neural.common import keep_full_speed
    keep_full_speed()
    training = [["-m", "experiments.neural.run_ladders", "--threads", str(args.threads), "--only", args.levels,
                 "--stop-at", args.stop_at]]
    references = [["-m", "experiments.reference.reference_check", "--workers", str(args.workers),
                   "--limit", str(args.phase0_limit)],
                  ["-m", "experiments.reference.reference_check", "--report"],
                  ["-m", "experiments.test_set.freeze_test_set", "--workers", str(args.workers)],
                  ["-m", "experiments.verify", "results/test_set_v1"]]
    threads = [threading.Thread(target=chain, args=("training", training)),
               threading.Thread(target=chain, args=("references", references))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    log("all chains finished")


if __name__ == "__main__":
    main()
