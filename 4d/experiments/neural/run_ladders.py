"""Run both ladders with at most --parallel levels at a time (each --threads threads).

Usage (from 4d/): python -m experiments.neural.run_ladders [--parallel 1] [--threads 8]

The default runs one level at a time, the setting under which training times compare
(see train.py); with --parallel above 1 each summary's machine_load records the overlap.

A level starts when its warm start (if any) is finished; D9_ensemble starts when its four
D8 members are finished. Finished levels (checkpoint present) are skipped, so re-running
after an interruption resumes: unfinished levels continue from their resume files.
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from experiments.neural.common import LADDER, checkpoint_path, level_settings, load_settings
from experiments.records import ROOT

LOG = ROOT / "artifacts" / "neural" / "run_ladders.log"


def jobs(settings):
    """(name, command args, prerequisites) in priority order."""
    out = []
    for level in settings["levels"]:
        s = level_settings(settings, level)
        if "members_of" in s:
            continue
        need = [(s["warm_start"], 0)] if s.get("warm_start") else []
        out.append(((level, 0), ["-m", "experiments.neural.train", level], need))
    for level in settings["levels"]:
        s = level_settings(settings, level)
        if "members_of" in s:
            members = [(s["members_of"], seed) for seed in s["seeds"]]
            for seed in s["seeds"]:
                if seed != 0:
                    out.append(((s["members_of"], seed), ["-m", "experiments.neural.train", s["members_of"],
                                                         "--seed", str(seed)], []))
            out.append(((level, 0), ["-m", "experiments.neural.ensemble", level], members))
    return out


def log(msg):
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--only", default=None,
                    help="comma-separated level names to run, in this order of priority (others skipped)")
    ap.add_argument("--stop-at", default=None,
                    help="local time (YYYY-MM-DDTHH:MM) after which no new level starts; running ones finish")
    args = ap.parse_args()
    settings = load_settings()
    stop = datetime.fromisoformat(args.stop_at) if args.stop_at else None
    pending = jobs(settings)
    if args.only:
        order = args.only.split(",")
        pending = sorted((j for j in pending if j[0][0] in order), key=lambda j: order.index(j[0][0]))
    done = lambda key: checkpoint_path(*key).exists()
    pending = [j for j in pending if not done(j[0])]
    running = {}
    failed = set()
    LADDER.mkdir(parents=True, exist_ok=True)
    while pending or running:
        for key, proc in list(running.items()):
            if proc.poll() is not None:
                del running[key]
                if proc.returncode == 0 and done(key):
                    log(f"finished {key}")
                else:
                    failed.add(key)
                    log(f"FAILED {key} (exit {proc.returncode})")
        if stop is not None and datetime.now() >= stop and pending:
            log(f"stop time reached; not starting {[j[0] for j in pending]}")
            pending = []
        for job in list(pending):
            if len(running) >= args.parallel:
                break
            key, cmd, need = job
            if any(n in failed for n in need):
                pending.remove(job)
                failed.add(key)
                log(f"skipped {key}: prerequisite failed")
                continue
            if all(done(n) for n in need):
                pending.remove(job)
                out = (LADDER / f"{key[0]}_seed{key[1]}.stdout").open("a", encoding="utf-8")
                full = [sys.executable, *cmd, "--threads", str(args.threads)]
                running[key] = subprocess.Popen(full, cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
                log(f"started {key}: {' '.join(cmd)}")
        time.sleep(20)
    log(f"all done; failed: {sorted(failed) if failed else 'none'}")


if __name__ == "__main__":
    main()
