import argparse
import csv
import gc
import json
import threading
from pathlib import Path
from time import sleep

import psutil

from src.classical_benchmark import METHODS, evaluate_batch


ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=METHODS)
    args = parser.parse_args()
    settings = json.loads((ROOT / "experiments" / "config" / "common_benchmark_settings.json").read_text())
    with (ROOT / "results" / "common_benchmark" / "tables" / "benchmark_cases.csv").open(newline="") as handle:
        cases = list(csv.DictReader(handle))[:settings["timing"]["case_count"]]
    process = psutil.Process()
    gc.collect()
    baseline = process.memory_info().rss
    peak = [baseline]
    stop = threading.Event()

    def sample():
        interval = settings["timing"]["memory_sample_interval_seconds"]
        while not stop.is_set():
            peak[0] = max(peak[0], process.memory_info().rss)
            sleep(interval)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        evaluate_batch(args.method, cases, settings)
    finally:
        stop.set()
        thread.join()
        peak[0] = max(peak[0], process.memory_info().rss)
    print(json.dumps({
        "method": args.method,
        "baseline_rss_mb": baseline / 1024**2,
        "peak_rss_mb": peak[0] / 1024**2,
        "incremental_peak_mb": (peak[0] - baseline) / 1024**2,
        "batch_size": len(cases),
    }))


if __name__ == "__main__":
    main()
