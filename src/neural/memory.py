"""Peak resident-set-size memory tracking for training/refinement runs.

A lightweight background-thread sampler records the highest RSS reached
between ``start``/``stop`` (or as a context manager), so runs can be
compared by peak as well as incremental memory without instrumenting the
optimisation loop itself.
"""

import threading
from time import sleep

import psutil


class MemoryTracker:
    """Sample peak RSS of this process from a background thread."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.process = psutil.Process()
        self._peak = None
        self._stop = None
        self._thread = None
        self.baseline_mb = None
        self.peak_mb = None

    def _sample(self):
        current = self.process.memory_info().rss
        if self._peak is None or current > self._peak:
            self._peak = current

    def _run(self):
        while not self._stop.is_set():
            self._sample()
            sleep(self.interval)

    def start(self):
        self.baseline_mb = self.process.memory_info().rss / 1024**2
        self._peak = self.process.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        if self._stop is not None:
            self._stop.set()
        if self._thread is not None:
            self._thread.join()
        self._sample()
        self.peak_mb = self._peak / 1024**2
        return self

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    @property
    def incremental_peak_mb(self):
        if self.peak_mb is None or self.baseline_mb is None:
            return None
        return self.peak_mb - self.baseline_mb