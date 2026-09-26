"""Verify the saved comparison without importing numerical libraries or running models."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/shared_comparison/v1'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(path, expected):
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f'Integrity mismatch: {path.relative_to(ROOT)}')


def main():
    verify(OUT/'protocol.json', (OUT/'protocol.sha256').read_text().strip())
    protocol = read(OUT/'protocol.json')
    verify(OUT/'cases.json', protocol['cases_sha256'])
    for name, expected in protocol['source_sha256'].items():
        verify(ROOT/Path(name.replace('\\', '/')), expected)
    for model in protocol['models'].values():
        verify(ROOT/Path(model['checkpoint'].replace('\\', '/')), model['sha256'])
    predictions = read(OUT/'accuracy_complete.json')['predictions_sha256']
    for name, expected in predictions.items():
        verify(OUT/'predictions'/name, expected)
    timings = read(OUT/'timing_complete.json')['files_sha256']
    for name, expected in timings.items():
        verify(OUT/'timing'/name, expected)
    seal = ROOT/'results/milestone_1d/artifact_hashes.json'
    if seal.exists():
        for name, expected in read(seal).items():
            verify(ROOT/name, expected)
    print(f'Verified protocol, cases, {len(protocol["source_sha256"])} source files, '
          f'{len(protocol["models"])} checkpoints, {len(predictions)} predictions and '
          f'{len(timings)} timing records. No models were evaluated.')


if __name__ == '__main__':
    main()
