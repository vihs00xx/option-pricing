import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / 'experiments/config/neural/deeponet_model_registry.json'
OUTPUT = ROOT / 'results/deeponet/primary/final_test_metrics.json'


def main():
    registry = json.loads(REGISTRY.read_text())
    result = {'protocol_id': registry['protocol_id'], 'registry_sha256': hashlib.sha256(REGISTRY.read_bytes()).hexdigest(),
              'model_selection_preceded_final_test': True, 'models': {}}
    for role in ('primary', 'greek_focused_reference', 'price_only_reference'):
        item = registry[role]
        checkpoint = ROOT / item['checkpoint']
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != item['checkpoint_sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {role}')
        completed = subprocess.run([sys.executable, '-m', 'experiments.deeponet.evaluate',
                                    str(checkpoint), '--final-test'], cwd=ROOT,
                                   text=True, capture_output=True, check=True)
        evaluation = json.loads(completed.stdout)
        result['models'][role] = dict(checkpoint=item['checkpoint'],
                                      checkpoint_sha256=item['checkpoint_sha256'],
                                      metrics=evaluation['metrics'])
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(OUTPUT), 'models': result['models']}, indent=2))


if __name__ == '__main__':
    main()
