"""Refit the recorded endpoints with the original unweighted least-squares code."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from analyze_tiny_scaling import fit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metric', choices=('move', 'expert2400', 'expert2600'), default='move')
    args = parser.parse_args()
    data = json.loads((ROOT / 'analysis/scaling_fit.json').read_text())
    result = {recipe: fit([r for r in data['rows'] if r['recipe'] == recipe], args.metric)
              for recipe in ('ours', 'qwen_recipe')}
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
