"""Train with battle-grouped split and equal per-battle sample weighting."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from src.training.model import OutcomeModel

def train(directories, output, seed=123):
    groups = []
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory/'manifest.json').read_text())
        if manifest.get('split') == 'test':
            raise ValueError('Refusing benchmark test data')
        result = json.loads((directory/'results.json').read_text())
        if result['warnings_and_errors']:
            raise ValueError('Collection has protocol errors')
        for filename, results in [('decisions.jsonl', result['games']),
                                  ('opponent-decisions.jsonl', json.loads((directory/'opponent-results.json').read_text()))]:
            labels = {g['battle']: float(g['won']) if g['won'] or g['lost'] else .5 for g in results}
            for line in (directory/filename).read_text().splitlines():
                row = json.loads(line)
                if 'features' in row:
                    groups.append((str(directory.resolve())+'/'+row['battle'], row['features'], labels[row['battle']]))
    keys = sorted({g[0] for g in groups})
    if len(keys) < 10:
        raise ValueError('Need at least ten complete games')
    rng = np.random.default_rng(seed)
    rng.shuffle(keys)
    validation = set(keys[:max(1,len(keys)//5)])
    model = OutcomeModel()
    metrics = {}
    for split in ('train', 'validation'):
        rows = [g for g in groups if (g[0] in validation) == (split == 'validation')]
        x = np.array([g[1] for g in rows]); y = np.array([g[2] for g in rows])
        counts = {k: sum(g[0] == k for g in rows) for k in {g[0] for g in rows}}
        w = np.array([1/counts[g[0]] for g in rows])
        if split == 'train':
            model.fit(x, y, w)
        pred = np.clip(model.predict(x), 1e-8, 1-1e-8)
        metrics[split] = {'games': len(counts), 'positions': len(rows),
            'log_loss': float(np.average(-y*np.log(pred)-(1-y)*np.log(1-pred), weights=w)),
            'constant_half_log_loss': float(np.log(2))}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    model.save(output)
    report = {'schema': 1, 'seed': seed, 'metrics': metrics, 'sources': list(map(str,directories)),
              'validation_game_keys': sorted(validation),
              'sha256': hashlib.sha256(output.read_bytes()).hexdigest()}
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['metrics'], indent=2))
    return report

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('directories', nargs='+'); p.add_argument('--output', required=True)
    a=p.parse_args(); train(a.directories,a.output)
