"""Bounded, one-command local collection, training and untouched evaluation.

Seeds and sample sizes are fixed before evaluation. No checkpoint selection on test.
WebSocket simulator RNG is not seeded; repeatability is statistical, not bitwise.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.training.train import train

def batch(agent, opponent, games, seed, split, model=None, backend='showdown'):
    cmd=[sys.executable,'-m','src.eval.runner','local','--agent',agent,'--opponent',opponent,
         '--games',str(games),'--seed',str(seed),'--backend',backend]
    if model: cmd += ['--model',str(model)]
    completed=subprocess.run(cmd,cwd=ROOT,check=True,text=True,capture_output=True)
    print(completed.stdout, end='')
    directory=Path(json.loads(completed.stdout.strip().splitlines()[-1])['output'])
    path=directory/'manifest.json'; manifest=json.loads(path.read_text());manifest['split']=split
    path.write_text(json.dumps(manifest,indent=2))
    return directory

def interval(w,n):
    z=1.96; p=w/n; d=1+z*z/n
    center=(p+z*z/(2*n))/d; half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [center-half,center+half]

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['showdown','pikapro'],default='showdown');p.add_argument('--collection-games',type=int,default=200)
    p.add_argument('--test-games',type=int,default=100);a=p.parse_args()
    out=ROOT/'runs'/f'experiment-{time.time_ns()}';out.mkdir(parents=True)
    plan={'backend':a.backend,'collection_games':a.collection_games,'test_games':a.test_games,
          'collection_seed':11001,'evaluation_seed':91001,'model_weight':.25,
          'criterion':'candidate Wilson lower bound > 0.5 against frozen current bot',
          'limitations':('one fixture team; seeded native RNG' if a.backend == 'pikapro' else 'one fixture team; unseeded server RNG') + '; no team generalization claim'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    collection=batch('cfr','cfr',a.collection_games,11001,'train',backend=a.backend)
    model=out/'outcome.npz'; train([collection],model)
    evaluation=batch('learned','frozen',a.test_games,91001,'test',model,backend=a.backend)
    r=json.loads((evaluation/'results.json').read_text())
    ci=interval(r['wins'],len(r['games']))
    report={'plan':plan,'collection':str(collection),'evaluation':str(evaluation),
            'model':str(model),'wins':r['wins'],'games':len(r['games']),
            'wilson_95':ci,'improvement_demonstrated':ci[0]>.5}
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
