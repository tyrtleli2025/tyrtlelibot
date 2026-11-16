"""Bounded on-policy Monte Carlo actor/critic self-play; no rollout search."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.training.policy import PolicyValue
from src.sim.pikapro_backend import play

def main():
    p=argparse.ArgumentParser();p.add_argument('--games',type=int,default=400)
    p.add_argument('--seed',type=int,default=51001);p.add_argument('--resume')
    a=p.parse_args()
    if not 20<=a.games<=2000:p.error('Use 20–2000 games for a bounded local run')
    out=ROOT/'runs'/f'policy-{time.time_ns()}';out.mkdir(parents=True)
    model=PolicyValue.load(a.resume) if a.resume else PolicyValue()
    teams=[(ROOT/x).read_text() for x in ('teams/balance.txt','teams/native-sample.txt')]
    plan={'schema':1,'algorithm':'terminal Monte Carlo policy gradient with learned baseline',
          'games':a.games,'seed':a.seed,'batch_games':20,'learning_rate':.03,
          'team_hashes':[hashlib.sha256(t.encode()).hexdigest() for t in teams],
          'source_sha256':hashlib.sha256(b''.join(x.read_bytes() for x in sorted((ROOT/'src').rglob('*.py')))).hexdigest(),
          'pikapro_version':'0.1.0','resume':a.resume,
          'evaluation':'fresh seeds 71001/72001; Showdown 50 games per fixture vs frozen; no test tuning'}
    (out/'plan.json').write_text(json.dumps(plan,indent=2))
    model.save(out/'initial.npz');start=time.perf_counter()
    for first in range(0,a.games,20):
        episodes=[];wins=0
        for game in range(first,min(first+20,a.games)):
            result,_=play(a.seed+game,teams[game%len(teams)],('policy','policy'),policy=model,episodes=episodes)
            if result['failures']:raise RuntimeError('Policy fallback in collection')
            wins+=result['won']
        metrics=model.update(episodes)
        metrics.update(games=min(first+20,a.games),side0_wins=wins,elapsed_seconds=time.perf_counter()-start)
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
        model.save(out/'policy.npz')
        print(json.dumps(metrics),flush=True)
    (out/'report.json').write_text(json.dumps({'output':str(out),'games':a.games,'seconds':time.perf_counter()-start},indent=2))
    print(str(out/'policy.npz'),flush=True)
if __name__=='__main__':main()
