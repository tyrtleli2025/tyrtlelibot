"""Seeded native backend producing the same training artifact schema as Showdown."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from src.sim.pikapro_backend import play

def run(config,args,directory):
    team=Path(config['team']).read_text()
    start=time.perf_counter();results=[]
    for game in range(config['games']):
        result,logs=play(config['seed']+game,team,(args.agent,args.opponent),args.model,
                         (directory/'decisions.jsonl',directory/'opponent-decisions.jsonl'))
        for side,log in enumerate(logs):
            (directory/'replays'/f'{game}-p{side+1}.log').write_text('\n'.join(log)+'\n')
        results.append(result)
    elapsed=time.perf_counter()-start
    report={'schema':1,'backend':'pikapro','games':results,'wins':sum(g['won'] for g in results),
            'decisions':sum(g['decisions'] for g in results),
            'inference_seconds':sum(g['inference_seconds'] for g in results),
            'warnings_and_errors':[], 'elapsed_seconds':elapsed,
            'games_per_hour':3600*len(results)/elapsed}
    (directory/'results.json').write_text(json.dumps(report,indent=2))
    (directory/'opponent-results.json').write_text(json.dumps([
        {'battle':g['battle'],'won':g['lost'],'lost':g['won']} for g in results]))
    if any(g['failures'] for g in results):
        raise RuntimeError('Agent fallback occurred; native run rejected')
    return report
