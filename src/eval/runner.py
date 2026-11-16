import argparse
import asyncio
from contextlib import nullcontext
import hashlib
import json
import logging
import os
from pathlib import Path
import platform
import resource
import subprocess
import time
from poke_env.ps_client import AccountConfiguration, ServerConfiguration, ShowdownServerConfiguration
from src.agents.player import Bot
from src.agents.frozen import FrozenBot
from src.sim.server import local_server, ROOT

class Errors(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []
    def emit(self, record):
        self.messages.append(record.getMessage())

async def run(config, args, directory):
    server = ShowdownServerConfiguration if args.command == 'ladder' else ServerConfiguration(
        f"ws://127.0.0.1:{config['port']}/showdown/websocket", 'http://127.0.0.1:1/unused')
    team = (ROOT / config['team']).read_text()
    common = dict(battle_format=config['format'], team=team, server_configuration=server,
                  max_concurrent_battles=1, accept_open_team_sheet=False,
                  save_replays=str(directory / 'replays'), log_level=logging.WARNING)
    players = []
    errors = Errors()
    try:
        account = AccountConfiguration(os.environ['PS_USERNAME'], os.environ['PS_PASSWORD']) if args.command == 'ladder' else None
        player_cls = FrozenBot if args.agent == 'frozen' else Bot
        if args.agent == 'ppo':
            from src.agents.ppo_player import PPOBot
            player_cls = PPOBot
        model_options = {'model_path': args.model} if args.agent in {'learned','policy','ppo'} else {}
        player = player_cls(**model_options,seed=config['seed'], mode=args.agent, iterations=config['iterations'],
                     candidate_limit=config['candidate_limit'], trace=directory/'decisions.jsonl',
                     account_configuration=account, **common)
        players.append(player)
        player.logger.addHandler(errors)
        if args.command == 'local':
            opponent_cls = FrozenBot if args.opponent == 'frozen' else Bot
            opponent = opponent_cls(trace=directory/'opponent-decisions.jsonl',seed=config['seed']+1, mode=args.opponent,
                           iterations=config['iterations'], candidate_limit=config['candidate_limit'], **common)
            players.append(opponent)
            opponent.logger.addHandler(errors)
            task = player.battle_against(opponent, n_battles=config['games'])
        else:
            task = player.ladder(config['games'])
        await asyncio.wait_for(task, timeout=config['timeout'])
        results = [{'battle': b.battle_tag, 'won': b.won, 'lost': b.lost,
                    'finished': b.finished, 'turns': b.turn} for b in player.battles.values()]
        report = {'schema': 1, 'games': results, 'wins': player.n_won_battles,
                  'decisions': player.decisions, 'inference_seconds': player.seconds,
                  'warnings_and_errors': errors.messages}
        (directory/'opponent-results.json').write_text(json.dumps([{'battle': b.battle_tag, 'won': b.won, 'lost': b.lost} for b in players[-1].battles.values()]))
        (directory/'results.json').write_text(json.dumps(report, indent=2))
        if len(results) != config['games'] or not all(g['finished'] for g in results) or errors.messages:
            raise RuntimeError('Incomplete games or protocol warnings; inspect results.json')
        return report
    finally:
        for player in players:
            await player.ps_client.stop_listening()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['local', 'ladder'])
    parser.add_argument('--config', default='configs/local.json')
    parser.add_argument('--games', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--agent', choices=['cfr', 'random', 'frozen', 'learned', 'policy', 'ppo'], default='cfr')
    parser.add_argument('--opponent', choices=['cfr', 'random', 'frozen'], default='random')
    parser.add_argument('--backend', choices=['showdown','pikapro'], default='showdown')
    parser.add_argument('--model')
    parser.add_argument('--registered-bot', action='store_true')
    parser.add_argument('--authorize-public-run', action='store_true')
    args = parser.parse_args()
    if args.agent in {'learned','policy','ppo'} and not args.model:
        parser.error('--agent learned/policy requires --model')
    config = json.loads(Path(args.config).read_text())
    for key in ('games', 'seed'):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if config['visibility'] != 'closed' or config['series_length'] != 1:
        parser.error('This prototype supports closed sheets, best-of-one only')
    if not config['format'].startswith('gen9championsvgc') or 'bo3' in config['format']:
        parser.error('This build supports Champions VGC Bo1 only; revalidate other rulesets first')
    if not 1 <= config['games'] <= 200 or config['iterations'] < 1 or config['candidate_limit'] < 1:
        parser.error('Use 1–200 games and positive search budgets')
    if args.backend == 'pikapro' and args.agent == 'ppo':
        parser.error('Use native PPO evaluation in src.training.ppo or Showdown deployment here')
    if args.backend == 'pikapro' and config['format'] != 'gen9championsvgc2026regmc':
        parser.error('Pikapro adapter supports exactly gen9championsvgc2026regmc')
    if args.command == 'ladder' and args.backend != 'showdown':
        parser.error('Public ladder uses Showdown, not the native training backend')
    if args.command == 'ladder':
        if not (args.registered_bot and args.authorize_public_run):
            parser.error('Public play requires --registered-bot and --authorize-public-run after checking current policy')
        if not os.environ.get('PS_USERNAME') or not os.environ.get('PS_PASSWORD'):
            parser.error('Set PS_USERNAME and PS_PASSWORD for your registered bot account')
    subprocess.run(['node', 'scripts/validate-team.cjs', str(Path(args.config).resolve())], cwd=ROOT, check=True)
    directory = ROOT / 'runs' / f'{time.time_ns()}-{args.command}-s{config["seed"]}'
    directory.mkdir(parents=True)
    (directory/'replays').mkdir()
    manifest = {'schema': 1, 'backend': args.backend, 'config': config, 'agent': args.agent, 'opponent': args.opponent,
                'model_sha256': hashlib.sha256(Path(args.model).read_bytes()).hexdigest() if args.model else None,
                'python': platform.python_version(), 'showdown': (ROOT/'configs/showdown-revision.txt').read_text().strip(),
                'team_sha256': hashlib.sha256((ROOT/config['team']).read_bytes()).hexdigest(),
                'simulator_seed': None, 'note': 'Agent RNG seeded; server battle RNG is not controlled.',
                'source_sha256': hashlib.sha256(b''.join(p.read_bytes() for p in sorted((ROOT/'src').rglob('*.py')))).hexdigest()}
    if args.backend == 'pikapro':
        import importlib.metadata
        manifest.update(pikapro_version=importlib.metadata.version('pikapro'), simulator_seed=config['seed'],
                        note='Native simulator seed is base seed plus game index; fast RNG.')
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2))
    cpu_start = time.process_time()
    start = time.monotonic()
    try:
        if args.backend == 'pikapro':
            from src.eval.pikapro_runner import run as native_run
            report = native_run(config,args,directory)
            print(json.dumps({'output': str(directory), 'wins':report['wins'], 'games':config['games'], 'seconds':report['elapsed_seconds']}))
            return
        context = local_server(config['port'], directory/'server.log') if args.command == 'local' else nullcontext()
        with context:
            report = asyncio.run(run(config, args, directory))
        elapsed = time.monotonic()-start
        report.update(elapsed_seconds=elapsed, games_per_hour=3600*config['games']/elapsed,
                      python_cpu_seconds=time.process_time()-cpu_start,
                      python_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if platform.system() == 'Darwin' else 1024),
                      paid_compute_cost=0)
        (directory/'results.json').write_text(json.dumps(report, indent=2))
        print(json.dumps({'output': str(directory), 'wins': report['wins'], 'games': config['games'], 'seconds': elapsed}))
    except BaseException as exc:
        (directory/'failure.json').write_text(json.dumps({'error': repr(exc), 'elapsed': time.monotonic()-start}))
        raise

if __name__ == '__main__':
    main()
