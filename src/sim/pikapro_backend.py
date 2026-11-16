"""Native simulation; agents see only their protocol log and own request.

Full native battle state never enters the agent. Decisions for both sides are
collected before either is submitted, preserving simultaneous information.
"""
import json
import logging
from poke_env.battle import DoubleBattle
from poke_env.battle import Pokemon
from src.actions.legal import legal_joint_actions
from src.agents.player import Bot
from src.agents.frozen import FrozenBot

class ClientView:
    def __init__(self, side, tag):
        self.side = side
        self.cursor = 0
        self.battle = DoubleBattle(tag, f'p{side+1}', logging.getLogger(tag), gen=9)
    def update(self, native):
        for line in native.log(f'p{self.side+1}', start=self.cursor):
            message = line.split('|')
            if len(message)>1 and message[1] not in {'win','tie','uhtmlchange','uhtml'}:
                self.battle.parse_message(message)
        self.cursor = native.event_count
        request = json.loads(native.request_json(self.side))
        self.battle.parse_request(request)
        return request

class DecisionView:
    def __init__(self, battle, actions):
        self.battle = battle
        self.tyrtle_legal_actions = actions
    def __getattr__(self, name):
        return getattr(self.battle, name)

def choice_text(order, request, preview=False):
    if preview:
        return order.removeprefix('/')
    parts=[]
    for single in (order.first_order,order.second_order):
        if isinstance(single.order,Pokemon):
            names=[p['ident'].split(': ',1)[1] for p in request['side']['pokemon']]
            parts.append('switch '+str(names.index(single.order.name)+1))
        else:
            parts.append(single.message.removeprefix('/choose '))
    return ', '.join(parts)

def play(seed, team, agents=('cfr','random'), model=None, trace_paths=(None,None), max_decisions=2000, policy=None, episodes=None):
    import pikapro
    native=pikapro.Battle('championsvgc',[team,team],seed=seed)
    players=[]
    for side,mode in enumerate(agents):
        cls=FrozenBot if mode=='frozen' else Bot
        extra={'model_path':model} if mode in {'learned','policy'} and policy is None else {}
        players.append(cls(start_listening=False,seed=seed+side,mode=mode,
                           trace=trace_paths[side],battle_format='gen9championsvgc2026regmc',**extra))
    if policy is not None:
        for player in players:
            if player.mode == 'policy': player.model = policy
    views=[ClientView(i,f'battle-pikapro-{seed}') for i in range(2)]
    choices=[]
    for _ in range(max_decisions):
        if native.over:
            if episodes is not None:
                for side, player in enumerate(players):
                    reward = 0. if native.winner is None else (1. if native.winner == side else -1.)
                    episodes.append((player.training_records, reward))
            return {'battle':f'battle-pikapro-{seed}','won':native.winner==0,'lost':native.winner==1,
                    'finished':True,'turns':native.turn,'seed':seed,
                    'failures':sum(p.failures for p in players),
                    'decisions':sum(p.decisions for p in players),
                    'inference_seconds':sum(p.seconds for p in players),
                    'choices':choices}, [native.log('p1'),native.log('p2')]
        pending=[]
        for side in range(2):
            kind=native.request_kind(side)
            if kind=='wait': continue
            request=views[side].update(native)
            battle=views[side].battle
            if kind != 'team_preview':
                allowed = set(native.legal_actions(side))
                actions = []
                for action in legal_joint_actions(battle):
                    try:
                        index = native.choice_index(side, choice_text(action, request))
                        if index in allowed:
                            actions.append(action)
                    except ValueError:
                        continue
                if not actions:
                    raise ValueError('No native-validated client actions')
                battle = DecisionView(battle, actions)
            order=players[side].teampreview(battle) if kind=='team_preview' else players[side].choose_move(battle)
            text=choice_text(order,request,kind=='team_preview')
            index=native.choice_index(side,text)
            if index not in native.legal_actions(side):
                raise ValueError(f'Illegal native choice: {text}; request={request}; legal={native.legal_choices(side)}')
            pending.append((side,index,text))
        if not pending: raise RuntimeError('Nonterminal battle has no requests')
        for side,index,text in pending:
            choices.append([side,text]); native.submit(side,index)
    raise TimeoutError('Pikapro battle exceeded decision limit')
