"""PPO deployment through the same Pikapro player-view encoder used in training."""
import json
import time
import torch
from pikapro.rl import ChannelSession
from poke_env.player.battle_order import BattleOrder
from src.agents.player import Bot
from src.training.ppo import load

class TextOrder(BattleOrder):
    def __init__(self,text):self.text=text
    @property
    def message(self):return self.text

class PPOBot(Bot):
    def __init__(self,*,model_path,**kwargs):
        super().__init__(model_path=None,**kwargs)
        torch.set_num_threads(2)
        self.network,self.metadata=load(model_path)
        self.sessions={}
        self.generator=torch.Generator().manual_seed(kwargs.get('seed',0))
    def decision(self,battle):
        start=time.perf_counter()
        if battle.battle_tag not in self.sessions:
            self.sessions[battle.battle_tag]=(ChannelSession('championsvgc',masks='joint'),0)
        session,cursor=self.sessions[battle.battle_tag]
        lines=battle._replay_data
        session.feed_many('|'.join(row) for row in lines[cursor:] if len(row)>1 and row[1]!='request')
        session.feed('|request|'+json.dumps(battle._last_request))
        self.sessions[battle.battle_tag]=(session,len(lines))
        if session.error:raise RuntimeError(session.error)
        with torch.no_grad():
            dist,_=self.network(torch.tensor(session.obs.copy()).reshape(1,-1),torch.tensor(session.mask.copy()).reshape(1,-1))
            action=int(torch.multinomial(dist.probs[0],1,generator=self.generator))
        legal_count=int(session.mask.sum())
        choice=session.choice([action])
        if not choice.startswith('/'):
            choice=('/' if choice.startswith('team ') else '/choose ')+choice
        self.decisions+=1;self.seconds+=time.perf_counter()-start
        if self.trace:
            with self.trace.open('a') as f:f.write(json.dumps({'battle':battle.battle_tag,'action':choice,'encoder':'pikapro-basic','legal_count':legal_count})+'\n')
        return choice
    def teampreview(self,battle):return self.decision(battle)
    def choose_move(self,battle):return TextOrder(self.decision(battle))
