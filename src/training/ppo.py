"""Batched current-policy self-play PPO. No teacher or search in collection."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

class Network(nn.Module):
    def __init__(self, obs_dim, actions):
        super().__init__()
        self.dimensions=(obs_dim,actions)
        self.body=nn.Sequential(nn.Linear(obs_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
        self.actor=nn.Linear(128,actions);self.value=nn.Linear(128,1)
    def forward(self,obs,mask):
        h=self.body(obs)
        # Waiting players receive a dummy distribution, excluded from actor loss.
        safe=mask.clone();safe[~safe.any(-1),0]=True
        return Categorical(logits=self.actor(h).masked_fill(~safe,-1e9)),self.value(h).squeeze(-1)

def advantages(rewards,values,next_values,terminated,done,gamma=1.,lam=.95):
    result=torch.zeros_like(rewards);carry=torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        delta=rewards[t]+gamma*next_values[t]*(~terminated[t])-values[t]
        carry=delta+gamma*lam*(~done[t])*carry
        result[t]=carry
    return result

def save(path,net,optimizer,step,config):
    torch.save({'version':1,'dimensions':net.dimensions,'network':net.state_dict(),
        'optimizer':optimizer.state_dict(),'step':step,'config':config,
        'torch_rng':torch.get_rng_state(),'numpy_rng':np.random.get_state()[1].tolist()},path)

def load(path):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if d['version']!=1:raise ValueError('Checkpoint version')
    m=Network(*d['dimensions']);m.load_state_dict(d['network']);return m,d

def evaluate(net,opponent,seed,games=100):
    import pikapro.rl as rl
    env=rl.VecEnv('championsvgc',n=16,threads=2,seed=seed,opponent=opponent,
                  masks='joint',autoreset='same_step',reward='terminal',max_turns=100,ots=False)
    s=env.reset(seed=seed);wins=losses=draws=truncations=0
    try:
        while wins+losses+draws<games:
            with torch.no_grad():
                dist,_=net(torch.tensor(s.obs[:,0].copy()),torch.tensor(s.masks[:,0].copy()))
                action=dist.sample().numpy()
            s=env.step(action)
            if (s.status<0).any():raise RuntimeError('Rejected evaluation action')
            for i in np.flatnonzero(s.terminated|s.truncated):
                if wins+losses+draws>=games:break
                truncations+=int(s.truncated[i]);w=int(s.winner[i])
                wins+=w==0;losses+=w==1;draws+=w not in (0,1)
        return dict(wins=wins,losses=losses,draws=draws,truncations=truncations)
    finally:env.close()

def main():
    import pikapro.rl as rl
    p=argparse.ArgumentParser();p.add_argument('--updates',type=int,default=40)
    p.add_argument('--envs',type=int,default=16);p.add_argument('--seed',type=int,default=81001)
    p.add_argument('--resume');a=p.parse_args()
    if not 1<=a.updates<=200 or not 2<=a.envs<=64:p.error('Bounded run: updates 1–200, envs 2–64')
    torch.set_num_threads(2);torch.manual_seed(a.seed);np.random.seed(a.seed);random.seed(a.seed)
    out=Path('runs')/f'ppo-{time.time_ns()}';out.mkdir(parents=True)
    env=rl.VecEnv('championsvgc',n=a.envs,threads=2,seed=a.seed,opponent='external',
                  masks='joint',autoreset='same_step',reward='terminal',max_turns=100,ots=False,teams='sets')
    state=env.reset(seed=a.seed)
    net=Network(state.obs.shape[-1],state.masks.shape[-1]);optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
    offset=0
    if a.resume:
        net,d=load(a.resume);optimizer=torch.optim.Adam(net.parameters(),lr=3e-4)
        optimizer.load_state_dict(d['optimizer']);offset=d['step']
    config=vars(a)|{'rollout':32,'epochs':4,'gamma':1.,'lambda':.95,'clip':.2,
        'entropy':.01,'reward':'terminal','opponents':'current policy both sides',
        'pikapro':'0.1.0','torch':str(torch.__version__),'format':'championsvgc','ots':False,
        'resume_semantics':'weights/optimizer resumed; fresh seeded environment, no midgame restore',
        'evaluation_seeds':[91011,91012],'team_source':'Pikapro sets generator',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'manifest.json').write_text(json.dumps(config,indent=2));save(out/'initial.pt',net,optimizer,offset,config)
    initial={name:evaluate(net,name,seed) for name,seed in [('random',91011),('simple_heuristics',91012)]}
    start=time.perf_counter();completed=truncated=0
    try:
        for update in range(a.updates):
            rollout=[]
            for _ in range(32):
                obs=torch.tensor(state.obs.copy()).reshape(-1,net.dimensions[0])
                mask=torch.tensor(state.masks.copy()).reshape(-1,net.dimensions[1])
                active=torch.tensor(state.needs_action.copy()).flatten()
                with torch.no_grad():dist,value=net(obs,mask);action=dist.sample();logp=dist.log_prob(action)
                state=env.step(action.numpy().reshape(a.envs,2))
                if (state.status<0).any():raise RuntimeError('Rejected training action')
                term=torch.tensor(np.repeat(state.terminated.copy(),2));done=torch.tensor(np.repeat(state.done.copy(),2))
                with torch.no_grad():
                    next_obs=state.obs.copy()
                    if state.truncated.any():next_obs[state.truncated]=state.final_obs[state.truncated]
                    _,nv=net(torch.tensor(next_obs).reshape(-1,net.dimensions[0]),torch.ones_like(mask))
                rollout.append((obs,mask,active,action,logp,value,torch.tensor(state.reward.copy()).flatten(),nv,term,done))
                completed+=int(state.terminated.sum());truncated+=int(state.truncated.sum())
            fields=list(zip(*rollout));reward,oldvalue,nv,term,done=[torch.stack(fields[i]) for i in (6,5,7,8,9)]
            adv=advantages(reward,oldvalue,nv,term,done);returns=(adv+oldvalue).flatten()
            obs,mask,active,actions,oldlog=[torch.cat(fields[i]) for i in range(5)]
            adv=adv.flatten();adv=(adv-adv[active].mean())/(adv[active].std()+1e-8)
            kls=[]
            for epoch in range(4):
                for idx in torch.randperm(len(obs)).split(256):
                    dist,v=net(obs[idx],mask[idx]);lp=dist.log_prob(actions[idx]);ratio=(lp-oldlog[idx]).exp()
                    valid=active[idx]
                    if not valid.any():continue
                    policy=-torch.minimum(ratio*adv[idx],ratio.clamp(.8,1.2)*adv[idx])[valid].mean()
                    loss=policy+.5*(v-returns[idx]).square().mean()-.01*dist.entropy()[valid].mean()
                    optimizer.zero_grad();loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),.5);optimizer.step()
                    kls.append(float(((ratio-1)-(lp-oldlog[idx]))[valid].mean().detach()))
                if np.mean(kls)>.03:break
            metrics={'update':offset+update+1,'loss':float(loss.detach()),'approx_kl':float(np.mean(kls)),
                     'gradient_norm':float(norm),'completed_games':completed,'truncated':truncated,
                     'seconds':time.perf_counter()-start}
            if not np.isfinite(list(metrics.values())).all():raise RuntimeError('Nonfinite training metrics')
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(metrics)+'\n')
            save(out/'checkpoint.pt',net,optimizer,offset+update+1,config)
            print(json.dumps(metrics),flush=True)
    finally:env.close()
    final={name:evaluate(net,name,seed) for name,seed in [('random',91011),('simple_heuristics',91012)]}
    (out/'evaluation.json').write_text(json.dumps({'initial':initial,'final':final},indent=2))
    print(out,initial,final,flush=True)
if __name__=='__main__':main()
