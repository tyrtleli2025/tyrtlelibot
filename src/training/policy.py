"""Small nonlinear actor/critic using NumPy and terminal Monte Carlo returns.

A fixed tanh feature layer supplies state/action interactions. Actor and value
heads learn by policy gradient and regression. Complete legal joint actions form
the categorical distribution, so sampling and log probabilities agree exactly.
"""
import numpy as np
from src.training.model import DIM, features

class PolicyValue:
    def __init__(self, seed=7):
        rng=np.random.default_rng(seed)
        self.projection=rng.normal(0,.5,(DIM,32))
        self.actor=np.zeros(33); self.critic=np.zeros(33)
    def encode(self,x):
        x=np.asarray(x)
        return np.concatenate([np.ones((*x.shape[:-1],1)),np.tanh(x@self.projection)],axis=-1)
    def distribution(self,x):
        h=self.encode(x); logits=h@self.actor
        p=np.exp(logits-logits.max());p/=p.sum()
        return p,h
    def value_features(self,x):
        state=np.array(x,copy=True);state[14:]=0
        return self.encode(state)
    def update(self,episodes,rate=.03):
        actor_grad=np.zeros(33);critic_grad=np.zeros(33);loss=[];entropy=[]
        count=0
        for records,reward in episodes:
            if not records: continue
            # Each game-side has equal weight; both sides use the frozen batch policy.
            for x,index in records:
                p,h=self.distribution(x);v=self.value_features(x[index]);value=v@self.critic
                advantage=reward-value
                actor_grad += advantage*(h[index]-p@h)/len(records)
                critic_grad += (reward-value)*v/len(records)
                loss.append(float((reward-value)**2));entropy.append(float(-p@np.log(p+1e-12)))
            count+=1
        if not count: raise ValueError('No complete trajectories')
        self.actor+=rate*np.clip(actor_grad/count,-1,1)
        self.critic+=rate*np.clip(critic_grad/count,-1,1)
        return {'value_mse':float(np.mean(loss)),'entropy':float(np.mean(entropy)),
                'actor_norm':float(np.linalg.norm(self.actor))}
    def save(self,path):
        np.savez(path,version=1,projection=self.projection,actor=self.actor,critic=self.critic)
    @classmethod
    def load(cls,path):
        m=cls()
        with np.load(path,allow_pickle=False) as d:
            if int(d['version'])!=1: raise ValueError('Unknown policy schema')
            for key,shape in [('projection',(DIM,32)),('actor',(33,)),('critic',(33,))]:
                value=d[key]
                if value.shape!=shape or not np.isfinite(value).all(): raise ValueError('Invalid policy checkpoint')
                setattr(m,key,value.copy())
        return m
