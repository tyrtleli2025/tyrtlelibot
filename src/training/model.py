"""Small NumPy logistic outcome model conditioned on visible state and action.

Predicts win probability under the collection policies, not causal action value.
Feature vocabulary is fixed, explicitly versioned, and has no species identifiers.
"""
import numpy as np
from src.search.tactical import own_intents

FEATURE_VERSION = 1
DIM = 22

def features(obs, action):
    intents = own_intents(action, obs)
    sides = (obs.own, obs.opponent)
    x = [1., min(obs.turn, 50)/50]
    for active, team in zip(sides, (obs.own_team, obs.opponent_team)):
        x += [sum(p.hp for p in active if p)/2, sum(p is not None for p in active)/2,
              sum(p.hp for p in team if p)/4, sum(p is not None and p.hp > 0 for p in team)/4,
              sum(p.status is not None for p in active if p)/2]
    x += list(map(float, obs.tailwind))
    ds = [sum(a.damage[t] for a in intents) for t in range(2)]
    x += [sum(ds), max(ds), sum(a.protect for a in intents)/2,
          sum(a.replacement is not None for a in intents)/2,
          sum(a.bonus for a in intents), sum(a.priority for a in intents)/12,
          sum(min(ds[i], p.hp) for i,p in enumerate(obs.opponent) if p),
          sum(ds[i] >= p.hp for i,p in enumerate(obs.opponent) if p)/2]
    assert len(x) == DIM
    return np.clip(np.asarray(x, dtype=float), -4, 4)

class OutcomeModel:
    def __init__(self, weights=None):
        self.weights = np.zeros(DIM) if weights is None else weights
    def predict(self, x):
        return 1 / (1 + np.exp(-np.clip(np.asarray(x) @ self.weights, -30, 30)))
    def save(self, path):
        np.savez(path, weights=self.weights, feature_version=FEATURE_VERSION)
    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data['feature_version']) != FEATURE_VERSION or data['weights'].shape != (DIM,):
                raise ValueError('Incompatible outcome model')
            return cls(data['weights'].copy())
    def fit(self, x, y, weights, epochs=600):
        for _ in range(epochs):
            error = (self.predict(x)-y)*weights
            self.weights -= .15 * (x.T @ error / weights.sum() + .002*self.weights)
