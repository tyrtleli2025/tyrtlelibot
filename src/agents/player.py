import json
import time
from pathlib import Path
import numpy as np
from poke_env.player import Player
from src.actions.legal import legal_joint_actions
from src.observations.view import observe
from src.search.mccfr import solve
from src.training.policy import PolicyValue
from src.training.model import OutcomeModel, features
from src.search.tactical import own_intents, opponent_candidates, matrix

class Bot(Player):
    def __init__(self, *, seed=0, mode='cfr', iterations=384, candidate_limit=32, trace=None, model_path=None, model_weight=.25, **kwargs):
        super().__init__(**kwargs)
        self.rng = np.random.default_rng(seed)
        self.mode, self.iterations, self.candidate_limit = mode, iterations, candidate_limit
        self.model = (PolicyValue.load(model_path) if mode == 'policy' else OutcomeModel.load(model_path)) if model_path else None
        self.training_records = []
        self.model_weight = model_weight
        self.trace = Path(trace) if trace else None
        self.decisions = 0
        self.seconds = 0.
        self.failures = 0

    def teampreview(self, battle):
        # Seeded bring-four/lead-order baseline. Preview learning is future work.
        members = self.rng.permutation(len(battle.team))[:4]
        for i in members:
            list(battle.team.values())[i]._selected_in_teampreview = True
        return '/team ' + ''.join(str(i+1) for i in members)

    def choose_move(self, battle):
        try:
            return self._choose_move(battle)
        except Exception:
            self.failures += 1
            self.logger.exception('Search failed; using a legal fallback')
            legal = legal_joint_actions(battle)
            return legal[int(self.rng.integers(len(legal)))]

    def _choose_move(self, battle):
        start = time.perf_counter()
        legal = legal_joint_actions(battle)
        obs = observe(battle)
        gap = None
        if self.mode == 'policy':
            if self.model is None:
                raise ValueError('policy mode requires checkpoint')
            x = np.array([features(obs, action) for action in legal])
            probabilities, _ = self.model.distribution(x)
            index = int(self.rng.choice(len(legal), p=probabilities))
            chosen = legal[index]
            self.training_records.append((x, index))
        elif self.mode == 'random' or any(battle.force_switch):
            chosen = legal[int(self.rng.integers(len(legal)))]
        else:
            intents = [own_intents(a, obs) for a in legal]
            # Rank attacks, retain randomized exploration so Protect/switch survive.
            scores = [sum(sum(x.damage) + x.bonus for x in a) for a in intents]
            width = min(self.candidate_limit, len(legal))
            best = list(np.argsort(scores)[::-1][:max(1, width*3//4)])
            remaining = [i for i in range(len(legal)) if i not in best]
            if remaining:
                best += list(self.rng.choice(remaining, size=width-len(best), replace=False))
            selected = [intents[i] for i in best]
            other = opponent_candidates(obs)
            a = matrix(selected, other, obs, self.rng)
            if self.mode == 'learned':
                if self.model is None:
                    raise ValueError('learned mode requires a model')
                predictions = self.model.predict(np.array([features(obs, legal[i]) for i in best]))
                a += self.model_weight * (2 * predictions[:, None] - 1)
            solution = solve(a, self.iterations, self.rng)
            gap = solution.exploitability
            chosen = legal[best[int(self.rng.choice(len(best), p=solution.row))]]
        assert chosen.message in {a.message for a in legal}
        elapsed = time.perf_counter() - start
        self.decisions += 1
        self.seconds += elapsed
        if self.trace:
            with self.trace.open('a') as f:
                f.write(json.dumps({'schema': 1, 'battle': battle.battle_tag,
                    'observation': obs.json(), 'action': chosen.message,
                    'features': features(obs, chosen).tolist(),
                    'legal_count': len(legal), 'forced_switch': any(battle.force_switch),
                    'seconds': elapsed, 'matrix_exploitability': gap}) + '\n')
        return chosen
