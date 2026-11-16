from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from poke_env.battle import Move, Pokemon
from poke_env.player.battle_order import DoubleBattleOrder, SingleBattleOrder
from src.observations.view import observe
from src.training.model import DIM, OutcomeModel, features

def test_model_learns_and_reload_is_exact(tmp_path):
    rng=np.random.default_rng(4)
    x=rng.normal(size=(200,DIM));y=(x[:,2]>0).astype(float)
    m=OutcomeModel();m.fit(x,y,np.ones(len(y)))
    assert np.mean((m.predict(x)>.5)==y)>.95
    path=tmp_path/'model.npz';m.save(path)
    assert np.array_equal(m.predict(x),OutcomeModel.load(path).predict(x))

def test_observation_preserves_fainted_status_and_public_roster():
    own=Pokemon(gen=9,species='garchomp');foe=Pokemon(gen=9,species='incineroar')
    foe.faint()
    b=SimpleNamespace(turn=2,active_pokemon=[own,None],opponent_active_pokemon=[None,None],
                      team={'own':own},opponent_team={'public':foe},private_opponent_moves=['toxic'])
    obs=observe(b)
    assert obs.version==2
    assert obs.opponent_team[0].fainted
    assert obs.opponent_team[0].moves==()
    b.private_opponent_moves=['protect']
    assert obs==observe(b)

def test_action_features_change_with_target_and_are_finite():
    own=Pokemon(gen=9,species='garchomp');foe=Pokemon(gen=9,species='incineroar')
    b=SimpleNamespace(turn=2,active_pokemon=[own,None],opponent_active_pokemon=[foe,None],
                      team={'a':own},opponent_team={'b':foe})
    obs=observe(b)
    attack=DoubleBattleOrder(SingleBattleOrder(Move('earthpower',gen=9),move_target=1))
    protect=DoubleBattleOrder(SingleBattleOrder(Move('protect',gen=9)))
    a,c=features(obs,attack),features(obs,protect)
    assert a.shape==(DIM,)
    assert np.isfinite(a).all() and not np.array_equal(a,c)

def test_trainer_rejects_test_partition(tmp_path):
    import json
    import pytest
    from src.training.train import train
    (tmp_path/'manifest.json').write_text(json.dumps({'split':'test'}))
    with pytest.raises(ValueError,match='test data'):
        train([tmp_path],tmp_path/'forbidden.npz')
