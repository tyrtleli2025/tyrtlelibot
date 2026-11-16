import numpy as np
from src.observations.view import Mon, Observation
from src.search.tactical import Intent, damage, payoff

def pokemon(typ='NORMAL', speed=50):
    return Mon('test', 1., (typ,), speed, (), None, None)

def test_immunity_and_super_effective():
    p = pokemon('ELECTRIC')
    assert damage(90, 'ELECTRIC', p, pokemon('GROUND')) == 0
    assert damage(90, 'ELECTRIC', p, pokemon('WATER')) > damage(90, 'ELECTRIC', p, pokemon())

def test_protect_and_fast_ko_change_payoff():
    obs = Observation(1, 1, (pokemon(), None), (pokemon(), None))
    attack = (Intent(0, (1.1, 0), speed=100), Intent(1))
    slow = (Intent(0, (1.1, 0), speed=40), Intent(1))
    protect = (Intent(0, protect=True, priority=4), Intent(1))
    assert payoff(attack, slow, obs, np.ones((2,2))) > 1
    assert payoff(attack, protect, obs, np.ones((2,2))) == 0

def test_fakeout_after_first_turn_is_not_scored_as_damage():
    from dataclasses import replace
    from poke_env.battle import Move
    from poke_env.player.battle_order import SingleBattleOrder
    from src.search.tactical import intent
    p = replace(pokemon(), first_turn=False)
    a = intent(SingleBattleOrder(Move('fakeout', gen=9), move_target=1), 0,
               (p, None), (pokemon(), None))
    assert a.damage == (0., 0.)
    assert a.bonus < 0
