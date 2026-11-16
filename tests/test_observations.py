from types import SimpleNamespace
from poke_env.battle import Pokemon
from src.observations.view import observe

def test_private_state_does_not_enter_view():
    own = Pokemon(gen=9, species='garchomp')
    foe = Pokemon(gen=9, species='incineroar')
    battle = SimpleNamespace(turn=1, active_pokemon=[own, None], opponent_active_pokemon=[foe, None],
                             hidden_opponent_team=['protect'], internal_seed=11)
    first = observe(battle)
    battle.hidden_opponent_team = ['toxic']
    battle.internal_seed = 99
    assert first == observe(battle)
    assert first.opponent[0].moves == ()
    assert first.opponent[0].item is None
