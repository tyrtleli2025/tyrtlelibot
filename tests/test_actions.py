from types import SimpleNamespace
from poke_env.battle import Pokemon, Move
from poke_env.player.battle_order import SingleBattleOrder, PassBattleOrder
from src.actions.legal import legal_joint_actions

def test_joint_mechanic_and_duplicate_switch_constraints():
    bench = Pokemon(gen=9, species='garchomp')
    options = [SingleBattleOrder(bench), SingleBattleOrder(Move('protect', gen=9), mega=True)]
    actions = legal_joint_actions(SimpleNamespace(valid_orders=[options, options]))
    assert len(actions) == 2
    assert all(not (a.first_order.mega and a.second_order.mega) for a in actions)

def test_forced_replacement_single_bench_and_pass():
    bench = Pokemon(gen=9, species='garchomp')
    options = [SingleBattleOrder(bench), PassBattleOrder()]
    actions = legal_joint_actions(SimpleNamespace(valid_orders=[options, options]))
    assert len(actions) == 2
    assert all('switch' in a.message and 'pass' in a.message for a in actions)
