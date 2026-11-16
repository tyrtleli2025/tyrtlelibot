from poke_env.player.battle_order import DoubleBattleOrder

def legal_joint_actions(battle):
    """Request-authoritative legal actions; encode using Showdown choice syntax v1."""
    if hasattr(battle, 'tyrtle_legal_actions'):
        return battle.tyrtle_legal_actions
    actions = DoubleBattleOrder.join_orders(*battle.valid_orders)
    unique = {a.message: a for a in actions}
    if not unique:
        raise ValueError('No legal joint action in decision request')
    return list(unique.values())
