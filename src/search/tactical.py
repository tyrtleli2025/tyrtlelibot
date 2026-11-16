"""Transparent ONE-TURN SURROGATE, not a replacement Pokemon simulator.

Unknown attacks use an 80-power STAB prior. Damage scales approximate fractional
HP. Protect, priority, focus fire and speed affect the joint payoff. Status and
switch values are hand-authored. No claim of full-game CFR convergence.
"""
from dataclasses import dataclass
from itertools import product
import numpy as np
from poke_env.battle import Move, Pokemon, PokemonType
from poke_env.data import GenData
from src.observations.view import Mon, Observation, mon

CHART = GenData.from_gen(9).type_chart

@dataclass(frozen=True)
class Intent:
    actor: int
    damage: tuple[float, float] = (0., 0.)
    protect: bool = False
    protect_chance: float = 1.
    priority: int = 0
    speed: float = 0.
    bonus: float = 0.
    replacement: Mon | None = None

def damage(power, move_type, attacker, defender, accuracy=1.):
    if not attacker or not defender or not power or move_type is None:
        return 0.
    typ = PokemonType[move_type]
    types = [PokemonType[t] for t in defender.types]
    effectiveness = typ.damage_multiplier(*types, type_chart=CHART)
    stab = 1.5 if move_type in attacker.types else 1.
    return min(1.5, power / 250 * effectiveness * stab * accuracy)

def intent(order, slot, own, foe):
    p = own[slot]
    if isinstance(order.order, Pokemon):
        replacement = mon(order.order)
        return Intent(slot, priority=6, bonus=-0.08, replacement=replacement)
    move = order.order
    if not isinstance(move, Move) or p is None:
        return Intent(slot)
    protect = move.id in {'protect', 'detect', 'spikyshield', 'kingsshield', 'banefulbunker'}
    target = order.move_target
    spread = move.target.name in {'ALL_ADJACENT_FOES', 'ALL_ADJACENT'}
    ds = tuple(damage(move.base_power, move.type.name if move.type else None, p, f, move.accuracy)
               * (0.75 if spread and all(foe) else 1.)
               if spread or target == i + 1 or (target == 0 and not spread and sum(x is not None for x in foe) == 1)
               else 0. for i, f in enumerate(foe))
    if move.id == 'fakeout' and not p.first_turn:
        ds = (0., 0.)
    # Discourage friendly fire and aimless status; utility is deliberately modest.
    bonus = {'tailwind': .20, 'spore': .25, 'willowisp': .12, 'thunderwave': .12,
             'ragepowder': .08, 'followme': .08, 'fakeout': .12}.get(move.id, 0.)
    if move.id == 'fakeout' and not p.first_turn:
        bonus = -.2
    if protect:
        bonus -= .025
    if move.base_power and target < 0:
        bonus -= .5
    if move.target.name == 'ALL_ADJACENT' and own[1-slot]:
        bonus -= damage(move.base_power, move.type.name, p, own[1-slot]) * .75
    if not move.base_power and not protect and not bonus:
        bonus = -.03
    if order.mega:
        bonus += .08
    return Intent(slot, ds, protect=protect, protect_chance=1 / (3 ** p.protect_counter),
                  priority=move.priority, speed=p.speed, bonus=bonus)

def own_intents(action, obs):
    from dataclasses import replace
    result = []
    for i, order in enumerate((action.first_order, action.second_order)):
        a = intent(order, i, obs.own, obs.opponent)
        if obs.tailwind[0]:
            a = replace(a, speed=a.speed * 2)
            if isinstance(order.order, Move) and order.order.id == 'tailwind':
                a = replace(a, bonus=-.1)
        result.append(a)
    return tuple(result)

def opponent_candidates(obs):
    choices = []
    for i, p in enumerate(obs.opponent):
        if p is None:
            choices.append([Intent(i)])
            continue
        options = [Intent(i, protect=True, protect_chance=1 / (3 ** p.protect_counter), priority=4, speed=p.speed)]
        # Revealed moves plus a transparent generic STAB prior, not a hidden set.
        attacks = [(80, t, 0, 1.) for t in p.types]
        for name in p.moves:
            m = Move(name, gen=9)
            if m.base_power and (m.id != 'fakeout' or p.first_turn):
                attacks.append((m.base_power, m.type.name, m.priority, m.accuracy))
        for power, typ, priority, accuracy in attacks:
            for target, defender in enumerate(obs.own):
                if defender:
                    ds = [0., 0.]
                    ds[target] = damage(power, typ, p, defender, accuracy)
                    options.append(Intent(i, tuple(ds), priority=priority, speed=p.speed * (2 if obs.tailwind[1] else 1)))
        choices.append(options)
    return list(product(*choices))

def payoff(ours, theirs, obs, rolls):
    hp = [[p.hp if p else 0. for p in side] for side in (obs.own, obs.opponent)]
    initial = [x.copy() for x in hp]
    protected = [[x.protect and rolls[s, x.actor] <= x.protect_chance for x in side]
                 for s, side in enumerate((ours, theirs))]
    events = [(a.priority, a.speed + rolls[s, a.actor] * .01, s, a) for s, side in enumerate((ours, theirs)) for a in side]
    value = 0.
    for _, _, side, a in sorted(events, key=lambda x: (x[0], x[1]), reverse=True):
        if hp[side][a.actor] <= 0:
            continue
        sign = 1 if side == 0 else -1
        value += sign * a.bonus
        if a.replacement:
            # Switch utility only: model does not reconstruct actual incoming attacks.
            continue
        for target, amount in enumerate(a.damage):
            if not protected[1-side][target]:
                before = hp[1-side][target]
                hp[1-side][target] = max(0., before - amount * (.85 + .15 * rolls[side, a.actor]))
                if before > 0 and hp[1-side][target] == 0:
                    value += sign * .35
    return value + sum(initial[1]) - sum(hp[1]) - sum(initial[0]) + sum(hp[0])

def matrix(ours, theirs, obs, rng, worlds=3):
    result = np.zeros((len(ours), len(theirs)))
    for _ in range(worlds):
        rolls = rng.random((2, 2))
        for i, a in enumerate(ours):
            for j, b in enumerate(theirs):
                result[i, j] += payoff(a, b, obs, rolls) / worlds
    return result
