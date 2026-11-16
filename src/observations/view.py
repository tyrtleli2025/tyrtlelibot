"""Versioned serializable projection of the network client's legal information.

No simulator state, opposing request, private seed, or opponent team file enters
this module. None means unknown, never 'no item/ability/moves'.
"""
from dataclasses import asdict, dataclass
from poke_env.battle import Pokemon

@dataclass(frozen=True)
class Mon:
    species: str
    hp: float
    types: tuple[str, ...]
    speed: float
    moves: tuple[str, ...]
    item: str | None
    ability: str | None
    first_turn: bool = True
    protect_counter: int = 0
    status: str | None = None
    boosts: tuple[tuple[str, int], ...] = ()
    fainted: bool = False

@dataclass(frozen=True)
class Observation:
    version: int
    turn: int
    own: tuple[Mon | None, ...]
    opponent: tuple[Mon | None, ...]
    tailwind: tuple[bool, bool] = (False, False)
    own_team: tuple[Mon | None, ...] = ()
    opponent_team: tuple[Mon | None, ...] = ()
    weather: tuple[str, ...] = ()
    fields: tuple[str, ...] = ()
    side_conditions: tuple[tuple[str, ...], ...] = ()
    def json(self):
        return asdict(self)

def mon(p: Pokemon | None) -> Mon | None:
    if p is None:
        return None
    unknown = {None, '', 'unknown_item'}
    return Mon(p.species, p.current_hp_fraction,
               tuple(t.name for t in p.types if t), float(p.base_stats['spe']),
               tuple(sorted(p.moves)), p.item if p.item not in unknown else None,
               p.ability if p.ability not in unknown else None, p.first_turn, p.protect_counter,
               p.status.name if p.status else None, tuple(sorted(p.boosts.items())), p.fainted)

def observe(battle) -> Observation:
    return Observation(2, battle.turn, tuple(map(mon, battle.active_pokemon)),
                       tuple(map(mon, battle.opponent_active_pokemon)),
                       tuple(any(getattr(k, "name", "") == "TAILWIND" for k in getattr(battle, attr, {}))
                             for attr in ("side_conditions", "opponent_side_conditions")),
                       tuple(map(mon, getattr(battle, 'team', {}).values())),
                       tuple(map(mon, getattr(battle, 'opponent_team', {}).values())),
                       tuple(sorted(k.name for k in getattr(battle, 'weather', {}))),
                       tuple(sorted(k.name for k in getattr(battle, 'fields', {}))),
                       tuple(tuple(sorted(k.name for k in getattr(battle, attr, {})))
                             for attr in ('side_conditions', 'opponent_side_conditions')))
