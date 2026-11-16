"""External-sampling MCCFR for a finite, simultaneous zero-sum matrix game.

Each player has one information set. Sample the other player's action; enumerate
all traverser's actions. This is an unbiased counterfactual regret estimate.
The returned average policies, not the final iterates, define play.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Solution:
    row: np.ndarray
    column: np.ndarray
    exploitability: float

def regret_match(regrets: np.ndarray) -> np.ndarray:
    positive = np.maximum(regrets, 0)
    return positive / positive.sum() if positive.sum() else np.ones(len(regrets)) / len(regrets)

def solve(payoffs: np.ndarray, iterations: int, rng: np.random.Generator) -> Solution:
    a = np.asarray(payoffs, dtype=float)
    if a.ndim != 2 or not a.size or not np.isfinite(a).all() or iterations < 1:
        raise ValueError('Expected a finite nonempty matrix and positive iteration count')
    r, c = np.zeros(a.shape[0]), np.zeros(a.shape[1])
    avg_r, avg_c = r.copy(), c.copy()
    for _ in range(iterations):
        p, q = regret_match(r), regret_match(c)
        avg_r += p
        avg_c += q
        i, j = rng.choice(len(p), p=p), rng.choice(len(q), p=q)
        row_values, col_values = a[:, j], -a[i, :]
        r += row_values - p @ row_values
        c += col_values - q @ col_values
    p, q = avg_r / iterations, avg_c / iterations
    gap = float(np.max(a @ q) - np.min(p @ a))
    return Solution(p, q, gap)
