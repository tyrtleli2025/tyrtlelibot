import numpy as np
import pytest
from src.search.mccfr import solve

def test_matching_pennies():
    s = solve(np.array([[1, -1], [-1, 1]]), 20000, np.random.default_rng(7))
    assert np.max(abs(s.row - .5)) < .04
    assert s.exploitability < .12

def test_dominant_action():
    s = solve(np.array([[2, 1], [0, -1]]), 2000, np.random.default_rng(3))
    assert s.row[0] > .99
    assert s.column[1] > .99

def test_seeded_and_rejects_invalid():
    a = np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]])
    assert np.array_equal(solve(a, 100, np.random.default_rng(4)).row,
                          solve(a, 100, np.random.default_rng(4)).row)
    with pytest.raises(ValueError):
        solve(np.array([[np.nan]]), 10, np.random.default_rng())
