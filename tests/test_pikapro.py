from pathlib import Path
import pytest
pikapro=pytest.importorskip('pikapro')
from src.sim.pikapro_backend import play,ClientView
from src.observations.view import observe
TEAM=Path('teams/balance.txt').read_text()

def test_native_reproducible_legal_complete_game():
    a,logs=play(17,TEAM,('random','random'))
    b,logs2=play(17,TEAM,('random','random'))
    assert a['finished'] and a['failures']==0
    assert a['choices']==b['choices'] and logs==logs2

def test_opponent_private_move_change_does_not_enter_observation():
    other=TEAM.replace('- Fake Out','- Taunt')
    a=pikapro.Battle('championsvgc',[TEAM,TEAM],seed=2)
    b=pikapro.Battle('championsvgc',[TEAM,other],seed=2)
    views=[ClientView(0,'test-a'),ClientView(0,'test-b')]
    views[0].update(a);views[1].update(b)
    assert observe(views[0].battle)==observe(views[1].battle)
