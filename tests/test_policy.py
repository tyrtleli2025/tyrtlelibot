import numpy as np
from src.training.policy import PolicyValue
from src.training.model import DIM

def test_policy_learns_rewarded_action_and_value_and_reloads(tmp_path):
    m=PolicyValue(); x=np.zeros((2,DIM));x[0,14]=1;x[1,14]=-1
    before=m.distribution(x)[0][0]
    for _ in range(300):m.update([([(x,0)],1.), ([(x,1)],-1.)])
    assert m.distribution(x)[0][0]>before+.4
    # A constant positive terminal target is learned by the value head.
    for _ in range(300):m.update([([(x,0)],1.)])
    assert abs(m.value_features(x[0])@m.critic-1)<.1
    path=tmp_path/'policy.npz';m.save(path)
    assert np.array_equal(m.distribution(x)[0],PolicyValue.load(path).distribution(x)[0])

def test_legal_distribution_finite_and_normalized():
    m=PolicyValue();x=np.ones((3,DIM));p,_=m.distribution(x)
    assert np.isfinite(p).all() and np.isclose(p.sum(),1)
    assert len(p)==3
