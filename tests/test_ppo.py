import pytest
import numpy as np
torch=pytest.importorskip('torch')
from src.training.ppo import Network,advantages,save,load

def test_gae_terminal_and_truncation_boundaries():
    reward=torch.tensor([[1.],[0.]])
    value=torch.zeros_like(reward);nv=torch.tensor([[9.],[2.]])
    term=torch.tensor([[True],[False]]);done=torch.tensor([[True],[True]])
    assert torch.allclose(advantages(reward,value,nv,term,done),torch.tensor([[1.],[2.]]))

def test_joint_mask_logprob_and_checkpoint(tmp_path):
    net=Network(3,4);obs=torch.ones((8,3));mask=torch.tensor([[False,True,False,True]]*8)
    dist,_=net(obs,mask);a=dist.sample()
    assert ((a==1)|(a==3)).all()
    assert torch.allclose(dist.log_prob(a),torch.log(dist.probs.gather(1,a[:,None]).squeeze()))
    optimizer=torch.optim.Adam(net.parameters());path=tmp_path/'m.pt';save(path,net,optimizer,2,{})
    restored,_=load(path)
    assert torch.equal(dist.probs,restored(obs,mask)[0].probs)

def test_trainable_network_learns_masked_bandit():
    torch.manual_seed(3);net=Network(2,2);optimizer=torch.optim.Adam(net.parameters(),lr=.01)
    obs=torch.ones((16,2));mask=torch.ones((16,2),dtype=torch.bool)
    for _ in range(30):
        dist,value=net(obs,mask)
        loss=-dist.log_prob(torch.ones(16,dtype=torch.long)).mean()+(value-1).square().mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
    assert net(obs,mask)[0].probs[0,1]>.99

def test_channel_preview_has_showdown_prefix(tmp_path):
    import pikapro
    from src.sim.pikapro_backend import ClientView
    from src.agents.ppo_player import PPOBot
    team=open('teams/balance.txt').read()
    b=pikapro.Battle('championsvgc',[team,team],seed=1)
    view=ClientView(0,'test-channel');view.update(b)
    net=Network(1688,2569);optimizer=torch.optim.Adam(net.parameters());path=tmp_path/'p.pt'
    save(path,net,optimizer,0,{})
    bot=PPOBot(start_listening=False,model_path=path,battle_format='gen9championsvgc2026regmc')
    choice=bot.teampreview(view.battle)
    assert choice.startswith('/team ')
    assert b.choice_index(0,choice[1:]) in b.legal_actions(0)

def test_clipped_on_policy_update_learns_bandit():
    torch.manual_seed(19);net=Network(2,2);opt=torch.optim.Adam(net.parameters(),lr=.003)
    obs=torch.ones((128,2));mask=torch.ones((128,2),dtype=torch.bool)
    for _ in range(20):
        with torch.no_grad():
            old,value=net(obs,mask);actions=old.sample();lp=old.log_prob(actions)
            reward=(actions==1).float();adv=reward-value
        for _ in range(2):
            dist,v=net(obs,mask);ratio=(dist.log_prob(actions)-lp).exp()
            loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()+.5*(v-reward).square().mean()
            opt.zero_grad();loss.backward();opt.step()
    assert net(obs,mask)[0].probs[0,1]>.9
