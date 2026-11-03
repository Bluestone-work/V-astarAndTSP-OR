"""Behavioral checks against Eqs.18--19 and Algorithm3, independent of training."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import numpy as np
import pytest
import torch
from torch import nn
import agents
from manuscript_replay import dual_priority, ManuscriptReplay, StandardPER, UniformReplay

class TableQ(nn.Module):
    def __init__(self,*args):
        super().__init__();self.q=nn.Parameter(torch.zeros(2,9))
    def forward(self,x):return self.q[x[:,-1,0].long()]


def make_agent(monkeypatch,**kwargs):
    monkeypatch.setattr(agents,'DuelingDQN_with_LSTM',TableQ)
    options=dict(env=SimpleNamespace(),device='cpu',batch_size=2,smoothing_factor=0.,gamma=.5)
    options.update(kwargs)
    return agents.RainbowDQN(**options)


def state(which=0):
    x=np.zeros((4,51),np.float32);x[:,0]=which;return x


@pytest.mark.parametrize('reward',(-10.,-1.,0.,.1,5.,10.))
@pytest.mark.parametrize('delta',(0.,-.1,1.,10.))
def test_priority_matches_written_equation(reward,delta):
    expected=.25*np.log(reward+abs(min(reward,0))+1)+.75*np.log(abs(delta)+1)+1e-5
    assert float(dual_priority(reward,delta))==pytest.approx(expected)


def test_nonpositive_rewards_do_not_gain_mass_from_absolute_signed_priorities():
    assert dual_priority(-5,0)==pytest.approx(1e-5)
    assert dual_priority(0,0)==pytest.approx(1e-5)
    assert dual_priority(10,0)>dual_priority(-5,0)


def test_single_fifo_pool_owns_immutable_transition_copies():
    replay=ManuscriptReplay(capacity=2);s=state()
    replay.append(s,0,-5,s,True,2);s[0,0]=99
    assert replay.memory[0][0][0,0]==0
    replay.append(state(),1,0,state(),False,1)
    replay.append(state(),2,10,state(),True,0)
    assert [x[1] for x in replay.memory]==[1,2]
    assert len(replay)==2 and not hasattr(replay,'priority_buffer')


def test_entire_batch_comes_from_one_directly_normalized_distribution(monkeypatch):
    replay=ManuscriptReplay(10)
    replay.append(state(),0,-5,state(),True,1)
    replay.append(state(),1,10,state(),True,0)
    expected=np.array([dual_priority(-5,1),dual_priority(10,0)])
    expected/=expected.sum()
    def sample(n,size,replace,p):
        assert n==2 and size==64 and replace and np.allclose(p,expected)
        return np.array([0,1]*32)
    monkeypatch.setattr(np.random,'choice',sample)
    indices,batch=replay.sample(64)
    assert len(batch)==len(indices)==64


def test_refresh_changes_sampled_priorities_only():
    replay=ManuscriptReplay(10)
    for i in range(3):replay.append(state(),i,float(i),state(),False,1)
    before=[item[5] for item in replay.memory]
    replay.refresh([1],[-10])
    assert replay.memory[0][5]==before[0] and replay.memory[2][5]==before[2]
    assert replay.memory[1][5]==pytest.approx(dual_priority(1,10))


def test_first_transition_uses_eq19_including_terminal_mask(monkeypatch):
    a=make_agent(monkeypatch)
    with torch.no_grad():
        a.q_network.q[0,0]=2.;a.target_network.q[1,2]=99.
    a.store_transition(state(),0,-5.,state(1),True)
    assert a.memory[0][5]==pytest.approx(dual_priority(-5,7))
    assert not hasattr(a,'priority_buffer')


def test_td_priority_uses_same_stored_composite_reward(monkeypatch):
    a=make_agent(monkeypatch,smoothing_factor=.5)
    a.last_actions.extend([0,1])
    a.store_transition(state(),1,1.,state(1),True)
    reward=1.-.5*.3
    assert a.memory[0][2]==pytest.approx(reward)
    assert a.memory[0][5]==pytest.approx(dual_priority(reward,reward))


def test_one_step_target_uses_target_maximum_and_mse_without_IS(monkeypatch):
    a=make_agent(monkeypatch)
    with torch.no_grad():
        a.q_network.q[1,2]=50.
        a.target_network.q[1,1]=4.
        a.target_network.q[1,2]=1.
    batch=[(state(),0,1.,state(1),False,1.),(state(),1,-5.,state(1),True,1.)]
    metrics=a.compute_bellman_metrics(batch)
    assert np.allclose(metrics['target_q_values'],[3.,-5.])
    assert np.allclose(metrics['td_errors'],[3.,-5.])
    assert metrics['loss_tensor'].item()==pytest.approx((9+25)/2)


def test_train_refreshes_priorities_from_pre_gradient_errors(monkeypatch):
    a=make_agent(monkeypatch)
    a.store_transition(state(),0,1.,state(),True)
    a.store_transition(state(),1,-5.,state(),True)
    monkeypatch.setattr(np.random,'choice',lambda *args,**kwargs:np.array([0,1]))
    with torch.no_grad():a.q_network.q[0,0]=.5;a.q_network.q[0,1]=-1.
    loss=a.train()
    assert float(loss)==pytest.approx((.5**2+4.**2)/2)
    assert [item[5] for item in a.memory]==pytest.approx([dual_priority(1,.5),dual_priority(-5,4)])
    assert a.training_steps==1


def test_standard_per_applies_IS_loss_and_soft_updates_on_first_optimizer_step(monkeypatch):
    a = make_agent(monkeypatch, replay_mode='per', alpha=1., beta=1.)
    a.store_transition(state(), 0, 1., state(), True)
    a.store_transition(state(), 1, 4., state(), True)
    monkeypatch.setattr(np.random, 'choice', lambda *args, **kwargs: np.array([0, 1]))
    weight = (1.+1e-5)/(4.+1e-5)
    loss = a.train()
    assert float(loss) == pytest.approx((1.+16.*weight)/2)
    assert a.training_steps == 1
    assert torch.allclose(a.target_network.q, .01*a.q_network.q)


def test_repeated_reward_multistep_mode_is_rejected(monkeypatch):
    with pytest.raises(ValueError,match='one-step'):make_agent(monkeypatch,n_step=4)


def test_checkpoint_identifies_corrected_replay(monkeypatch,tmp_path):
    a=make_agent(monkeypatch);p=tmp_path/'corrected.pth';a.save_model(p)
    assert torch.load(p,weights_only=False)['replay_protocol']=='submitted-eq18-eq19-one-step'


@pytest.mark.parametrize('reward,delta',((np.nan,0),(1,np.inf)))
def test_invalid_priorities_are_rejected(reward,delta):
    with pytest.raises(ValueError):dual_priority(reward,delta)


def test_replay_comparison_baselines_have_matching_sampling_interfaces():
    for replay in (UniformReplay(4), StandardPER(4, alpha=.6, beta=.4)):
        replay.append(state(), 0, -1., state(), True, 2.)
        replay.append(state(), 1, 1., state(), False, .5)
        indices, batch = replay.sample(8)
        assert len(indices) == len(batch) == 8
        assert np.isclose(replay.probabilities().sum(), 1.0)
        replay.refresh(indices, np.ones(len(indices)))


def test_real_network_forward_and_optimizer_update():
    torch.set_num_threads(1)
    a=agents.RainbowDQN(SimpleNamespace(),device='cpu',batch_size=2,smoothing_factor=0.)
    assert a.q_network(torch.zeros(2,4,51)).shape==(2,9)
    a.store_transition(state(),0,1.,state(),True)
    a.store_transition(state(),1,-5.,state(),True)
    assert torch.isfinite(a.train())


def test_original_public_source_archive_matches_recorded_commit_blob():
    file=Path(__file__).resolve().parents[1]/'legacy/agents_original.py'
    assert hashlib.sha256(file.read_bytes()).hexdigest()=='19f1070e31254ad4233e346bea6eeba8f54c1be79d2db9a574dbb2bc996de797'
