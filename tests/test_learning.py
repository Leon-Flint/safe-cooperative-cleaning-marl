"""网络、训练与存取测试。"""
import numpy as np
import pytest
import torch
from safe_cleaning.config import TrainConfig
from safe_cleaning.networks import ActorNet, CriticNet, CentralCritic, SharedActor
from safe_cleaning.rollout import Rollout
from safe_cleaning.environment import CleanEnv
from safe_cleaning.ippo import train_ippo, get_ippo_roll
from safe_cleaning.mappo import train_mappo, get_mappo_roll
from safe_cleaning.storage import load_ippo, load_mappo, read_arrays, save_roll
from safe_cleaning.evaluation import evaluate_diagnostics, jsd

def test_network_shapes():
    """检查网络输出形状。"""
    obs = torch.zeros(3, 12, 12, 5)
    assert ActorNet()(obs).shape == (3, 5)
    assert CriticNet()(obs).shape == (3,)
    assert CentralCritic()(obs.reshape(1, -1)).shape == (1,)
    assert SharedActor()(obs, [0, 1, 2]).shape == (3, 5)

def test_gae_by_hand_and_copy():
    """核对优势计算与观测副本。"""
    buffer = Rollout()
    obs = np.zeros((12, 12, 5))
    buffer.add(obs, 0, -1, 1, 0.5, False)
    buffer.add(obs, 1, -1, 2, 0.6, True)
    obs[:] = 1
    assert buffer.observations[0].sum() == 0
    buffer.compute_gae(99, gamma=0.9, gae_lambda=0.8)
    # 手算TD误差分别为1.04和1.4。
    np.testing.assert_allclose(buffer.advantages, [2.048, 1.4], rtol=1e-6)
    np.testing.assert_allclose(buffer.returns, [2.548, 2.0], rtol=1e-6)
    assert buffer.as_tensors()['actions'].dtype == torch.long
    assert buffer.as_tensors()['returns'].shape == (2,)
    with pytest.raises(ValueError):
        Rollout().compute_gae(0)
    with pytest.raises(ValueError):
        Rollout().as_tensors()
@pytest.mark.parametrize('algorithm', ['ippo', 'mappo'])

def test_train_reload_evaluate(tmp_path, algorithm):
    """检查参数更新、重载与评估。"""
    cfg = TrainConfig(epochs=1, batch_size=4)
    torch.manual_seed(31)
    initial = ActorNet() if algorithm == 'ippo' else SharedActor()
    before = {name: p.detach().clone() for name, p in initial.named_parameters()}
    train = train_ippo if algorithm == 'ippo' else train_mappo
    loader = load_ippo if algorithm == 'ippo' else load_mappo
    first, second, metrics, best = train('fixed', 2, 8, 31, tmp_path, cfg)
    actor = first[0] if algorithm == 'ippo' else first
    assert any(not torch.equal(before[n], p) for n, p in actor.named_parameters())
    for values in metrics.values():
        assert values.shape == (2, 3)
        assert np.isfinite(values).all()
    assert best.is_file()
    last = tmp_path / f'{algorithm}_fixed_last.npz'
    loaded = loader(last)
    loaded_actor = loaded[0][0] if algorithm == 'ippo' else loaded[0]
    for name, value in actor.state_dict().items():
        torch.testing.assert_close(value, loaded_actor.state_dict()[name], rtol=0, atol=0)
    result = evaluate_diagnostics(loaded, algorithm, episodes=2, seed=101)
    assert result['episode_returns'].shape == (2, 3)
    assert np.isfinite(result['episode_returns']).all()
    assert np.all((result['success_fractions'] >= 0) & (result['success_fractions'] <= 1))
    assert read_arrays(best)['update'] >= 0

def test_collect_save_and_random(tmp_path):
    """检查随机环境经历收集与保存。"""
    actors = [ActorNet() for _ in range(3)]
    critics = [CriticNet() for _ in range(3)]
    buffers = get_ippo_roll(CleanEnv('random'), actors, critics, rollout_length=4)
    save_roll(buffers, tmp_path / 'roll.npz')
    assert read_arrays(tmp_path / 'roll.npz')['agent_1_actions'].shape == (4,)
    buffers = get_mappo_roll(CleanEnv('random'), SharedActor(), CentralCritic(), rollout_length=4)
    assert buffers[0].as_tensors()['global_states'].shape == (4, 2160)

def test_jsd_and_config():
    """检查分布差异与配置校验。"""
    assert jsd([0.2, 0.8], [0.2, 0.8]) == pytest.approx(0)
    assert jsd([1, 0], [0, 1]) == pytest.approx(np.log(2), abs=1e-9)
    with pytest.raises(ValueError):
        TrainConfig(batch_size=0).validate()
