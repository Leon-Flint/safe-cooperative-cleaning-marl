"""环境规则测试。"""
import numpy as np
import pytest
from safe_cleaning.config import AGENTS, FRAGILE, GRID, STARTS
from safe_cleaning.environment import CleanEnv, safety_check
@pytest.mark.parametrize('mode', ['fixed', 'random'])

def test_reset_and_seed(mode):
    """检查重置与随机种子。"""
    env = CleanEnv(mode)
    obs, info = env.reset(seed=7)
    dirt = env.normal_dirt.copy(), env.large_dirt.copy()
    assert env.observation_space.contains(obs)
    assert obs.shape == (AGENTS, GRID, GRID, 5)
    assert len(dirt[0]) == len(dirt[1]) == 8
    assert not dirt[0] & dirt[1]
    assert not (dirt[0] | dirt[1]) & FRAGILE
    env.step([0, 0, 0])
    obs2, _ = env.reset(seed=7)
    np.testing.assert_array_equal(obs, obs2)
    assert dirt == (env.normal_dirt, env.large_dirt)
    np.testing.assert_array_equal(env.agent_positions, STARTS)
    assert info['step'] == env.step_count == 0

def test_idle_and_safety():
    """检查停留奖励与安全拦截。"""
    env = CleanEnv()
    env.reset()
    _, rewards, terminated, truncated, _ = env.step([0, 0, 0])
    np.testing.assert_allclose(rewards, [-0.2] * 3)
    assert not terminated and not truncated
    before = env.agent_positions.copy()
    _, rewards, terminated, truncated, info = env.step([1, 0, 0])
    assert truncated and not terminated
    assert info['violating_agents'] == (0,)
    assert rewards[0] < -20
    np.testing.assert_array_equal(env.agent_positions, before)

def test_collision_and_swap():
    """检查同格与交换位置碰撞。"""
    positions = np.array([[0, 0], [0, 1], [11, 11]])
    same = np.array([[0, 1], [0, 1], [11, 11]])
    assert 'same_cell_collision' in safety_check(positions, same)[0]
    swapped = np.array([[0, 1], [0, 0], [11, 11]])
    assert 'position_swap_collision' in safety_check(positions, swapped)[1]
    fragile = positions.copy()
    fragile[0] = next(iter(FRAGILE))
    assert 'fragile_object' in safety_check(positions, fragile)[0]

def test_completion_and_timeout():
    """检查完成与超时。"""
    env = CleanEnv()
    env.reset()
    env.normal_dirt.clear()
    env.large_dirt.clear()
    _, rewards, term, trunc, info = env.step([0, 0, 0])
    assert term and not trunc and info['termination_reason'] == 'all_dirt_cleaned'
    np.testing.assert_allclose(rewards, [19.8] * 3)
    env.reset()
    env.step_count = 49
    _, _, term, trunc, info = env.step([0, 0, 0])
    assert not term and trunc and info['termination_reason'] == 'time_limit'

def test_render_and_invalid_actions():
    """检查渲染与非法动作。"""
    env = CleanEnv(render_mode='rgb_array')
    env.reset()
    rgb = env.render()
    assert rgb.ndim == 3 and rgb.shape[-1] == 3 and rgb.dtype == np.uint8
    with pytest.raises(ValueError):
        env.step([5, 0, 0])
