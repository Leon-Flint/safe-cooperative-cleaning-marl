"""策略评估与协作诊断。"""
import numpy as np
import torch
from .config import AGENTS, GRID, IDLE, MAX_STEPS, SEED
from .environment import CleanEnv

def jsd(probability_p, probability_q):
    """计算两个动作分布的JSD差异。"""
    # 避免对零取对数。
    p = np.asarray(probability_p, dtype=np.float64)
    p = np.clip(p, 1e-12, 1.0)
    q = np.asarray(probability_q, dtype=np.float64)
    q = np.clip(q, 1e-12, 1.0)
    p /= p.sum()
    q /= q.sum()
    mixture = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / mixture)) + 0.5 * np.sum(q * np.log(q / mixture)))

def evaluate_ippo(actors, dirt_mode='random', episodes=200, seed=SEED + 10000):
    """以贪心动作评估IPPO。"""
    parameter = next(actors[0].parameters())
    device = parameter.device
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    environment = CleanEnv(dirt_mode=dirt_mode)
    episode_returns, episode_lengths, success_fractions, safety_violations = [], [], [], []
    successful_lengths = []
    for episode in range(episodes):
        observations, _ = environment.reset(seed=seed + episode)
        returns = np.zeros(AGENTS, dtype=np.float32)
        final_info = None
        for step in range(MAX_STEPS):
            actions = []
            with torch.no_grad():
                for agent_id in range(AGENTS):
                    obs = observations[agent_id][None]
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    logits = actors[agent_id](tensor)
                    action = torch.argmax(logits, dim=-1)
                    actions.append(int(action.item()))
            observations, rewards, terminated, truncated, final_info = environment.step(actions)
            returns += rewards
            if terminated or truncated:
                break
        cleaned_weight = (8 - len(environment.normal_dirt)) + 2 * (8 - len(environment.large_dirt))
        success = bool(final_info and final_info['termination_reason'] == 'all_dirt_cleaned')
        episode_returns.append(returns)
        # 失败回合长度记为缺失值。
        if success:
            successful_length = float(step + 1)
        else:
            successful_length = np.nan
        episode_lengths.append(successful_length)
        success_fractions.append(cleaned_weight / 24.0)
        safety_violations.append(float(final_info and final_info['termination_reason'] == 'safety_violation'))
        if success:
            successful_lengths.append(step + 1)
    if successful_lengths:
        mean_length = float(np.mean(successful_lengths))
    else:
        mean_length = np.nan
    return {
        'episode_returns': np.asarray(episode_returns, dtype=np.float32),
        'episode_lengths': np.asarray(episode_lengths, dtype=np.float32),
        'success_fractions': np.asarray(success_fractions, dtype=np.float32),
        'safety_violations': np.asarray(safety_violations, dtype=np.float32),
        'successful_episode_length_mean': mean_length,
    }

def evaluate_mappo(actor, dirt_mode='random', episodes=200, seed=SEED + 20000):
    """以贪心动作评估MAPPO。"""
    parameter = next(actor.parameters())
    device = parameter.device
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    environment = CleanEnv(dirt_mode=dirt_mode)
    episode_returns, episode_lengths = [], []
    success_fractions, safety_violations, successful_lengths = [], [], []
    for episode in range(episodes):
        observations, _ = environment.reset(seed=seed + episode)
        returns = np.zeros(AGENTS, dtype=np.float32)
        final_info = None
        for step in range(MAX_STEPS):
            actions = []
            with torch.no_grad():
                for agent_id in range(AGENTS):
                    obs = observations[agent_id][None]
                    tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                    logits = actor(tensor, [agent_id])
                    action = torch.argmax(logits, dim=-1)
                    actions.append(int(action.item()))
            observations, rewards, terminated, truncated, final_info = environment.step(actions)
            returns += rewards
            if terminated or truncated:
                break
        cleaned_weight = (8 - len(environment.normal_dirt)) + 2 * (8 - len(environment.large_dirt))
        success = bool(final_info and final_info['termination_reason'] == 'all_dirt_cleaned')
        episode_returns.append(returns)
        # 失败回合长度记为缺失值。
        if success:
            successful_length = float(step + 1)
        else:
            successful_length = np.nan
        episode_lengths.append(successful_length)
        success_fractions.append(cleaned_weight / 24.0)
        safety_violations.append(float(final_info and final_info['termination_reason'] == 'safety_violation'))
        if success:
            successful_lengths.append(step + 1)
    if successful_lengths:
        mean_length = float(np.mean(successful_lengths))
    else:
        mean_length = np.nan
    return {
        'episode_returns': np.asarray(episode_returns, dtype=np.float32),
        'episode_lengths': np.asarray(episode_lengths, dtype=np.float32),
        'success_fractions': np.asarray(success_fractions, dtype=np.float32),
        'safety_violations': np.asarray(safety_violations, dtype=np.float32),
        'successful_episode_length_mean': mean_length,
    }

def evaluate_diagnostics(model, algorithm, dirt_mode='random', episodes=400, seed=SEED + 30000, gamma=0.95):
    """评估清扫表现、安全与协作指标。"""
    if algorithm == 'ippo':
        actors, critics = model
    elif algorithm == 'mappo':
        actor, critic = model
    else:
        raise ValueError("algorithm must be 'ippo' or 'mappo'.")
    if algorithm == 'ippo':
        parameter = next(actors[0].parameters())
    else:
        parameter = next(actor.parameters())
    device = parameter.device
    if episodes < 1:
        raise ValueError('episodes must be positive.')
    environment = CleanEnv(dirt_mode=dirt_mode)
    returns_history, lengths = [], []
    success_fractions, safety_flags, successful_lengths = [], [], []
    # 累计访问、贡献与策略差异。
    visit_map = np.zeros((GRID, GRID), dtype=np.int32)
    revisit_map = np.zeros_like(visit_map)
    contribution = np.zeros(AGENTS, dtype=np.float64)
    jsd_sum = np.zeros((AGENTS, AGENTS))
    jsd_count = 0
    for episode in range(episodes):
        observations, _ = environment.reset(seed=seed + episode)
        episode_return = np.zeros(AGENTS)
        final_info = None
        for step in range(MAX_STEPS):
            # 保存前一步地图，供访问统计使用。
            previous_positions = environment.agent_positions.copy()
            previous_owner = environment.explored_owner.copy()
            action_distributions, values, actions = [], [], []
            reference_obs = observations[0][None]
            reference = torch.as_tensor(reference_obs, dtype=torch.float32, device=device)
            with torch.no_grad():
                if algorithm == 'ippo':
                    for agent_id in range(AGENTS):
                        obs = observations[agent_id][None]
                        own = torch.as_tensor(obs, dtype=torch.float32, device=device)
                        logits = actors[agent_id](own)
                        value = critics[agent_id](own)
                        values.append(value.item())
                        # 用同一参考观测比较策略。
                        reference_logits = actors[agent_id](reference)
                        probs = torch.softmax(reference_logits, dim=-1)
                        probs = probs.cpu()
                        probs = probs.numpy()
                        action_distributions.append(probs[0])
                        action = torch.argmax(logits, dim=-1)
                        actions.append(int(action.item()))
                else:
                    # 共享联合状态价值。
                    state = environment.global_state()
                    state = state[None]
                    global_tensor = torch.as_tensor(state, dtype=torch.float32, device=device)
                    value = critic(global_tensor)
                    central_value = value.item()
                    for agent_id in range(AGENTS):
                        obs = observations[agent_id][None]
                        own = torch.as_tensor(obs, dtype=torch.float32, device=device)
                        logits = actor(own, [agent_id])
                        reference_logits = actor(reference, [agent_id])
                        values.append(central_value)
                        probs = torch.softmax(reference_logits, dim=-1)
                        probs = probs.cpu()
                        probs = probs.numpy()
                        action_distributions.append(probs[0])
                        action = torch.argmax(logits, dim=-1)
                        actions.append(int(action.item()))
            next_observations, rewards, terminated, truncated, final_info = environment.step(actions)
            episode_return += rewards
            next_values = np.zeros(AGENTS)
            if not (terminated or truncated):
                with torch.no_grad():
                    if algorithm == 'ippo':
                        for agent_id in range(AGENTS):
                            next_obs = next_observations[agent_id][None]
                            next_tensor = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
                            value = critics[agent_id](next_tensor)
                            next_values[agent_id] = value.item()
                    else:
                        state = environment.global_state()
                        state = state[None]
                        next_state = torch.as_tensor(state, dtype=torch.float32, device=device)
                        value = critic(next_state)
                        next_values[:] = value.item()
            # 累计TD贡献估计。
            contribution += rewards + gamma * next_values * float(not (terminated or truncated)) - np.asarray(values)
            # 累计访问与策略差异。
            for agent_id, action in enumerate(actions):
                if action != IDLE and not np.array_equal(previous_positions[agent_id], environment.agent_positions[agent_id]):
                    row, col = environment.agent_positions[agent_id]
                    visit_map[row, col] += 1
                    revisit_map[row, col] += int(previous_owner[row, col] >= 0)
            for first in range(AGENTS):
                for second in range(first + 1, AGENTS):
                    divergence = jsd(action_distributions[first], action_distributions[second])
                    jsd_sum[first, second] += divergence
                    jsd_sum[second, first] += divergence
            jsd_count += 1
            observations = next_observations
            if terminated or truncated:
                break
        cleaned_weight = (8 - len(environment.normal_dirt)) + 2 * (8 - len(environment.large_dirt))
        successful = bool(final_info and final_info['termination_reason'] == 'all_dirt_cleaned')
        returns_history.append(episode_return)
        # 失败回合长度记为缺失值。
        if successful:
            successful_length = float(step + 1)
        else:
            successful_length = np.nan
        lengths.append(successful_length)
        success_fractions.append(cleaned_weight / 24.0)
        safety_flags.append(float(final_info and final_info['termination_reason'] == 'safety_violation'))
        if successful:
            successful_lengths.append(step + 1)
    # 汇总重复覆盖率、贡献与策略差异。
    rcr_map = np.divide(revisit_map, visit_map, out=np.zeros_like(revisit_map, dtype=float), where=visit_map > 0)
    if successful_lengths:
        mean_length = float(np.mean(successful_lengths))
    else:
        mean_length = np.nan
    return {
        'episode_returns': np.asarray(returns_history, dtype=np.float32),
        'episode_lengths': np.asarray(lengths, dtype=np.float32),
        'successful_episode_length_mean': mean_length,
        'success_fractions': np.asarray(success_fractions, dtype=np.float32),
        'safety_violations': np.asarray(safety_flags, dtype=np.float32),
        'visit_map': visit_map,
        'revisit_map': revisit_map,
        'rcr_map': rcr_map,
        'global_rcr': revisit_map.sum() / max(visit_map.sum(), 1),
        'agent_contribution': contribution / max(episodes, 1),
        'jsd_matrix': jsd_sum / max(jsd_count, 1),
    }
