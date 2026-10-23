"""MAPPO经历收集与训练。"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .config import AGENTS, SEED
from .config import TrainConfig, resolve_device, seed_everything
from .environment import CleanEnv
from .networks import CentralCritic, SharedActor
from .rollout import Rollout
from .storage import save_checkpoint

def get_mappo_roll(env, actor, critic, rollout_length=200, gamma=0.95, gae_lambda=0.95, seed=None, device='cpu'):
    """收集共享策略经历，计算各自优势。"""
    if rollout_length < 2:
        raise ValueError('rollout_length must be at least 2.')
    buffers = []
    for _ in range(AGENTS):
        buffers.append(Rollout())
    if seed is None:
        rollout_seed = SEED
    else:
        rollout_seed = seed
    (observations, _) = env.reset(seed=rollout_seed)
    for _ in range(rollout_length):
        # 策略使用各自观测，价值网络使用联合状态。
        global_state = env.global_state()
        actions, log_probs = [], []
        with torch.no_grad():
            global_tensor = torch.as_tensor(global_state[None], dtype=torch.float32, device=device)
            value = critic(global_tensor)
            centralized_value = value.item()
            for agent_id in range(AGENTS):
                obs = observations[agent_id][None]
                observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                logits = actor(observation_tensor, [agent_id])
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                actions.append(action.item())
                log_prob = distribution.log_prob(action)
                log_probs.append(log_prob.item())
        # 各智能体共享价值估计。
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        for agent_id in range(AGENTS):
            buffers[agent_id].add(
                observations[agent_id], actions[agent_id], log_probs[agent_id],
                rewards[agent_id], centralized_value, done, global_state,
            )
        observations = next_observations
        if done:
            observations, _ = env.reset()
    with torch.no_grad():
        # 估计末步价值。
        state = env.global_state()
        tensor = torch.as_tensor(state[None], dtype=torch.float32, device=device)
        value = critic(tensor)
        final_value = value.item()
    for buffer in buffers:
        buffer.compute_gae(final_value, gamma, gae_lambda)
    return buffers

def update_mappo(actor, critic, actor_optimizer, critic_optimizer, buffers, cfg=None):
    """更新共享策略和集中式价值网络。"""
    if cfg is None:
        cfg = TrainConfig()
    cfg.validate()
    device = next(actor.parameters()).device
    metrics = {}
    for name in ('total_loss', 'actor_loss', 'critic_loss', 'entropy'):
        agent_metrics = []
        for _ in range(AGENTS):
            agent_metrics.append([])
        metrics[name] = agent_metrics
    for _ in range(cfg.epochs):
        for agent_id, buffer in enumerate(buffers):
            # 分别标准化各智能体的优势。
            data = buffer.as_tensors(device)
            advantages = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)
            count = len(data['actions'])
            order = torch.randperm(count, device=device)
            batches = order.split(cfg.batch_size)
            for indices in batches:
                observations = data['observations'][indices]
                agent_ids = torch.full((len(indices),), agent_id, dtype=torch.long, device=device)
                # 计算裁剪策略损失。
                logits = actor(observations, agent_ids)
                distribution = Categorical(logits=logits)
                new_log_probs = distribution.log_prob(data['actions'][indices])
                ratio = torch.exp(new_log_probs - data['log_probs'][indices])
                unclipped = ratio * advantages[indices]
                clipped = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages[indices]
                actor_loss = -torch.min(unclipped, clipped).mean()
                # 联合状态用于价值损失。
                new_values = critic(data['global_states'][indices])
                critic_loss = nn.functional.mse_loss(new_values, data['returns'][indices])
                entropy = distribution.entropy().mean()
                total_loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy
                # 反向传播并裁剪梯度。
                actor_optimizer.zero_grad()
                critic_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                actor_optimizer.step()
                critic_optimizer.step()
                metrics['total_loss'][agent_id].append(total_loss.item())
                metrics['actor_loss'][agent_id].append(actor_loss.item())
                metrics['critic_loss'][agent_id].append(critic_loss.item())
                metrics['entropy'][agent_id].append(entropy.item())
    mean_metrics = {}
    for (name, values) in metrics.items():
        agent_means = []
        for agent_values in values:
            agent_means.append(np.mean(agent_values))
        mean_metrics[name] = np.asarray(agent_means, dtype=np.float32)
    return mean_metrics

def train_mappo(dirt_mode, updates=20, rollout_length=200, seed=SEED, artifact_dir='artifacts/mappo', cfg=None):
    """训练MAPPO网络并保存结果。"""
    if cfg is None:
        cfg = TrainConfig()
    cfg.validate()
    if updates < 1 or rollout_length < 2:
        raise ValueError('updates >= 1 and rollout_length >= 2 are required.')
    device = resolve_device(cfg.device)
    seed_everything(seed)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    actor = SharedActor()
    critic = CentralCritic()
    actor = actor.to(device)
    critic = critic.to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)
    environment = CleanEnv(dirt_mode=dirt_mode)
    history = {}
    for name in ('total_loss', 'actor_loss', 'critic_loss', 'entropy', 'rollout_return'):
        history[name] = []
    best_return, best_update = -np.inf, -1
    checkpoint_path = artifact_dir / f'mappo_{dirt_mode}_best.npz'
    # 交替收集经历与更新网络。
    for update in range(updates):
        buffers = get_mappo_roll(
            environment, actor, critic, rollout_length, cfg.gamma,
            cfg.gae_lambda, seed + update, device=device,
        )
        report = update_mappo(actor, critic, actor_optimizer, critic_optimizer, buffers, cfg=cfg)
        for name in ('total_loss', 'actor_loss', 'critic_loss', 'entropy'):
            history[name].append(report[name])
        roll_returns = []
        for buffer in buffers:
            roll_returns.append(np.sum(buffer.rewards))
        agent_returns = np.asarray(roll_returns, dtype=np.float32)
        history['rollout_return'].append(agent_returns)
        # 按更新前回报选择更新后权重，并非独立验证最优。
        mean_return = float(agent_returns.mean())
        if mean_return > best_return:
            best_return, best_update = mean_return, update
            save_checkpoint((actor, critic), 'mappo', checkpoint_path, dirt_mode, update)
        print(f'MAPPO update {update + 1}/{updates}: mean rollout return={mean_return:.3f}', flush=True)
    environment.close()
    # 保存逐轮指标。
    metrics = {}
    for (name, values) in history.items():
        metrics[name] = np.asarray(values, dtype=np.float32)
    metrics_path = artifact_dir / f'mappo_{dirt_mode}_training_metrics.npz'
    last_path = artifact_dir / f'mappo_{dirt_mode}_last.npz'
    np.savez_compressed(metrics_path, **metrics)
    save_checkpoint((actor, critic), 'mappo', last_path, dirt_mode, updates - 1)
    print(f'MAPPO {dirt_mode}: best update={best_update + 1}, checkpoint={checkpoint_path}')
    return actor, critic, metrics, checkpoint_path
