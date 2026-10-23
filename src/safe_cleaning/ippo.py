"""IPPO经历收集与训练。"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from .config import AGENTS, SEED
from .config import TrainConfig, resolve_device, seed_everything
from .environment import CleanEnv
from .networks import ActorNet, CriticNet
from .rollout import Rollout
from .storage import save_checkpoint

def get_ippo_roll(env, actors, critics, rollout_length=200, gamma=0.95, gae_lambda=0.95, seed=None, device='cpu'):
    """收集独立策略经历，计算各自优势。"""
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
    # 采样动作并记录经历。
    for _ in range(rollout_length):
        actions, log_probs, values = [], [], []
        with torch.no_grad():
            for agent_id in range(AGENTS):
                obs = observations[agent_id][None]
                observation_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
                logits = actors[agent_id](observation_tensor)
                distribution = Categorical(logits=logits)
                action = distribution.sample()
                actions.append(action.item())
                log_prob = distribution.log_prob(action)
                log_probs.append(log_prob.item())
                value = critics[agent_id](observation_tensor)
                values.append(value.item())
        next_observations, rewards, terminated, truncated, _ = env.step(actions)
        done = terminated or truncated
        for agent_id in range(AGENTS):
            buffers[agent_id].add(
                observations[agent_id], actions[agent_id], log_probs[agent_id], rewards[agent_id], values[agent_id], done
            )
        observations = next_observations
        if done:
            observations, _ = env.reset()
    # 估计末步价值，计算优势。
    with torch.no_grad():
        for agent_id in range(AGENTS):
            obs = observations[agent_id][None]
            final_observation = torch.as_tensor(obs, dtype=torch.float32, device=device)
            value = critics[agent_id](final_observation)
            buffers[agent_id].compute_gae(value.item(), gamma, gae_lambda)
    return buffers

def ppo_update(actor, critic, actor_optimizer, critic_optimizer, buffer, cfg=None):
    """用PPO更新独立策略和价值网络。"""
    if cfg is None:
        cfg = TrainConfig()
    cfg.validate()
    device = next(actor.parameters()).device
    # 标准化优势。
    data = buffer.as_tensors(device)
    advantages = (data['advantages'] - data['advantages'].mean()) / (data['advantages'].std() + 1e-8)
    metrics = {'total_loss': [], 'actor_loss': [], 'critic_loss': [], 'entropy': []}
    # 打乱数据，分批更新。
    for _ in range(cfg.epochs):
        count = len(data['actions'])
        order = torch.randperm(count, device=device)
        batches = order.split(cfg.batch_size)
        for indices in batches:
            observations = data['observations'][indices]
            logits = actor(observations)
            distribution = Categorical(logits=logits)
            new_log_probs = distribution.log_prob(data['actions'][indices])
            probability_ratio = torch.exp(new_log_probs - data['log_probs'][indices])
            # 计算裁剪策略损失。
            unclipped_objective = probability_ratio * advantages[indices]
            clipped_objective = torch.clamp(probability_ratio, 1 - cfg.clip, 1 + cfg.clip) * advantages[indices]
            actor_loss = -torch.min(unclipped_objective, clipped_objective).mean()
            # 合并价值损失与熵正则。
            new_values = critic(observations)
            critic_loss = nn.functional.mse_loss(new_values, data['returns'][indices])
            entropy = distribution.entropy().mean()
            total_loss = actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy
            # 反向传播并裁剪梯度。
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_norm=0.5)
            actor_optimizer.step()
            critic_optimizer.step()
            metrics['total_loss'].append(total_loss.item())
            metrics['actor_loss'].append(actor_loss.item())
            metrics['critic_loss'].append(critic_loss.item())
            metrics['entropy'].append(entropy.item())
    summary = {}
    for (name, values) in metrics.items():
        summary[name] = float(np.mean(values))
    return summary

def train_ippo(dirt_mode, updates=20, rollout_length=200, seed=SEED, artifact_dir='artifacts/ippo', cfg=None):
    """训练各智能体的独立网络。"""
    if cfg is None:
        cfg = TrainConfig()
    cfg.validate()
    if updates < 1 or rollout_length < 2:
        raise ValueError('updates >= 1 and rollout_length >= 2 are required.')
    device = resolve_device(cfg.device)
    seed_everything(seed)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    # 创建独立网络与优化器。
    actors = []
    for _ in range(AGENTS):
        actor = ActorNet()
        actor = actor.to(device)
        actors.append(actor)
    critics = []
    for _ in range(AGENTS):
        critic = CriticNet()
        critic = critic.to(device)
        critics.append(critic)
    actor_optimizers = []
    for actor in actors:
        optimizer = optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        actor_optimizers.append(optimizer)
    critic_optimizers = []
    for critic in critics:
        optimizer = optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)
        critic_optimizers.append(optimizer)
    environment = CleanEnv(dirt_mode=dirt_mode)
    metric_history = {}
    for name in ('total_loss', 'actor_loss', 'critic_loss', 'entropy', 'rollout_return'):
        metric_history[name] = []
    best_team_return, best_update = -np.inf, -1
    checkpoint_path = artifact_dir / f'ippo_{dirt_mode}_best.npz'
    # 交替收集经历与更新网络。
    for update in range(updates):
        buffers = get_ippo_roll(
            environment, actors, critics, rollout_length=rollout_length,
            gamma=cfg.gamma, gae_lambda=cfg.gae_lambda, seed=seed + update, device=device,
        )
        reports = []
        for agent_id in range(AGENTS):
            report = ppo_update(
                actors[agent_id], critics[agent_id], actor_optimizers[agent_id],
                critic_optimizers[agent_id], buffers[agent_id], cfg=cfg,
            )
            reports.append(report)
        for name in ('total_loss', 'actor_loss', 'critic_loss', 'entropy'):
            agent_losses = []
            for report in reports:
                agent_losses.append(report[name])
            metric_history[name].append(agent_losses)
        roll_returns = []
        for buffer in buffers:
            roll_returns.append(np.sum(buffer.rewards))
        agent_returns = np.asarray(roll_returns, dtype=np.float32)
        metric_history['rollout_return'].append(agent_returns)
        # 按更新前回报选择更新后权重，并非独立验证最优。
        team_return = float(agent_returns.mean())
        if team_return > best_team_return:
            best_team_return, best_update = team_return, update
            save_checkpoint((actors, critics), 'ippo', checkpoint_path, dirt_mode, update)
        print(f'IPPO update {update + 1}/{updates}: mean rollout return={team_return:.3f}', flush=True)
    environment.close()
    # 保存逐轮指标与最终权重。
    metrics = {}
    for (name, values) in metric_history.items():
        metrics[name] = np.asarray(values, dtype=np.float32)
    metrics_path = artifact_dir / f'ippo_{dirt_mode}_training_metrics.npz'
    last_path = artifact_dir / f'ippo_{dirt_mode}_last.npz'
    np.savez_compressed(metrics_path, **metrics)
    save_checkpoint((actors, critics), 'ippo', last_path, dirt_mode, updates - 1)
    print(f'IPPO {dirt_mode}: best update={best_update + 1}, checkpoint={checkpoint_path}')
    return actors, critics, metrics, checkpoint_path
