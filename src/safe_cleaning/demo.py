"""已训练策略的运行演示。"""
import torch
from .config import AGENTS, SEED
from .environment import CleanEnv
from .storage import load_ippo, load_mappo

def show_ippo(checkpoint_path, dirt_mode='fixed', steps=10, render_mode='human', device='cpu'):
    """加载并演示IPPO策略。"""
    actors, _ = load_ippo(checkpoint_path, device=device)
    environment = CleanEnv(dirt_mode=dirt_mode, render_mode=render_mode)
    observations, _ = environment.reset(seed=SEED + 50000)
    for step in range(steps):
        actions = []
        with torch.no_grad():
            for agent_id, actor in enumerate(actors):
                obs = observations[agent_id][None]
                observation = torch.as_tensor(obs, dtype=torch.float32, device=device)
                logits = actor(observation)
                action = torch.argmax(logits, dim=-1)
                actions.append(int(action.item()))
        observations, rewards, terminated, truncated, info = environment.step(actions)
        environment.render()
        if terminated or truncated:
            break
    return environment

def show_mappo(checkpoint_path, dirt_mode='fixed', steps=10, render_mode='human', device='cpu'):
    """加载并演示MAPPO策略。"""
    actor, _ = load_mappo(checkpoint_path, device=device)
    environment = CleanEnv(dirt_mode=dirt_mode, render_mode=render_mode)
    observations, _ = environment.reset(seed=SEED + 60000)
    for step in range(steps):
        actions = []
        with torch.no_grad():
            for agent_id in range(AGENTS):
                obs = observations[agent_id][None]
                observation = torch.as_tensor(obs, dtype=torch.float32, device=device)
                logits = actor(observation, [agent_id])
                action = torch.argmax(logits, dim=-1)
                actions.append(int(action.item()))
        observations, rewards, terminated, truncated, info = environment.step(actions)
        environment.render()
        if terminated or truncated:
            break
    return environment
