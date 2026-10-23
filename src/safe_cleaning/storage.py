"""模型与经历数据存取。"""
from pathlib import Path
import numpy as np
import torch
from .config import AGENTS
from .networks import ActorNet, CriticNet, SharedActor, CentralCritic

def read_arrays(path):
    """读取数值NPZ，禁用pickle。"""
    arrays = {}
    with np.load(path, allow_pickle=False) as archive:
        for name in archive.files:
            arrays[name] = archive[name].copy()
    return arrays

def save_checkpoint(model, algorithm, path, dirt_mode, update):
    """仅保存推理权重，不支持精确断点续训。"""
    data = {'dirt_mode': np.asarray(dirt_mode), 'update': np.asarray(update)}
    actors, critics = model
    if algorithm == 'ippo':
        for agent_id in range(AGENTS):
            for name, value in copy_state(actors[agent_id]).items():
                data[f'actor_{agent_id}_{name}'] = value.numpy()
            for name, value in copy_state(critics[agent_id]).items():
                data[f'critic_{agent_id}_{name}'] = value.numpy()
    elif algorithm == 'mappo':
        for name, value in copy_state(actors).items():
            data[f'actor_{name}'] = value.numpy()
        for name, value in copy_state(critics).items():
            data[f'critic_{name}'] = value.numpy()
    else:
        raise ValueError('Unknown algorithm.')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)

def copy_state(model):
    """复制模型参数到CPU。"""
    params_copy = {}
    for (name, parameter) in model.state_dict().items():
        param = parameter.detach()
        param = param.cpu()
        params_copy[name] = param.clone()
    return params_copy

def save_roll(buffers, file_path):
    """按智能体保存经历数组。"""
    archive = {}
    for agent_id, buffer in enumerate(buffers):
        prefix = f'agent_{agent_id + 1}'
        archive[f'{prefix}_observations'] = np.stack(buffer.observations)
        archive[f'{prefix}_actions'] = np.asarray(buffer.actions)
        archive[f'{prefix}_log_probs'] = np.asarray(buffer.log_probs)
        archive[f'{prefix}_rewards'] = np.asarray(buffer.rewards)
        archive[f'{prefix}_values'] = np.asarray(buffer.values)
        archive[f'{prefix}_advantages'] = buffer.advantages
        archive[f'{prefix}_returns'] = buffer.returns
    np.savez_compressed(file_path, **archive)

def load_ippo(checkpoint_path, device='cpu'):
    """加载IPPO网络，切换评估模式。"""
    checkpoint = read_arrays(checkpoint_path)
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
    for agent_id in range(AGENTS):
        actor_state = {}
        for name in actors[agent_id].state_dict():
            actor_state[name] = torch.as_tensor(checkpoint[f'actor_{agent_id}_{name}'], device=device)
        critic_state = {}
        for name in critics[agent_id].state_dict():
            critic_state[name] = torch.as_tensor(checkpoint[f'critic_{agent_id}_{name}'], device=device)
        actors[agent_id].load_state_dict(actor_state)
        critics[agent_id].load_state_dict(critic_state)
        actors[agent_id].eval()
        critics[agent_id].eval()
    return actors, critics

def load_mappo(checkpoint_path, device='cpu'):
    """加载MAPPO网络，切换评估模式。"""
    checkpoint = read_arrays(checkpoint_path)
    actor = SharedActor()
    critic = CentralCritic()
    actor = actor.to(device)
    critic = critic.to(device)
    actor_state = {}
    for name in actor.state_dict():
        actor_state[name] = torch.as_tensor(checkpoint[f'actor_{name}'], device=device)
    critic_state = {}
    for name in critic.state_dict():
        critic_state[name] = torch.as_tensor(checkpoint[f'critic_{name}'], device=device)
    actor.load_state_dict(actor_state)
    critic.load_state_dict(critic_state)
    actor.eval()
    critic.eval()
    return actor, critic
