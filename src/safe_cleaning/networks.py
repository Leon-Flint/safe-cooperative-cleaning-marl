"""IPPO与MAPPO网络定义。"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import ACTS, AGENTS, HIDDEN, OBS_DIM, STATE_DIM

def mlp(input_dim, output_dim, hidden_dims=HIDDEN):
    """构建多层感知机。"""
    layers = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend([nn.Linear(previous_dim, hidden_dim), nn.ReLU()])
        previous_dim = hidden_dim
    # 构建输出层。
    layers.append(nn.Linear(previous_dim, output_dim))
    return nn.Sequential(*layers)

class ActorNet(nn.Module):
    """IPPO独立策略，根据观测计算动作分数。"""
    def __init__(self, observation_dim=OBS_DIM, action_dim=len(ACTS)):
        super().__init__()
        self.network = mlp(observation_dim, action_dim)
    def forward(self, observations):
        batch = observations.shape[0]
        flat = observations.reshape(batch, -1)
        flat = flat.float()
        return self.network(flat)

class CriticNet(nn.Module):
    """IPPO价值网络，根据观测估计价值。"""
    def __init__(self, observation_dim=OBS_DIM):
        super().__init__()
        self.network = mlp(observation_dim, 1)
    def forward(self, observations):
        batch = observations.shape[0]
        flat = observations.reshape(batch, -1)
        flat = flat.float()
        values = self.network(flat)
        return values.squeeze(-1)

class CentralCritic(nn.Module):
    """MAPPO集中式价值网络，根据联合观测估计价值。"""
    def __init__(self, state_dim=STATE_DIM):
        super().__init__()
        self.network = mlp(state_dim, 1)
    def forward(self, global_states):
        batch = global_states.shape[0]
        flat = global_states.reshape(batch, -1)
        flat = flat.float()
        values = self.network(flat)
        return values.squeeze(-1)

class SharedActor(nn.Module):
    """MAPPO共享策略，根据观测和编号计算动作分数。"""
    def __init__(self, observation_dim=OBS_DIM, n_agents=AGENTS, action_dim=len(ACTS)):
        super().__init__()
        self.n_agents = n_agents
        self.network = mlp(observation_dim + n_agents, action_dim)
    def forward(self, observations, agent_ids):
        batch = observations.shape[0]
        flat_observations = observations.reshape(batch, -1)
        flat_observations = flat_observations.float()
        # 编码智能体身份。
        agent_ids = torch.as_tensor(agent_ids, dtype=torch.long, device=observations.device)
        agent_ids = agent_ids.reshape(-1)
        one_hot_ids = F.one_hot(agent_ids, num_classes=self.n_agents)
        one_hot_ids = one_hot_ids.float()
        # 融合观测与身份。
        inputs = torch.cat([flat_observations, one_hot_ids], dim=-1)
        return self.network(inputs)
