"""经历存储与优势计算。"""
import numpy as np
import torch

class Rollout:
    """保存单个智能体的训练经历。"""
    def __init__(self):
        self.observations = []
        self.global_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.advantages = None
        self.returns = None
    def add(self, observation, action, log_prob, reward, value, done, global_state=None):
        """记录一步经历。"""
        obs = np.asarray(observation, dtype=np.float32)
        self.observations.append(obs.copy())
        if global_state is None:
            saved_state = None
        else:
            saved_state = np.asarray(global_state, dtype=np.float32)
            saved_state = saved_state.copy()
        self.global_states.append(saved_state)
        self.actions.append(int(action))
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.dones.append(bool(done))
    def compute_gae(self, last_value, gamma=0.95, gae_lambda=0.95):
        """计算优势和回报目标。"""
        if not self.rewards:
            raise ValueError('Cannot compute GAE for an empty rollout.')
        values = self.values + [float(last_value)]
        values = np.asarray(values, dtype=np.float32)
        rewards = np.asarray(self.rewards, dtype=np.float32)
        dones = np.asarray(self.dones, dtype=np.float32)
        advantages = np.zeros_like(rewards)
        gae = 0.0
        # 反向计算优势。
        for timestep in reversed(range(len(rewards))):
            non_terminal = 1.0 - dones[timestep]
            delta = rewards[timestep] + gamma * values[timestep + 1] * non_terminal - values[timestep]
            gae = delta + gamma * gae_lambda * non_terminal * gae
            advantages[timestep] = gae
        self.advantages = advantages
        self.returns = advantages + values[:-1]
    def as_tensors(self, device='cpu'):
        """转换为训练张量。"""
        if not self.observations:
            raise ValueError('Cannot convert an empty rollout.')
        if self.advantages is None or self.returns is None:
            raise RuntimeError('Call compute_gae before converting the rollout to tensors.')
        obs = np.stack(self.observations)
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        payload = {
            'observations': obs,
            'actions': torch.as_tensor(self.actions, dtype=torch.long, device=device),
            'log_probs': torch.as_tensor(self.log_probs, dtype=torch.float32, device=device),
            'advantages': torch.as_tensor(self.advantages, dtype=torch.float32, device=device),
            'returns': torch.as_tensor(self.returns, dtype=torch.float32, device=device),
            'values': torch.as_tensor(self.values, dtype=torch.float32, device=device),
        }
        # 补充联合状态。
        if self.global_states[0] is not None:
            states = np.stack(self.global_states)
            states = torch.as_tensor(states, dtype=torch.float32, device=device)
            payload['global_states'] = states
        return payload
