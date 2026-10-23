"""地图、奖励与训练配置。"""
import numpy as np
import random
from dataclasses import dataclass
import math
import torch
SEED = 2026
GRID = 12
AGENTS = 3
MAX_STEPS = 50
# 动作对应行列偏移。
IDLE, UP, DOWN, LEFT, RIGHT = range(5)
ACTS = ('IDLE', 'UP', 'DOWN', 'LEFT', 'RIGHT')
DELTAS = np.array([(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)], dtype=np.int8)
STARTS = np.array([(1, 3), (12, 3), (7, 12)], dtype=np.int8) - 1
fragile_cells = []
for (row, col) in [(2, 2), (4, 4), (6, 6), (7, 2), (8, 4), (10, 6), (1, 7), (4, 7), (8, 7), (12, 7), (3, 11), (8, 9), (6, 10), (11, 11)]:
    fragile_cells.append((row - 1, col - 1))
FRAGILE = frozenset(fragile_cells)
large_cells = []
for (row, col) in [(2, 5), (3, 9), (4, 2), (6, 7), (8, 8), (9, 4), (10, 11), (12, 2)]:
    large_cells.append((row - 1, col - 1))
FIXED_LARGE = frozenset(large_cells)
normal_cells = []
for (row, col) in [(1, 4), (1, 10), (3, 2), (4, 8), (5, 9), (7, 3), (9, 11), (12, 6)]:
    normal_cells.append((row - 1, col - 1))
FIXED_NORMAL = frozenset(normal_cells)
OBS_SHAPE = (GRID, GRID, 5)
OBS_DIM = int(np.prod(OBS_SHAPE))
STATE_DIM = AGENTS * OBS_DIM
HIDDEN = (128, 128)

class Rewards:
    """清扫奖励与惩罚。"""
    def __init__(self):
        self.normal_dirt = 2.0
        self.large_dirt = 4.0
        self.new_cell = 2.0
        self.revisit = -1.0
        self.idle = -0.1
        self.time_step = -0.1
        self.safety_violation = -20.0
        self.completion = 20.0
@dataclass

class TrainConfig:
    """训练参数配置。"""
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.2
    actor_lr: float = 0.0003
    critic_lr: float = 0.001
    weight_decay: float = 0.00001
    epochs: int = 4
    batch_size: int = 100
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    device: str = 'cpu'
    def validate(self):
        """检查训练参数。"""
        for name in ('gamma', 'gae_lambda', 'clip'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f'{name} must be in [0, 1].')
        for name in ('actor_lr', 'critic_lr', 'weight_decay', 'value_coef', 'entropy_coef'):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and non-negative.')
        if self.actor_lr == 0 or self.critic_lr == 0:
            raise ValueError('Learning rates must be positive.')
        for name in ('epochs', 'batch_size'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer.')
        resolve_device(self.device)

def resolve_device(name):
    """选择设备，不可用时报错。"""
    if name == 'cpu':
        return torch.device('cpu')
    if name == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    if name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    raise ValueError(f'Device is unavailable or unsupported: {name}')

def seed_everything(seed):
    """设置随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
