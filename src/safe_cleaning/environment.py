"""协同清扫环境。"""
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .config import ACTS, AGENTS, DELTAS, FIXED_LARGE, FIXED_NORMAL, FRAGILE, GRID, IDLE, MAX_STEPS, STARTS, Rewards
from .rendering import draw_env

def next_pos(current_positions, actions):
    """计算候选位置，不修改环境。"""
    current_positions = np.asarray(current_positions, dtype=np.int16)
    actions = np.asarray(actions, dtype=np.int64)
    if current_positions.shape != (AGENTS, 2):
        raise ValueError(f'Expected positions with shape ({AGENTS}, 2).')
    if actions.shape != (AGENTS,) or not np.all((0 <= actions) & (actions < len(ACTS))):
        raise ValueError(f'Expected {AGENTS} actions with values from 0 to 4.')
    return current_positions + DELTAS[actions]

def safety_check(current_positions, candidate_positions):
    """返回违规智能体及原因。"""
    violations = {}
    for agent_id in range(AGENTS):
        violations[agent_id] = set()
    # 检查越界与易碎品。
    for agent_id, (row, col) in enumerate(candidate_positions):
        if not (0 <= row < GRID and 0 <= col < GRID):
            violations[agent_id].add('boundary')
        elif (int(row), int(col)) in FRAGILE:
            violations[agent_id].add('fragile_object')
    # 检查同格与交换位置碰撞。
    for first_agent in range(AGENTS):
        for second_agent in range(first_agent + 1, AGENTS):
            same_target = np.array_equal(candidate_positions[first_agent], candidate_positions[second_agent])
            swapped_positions = (
                np.array_equal(candidate_positions[first_agent], current_positions[second_agent])
                and np.array_equal(candidate_positions[second_agent], current_positions[first_agent])
            )
            if same_target:
                violations[first_agent].add('same_cell_collision')
                violations[second_agent].add('same_cell_collision')
            if swapped_positions:
                violations[first_agent].add('position_swap_collision')
                violations[second_agent].add('position_swap_collision')
    unsafe_agents = {}
    for (agent_id, reasons) in violations.items():
        if reasons:
            unsafe_agents[agent_id] = reasons
    return unsafe_agents

class CleanEnv(gym.Env):
    """管理清扫环境和回合状态。"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    def __init__(self, dirt_mode='fixed', render_mode=None):
        super().__init__()
        if dirt_mode not in {'fixed', 'random'}:
            raise ValueError("dirt_mode must be either 'fixed' or 'random'.")
        self.dirt_mode = dirt_mode
        self.render_mode = render_mode
        self.rewards = Rewards()
        # 定义动作与观测规范。
        self.action_space = spaces.MultiDiscrete([len(ACTS)] * AGENTS)
        self.observation_space = spaces.Box(low=0, high=1, shape=(AGENTS, GRID, GRID, 5), dtype=np.uint8)
        # 建立静态地图。
        self.fragile_map = np.zeros((GRID, GRID), dtype=np.uint8)
        for fragile_cell in FRAGILE:
            self.fragile_map[fragile_cell] = 1
        # 初始化回合状态。
        self.agent_positions = STARTS.copy()
        self.normal_dirt = set()
        self.large_dirt = set()
        self.explored_owner = np.full((GRID, GRID), fill_value=-1, dtype=np.int8)
        self.step_count = 0
    def reset(self, *, seed=None, options=None):
        """重置回合，返回观测和信息。"""
        super().reset(seed=seed)
        if options is None:
            options = {}
        mode = options.get('dirt_mode', self.dirt_mode)
        if mode not in {'fixed', 'random'}:
            raise ValueError("options['dirt_mode'] must be 'fixed' or 'random'.")
        self.dirt_mode = mode
        self.agent_positions = STARTS.copy()
        self.step_count = 0
        self.explored_owner.fill(-1)
        # 起点记为已探索。
        for agent_id, position in enumerate(self.agent_positions):
            self.explored_owner[tuple(position)] = agent_id
        # 按模式放置垃圾。
        if mode == 'fixed':
            self.normal_dirt = set(FIXED_NORMAL)
            self.large_dirt = set(FIXED_LARGE)
        else:
            self.normal_dirt, self.large_dirt = self._random_dirt()
        observations = self._obs()
        info = self._info()
        return observations, info
    def step(self, actions):
        """执行联合动作，返回观测、奖励和结束信息。"""
        actions = np.asarray(actions, dtype=np.int64)
        if not self.action_space.contains(actions):
            raise ValueError(f'Expected {AGENTS} integer actions from 0 to 4.')
        # 计算时间与停留惩罚。
        self.step_count += 1
        rewards = np.full(AGENTS, self.rewards.time_step, dtype=np.float32)
        rewards[actions == IDLE] += self.rewards.idle
        candidates = next_pos(self.agent_positions, actions)
        violations = safety_check(self.agent_positions, candidates)
        # 违规时保持位置不变，并结束回合。
        if violations:
            violating_agents = tuple(sorted(violations))
            rewards[list(violating_agents)] += self.rewards.safety_violation
            observations = self._obs()
            info = self._info('safety_violation', violating_agents)
            reason_map = {}
            for (agent_id, reasons) in violations.items():
                reason_map[agent_id] = sorted(reasons)
            info['violation_reasons'] = reason_map
            return observations, rewards, False, True, info
        # 更新位置与行为奖励。
        self.agent_positions = candidates.astype(np.int8)
        for agent_id, position in enumerate(self.agent_positions):
            if actions[agent_id] == IDLE:
                continue
            cell = tuple(position)
            if self.explored_owner[cell] == -1:
                self.explored_owner[cell] = agent_id
                rewards[agent_id] += self.rewards.new_cell
            else:
                rewards[agent_id] += self.rewards.revisit
            if cell in self.normal_dirt:
                self.normal_dirt.remove(cell)
                rewards[agent_id] += self.rewards.normal_dirt
            elif cell in self.large_dirt:
                self.large_dirt.remove(cell)
                rewards[agent_id] += self.rewards.large_dirt
        # 判断完成或超时。
        terminated = not self.normal_dirt and not self.large_dirt
        if terminated:
            rewards += self.rewards.completion
        truncated = self.step_count >= MAX_STEPS and not terminated
        if terminated:
            reason = 'all_dirt_cleaned'
        elif truncated:
            reason = 'time_limit'
        else:
            reason = None
        observations = self._obs()
        return observations, rewards, terminated, truncated, self._info(reason, ())
    def global_state(self):
        """拼接联合观测，供集中式价值网络使用。"""
        obs = self._obs()
        obs = obs.reshape(-1)
        return obs.astype(np.float32)
    def _random_dirt(self):
        """随机放置两类垃圾，各八个。"""
        # 排除易碎品与起点。
        start_cells = set()
        for position in STARTS:
            start_cells.add(tuple(position))
        forbidden_cells = set(FRAGILE) | start_cells
        candidates = []
        for row in range(GRID):
            for col in range(GRID):
                if (row, col) not in forbidden_cells:
                    candidates.append((row, col))
        selected_indices = self.np_random.choice(len(candidates), size=16, replace=False)
        selected_cells = []
        for index in selected_indices:
            selected_cells.append(candidates[index])
        return set(selected_cells[:8]), set(selected_cells[8:])
    def _obs(self):
        """生成每个智能体的五通道观测。"""
        observations = np.zeros((AGENTS, GRID, GRID, 5), dtype=np.uint8)
        explored_map = (self.explored_owner >= 0).astype(np.uint8)
        for agent_id in range(AGENTS):
            # 通道依次为易碎品、自身、其他智能体、已探索区域、停靠点。
            observations[agent_id, :, :, 0] = self.fragile_map
            row, col = self.agent_positions[agent_id]
            observations[agent_id, row, col, 1] = 1
            for other_id, position in enumerate(self.agent_positions):
                if other_id != agent_id:
                    other_row, other_col = position
                    observations[agent_id, other_row, other_col, 2] = 1
            observations[agent_id, :, :, 3] = explored_map
            start_row, start_col = STARTS[agent_id]
            observations[agent_id, start_row, start_col, 4] = 1
        return observations
    def render(self):
        """按指定模式渲染地图。"""
        mode = self.render_mode
        if mode is None:
            mode = 'human'
        return draw_env(self, mode=mode)
    def _info(self, termination_reason=None, violating_agents=()):
        """返回回合诊断信息。"""
        return {
            'dirt_mode': self.dirt_mode,
            'step': self.step_count,
            'remaining_normal_dirt': len(self.normal_dirt),
            'remaining_large_dirt': len(self.large_dirt),
            'agent_positions_one_indexed': (self.agent_positions + 1).copy(),
            'termination_reason': termination_reason,
            'violating_agents': violating_agents,
        }
