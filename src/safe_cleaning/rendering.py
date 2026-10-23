"""清扫地图绘制。"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from .config import AGENTS, GRID, STARTS

def draw_env(env, mode='human'):
    """显示地图或返回图像数组。"""
    from matplotlib.patches import Patch
    if mode not in {'human', 'rgb_array'}:
        raise ValueError("mode must be 'human' or 'rgb_array'.")
    # 按图层填充显示编号。
    display_map = np.zeros((GRID, GRID), dtype=np.int8)
    for agent_id in range(AGENTS):
        display_map[env.explored_owner == agent_id] = 1 + agent_id
    display_map[env.fragile_map.astype(bool)] = 4
    for cell in env.normal_dirt:
        display_map[cell] = 5
    for cell in env.large_dirt:
        display_map[cell] = 6
    # 最后绘制智能体，避免遮挡。
    for agent_id, position in enumerate(env.agent_positions):
        display_map[tuple(position)] = 7 + agent_id
    colours = [
        '#ffffff', '#fff9c4', '#c8e6c9', '#bbdefb', '#f44336',
        '#bdbdbd', '#616161', '#ffeb3b', '#4caf50', '#2196f3',
    ]
    figure, axis = plt.subplots(figsize=(8, 7))
    axis.imshow(display_map, cmap=ListedColormap(colours), vmin=0, vmax=9, interpolation='none')
    # 添加网格与坐标。
    axis.set_xticks(np.arange(-0.5, GRID, 1), minor=True)
    axis.set_yticks(np.arange(-0.5, GRID, 1), minor=True)
    axis.grid(which='minor', color='#bdbdbd', linewidth=0.8)
    axis.tick_params(which='minor', bottom=False, left=False)
    axis.set_xticks(np.arange(GRID), labels=np.arange(1, GRID + 1))
    axis.set_yticks(np.arange(GRID), labels=np.arange(1, GRID + 1))
    axis.set_xlabel('Column')
    axis.set_ylabel('Row')
    axis.set_title(f'Collaborative Cleaning | step {env.step_count} | remaining dirt: {len(env.normal_dirt) + len(env.large_dirt)}')
    for start in STARTS:
        axis.text(start[1], start[0], '☆', ha='center', va='center', fontsize=16, color='black')
    # 添加图例。
    legend_items = [
        Patch(facecolor='#f44336', label='Fragile object'),
        Patch(facecolor='#bdbdbd', label='Normal dirt'),
        Patch(facecolor='#616161', label='Large dirt'),
        Patch(facecolor='#ffeb3b', label='Agent 1'),
        Patch(facecolor='#4caf50', label='Agent 2'),
        Patch(facecolor='#2196f3', label='Agent 3'),
    ]
    axis.legend(handles=legend_items, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    # 预留图例空间。
    figure.subplots_adjust(right=0.78)
    # 返回图像数组或直接显示。
    if mode == 'rgb_array':
        figure.canvas.draw()
        rgba = figure.canvas.buffer_rgba()
        rgba = np.asarray(rgba)
        rgb_image = rgba[..., :3].copy()
        plt.close(figure)
        return rgb_image
    plt.show()
    return None
