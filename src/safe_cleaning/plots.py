"""训练与评估结果绘图。"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .config import AGENTS

def plot_training(metrics, path):
    """绘制单次训练曲线。"""
    names = ('total_loss', 'actor_loss', 'critic_loss', 'entropy', 'rollout_return')
    figure, axes = plt.subplots(1, len(names), figsize=(20, 4))
    for axis, name in zip(axes, names):
        values = metrics[name]
        for agent_id in range(AGENTS):
            axis.plot(np.arange(1, len(values) + 1), values[:, agent_id], label=f'Agent {agent_id + 1}')
        axis.set_title(name)
        axis.set_xlabel('Update')
        axis.grid(alpha=0.3)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)

def plot_performance(diagnostics, mode, artifact_dir):
    """比较两个算法的清扫表现。"""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    names = ('ippo', 'mappo')
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    # 比较团队回报与清扫率。
    for name in names:
        team_returns = diagnostics[name]['episode_returns'].sum(axis=1)
        axes[0, 0].plot(team_returns, label=name.upper(), alpha=0.8)
        axes[0, 1].plot(diagnostics[name]['success_fractions'], label=name.upper(), alpha=0.8)
    axes[0, 0].set_title(f'{mode}: collective team return by episode')
    axes[0, 1].set_title(f'{mode}: weighted cleaning fraction by episode')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    return_boxes = []
    # 比较各智能体的回报分布。
    for name in names:
        for agent_id in range(AGENTS):
            return_boxes.append(diagnostics[name]['episode_returns'][:, agent_id])
    return_labels = []
    for name in names:
        for agent_id in range(AGENTS):
            return_labels.append(f'{name.upper()[0]}{agent_id + 1}')
    axes[0, 2].boxplot(return_boxes, tick_labels=return_labels)
    axes[0, 2].set_title(f'{mode}: return per agent')
    success_boxes = []
    for name in names:
        success_boxes.append(diagnostics[name]['success_fractions'])
    axes[1, 0].boxplot(success_boxes, tick_labels=['IPPO', 'MAPPO'])
    axes[1, 0].set_title(f'{mode}: weighted cleaning fraction')
    safety_rates = []
    # 比较违规率与成功回合长度。
    for name in names:
        safety_rates.append(diagnostics[name]['safety_violations'].mean())
    axes[1, 1].bar(['IPPO', 'MAPPO'], safety_rates)
    axes[1, 1].set_title('Average safety violation rate')
    length_values = []
    for name in names:
        length_values.append(diagnostics[name]['successful_episode_length_mean'])
    axes[1, 2].bar(['IPPO', 'MAPPO'], np.nan_to_num(length_values, nan=0.0))
    axes[1, 2].set_title('Successful episode length')
    for index, value in enumerate(length_values):
        if not np.isfinite(value):
            label_height = 0.03
        else:
            label_height = value
        if not np.isfinite(value):
            label_text = 'N/A'
        else:
            label_text = f'{value:.1f}'
        axes[1, 2].text(index, label_height, label_text, ha='center')
    for axis in axes.flat:
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(artifact_dir / f'comparison_{mode}_performance.png', dpi=160)
    return figure

def plot_diag(diagnostics, mode, artifact_dir):
    """绘制协作诊断图。"""
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes[0, 0].bar(np.arange(AGENTS) - 0.18, diagnostics['ippo']['agent_contribution'], width=0.36, label='IPPO')
    axes[0, 0].bar(np.arange(AGENTS) + 0.18, diagnostics['mappo']['agent_contribution'], width=0.36, label='MAPPO')
    axes[0, 0].set_title('Agent contribution')
    axes[0, 0].legend()
    axes[0, 1].bar(['IPPO', 'MAPPO'], [diagnostics['ippo']['global_rcr'], diagnostics['mappo']['global_rcr']])
    axes[0, 1].set_title('Global RCR (lower is better)')
    mean_jsd = []
    for name in ('ippo', 'mappo'):
        mean_jsd.append(diagnostics[name]['jsd_matrix'][np.triu_indices(AGENTS, 1)].mean())
    axes[0, 2].bar(['IPPO', 'MAPPO'], mean_jsd)
    axes[0, 2].set_title('Mean pairwise JSD')
    # 添加指标说明。
    axes[0, 3].axis('off')
    axes[0, 3].text(0.05, 0.55, f'{mode} mode\nLower RCR = less duplicate travel\nHigher JSD = more distinct policies', va='center')
    axes[1, 0].imshow(diagnostics['ippo']['rcr_map'], vmin=0, vmax=1, cmap='magma')
    axes[1, 0].set_title(f'IPPO RCR map ({diagnostics["ippo"]["global_rcr"]:.3f})')
    axes[1, 1].imshow(diagnostics['mappo']['rcr_map'], vmin=0, vmax=1, cmap='magma')
    axes[1, 1].set_title(f'MAPPO RCR map ({diagnostics["mappo"]["global_rcr"]:.3f})')
    axes[1, 2].imshow(diagnostics['ippo']['jsd_matrix'], vmin=0, vmax=np.log(2), cmap='viridis')
    axes[1, 2].set_title('IPPO pairwise JSD')
    axes[1, 3].imshow(diagnostics['mappo']['jsd_matrix'], vmin=0, vmax=np.log(2), cmap='viridis')
    axes[1, 3].set_title('MAPPO pairwise JSD')
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(artifact_dir / f'comparison_{mode}_summary.png', dpi=160)
    return figure
