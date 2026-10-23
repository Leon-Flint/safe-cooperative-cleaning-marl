# Safety-Constrained Multi-Agent Reinforcement Learning for Cooperative Cleaning

一个研究多智能体如何在安全限制下协同完成清扫任务的强化学习项目。三个智能体需要在12×12网格中清理不同类型的垃圾，同时避免越界、碰撞和进入易碎区域。

## 项目简介

现实中的多机器人协作不仅要完成任务，也必须遵守安全规则。本项目从零构建了可训练、可评估、可视化的协同清扫环境，并在同一任务中实现和比较两种多智能体强化学习方法。

## 核心工作

- 自建3智能体协同清扫环境，支持固定和随机两种任务模式。
- 设计边界、碰撞和易碎区域等安全约束，以及探索、重复访问和清扫奖励。
- 从零实现PPO训练所需的策略网络、价值网络、经历记录和GAE优势估计。
- 实现IPPO：每个智能体分别学习自己的策略与价值网络。
- 实现MAPPO：执行时各智能体独立观察，训练时共享策略并使用联合状态价值网络。
- 实现训练、模型保存、评估、结果图表和策略演示的完整流程。

## 当前状态

当前版本已经完成环境、算法和评估流程的端到端实现，并通过基础功能测试。现有实验主要用于验证流程；由于尚未进行充分训练、多随机种子实验和基线比较，暂不对算法性能或协作效果作结论。

## 代码结构

| 文件或文件夹 | 用途 |
|---|---|
| `src/safe_cleaning/` | 项目主体代码 |
| `configs/default.json` | 默认训练参数 |
| `tests/` | 环境与训练流程的基础测试 |
| `pyproject.toml` | 安装信息与依赖配置 |

主体代码的分工：

- `environment.py`：清扫环境、安全规则和奖励反馈。
- `networks.py`：策略网络与价值网络。
- `ippo.py` / `mappo.py`：两种算法的训练流程。
- `evaluation.py` / `plots.py`：策略评估与结果图表。
- `main.py`：训练、评估、比较和演示的统一入口。

辅助函数由不同模块导入使用，不复制多份。源码和测试均不依赖 Notebook。

## 快速开始

需要 Python 3.11 或以上。在项目目录打开终端：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
```

Windows激活环境使用`.venv\Scripts\activate`。安装需要网络。
`.[dev]`表示安装本项目、运行依赖以及测试工具pytest。

不训练，只查看可用命令：

```bash
python -m safe_cleaning --help
```

## 训练与评估

训练一种算法：

```bash
python -m safe_cleaning train --algorithm ippo --mode fixed --updates 20 --steps 200 --seed 1 --config configs/default.json --output artifacts/ippo1
```

- `--algorithm`：选择`ippo`或`mappo`。
- `--mode`：选择固定垃圾`fixed`或随机位置`random`。
- `--updates`：更新轮数；`--steps`：每轮收集的环境步数。
- `--seed`：随机种子；`--output`：结果目录。

20轮只是流程验证的起始配置，不代表训练充分。训练完成后可评估最终模型：

```bash
python -m safe_cleaning evaluate --algorithm ippo --checkpoint artifacts/ippo1/ippo_fixed_last.npz --mode fixed --episodes 200 --seed 30001 --output artifacts/eval1
```

展示动作：

```bash
python -m safe_cleaning demo --algorithm ippo --checkpoint artifacts/ippo1/ippo_fixed_last.npz --steps 10
```

将训练命令中的算法改为`mappo`、输出目录改为`artifacts/mappo1`后，可以比较两种算法：

```bash
python -m safe_cleaning compare --ippo-checkpoint artifacts/ippo1/ippo_fixed_last.npz --mappo-checkpoint artifacts/mappo1/mappo_fixed_last.npz --mode fixed --episodes 200 --seed 30001 --output artifacts/compare1
```

对比会使用相同评估种子。输出目录必须是新的或空的，避免覆盖旧结果。运行生成的`artifacts/`不上传GitHub。

## 输出结果

| 输出 | 内容 |
|---|---|
| `*_last.npz` | 最后一次更新后的模型权重 |
| `summary.json` | 评估摘要，包括完成率、清扫比例和安全违规率 |
| `*.png` | 训练曲线或算法对比图 |
| `config.json` | 本次训练使用的参数 |

完整清扫率表示成功完成全部清扫任务的回合比例；加权清扫比例表示已清理垃圾的比例，两者并不相同。

