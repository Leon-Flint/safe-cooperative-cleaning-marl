"""训练、评估与演示入口。"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from .config import SEED, TrainConfig
from .storage import load_ippo, load_mappo

def write_json(path, data):
    """保存JSON数据。"""
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    text = text + '\n'
    Path(path).write_text(text, encoding='utf-8')

def make_output(path):
    """创建输出目录，防止覆盖结果。"""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output directory is not empty: {path}. Choose a new directory.')
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_config(path):
    """读取并检查训练配置。"""
    raw = {}
    if path:
        raw = json.loads(Path(path).read_text(encoding='utf-8'))
    cfg = TrainConfig(**raw)
    cfg.validate()
    return cfg

def summarize(result):
    """汇总评估指标。"""
    lengths = result['episode_lengths']
    mean_length = result['successful_episode_length_mean']
    if not np.isfinite(mean_length):
        mean_length = None
    return {
        'episodes': len(lengths),
        'team_return_mean': float(result['episode_returns'].sum(axis=1).mean()),
        'completion_rate': float(np.isfinite(lengths).mean()),
        'weighted_cleaning_fraction': float(result['success_fractions'].mean()),
        'safety_violation_rate': float(result['safety_violations'].mean()),
        'successful_episode_length_mean': mean_length,
    }

def load_model(algorithm, checkpoint, device):
    """按算法加载模型。"""
    if algorithm == 'ippo':
        return load_ippo(checkpoint, device=device)
    return load_mappo(checkpoint, device=device)

def parser():
    """定义命令行参数。"""
    cli = argparse.ArgumentParser(description='协同清扫MARL：训练、评估、比较和演示')
    commands = cli.add_subparsers(dest='command', required=True)
    train = commands.add_parser('train', help='训练一个算法和一种垃圾模式')
    train.add_argument('--algorithm', choices=['ippo', 'mappo'], required=True)
    train.add_argument('--mode', choices=['fixed', 'random'], default='fixed')
    train.add_argument('--updates', type=int, default=20)
    train.add_argument('--steps', type=int, default=200)
    train.add_argument('--seed', type=int, default=SEED)
    train.add_argument('--config', help='JSON训练参数文件')
    train.add_argument('--output', required=True, help='新的或空的输出目录')
    evaluate = commands.add_parser('evaluate', help='从检查点评估，不重新训练')
    evaluate.add_argument('--algorithm', choices=['ippo', 'mappo'], required=True)
    evaluate.add_argument('--checkpoint', required=True)
    compare = commands.add_parser('compare', help='相同评估种子下比较两个检查点')
    compare.add_argument('--ippo-checkpoint', required=True)
    compare.add_argument('--mappo-checkpoint', required=True)
    for cmd in (evaluate, compare):
        cmd.add_argument('--mode', choices=['fixed', 'random'], default='fixed')
        cmd.add_argument('--episodes', type=int, default=200)
        cmd.add_argument('--seed', type=int, default=SEED + 30000)
        cmd.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
        cmd.add_argument('--gamma', type=float, default=0.95, help='贡献度诊断使用的折扣，应与训练一致')
        cmd.add_argument('--output', required=True)
    demo = commands.add_parser('demo', help='显示策略动作，不作为性能证明')
    demo.add_argument('--algorithm', choices=['ippo', 'mappo'], required=True)
    demo.add_argument('--checkpoint', required=True)
    demo.add_argument('--mode', choices=['fixed', 'random'], default='fixed')
    demo.add_argument('--steps', type=int, default=10)
    demo.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    return cli

def main(argv=None):
    """解析命令并处理输入错误。"""
    cli = parser()
    args = cli.parse_args(argv)
    try:
        run(args)
    except (ValueError, TypeError, FileNotFoundError) as error:
        cli.error(str(error))

def run(args):
    """执行训练、评估或演示。"""
    if args.command == 'demo':
        from .demo import show_ippo, show_mappo
        if args.steps < 1:
            raise ValueError('steps must be positive.')
        show = show_ippo if args.algorithm == 'ippo' else show_mappo
        show(args.checkpoint, args.mode, args.steps, device=args.device)
        return
    # 使用非交互绘图，不弹窗。
    import matplotlib
    matplotlib.use('Agg')
    from .evaluation import evaluate_diagnostics
    from .plots import plot_training, plot_performance, plot_diag
    from .ippo import train_ippo
    from .mappo import train_mappo
    import matplotlib.pyplot as plt
    if args.command == 'train':
        cfg = load_config(args.config)
        if args.updates < 1 or args.steps < 2:
            raise ValueError('updates >= 1 and steps >= 2 are required.')
        out = make_output(args.output)
        write_json(out / 'config.json', asdict(cfg))
        train = train_ippo if args.algorithm == 'ippo' else train_mappo
        _, _, metrics, _ = train(args.mode, args.updates, args.steps, args.seed, out, cfg)
        plot_training(metrics, out / 'training_curves.png')
        return
    from .config import resolve_device
    resolve_device(args.device)
    if args.episodes < 1 or not 0 <= args.gamma <= 1:
        raise ValueError('episodes must be positive and gamma must be in [0, 1].')
    paths = {}
    if args.command == 'evaluate':
        paths[args.algorithm] = args.checkpoint
    else:
        paths = {'ippo': args.ippo_checkpoint, 'mappo': args.mappo_checkpoint}
    for path in paths.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = make_output(args.output)
    results = {}
    for alg, path in paths.items():
        model = load_model(alg, path, args.device)
        results[alg] = evaluate_diagnostics(model, alg, args.mode, args.episodes, args.seed, args.gamma)
    summary = {}
    for alg, result in results.items():
        np.savez_compressed(out / f'{alg}_evaluation.npz', **result)
        summary[alg] = summarize(result)
    write_json(out / 'summary.json', summary)
    if len(results) == 2:
        mode = getattr(args, 'mode', 'fixed')
        for plot in (plot_performance, plot_diag):
            figure = plot(results, mode, out)
            plt.close(figure)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
