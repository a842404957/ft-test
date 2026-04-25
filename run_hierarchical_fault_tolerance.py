#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
运行三级分层容错策略仿真
测试并比较不同容错策略的效果
"""

import os
import sys
import torch
import argparse
import csv
import json
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simulator.Fault_Tolerance.fault_tolerance_simulation import FaultToleranceSimulator
from simulator.Fault_Tolerance.report_generator import ReportGenerator
from model import Vgg16, Res18, Res50, WRN
from torchvision import transforms, datasets


DEFAULT_CONFIG_FILE = 'fault_tolerance_config_high_fault_rate.json'


def load_model_config(config_file=None):
    config_model_name = None
    num_classes = 10
    if config_file and os.path.exists(config_file):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            config_model_name = config.get('model', {}).get('name')
            num_classes = config.get('model', {}).get('num_classes', 10)
        except Exception as e:
            print(f"  ⚠️ 读取配置文件失败: {e}，使用默认参数")
    return config_model_name, num_classes


def resolve_runtime_model_name(cli_model_name, config_file=None):
    config_model_name, num_classes = load_model_config(config_file)
    runtime_model_name = cli_model_name or config_model_name or 'Vgg16'
    if config_model_name:
        if config_model_name != runtime_model_name:
            print(f"  ⚠️ 配置文件 model={config_model_name} 与命令行 model={runtime_model_name} 不一致，使用命令行参数")
        else:
            print(f"  📄 从配置文件读取: model={config_model_name}, num_classes={num_classes}")
    return runtime_model_name, num_classes


def load_model(model_name, translate_name='ft_group_cluster_translate', device='cuda', config_file=None, data_dir='.', num_classes=None):
    """加载训练好的模型"""
    if num_classes is None:
        model_name, num_classes = resolve_runtime_model_name(model_name, config_file)

    if model_name == 'Vgg16':
        model = Vgg16(num_classes=num_classes)
    elif model_name == 'Res18':
        model = Res18(num_classes=num_classes)
    elif model_name == 'Res50':
        model = Res50(num_classes=num_classes)
    elif model_name == 'WRN':
        model = WRN(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 加载模型参数
    model_file = Path(data_dir) / f'model_{model_name}_{translate_name}_after_translate_parameters.pth'

    if not model_file.exists():
        print(f"❌ 模型文件未找到: {model_file}")
        print("  请先运行 main.py 进行模型训练")
        return None

    model.load_state_dict(torch.load(model_file, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"✅ 模型加载成功\n")
    return model


def load_test_data(batch_size=128, config_file=None):
    """加载测试数据"""
    print("📦 加载测试数据...")

    # 🔧 从配置文件读取批大小
    if config_file and os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                config_batch_size = config.get('simulation', {}).get('batch_size', batch_size)
                if config_batch_size != batch_size:
                    print(f"  📄 从配置文件读取批大小: {batch_size} → {config_batch_size}")
                    batch_size = config_batch_size
        except Exception as e:
            print(f"  ⚠️ 读取配置文件失败: {e}，使用默认批大小 {batch_size}")

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_dataset = datasets.CIFAR10('./cifar10_data', train=False,
                                   transform=transform_test, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                             shuffle=False)

    print(f"✅ 测试数据加载完成: {len(test_dataset)} 样本，批大小: {batch_size}\n")
    return test_loader


def _build_runtime_config(base_config_file=None, config_overrides=None, report_output_dir=None,
                          runtime_model_name=None, fault_seed=None):
    base_config = {}
    if base_config_file and os.path.exists(base_config_file):
        with open(base_config_file, 'r', encoding='utf-8') as f:
            base_config = json.load(f)

    if not base_config:
        final_config = {
            'model': {'name': 'Unknown', 'num_classes': 10},
            'hierarchical_fault_tolerance': {
                'enabled': True,
                'level1': {
                    'name': 'redundancy_group',
                    'enabled': True,
                    'description': '冗余组内替换（主策略）'
                },
                'level2': {
                    'name': 'nearest_pattern',
                    'enabled': True,
                    'description': '相似模式近似替换',
                    'similarity_threshold': 0.85,
                    'k_nearest': 3,
                    'use_weighted_average': True,
                    'fallback_latency_ns': 15,
                    'fallback_energy_pj': 75,
                },
                'level3': {
                    'name': 'adaptive_masking',
                    'enabled': True,
                    'description': '自适应屏蔽策略',
                    'masking_strategy': 'weighted_neighbors',
                    'neighbor_radius': 2,
                    'fallback_latency_ns': 20,
                    'fallback_energy_pj': 100,
                },
            },
            'report': {'output_dir': './fault_tolerance_results'}
        }
    else:
        final_config = json.loads(json.dumps(base_config))

    if config_overrides:
        if 'hierarchical_fault_tolerance' not in final_config:
            final_config['hierarchical_fault_tolerance'] = {}
        for key, value in config_overrides.items():
            parts = key.split('.')
            current = final_config['hierarchical_fault_tolerance']
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value

    if report_output_dir:
        final_config.setdefault('report', {})
        final_config['report']['output_dir'] = report_output_dir

    if runtime_model_name:
        final_config.setdefault('model', {})
        final_config['model']['name'] = runtime_model_name
    if fault_seed is not None:
        final_config.setdefault('fault_injection', {})
        final_config['fault_injection']['random_seed'] = int(fault_seed)

    return final_config, base_config


def resolve_levels_overrides(levels: str):
    normalized = (levels or 'all').strip().lower()
    if normalized == 'level1':
        return {
            'level1.enabled': True,
            'level2.enabled': False,
            'level3.enabled': False,
        }
    if normalized in ('level1_level2', 'level1+level2'):
        return {
            'level1.enabled': True,
            'level2.enabled': True,
            'level3.enabled': False,
        }
    return {
        'level1.enabled': True,
        'level2.enabled': True,
        'level3.enabled': True,
    }


def resolve_level1_selection_overrides(args):
    overrides = {}
    if getattr(args, 'level1_selection', None):
        overrides['level1.selection'] = args.level1_selection
    if getattr(args, 'level1_topk', None) is not None:
        overrides['level1.topk'] = int(args.level1_topk)
    if getattr(args, 'level1_max_expected_error', None) is not None:
        overrides['level1.max_expected_error'] = float(args.level1_max_expected_error)
    if getattr(args, 'level1_min_expected_improvement', None) is not None:
        overrides['level1.min_expected_improvement'] = float(args.level1_min_expected_improvement)
    if getattr(args, 'level1_critical_layer_config', None):
        overrides['level1.critical_layer_config'] = args.level1_critical_layer_config
    if getattr(args, 'level1_cache_max_group_size', None) is not None:
        overrides['level1.cache_max_group_size'] = int(args.level1_cache_max_group_size)
    if getattr(args, 'level1_cache_critical_layers_only', False):
        overrides['level1.cache_critical_layers_only'] = True
    return overrides


def _export_compare_summary(all_results, output_dir):
    if not output_dir:
        return []

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for item in all_results:
        metrics = item['results']
        reliability = metrics.get('reliability', {})
        accuracy = metrics.get('accuracy', {})
        hierarchical = reliability.get('hierarchical_correction', {})
        repair_quality = reliability.get('repair_quality', {})
        summary_rows.append({
            'strategy': item['name'],
            'fault_correction_rate': reliability.get('fault_correction_rate', 0.0),
            'ft_accuracy': accuracy.get('ft_accuracy', 0.0),
            'accuracy_recovery_rate': accuracy.get('accuracy_recovery_rate', 0.0),
            'level1_corrections': hierarchical.get('level1_corrections', 0),
            'level2_corrections': hierarchical.get('level2_corrections', 0),
            'level3_corrections': hierarchical.get('level3_corrections', 0),
            'level1_failed_singleton': hierarchical.get('level1_failed_singleton', 0),
            'level1_zero_scale_failed': hierarchical.get('level1_zero_scale_failed', 0),
            'repair_mode': hierarchical.get('repair_mode', 'normal'),
            'level1_improved_rate': repair_quality.get('level1', {}).get('improved_rate', 0.0),
            'level2_improved_rate': repair_quality.get('level2', {}).get('improved_rate', 0.0),
        })

    json_path = output_path / 'comparison_summary.json'
    csv_path = output_path / 'comparison_summary.csv'
    md_path = output_path / 'comparison_summary.md'

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(summary_rows, handle, indent=2, ensure_ascii=False)

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('# Comparison Summary\n\n')
        handle.write('| strategy | fault_correction_rate | ft_accuracy | accuracy_recovery_rate | level1 | level2 | level3 | failed_singleton | zero_scale_failed | repair_mode | level1_improved_rate | level2_improved_rate |\n')
        handle.write('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for row in summary_rows:
            handle.write(
                '| {strategy} | {fault_correction_rate:.6f} | {ft_accuracy:.6f} | '
                '{accuracy_recovery_rate:.6f} | {level1_corrections} | '
                '{level2_corrections} | {level3_corrections} | {level1_failed_singleton} | {level1_zero_scale_failed} | '
                '{repair_mode} | {level1_improved_rate:.6f} | {level2_improved_rate:.6f} |\n'.format(**row)
            )

    return [str(json_path), str(csv_path), str(md_path)]


def run_simulation_with_config(model, test_loader, config_name, config_overrides=None,
                                base_config_file=None, num_samples=1000, model_name=None,
                                translate_name='ft_group_cluster_translate',
                                report_output_dir=None, data_dir='.', fault_seed=None):
    """运行带有特定配置的仿真"""
    print("=" * 80)
    print(f"🚀 运行仿真配置: {config_name}")
    print("=" * 80)

    import tempfile
    final_config, base_config = _build_runtime_config(
        base_config_file=base_config_file,
        config_overrides=config_overrides,
        report_output_dir=report_output_dir,
        runtime_model_name=model_name,
        fault_seed=fault_seed,
    )

    # 决定使用哪个配置文件
    config_file_path = None
    temp_file_path = None  # 临时文件路径（需要清理）

    if config_overrides or report_output_dir or fault_seed is not None or not (base_config_file and os.path.exists(base_config_file)):
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(final_config, f, indent=2)
            config_file_path = f.name
            temp_file_path = config_file_path  # 标记这是临时文件

        if config_overrides:
            print(f"📝 应用配置覆盖:")
            for key, value in config_overrides.items():
                print(f"  {key}: {value}")
        if report_output_dir:
            print(f"🗂️  输出目录覆盖: {report_output_dir}")
        if fault_seed is not None:
            print(f"🎲 故障随机种子覆盖: {fault_seed}")
        print()
    elif base_config_file and os.path.exists(base_config_file):
        # 如果没有覆盖配置，直接使用原始配置文件
        config_file_path = base_config_file
        final_config = base_config  # 使用原始配置
        # temp_file_path 保持为 None，不需要清理

    # 创建仿真器（使用配置文件）
    # 优先使用传入的 model_name，否则从配置中读取
    final_model_name = model_name if model_name else final_config.get('model', {}).get('name', 'Unknown')
    simulator = FaultToleranceSimulator(
        model=model,
        model_name=final_model_name,
        translate_name=translate_name,
        config_file=config_file_path,
        data_dir=data_dir
    )
    
    # 只清理临时配置文件（不删除用户提供的配置文件）
    if temp_file_path:
        os.unlink(temp_file_path)
    
    # 运行仿真
    results = simulator.run_simulation(
        test_loader=test_loader,
        num_samples=num_samples
    )
    
    return results


def compare_strategies(config_file=None, num_samples=1000, translate_name='ft_group_cluster_translate',
                       model_name='Vgg16', report_output_dir=None, data_dir='.',
                       repair_mode='normal', levels='all', fault_seed=None, level1_overrides=None):
    """比较不同容错策略的效果"""
    print("\n" + "=" * 80)
    print("📊 三级容错策略对比实验")
    print("=" * 80 + "\n")
    
    # 检查CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🔧 使用设备: {device}\n")
    
    # 显示使用的配置文件
    if config_file:
        if os.path.exists(config_file):
            print(f"📄 使用配置文件: {config_file}\n")
        else:
            print(f"⚠️  配置文件不存在: {config_file}")
            print(f"   将使用默认配置\n")
            config_file = None
    
    # 🔧 从配置文件读取模型名称（或使用默认值）
    runtime_model_name, runtime_num_classes = resolve_runtime_model_name(model_name, config_file)

    # 加载模型和数据（使用配置文件中的模型名称）
    model = load_model(
        model_name=runtime_model_name,
        translate_name=translate_name,
        device=device,
        config_file=None,
        data_dir=data_dir,
        num_classes=runtime_num_classes,
    )
    if model is None:
        return

    test_loader = load_test_data(config_file=config_file)
    
    # 定义不同的测试配置
    level1_only = {
        'name': '仅Level 1 (冗余组)',
        'overrides': {
            'repair_mode': repair_mode,
            **resolve_levels_overrides('level1'),
            **(level1_overrides or {}),
        }
    }
    level1_level2 = {
        'name': 'Level 1 + Level 2 (冗余组 + 相似模式)',
        'overrides': {
            'repair_mode': repair_mode,
            **resolve_levels_overrides('level1_level2'),
            **(level1_overrides or {}),
        }
    }
    all_levels = {
        'name': 'Level 1 + Level 2 + Level 3 (完整三级容错)',
        'overrides': {
            'repair_mode': repair_mode,
            **resolve_levels_overrides('all'),
            **(level1_overrides or {}),
        }
    }
    if levels == 'level1':
        test_configs = [level1_only]
    elif levels == 'level1_level2':
        test_configs = [level1_level2]
    else:
        test_configs = [level1_only, level1_level2, all_levels]
    
    # 存储结果
    all_results = []
    
    # 🔧 为本次对比实验生成随机种子，确保同一次运行中三个策略使用相同故障
    import numpy as np
    import random
    import time
    
    # 使用时间戳生成随机种子，每次运行都不同
    random_seed = int(time.time() * 1000) % (2**32)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    print(f"🎲 本次实验随机种子: {random_seed}（确保三个策略使用相同故障进行对比）\n")
    
    # 运行每个配置
    for i, config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"实验 {i+1}/{len(test_configs)}: {config['name']}")
        print(f"{'='*80}\n")
        
        # 🔧 在每次实验前重置随机种子，确保使用相同的故障
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        results = run_simulation_with_config(
            model=model,
            test_loader=test_loader,
            config_name=config['name'],
            config_overrides=config['overrides'],
            base_config_file=config_file,
            num_samples=num_samples,
            model_name=runtime_model_name,
            translate_name=translate_name,
            report_output_dir=report_output_dir,
            data_dir=data_dir,
            fault_seed=fault_seed,
        )
        
        all_results.append({
            'name': config['name'],
            'results': results
        })
    
    # 打印对比总结
    print("\n" + "=" * 80)
    print("📊 容错策略对比总结")
    print("=" * 80)
    
    print(f"\n{'策略':<40} {'纠正率':<12} {'容错准确率':<12} {'准确率恢复':<12}")
    print("-" * 80)
    
    for item in all_results:
        name = item['name']
        results = item['results']
        reliability = results['reliability']
        accuracy = results['accuracy']
        
        correction_rate = reliability['fault_correction_rate']
        ft_accuracy = accuracy['ft_accuracy']
        recovery_rate = accuracy['accuracy_recovery_rate']
        
        print(f"{name:<40} {correction_rate:>10.2%}  {ft_accuracy:>10.2%}  {recovery_rate:>10.2%}")
    
    # 详细的三级容错统计（仅最后一个完整配置）
    if all_results:
        final_results = all_results[-1]['results']
        hierarchical = final_results['reliability'].get('hierarchical_correction', {})
        
        if hierarchical.get('level1_corrections', 0) > 0:
            print("\n" + "=" * 80)
            print("🔧 完整三级容错策略详细统计")
            print("=" * 80)
            
            level1 = hierarchical['level1_corrections']
            level2 = hierarchical['level2_corrections']
            level3 = hierarchical['level3_corrections']
            total = level1 + level2 + level3
            
            print(f"\n总纠正故障: {total}")
            print(f"  Level 1 (冗余组):       {level1:4d} ({level1/total*100:.1f}%)")
            print(f"  Level 2 (相似模式):     {level2:4d} ({level2/total*100:.1f}%)")
            if hierarchical.get('level2_similarity_avg', 0) > 0:
                print(f"    - 平均相似度: {hierarchical['level2_similarity_avg']:.4f}")
            print(f"  Level 3 (自适应屏蔽):   {level3:4d} ({level3/total*100:.1f}%)")
            
            # 硬件开销
            hw = final_results['hardware_overhead']
            print(f"\n硬件开销:")
            print(f"  总延迟: {hw['total_latency_ns']:.2f} ns")
            print(f"  总能耗: {hw['total_energy_pj']:.2f} pJ")
            print(f"  延迟开销比: {hw.get('latency_overhead_ratio', 0):.2f}x")
            print(f"  能耗开销比: {hw.get('energy_overhead_ratio', 0):.2f}x")
    
    print("\n" + "=" * 80)
    print("✅ 对比实验完成")
    print("=" * 80 + "\n")

    generated_files = []
    if report_output_dir:
        report_generator = ReportGenerator(output_dir=report_output_dir)
        generated_files.append(
            report_generator.generate_comparison_report(
                [item['results'] for item in all_results],
                [item['name'] for item in all_results],
            )
        )
        generated_files.extend(_export_compare_summary(all_results, report_output_dir))
        print("🗂️ compare 汇总文件:")
        for path in generated_files:
            print(f"  - {path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='三级分层容错策略仿真')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['compare', 'single'],
                       help='运行模式: compare(对比实验) 或 single(单次运行)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='测试样本数量')
    parser.add_argument('--model', type=str, default='Vgg16',
                       choices=['Vgg16', 'Res18', 'Res50', 'WRN'],
                       help='模型名称')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_FILE,
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--translate', type=str, default='ft_group_cluster_translate',
                       help='转换方法名称')
    parser.add_argument('--repair-mode', type=str, default='normal',
                       choices=['normal', 'oracle'],
                       help='修复模式: normal(当前FT/PRAP路径) 或 oracle(直接恢复为原始权重)')
    parser.add_argument('--levels', type=str, default='all',
                       choices=['level1', 'level1_level2', 'all'],
                       help='单次仿真或裁剪后的 compare 级别选择')
    parser.add_argument('--output-dir', type=str, default='',
                       help='可选：覆盖仿真报告输出目录；建议同一轮 single/compare 使用同一个目录')
    parser.add_argument('--artifact-dir', type=str, default='.',
                       help='artifact 目录；默认当前目录，可指向 results/ft_runs/<model>/<translate>/<tag>/artifacts')
    parser.add_argument('--fault-seed', type=int, default=None,
                       help='覆盖配置文件中的 fault_injection.random_seed，用于多 seed 容错实验')
    parser.add_argument('--level1-selection', type=str, default=None,
                       choices=['default', 'best_pair', 'weighted_average'],
                       help='Level1 repair candidate selection mode; omitted keeps V1.3.12 default behavior')
    parser.add_argument('--level1-topk', type=int, default=None,
                       help='Top-k cached Level1 candidates for best_pair/weighted_average')
    parser.add_argument('--level1-max-expected-error', type=float, default=None,
                       help='Optional max expected repair error before low-confidence fallback')
    parser.add_argument('--level1-min-expected-improvement', type=float, default=None,
                       help='Required expected-error improvement over default before replacing default candidate')
    parser.add_argument('--level1-critical-layer-config', type=str, default='',
                       help='JSON config overriding Level1 selection policy per layer')
    parser.add_argument('--level1-cache-max-group-size', type=int, default=None,
                       help='Max group size before Level1 cache candidate prefilter is used')
    parser.add_argument('--level1-cache-critical-layers-only', action='store_true',
                       help='Build Level1 repair cache only for layers listed in the critical layer config')
    
    args = parser.parse_args()
    level1_overrides = resolve_level1_selection_overrides(args)
    
    if args.mode == 'compare':
        # 运行对比实验
        compare_strategies(
            config_file=args.config,
            num_samples=args.samples,
            translate_name=args.translate,
            model_name=args.model,
            repair_mode=args.repair_mode,
            levels=args.levels,
            report_output_dir=args.output_dir or None,
            data_dir=args.artifact_dir,
            fault_seed=args.fault_seed,
            level1_overrides=level1_overrides,
        )
    else:
        # 单次运行
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        runtime_model_name, runtime_num_classes = resolve_runtime_model_name(args.model, args.config)
        model = load_model(
            runtime_model_name,
            translate_name=args.translate,
            device=device,
            config_file=None,
            data_dir=args.artifact_dir,
            num_classes=runtime_num_classes,
        )
        if model is None:
            return

        test_loader = load_test_data(config_file=args.config)

        # 从配置文件或参数中获取模型名称
        model_name = runtime_model_name

        run_simulation_with_config(
            model=model,
            test_loader=test_loader,
            config_name='单次容错实验',
            config_overrides={
                'repair_mode': args.repair_mode,
                **resolve_levels_overrides(args.levels),
                **level1_overrides,
            },
            base_config_file=args.config,
            num_samples=args.samples,
            model_name=model_name,
            translate_name=args.translate,
            report_output_dir=args.output_dir or None,
            data_dir=args.artifact_dir,
            fault_seed=args.fault_seed,
        )


if __name__ == "__main__":
    main()
