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
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from simulator.Fault_Tolerance.fault_tolerance_simulation import FaultToleranceSimulator
from model import Vgg16, Res18, Res50
from torchvision import transforms, datasets


def load_model(model_name, translate_name='ft_group_cluster_translate', device='cuda', config_file=None):
    """加载训练好的模型"""
    # 如果提供了配置文件，尝试从中读取模型名称和类别数
    num_classes = 10  # 默认值
    if config_file and os.path.exists(config_file):
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_name = config.get('model', {}).get('name', model_name)
                num_classes = config.get('model', {}).get('num_classes', 10)
                print(f"  📄 从配置文件读取: model={model_name}, num_classes={num_classes}")
        except Exception as e:
            print(f"  ⚠️ 读取配置文件失败: {e}，使用默认参数")

    if model_name == 'Vgg16':
        model = Vgg16(num_classes=num_classes)
    elif model_name == 'Res18':
        model = Res18(num_classes=num_classes)
    elif model_name == 'Res50':
        model = Res50(num_classes=num_classes)
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 加载模型参数
    model_file = f'model_{model_name}_{translate_name}_after_translate_parameters.pth'

    if not os.path.exists(model_file):
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


def run_simulation_with_config(model, test_loader, config_name, config_overrides=None,
                                base_config_file=None, num_samples=1000, model_name=None,
                                translate_name='ft_group_cluster_translate'):
    """运行带有特定配置的仿真"""
    print("=" * 80)
    print(f"🚀 运行仿真配置: {config_name}")
    print("=" * 80)

    import json
    import tempfile
    # os 已经在文件顶部导入了，这里不需要再导入

    # 加载基础配置（如果有）
    base_config = {}
    if base_config_file and os.path.exists(base_config_file):
        with open(base_config_file, 'r', encoding='utf-8') as f:
            base_config = json.load(f)

    # 决定使用哪个配置文件
    config_file_path = None
    temp_file_path = None  # 临时文件路径（需要清理）
    final_config = {}  # 最终使用的配置

    # 如果有覆盖配置，需要创建临时文件（合并基础配置和覆盖配置）
    if config_overrides:
        # 从基础配置开始，如果没有基础配置则使用默认配置
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
                }
            }
        else:
            # 使用基础配置作为起点
            final_config = json.loads(json.dumps(base_config))  # 深拷贝

        # 应用覆盖配置
        if config_overrides:
            # 确保 hierarchical_fault_tolerance 存在
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

        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(final_config, f, indent=2)
            config_file_path = f.name
            temp_file_path = config_file_path  # 标记这是临时文件

        print(f"📝 应用配置覆盖:")
        for key, value in config_overrides.items():
            print(f"  {key}: {value}")
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
        data_dir='./'
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


def compare_strategies(config_file=None, num_samples=1000, translate_name='ft_group_cluster_translate'):
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
    model_name_from_config = 'unknown'  # 默认值
    if config_file:
        try:
            import json
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
                model_name_from_config = config.get('model', {}).get('name', 'Res18')
                print(f"  📄 从配置文件读取模型名称: {model_name_from_config}")
        except Exception as e:
            print(f"  ⚠️ 读取模型名称失败: {e}，使用默认值")

    # 加载模型和数据（使用配置文件中的模型名称）
    model = load_model(model_name=model_name_from_config, translate_name=translate_name, device=device, config_file=config_file)
    if model is None:
        return

    test_loader = load_test_data(config_file=config_file)
    
    # 定义不同的测试配置
    test_configs = [
        {
            'name': '仅Level 1 (冗余组)',
            'overrides': {
                'level1.enabled': True,
                'level2.enabled': False,
                'level3.enabled': False,
            }
        },
        {
            'name': 'Level 1 + Level 2 (冗余组 + 相似模式)',
            'overrides': {
                'level1.enabled': True,
                'level2.enabled': True,
                'level3.enabled': False,
            }
        },
        {
            'name': 'Level 1 + Level 2 + Level 3 (完整三级容错)',
            'overrides': {
                'level1.enabled': True,
                'level2.enabled': True,
                'level3.enabled': True,
            }
        },
    ]
    
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
            translate_name=translate_name,
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


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='三级分层容错策略仿真')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['compare', 'single'],
                       help='运行模式: compare(对比实验) 或 single(单次运行)')
    parser.add_argument('--samples', type=int, default=1000,
                       help='测试样本数量')
    parser.add_argument('--model', type=str, default='Vgg16',
                       help='模型名称')
    parser.add_argument('--config', type=str, default='fault_tolerance_config_high_fault_rate.json',
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--translate', type=str, default='ft_group_cluster_translate',
                       help='转换方法名称')
    
    args = parser.parse_args()
    
    if args.mode == 'compare':
        # 运行对比实验
        compare_strategies(config_file=args.config, num_samples=args.samples, translate_name=args.translate)
    else:
        # 单次运行
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = load_model(args.model, translate_name=args.translate, device=device, config_file=args.config)
        if model is None:
            return

        test_loader = load_test_data(config_file=args.config)

        # 从配置文件或参数中获取模型名称
        import json
        model_name = args.model
        if os.path.exists(args.config):
            try:
                with open(args.config, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    model_name = config.get('model', {}).get('name', model_name)
            except Exception as e:
                print(f"  ⚠️ 读取配置文件失败: {e}，使用参数模型名称")

        run_simulation_with_config(
            model=model,
            test_loader=test_loader,
            config_name='完整三级容错',
            config_overrides={
                'level1.enabled': True,
                'level2.enabled': True,
                'level3.enabled': True,
            },
            base_config_file=args.config,
            num_samples=args.samples,
            model_name=model_name,
            translate_name=args.translate,
        )


if __name__ == "__main__":
    main()
