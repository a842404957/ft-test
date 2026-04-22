#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
容错机制全局配置
所有可调参数的集中管理
"""

import json
import os
import copy
from typing import Dict, Any


class FaultToleranceConfig:
    """容错机制配置类"""
    
    # 默认配置
    DEFAULT_CONFIG = {
        # 故障注入配置
        'fault_injection': {
            'enabled': True,
            'fault_rate': 0.01,  # 1% 故障率
            'fault_models': ['bit_flip'],  # 故障模型: bit_flip, output_corruption, stuck_at
            'target_layers': 'all',  # 'all' 或指定层列表
            'exclude_critical_layers': ['__first__', '__last__'],  # 动态排除首层和末层
            'random_seed': 42,  # 随机种子，用于可重现性
            'bit_flip_positions': 'random',  # 'random' 或 'msb' 或 'lsb'
            'bit_flip_ratio': 0.25,
        },
        
        # 多数表决器配置
        'majority_voter': {
            'enabled': True,
            'voting_strategy': 'simple_majority',  # simple_majority, weighted_majority
            'tie_breaking': 'detection_failure',  # detection_failure, first_vote, random
            'voter_latency_ns': 10,  # 纳秒
            'voter_energy_pj': 50,  # 皮焦耳
            'voter_area_um2': 100,  # 平方微米
        },
        
        # 三级容错策略配置
        'hierarchical_fault_tolerance': {
            'enabled': True,  # 启用分层容错
            'repair_mode': 'normal',  # normal, oracle
            'level1': {
                'name': 'redundancy_group',
                'enabled': True,
                'description': '冗余组内替换（主策略）'
            },
            'level2': {
                'name': 'nearest_pattern',
                'enabled': True,
                'description': '相似模式近似替换',
                'similarity_threshold': 0.85,  # 相似度阈值
                'k_nearest': 3,  # 考虑k个最近邻
                'use_weighted_average': True,  # 使用加权平均
                'fallback_latency_ns': 15,  # 额外延迟开销
                'fallback_energy_pj': 75,  # 额外能耗开销
            },
            'level3': {
                'name': 'adaptive_masking',
                'enabled': True,
                'description': '自适应屏蔽策略',
                'masking_strategy': 'weighted_neighbors',  # weighted_neighbors, zero_out, interpolation
                'neighbor_radius': 2,  # 邻域半径（对于Conv层）
                'fallback_latency_ns': 20,
                'fallback_energy_pj': 100,
            },
            'statistics': {
                'track_correction_level': True,  # 跟踪每个故障由哪一级纠正
                'track_correction_quality': True,  # 跟踪纠正质量
            }
        },
        
        # 冗余组配置
        'redundancy_group': {
            'min_group_size': 2,  # 最小冗余组大小
            'max_group_size': 8,  # 最大冗余组大小
            'grouping_strategy': 'pattern_based',  # pattern_based, random
        },
        
        # 硬件开销配置
        'hardware_overhead': {
            'parallel_computation': True,  # 冗余组内并行计算
            'include_voter_overhead': True,  # 包含表决器开销
            'include_interconnect_overhead': True,  # 包含互连开销
        },
        
        # 仿真配置
        'simulation': {
            'use_gpu': True,  # 使用GPU加速
            'batch_size': 128,  # 批大小
            'num_test_samples': 1000,  # 测试样本数（-1表示全部）
            'verbose': True,  # 详细输出
            'save_intermediate_results': False,  # 保存中间结果
        },
        
        # 评估指标配置
        'metrics': {
            'track_layer_wise': True,  # 逐层跟踪
            'compute_confidence_interval': True,  # 计算置信区间
            'generate_plots': True,  # 生成图表
            'save_raw_data': True,  # 保存原始数据
        },
        
        # 报告配置
        'report': {
            'output_dir': './fault_tolerance_results',
            'formats': ['json', 'csv', 'md'],  # 输出格式
            'include_summary': True,
            'include_detailed_stats': True,
        }
    }
    
    def __init__(self, config_file: str = None):
        """
        初始化配置
        
        Args:
            config_file: 配置文件路径（可选）
        """
        self.config = copy.deepcopy(self.DEFAULT_CONFIG)
        
        # 如果提供了配置文件，加载并合并
        if config_file and os.path.exists(config_file):
            self.load_from_file(config_file)
        
    def load_from_file(self, config_file: str):
        """从文件加载配置"""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            self._merge_config(user_config)
            print(f"✅ 配置已从 {config_file} 加载")
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}，使用默认配置")
    
    def _merge_config(self, user_config: Dict[str, Any]):
        """合并用户配置到默认配置"""
        for section, values in user_config.items():
            if section in self.config:
                if isinstance(values, dict):
                    self.config[section].update(values)
                else:
                    self.config[section] = values
    
    def save_to_file(self, config_file: str):
        """保存配置到文件"""
        try:
            os.makedirs(os.path.dirname(config_file) or '.', exist_ok=True)
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, indent=4, ensure_ascii=False)
            print(f"✅ 配置已保存到 {config_file}")
        except Exception as e:
            print(f"❌ 保存配置文件失败: {e}")
    
    def get(self, section: str, key: str = None, default=None):
        """获取配置值"""
        if key is None:
            return self.config.get(section, default)
        return self.config.get(section, {}).get(key, default)
    
    def set(self, section: str, key: str, value: Any):
        """设置配置值"""
        if section not in self.config:
            self.config[section] = {}
        self.config[section][key] = value
    
    def print_config(self):
        """打印当前配置"""
        print("=" * 60)
        print("容错机制配置")
        print("=" * 60)
        for section, values in self.config.items():
            print(f"\n[{section}]")
            if isinstance(values, dict):
                for key, value in values.items():
                    print(f"  {key}: {value}")
            else:
                print(f"  {values}")
        print("=" * 60)


# 创建全局配置实例
global_config = FaultToleranceConfig()


def get_config() -> FaultToleranceConfig:
    """获取全局配置实例"""
    return global_config


if __name__ == "__main__":
    # 测试配置模块
    config = FaultToleranceConfig()
    config.print_config()
    
    # 保存示例配置
    config.save_to_file('fault_tolerance_config_example.json')
