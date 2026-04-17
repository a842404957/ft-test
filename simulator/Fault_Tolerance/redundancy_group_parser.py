#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
冗余组解析器
从权重模式映射信息中识别和构建冗余计算组
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from .pattern_data_loader import PatternDataLoader


class RedundancyGroup:
    """冗余计算组数据结构"""
    
    def __init__(self, group_id: int, layer_name: str, pattern_id: int):
        """
        初始化冗余组
        
        Args:
            group_id: 组ID
            layer_name: 所属层名称
            pattern_id: 权重模式ID
        """
        self.group_id = group_id
        self.layer_name = layer_name
        self.pattern_id = pattern_id
        self.ou_indices = []  # OU索引列表 [(out_ch, in_ch), ...]
        self.multipliers = []  # 对应的倍数关系
        
    def add_ou(self, out_ch: int, in_ch: int, multiplier: float = 1.0):
        """添加OU到组"""
        self.ou_indices.append((out_ch, in_ch))
        self.multipliers.append(multiplier)
    
    def size(self) -> int:
        """返回组大小（OU数量）"""
        return len(self.ou_indices)
    
    def is_valid(self, min_size: int = 2) -> bool:
        """检查是否为有效的冗余组（至少有min_size个OU）"""
        return self.size() >= min_size
    
    def __repr__(self):
        return (f"RedundancyGroup(id={self.group_id}, layer={self.layer_name}, "
                f"pattern={self.pattern_id}, size={self.size()})")


class RedundancyGroupParser:
    """冗余组解析器"""
    
    def __init__(self, data_loader: PatternDataLoader, config: Dict = None):
        """
        初始化冗余组解析器
        
        Args:
            data_loader: 数据加载器实例
            config: 配置参数
        """
        self.data_loader = data_loader
        self.config = config or {
            'min_group_size': 2,
            'max_group_size': 8,
            'grouping_strategy': 'pattern_based'
        }
        
        # 存储解析结果
        self.redundancy_groups = {}  # {layer_name: [RedundancyGroup]}
        self.pattern_to_groups = defaultdict(list)  # {pattern_id: [group_ids]}
        self.statistics = {}
        
        print("🔍 初始化冗余组解析器")
    
    def parse_all_layers(self) -> bool:
        """
        解析所有层的冗余组
        
        Returns:
            bool: 是否成功解析
        """
        try:
            print("\n🔍 开始解析冗余计算组...")
            
            layer_names = self.data_loader.get_all_layer_names()
            total_groups = 0
            
            for layer_name in layer_names:
                groups = self.parse_layer(layer_name)
                if groups:
                    self.redundancy_groups[layer_name] = groups
                    total_groups += len(groups)
                    print(f"  ✓ {layer_name}: 发现 {len(groups)} 个冗余组")
            
            print(f"\n✅ 解析完成: 共 {total_groups} 个冗余组")
            self._compute_statistics()
            self._print_statistics()
            
            return True
            
        except Exception as e:
            print(f"❌ 解析失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def parse_layer(self, layer_name: str) -> List[RedundancyGroup]:
        """
        解析单层的冗余组
        
        Args:
            layer_name: 层名称
            
        Returns:
            该层的冗余组列表
        """
        # 获取映射信息
        map_table = self.data_loader.get_layer_map(layer_name)
        if map_table is None:
            return []
        if len(map_table.shape) < 2:
            return []

        multiplier_table = self.data_loader.get_layer_multiplier(layer_name)

        # 构建模式到OU的映射（确保去重）
        pattern_to_ous = defaultdict(dict)  # {pattern_id: {(out_ch, in_ch): multiplier}}
        in_channels, out_channels = map_table.shape[:2]

        def get_multiplier(out_ch: int, in_ch: int) -> float:
            if multiplier_table is None:
                return 1.0
            if len(multiplier_table.shape) == 2:  # FC层
                raw_multiplier = multiplier_table[out_ch, in_ch].item()
                return raw_multiplier if raw_multiplier != 0 else 1.0
            # Conv层
            multiplier_matrix = multiplier_table[out_ch, in_ch, :, :]
            valid_multipliers = multiplier_matrix[multiplier_matrix != 0]
            if len(valid_multipliers) > 0:
                unique_vals, counts = torch.unique(valid_multipliers, return_counts=True)
                most_common_idx = torch.argmax(counts)
                return unique_vals[most_common_idx].item()
            return 1.0

        for in_ch in range(in_channels):
            for entry_idx in range(out_channels):
                if len(map_table.shape) >= 3:
                    source_idx, target_idx = map_table[in_ch, entry_idx].tolist()
                else:
                    # 兼容二维映射：将当前列视为目标索引
                    source_idx = entry_idx
                    target_idx = map_table[in_ch, entry_idx].item()

                # 结束标记（用于剪枝后的稀疏表）
                if source_idx < 0:
                    break
                if target_idx < 0:
                    continue

                source_out = int(source_idx)
                target_out = int(target_idx)
                if source_out < 0 or source_out >= out_channels:
                    continue
                if target_out < 0 or target_out >= out_channels:
                    continue

                # 记录被复用OU
                pattern_to_ous[target_out][(source_out, in_ch)] = get_multiplier(source_out, in_ch)
                # 记录保留OU（确保冗余组完整）
                if (target_out, in_ch) not in pattern_to_ous[target_out]:
                    pattern_to_ous[target_out][(target_out, in_ch)] = get_multiplier(target_out, in_ch)
        
        # 创建冗余组
        groups = []
        group_id = 0
        
        for pattern_id, ou_map in pattern_to_ous.items():
            if len(ou_map) >= self.config['min_group_size']:
                # 创建冗余组
                group = RedundancyGroup(group_id, layer_name, pattern_id)
                
                for (out_ch, in_ch), multiplier in sorted(ou_map.items()):
                    group.add_ou(out_ch, in_ch, multiplier)
                
                groups.append(group)
                self.pattern_to_groups[pattern_id].append(group_id)
                group_id += 1
        
        return groups
    
    def get_layer_groups(self, layer_name: str) -> List[RedundancyGroup]:
        """获取指定层的冗余组"""
        return self.redundancy_groups.get(layer_name, [])
    
    def get_group_by_id(self, layer_name: str, group_id: int) -> Optional[RedundancyGroup]:
        """根据ID获取冗余组"""
        groups = self.get_layer_groups(layer_name)
        for group in groups:
            if group.group_id == group_id:
                return group
        return None
    
    def _compute_statistics(self):
        """计算统计信息"""
        total_groups = 0
        total_ous = 0
        group_sizes = []
        redundancy_ratios = []
        
        for layer_name, groups in self.redundancy_groups.items():
            layer_total_ous = 0
            layer_redundant_ous = 0
            
            for group in groups:
                total_groups += 1
                group_size = group.size()
                group_sizes.append(group_size)
                total_ous += group_size
                layer_redundant_ous += group_size
            
            # 计算该层的总OU数
            map_table = self.data_loader.get_layer_map(layer_name)
            if map_table is not None:
                in_channels, out_channels, _ = map_table.shape
                layer_total_ous = in_channels * out_channels
            
            # 冗余率 = 冗余OU数 / 总OU数
            if layer_total_ous > 0:
                redundancy_ratio = layer_redundant_ous / layer_total_ous
                redundancy_ratios.append(redundancy_ratio)
        
        self.statistics = {
            'total_layers': len(self.redundancy_groups),
            'total_groups': total_groups,
            'total_redundant_ous': total_ous,
            'avg_group_size': np.mean(group_sizes) if group_sizes else 0,
            'max_group_size': max(group_sizes) if group_sizes else 0,
            'min_group_size': min(group_sizes) if group_sizes else 0,
            'avg_redundancy_ratio': np.mean(redundancy_ratios) if redundancy_ratios else 0,
            'group_size_distribution': self._compute_distribution(group_sizes)
        }
    
    def _compute_distribution(self, sizes: List[int]) -> Dict[int, int]:
        """计算组大小分布"""
        distribution = defaultdict(int)
        for size in sizes:
            distribution[size] += 1
        return dict(distribution)
    
    def _print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("冗余组统计信息")
        print("=" * 60)
        print(f"总层数: {self.statistics['total_layers']}")
        print(f"总冗余组数: {self.statistics['total_groups']}")
        print(f"总冗余OU数: {self.statistics['total_redundant_ous']}")
        print(f"平均组大小: {self.statistics['avg_group_size']:.2f}")
        print(f"最大组大小: {self.statistics['max_group_size']}")
        print(f"最小组大小: {self.statistics['min_group_size']}")
        print(f"平均冗余率: {self.statistics['avg_redundancy_ratio']:.2%}")
        
        # print("\n组大小分布:")
        # for size, count in sorted(self.statistics['group_size_distribution'].items()):
        #     print(f"  大小 {size}: {count} 个组")
        
        print("=" * 60 + "\n")
    
    def export_groups(self, output_file: str = 'redundancy_groups.json'):
        """导出冗余组信息到文件"""
        import json
        
        export_data = {
            'statistics': self.statistics,
            'layers': {}
        }
        
        for layer_name, groups in self.redundancy_groups.items():
            layer_groups = []
            for group in groups:
                group_data = {
                    'group_id': group.group_id,
                    'pattern_id': group.pattern_id,
                    'size': group.size(),
                    'ou_indices': group.ou_indices,
                    'multipliers': group.multipliers
                }
                layer_groups.append(group_data)
            
            export_data['layers'][layer_name] = layer_groups
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 冗余组信息已导出到: {output_file}")
    
    def visualize_layer_groups(self, layer_name: str, max_display: int = 5):
        """可视化显示层的冗余组（用于调试）"""
        groups = self.get_layer_groups(layer_name)
        
        print(f"\n{'=' * 60}")
        print(f"层 {layer_name} 的冗余组详情")
        print(f"{'=' * 60}")
        
        for i, group in enumerate(groups[:max_display]):
            print(f"\n组 {group.group_id} (模式 {group.pattern_id}):")
            print(f"  大小: {group.size()} 个OU")
            print(f"  OU索引 (前5个): {group.ou_indices[:5]}")
            if group.multipliers:
                unique_multipliers = set(group.multipliers)
                print(f"  倍数关系: {unique_multipliers}")
        
        if len(groups) > max_display:
            print(f"\n... 还有 {len(groups) - max_display} 个组未显示")
        
        print(f"{'=' * 60}\n")


def test_parser():
    """测试冗余组解析器"""
    print("🧪 测试冗余组解析器\n")
    
    # 创建数据加载器
    loader = PatternDataLoader(
        model_name='Vgg16',
        translate_name='weight_pattern_shape_and_value_similar_translate',
        data_dir='./'
    )
    
    if not loader.load_all_data():
        print("❌ 数据加载失败")
        return
    
    # 创建解析器
    parser = RedundancyGroupParser(loader)
    
    # 解析所有层
    if parser.parse_all_layers():
        print("\n✅ 解析测试通过")
        
        # 可视化第一层
        layer_names = loader.get_all_layer_names()
        if layer_names:
            parser.visualize_layer_groups(layer_names[0])
        
        # 导出结果
        parser.export_groups()
    else:
        print("\n❌ 解析测试失败")


if __name__ == "__main__":
    test_parser()
