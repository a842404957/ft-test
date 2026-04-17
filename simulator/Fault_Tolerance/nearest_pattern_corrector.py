#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最近邻模式纠错器 (Level 2容错策略)
对于不在冗余组中的故障OU，使用最相似的权重模式作为近似替代
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


class NearestPatternCorrector:
    """最近邻模式纠错器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化最近邻模式纠错器
        
        Args:
            config: 配置参数
        """
        # 默认配置
        default_config = {
            'enabled': True,
            'similarity_threshold': 0.85,
            'k_nearest': 3,
            'use_weighted_average': True,
            'fallback_latency_ns': 15,
            'fallback_energy_pj': 75,
        }
        
        # 合并用户配置
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # 权重模式相似度索引
        self.pattern_similarity_index = {}  # {layer_name: similarity_matrix}
        self.ou_to_pattern_index = {}  # {layer_name: {ou_idx: pattern_representation}}
        
        # 统计信息
        self.correction_statistics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'corrections_by_layer': {},
            'average_similarity': 0.0,
            'similarity_distribution': []
        }
        
        print("🔍 最近邻模式纠错器已初始化")
        print(f"  相似度阈值: {self.config['similarity_threshold']}")
        print(f"  K最近邻: {self.config['k_nearest']}")
    
    def build_similarity_index(self, 
                               model: torch.nn.Module,
                               layer_names: List[str],
                               data_loader=None):
        """
        为所有层构建权重模式相似度索引
        
        Args:
            model: PyTorch模型
            layer_names: 层名称列表
            data_loader: 数据加载器（可选，用于基于激活的相似度）
        """
        print("\n🔍 构建权重模式相似度索引...")
        
        for layer_name in layer_names:
            module = self._get_module_by_name(model, layer_name)
            if module is None or not hasattr(module, 'weight'):
                continue
            
            weight = module.weight.data
            out_channels = weight.shape[0]
            
            # 为每个OU（输出通道）提取权重模式表示
            pattern_representations = []
            for ou_idx in range(out_channels):
                # 将该OU的所有权重展平作为模式表示
                pattern = weight[ou_idx].flatten()
                pattern_representations.append(pattern)
            
            # 构建相似度矩阵
            similarity_matrix = self._compute_pattern_similarity_matrix(
                pattern_representations
            )
            
            # 保存索引
            self.pattern_similarity_index[layer_name] = similarity_matrix
            self.ou_to_pattern_index[layer_name] = {
                i: pattern_representations[i] 
                for i in range(len(pattern_representations))
            }
            
            print(f"  ✓ {layer_name}: {out_channels} 个OU模式")
        
        print("✅ 相似度索引构建完成\n")
    
    def find_nearest_patterns(self,
                             layer_name: str,
                             faulty_ou_idx: int,
                             exclude_ous: List[int] = None,
                             k: int = None) -> List[Tuple[int, float]]:
        """
        为故障OU找到k个最相似的正常OU
        
        Args:
            layer_name: 层名称
            faulty_ou_idx: 故障OU索引
            exclude_ous: 要排除的OU列表（如其他故障OU）
            k: 返回k个最近邻（None则使用配置值）
            
        Returns:
            List[(ou_idx, similarity_score)]: k个最近邻及其相似度
        """
        if layer_name not in self.pattern_similarity_index:
            return []
        
        if k is None:
            k = self.config['k_nearest']
        
        exclude_ous = exclude_ous or []
        similarity_matrix = self.pattern_similarity_index[layer_name]
        
        # 获取该故障OU与所有其他OU的相似度
        similarities = similarity_matrix[faulty_ou_idx].clone()
        
        # 排除自己和其他故障OU
        similarities[faulty_ou_idx] = -1.0
        for ou_idx in exclude_ous:
            if ou_idx < len(similarities):
                similarities[ou_idx] = -1.0
        
        # 找到top-k最相似的
        top_k_values, top_k_indices = torch.topk(
            similarities, 
            k=min(k, len(similarities)), 
            largest=True
        )
        
        # 过滤低于阈值的
        threshold = self.config['similarity_threshold']
        nearest_patterns = [
            (idx.item(), sim.item()) 
            for idx, sim in zip(top_k_indices, top_k_values)
            if sim.item() >= threshold
        ]
        
        return nearest_patterns
    
    def correct_faulty_output(self,
                             output: torch.Tensor,
                             faulty_ou_idx: int,
                             layer_name: str,
                             exclude_ous: List[int] = None) -> Tuple[torch.Tensor, bool, Dict]:
        """
        使用最近邻模式纠正故障OU的输出
        
        Args:
            output: 层输出 [batch, channels, H, W] 或 [batch, features]
            faulty_ou_idx: 故障OU索引
            layer_name: 层名称
            exclude_ous: 要排除的OU列表
            
        Returns:
            Tuple[corrected_output, success, details]
        """
        self.correction_statistics['total_corrections'] += 1
        
        # 查找最近邻
        nearest_patterns = self.find_nearest_patterns(
            layer_name, faulty_ou_idx, exclude_ous
        )
        
        if not nearest_patterns:
            # 没有找到合适的最近邻
            self.correction_statistics['failed_corrections'] += 1
            return output, False, {
                'error': 'no_nearest_pattern_found',
                'faulty_ou': faulty_ou_idx
            }
        
        # 使用加权平均纠正
        if self.config['use_weighted_average'] and len(nearest_patterns) > 1:
            corrected_output = self._weighted_average_correction(
                output, faulty_ou_idx, nearest_patterns
            )
            avg_similarity = np.mean([sim for _, sim in nearest_patterns])
        else:
            # 使用最相似的单个OU
            nearest_ou_idx, similarity = nearest_patterns[0]
            corrected_output = self._single_pattern_correction(
                output, faulty_ou_idx, nearest_ou_idx
            )
            avg_similarity = similarity
        
        # 更新统计
        self.correction_statistics['successful_corrections'] += 1
        self.correction_statistics['similarity_distribution'].append(avg_similarity)
        
        if layer_name not in self.correction_statistics['corrections_by_layer']:
            self.correction_statistics['corrections_by_layer'][layer_name] = 0
        self.correction_statistics['corrections_by_layer'][layer_name] += 1
        
        details = {
            'nearest_patterns': nearest_patterns,
            'average_similarity': avg_similarity,
            'num_patterns_used': len(nearest_patterns),
            'correction_method': 'weighted_average' if len(nearest_patterns) > 1 else 'single_pattern'
        }
        
        return corrected_output, True, details
    
    def _weighted_average_correction(self,
                                    output: torch.Tensor,
                                    faulty_ou_idx: int,
                                    nearest_patterns: List[Tuple[int, float]]) -> torch.Tensor:
        """
        使用加权平均纠正
        
        Args:
            output: 层输出
            faulty_ou_idx: 故障OU索引
            nearest_patterns: [(ou_idx, similarity), ...]
            
        Returns:
            纠正后的输出
        """
        # 提取相似度作为权重
        similarities = torch.tensor([sim for _, sim in nearest_patterns])
        weights = similarities / similarities.sum()
        
        # 根据输出维度选择纠正方式
        if len(output.shape) == 4:  # Conv层: [batch, channels, H, W]
            corrected = torch.zeros_like(output[:, faulty_ou_idx:faulty_ou_idx+1, :, :])
            for (ou_idx, _), weight in zip(nearest_patterns, weights):
                corrected += weight * output[:, ou_idx:ou_idx+1, :, :]
            output[:, faulty_ou_idx:faulty_ou_idx+1, :, :] = corrected
            
        elif len(output.shape) == 2:  # Linear层: [batch, features]
            corrected = torch.zeros_like(output[:, faulty_ou_idx:faulty_ou_idx+1])
            for (ou_idx, _), weight in zip(nearest_patterns, weights):
                corrected += weight * output[:, ou_idx:ou_idx+1]
            output[:, faulty_ou_idx:faulty_ou_idx+1] = corrected
        
        return output
    
    def _single_pattern_correction(self,
                                  output: torch.Tensor,
                                  faulty_ou_idx: int,
                                  reference_ou_idx: int) -> torch.Tensor:
        """
        使用单个最相似的OU纠正
        
        Args:
            output: 层输出
            faulty_ou_idx: 故障OU索引
            reference_ou_idx: 参考OU索引
            
        Returns:
            纠正后的输出
        """
        if len(output.shape) == 4:  # Conv层
            output[:, faulty_ou_idx:faulty_ou_idx+1, :, :] = \
                output[:, reference_ou_idx:reference_ou_idx+1, :, :]
        elif len(output.shape) == 2:  # Linear层
            output[:, faulty_ou_idx:faulty_ou_idx+1] = \
                output[:, reference_ou_idx:reference_ou_idx+1]
        
        return output
    
    def _compute_pattern_similarity_matrix(self,
                                          patterns: List[torch.Tensor]) -> torch.Tensor:
        """
        计算权重模式之间的相似度矩阵
        使用余弦相似度
        
        Args:
            patterns: 权重模式列表
            
        Returns:
            相似度矩阵 [n, n]
        """
        n = len(patterns)
        similarity_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 余弦相似度
                    pattern_i = patterns[i].flatten()
                    pattern_j = patterns[j].flatten()
                    
                    norm_i = torch.norm(pattern_i)
                    norm_j = torch.norm(pattern_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = torch.dot(pattern_i, pattern_j) / (norm_i * norm_j)
                        # 映射到[0, 1]
                        similarity = (cosine_sim + 1) / 2
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity
        
        return similarity_matrix
    
    def _get_module_by_name(self, model: torch.nn.Module, layer_name: str):
        """根据层名称获取模块"""
        module_name = layer_name.replace('.weight', '')
        try:
            module = model
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def get_hardware_overhead(self) -> Dict:
        """获取硬件开销"""
        num_corrections = self.correction_statistics['successful_corrections']
        return {
            'total_latency_ns': num_corrections * self.config['fallback_latency_ns'],
            'total_energy_pj': num_corrections * self.config['fallback_energy_pj'],
            'per_correction_latency_ns': self.config['fallback_latency_ns'],
            'per_correction_energy_pj': self.config['fallback_energy_pj']
        }
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        stats = self.correction_statistics.copy()
        
        # 计算平均相似度
        if stats['similarity_distribution']:
            stats['average_similarity'] = np.mean(stats['similarity_distribution'])
            stats['min_similarity'] = np.min(stats['similarity_distribution'])
            stats['max_similarity'] = np.max(stats['similarity_distribution'])
        
        # 计算成功率
        if stats['total_corrections'] > 0:
            stats['success_rate'] = (
                stats['successful_corrections'] / stats['total_corrections']
            )
        else:
            stats['success_rate'] = 0.0
        
        # 添加硬件开销
        stats['hardware_overhead'] = self.get_hardware_overhead()
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.correction_statistics = {
            'total_corrections': 0,
            'successful_corrections': 0,
            'failed_corrections': 0,
            'corrections_by_layer': {},
            'average_similarity': 0.0,
            'similarity_distribution': []
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("最近邻模式纠错统计 (Level 2)")
        print("=" * 60)
        print(f"总纠正次数: {stats['total_corrections']}")
        print(f"成功纠正: {stats['successful_corrections']}")
        print(f"失败纠正: {stats['failed_corrections']}")
        print(f"成功率: {stats['success_rate']:.2%}")
        
        if stats['similarity_distribution']:
            print(f"\n相似度统计:")
            print(f"  平均: {stats['average_similarity']:.4f}")
            print(f"  最小: {stats['min_similarity']:.4f}")
            print(f"  最大: {stats['max_similarity']:.4f}")
        
        print("\n硬件开销:")
        hw = stats['hardware_overhead']
        print(f"  总延迟: {hw['total_latency_ns']:.2f} ns")
        print(f"  总能耗: {hw['total_energy_pj']:.2f} pJ")
        
        if stats['corrections_by_layer']:
            print("\n按层纠正统计 (前10层):")
            sorted_layers = sorted(
                stats['corrections_by_layer'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for layer, count in sorted_layers[:10]:
                print(f"  {layer}: {count}")
        
        print("=" * 60 + "\n")


def test_nearest_pattern_corrector():
    """测试最近邻模式纠错器"""
    print("🧪 测试最近邻模式纠错器\n")
    
    # 创建简单的测试模型
    import torch.nn as nn
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.fc1 = nn.Linear(64, 10)
        
        def forward(self, x):
            x = self.conv1(x)
            x = x.mean(dim=[2, 3])
            x = self.fc1(x)
            return x
    
    model = SimpleModel()
    
    # 创建纠错器
    corrector = NearestPatternCorrector({
        'similarity_threshold': 0.85,
        'k_nearest': 3,
        'use_weighted_average': True
    })
    
    # 构建相似度索引
    corrector.build_similarity_index(
        model,
        ['conv1.weight', 'fc1.weight']
    )
    
    # 测试纠正
    print("测试场景: 纠正故障OU的输出")
    test_output = torch.randn(32, 64, 8, 8)  # Conv层输出
    faulty_ou_idx = 10
    exclude_ous = [5, 15]  # 其他故障OU
    
    corrected_output, success, details = corrector.correct_faulty_output(
        test_output.clone(),
        faulty_ou_idx,
        'conv1.weight',
        exclude_ous
    )
    
    print(f"  纠正成功: {success}")
    if success:
        print(f"  使用的最近邻: {len(details['nearest_patterns'])}")
        print(f"  平均相似度: {details['average_similarity']:.4f}")
        print(f"  纠正方法: {details['correction_method']}")
    
    # 打印统计
    corrector.print_statistics()
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_nearest_pattern_corrector()

