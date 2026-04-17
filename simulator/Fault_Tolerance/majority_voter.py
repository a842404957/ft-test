#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数表决器
对冗余组内的OU输出进行多数表决，检测和纠正故障
"""

import torch
import numpy as np
from typing import List, Tuple, Dict, Optional
from enum import Enum


class VotingStrategy(Enum):
    """表决策略"""
    SIMPLE_MAJORITY = "simple_majority"      # 简单多数
    WEIGHTED_MAJORITY = "weighted_majority"  # 加权多数
    EXACT_MATCH = "exact_match"              # 精确匹配


class TieBreaking(Enum):
    """平局处理策略"""
    DETECTION_FAILURE = "detection_failure"  # 判定为检测失败
    FIRST_VOTE = "first_vote"                # 选择第一个
    RANDOM = "random"                         # 随机选择


class MajorityVoter:
    """多数表决器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化多数表决器
        
        Args:
            config: 表决器配置
        """
        self.config = config or {
            'enabled': True,
            'voting_strategy': 'simple_majority',
            'tie_breaking': 'detection_failure',
            'voter_latency_ns': 10,
            'voter_energy_pj': 50,
            'voter_area_um2': 100,
            'similarity_threshold': 0.95,  # 相似度阈值（用于浮点数比较）
        }
        
        # 统计信息
        self.voting_statistics = {
            'total_votes': 0,
            'successful_corrections': 0,
            'detection_failures': 0,
            'tie_breaker_used': 0,
            'corrections_by_layer': {},
            'failures_by_layer': {}
        }
        
        print(f"🗳️ 多数表决器已初始化")
        print(f"  表决策略: {self.config['voting_strategy']}")
        print(f"  平局处理: {self.config['tie_breaking']}")
    
    def vote(self, 
             outputs: List[torch.Tensor],
             layer_name: str = "unknown",
             multipliers: Optional[List[float]] = None) -> Tuple[torch.Tensor, bool, Dict]:
        """
        对一组输出进行多数表决
        
        Args:
            outputs: 冗余组内各OU的输出列表
            layer_name: 层名称
            multipliers: 倍数关系列表（用于加权表决）
            
        Returns:
            Tuple[torch.Tensor, bool, Dict]: 
                (纠正后的输出, 是否检测到并纠正了故障, 投票详情)
        """
        if not self.config['enabled']:
            return outputs[0], False, {}
        
        if len(outputs) < 2:
            return outputs[0], False, {'error': 'insufficient_outputs'}
        
        self.voting_statistics['total_votes'] += 1
        
        # 选择表决策略
        strategy = self.config['voting_strategy']
        
        if strategy == 'simple_majority':
            corrected_output, correction_made, details = self._simple_majority_vote(outputs)
        elif strategy == 'weighted_majority':
            corrected_output, correction_made, details = self._weighted_majority_vote(
                outputs, multipliers
            )
        elif strategy == 'exact_match':
            corrected_output, correction_made, details = self._exact_match_vote(outputs)
        else:
            corrected_output, correction_made, details = self._simple_majority_vote(outputs)
        
        # 更新统计
        if correction_made:
            self.voting_statistics['successful_corrections'] += 1
            if layer_name not in self.voting_statistics['corrections_by_layer']:
                self.voting_statistics['corrections_by_layer'][layer_name] = 0
            self.voting_statistics['corrections_by_layer'][layer_name] += 1
        
        if details.get('detection_failure', False):
            self.voting_statistics['detection_failures'] += 1
            if layer_name not in self.voting_statistics['failures_by_layer']:
                self.voting_statistics['failures_by_layer'][layer_name] = 0
            self.voting_statistics['failures_by_layer'][layer_name] += 1
        
        if details.get('tie_breaker_used', False):
            self.voting_statistics['tie_breaker_used'] += 1
        
        return corrected_output, correction_made, details
    
    def _simple_majority_vote(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, bool, Dict]:
        """
        简单多数表决
        通过计算输出之间的相似度来判断多数
        """
        n = len(outputs)
        
        # 计算所有输出对之间的相似度
        similarity_matrix = self._compute_similarity_matrix(outputs)
        
        # 找出与其他输出最相似的那个（即"多数"）
        similarity_scores = similarity_matrix.sum(dim=1)
        majority_idx = torch.argmax(similarity_scores).item()
        
        # 判断是否存在明确的多数
        max_similarity = similarity_scores[majority_idx].item()
        second_max = torch.topk(similarity_scores, k=min(2, n))[0]
        
        if n > 1 and len(second_max) > 1:
            second_max_value = second_max[1].item()
        else:
            second_max_value = 0
        
        # 检查是否有明确的多数（相似度显著高于其他）
        threshold = self.config['similarity_threshold'] * (n - 1)
        
        if max_similarity >= threshold:
            # 找到明确的多数
            correction_made = True
            detection_failure = False
        elif abs(max_similarity - second_max_value) < 0.5:
            # 平局情况
            correction_made = False
            detection_failure = True
            majority_idx = self._handle_tie(outputs, similarity_scores)
        else:
            # 没有明确多数，但选择最相似的
            correction_made = True
            detection_failure = False
        
        details = {
            'num_outputs': n,
            'majority_idx': majority_idx,
            'max_similarity': max_similarity,
            'detection_failure': detection_failure,
            'tie_breaker_used': detection_failure,
            'similarity_matrix': similarity_matrix.tolist()
        }
        
        return outputs[majority_idx], correction_made, details
    
    def _weighted_majority_vote(self,
                               outputs: List[torch.Tensor],
                               multipliers: Optional[List[float]]) -> Tuple[torch.Tensor, bool, Dict]:
        """
        加权多数表决
        考虑倍数关系的加权投票
        """
        if multipliers is None or len(multipliers) != len(outputs):
            # 如果没有提供倍数关系，退化为简单多数表决
            return self._simple_majority_vote(outputs)
        
        # 归一化倍数作为权重
        weights = torch.tensor(multipliers, dtype=torch.float32)
        weights = weights / weights.sum()
        
        # 计算加权相似度
        similarity_matrix = self._compute_similarity_matrix(outputs)
        weighted_similarity = similarity_matrix @ weights
        
        majority_idx = torch.argmax(weighted_similarity).item()
        
        details = {
            'num_outputs': len(outputs),
            'majority_idx': majority_idx,
            'weights': weights.tolist(),
            'weighted_similarity': weighted_similarity.tolist(),
            'detection_failure': False,
            'tie_breaker_used': False
        }
        
        return outputs[majority_idx], True, details
    
    def _exact_match_vote(self, outputs: List[torch.Tensor]) -> Tuple[torch.Tensor, bool, Dict]:
        """
        精确匹配表决
        要求至少有两个输出完全相同
        """
        n = len(outputs)
        
        # 统计每个输出的匹配次数
        match_counts = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i != j and torch.allclose(outputs[i], outputs[j], rtol=1e-5):
                    match_counts[i] += 1
        
        max_matches = max(match_counts)
        
        if max_matches > 0:
            # 找到了匹配的输出
            majority_idx = match_counts.index(max_matches)
            correction_made = True
            detection_failure = False
        else:
            # 没有找到匹配，选择第一个
            majority_idx = 0
            correction_made = False
            detection_failure = True
        
        details = {
            'num_outputs': n,
            'majority_idx': majority_idx,
            'match_counts': match_counts,
            'max_matches': max_matches,
            'detection_failure': detection_failure,
            'tie_breaker_used': detection_failure
        }
        
        return outputs[majority_idx], correction_made, details
    
    def _compute_similarity_matrix(self, outputs: List[torch.Tensor]) -> torch.Tensor:
        """
        计算输出之间的相似度矩阵
        使用余弦相似度
        """
        n = len(outputs)
        similarity_matrix = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    # 计算余弦相似度
                    output_i_flat = outputs[i].flatten()
                    output_j_flat = outputs[j].flatten()
                    
                    # 避免除零
                    norm_i = torch.norm(output_i_flat)
                    norm_j = torch.norm(output_j_flat)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = torch.dot(output_i_flat, output_j_flat) / (norm_i * norm_j)
                        # 将相似度映射到[0, 1]
                        similarity = (similarity + 1) / 2
                    else:
                        similarity = 0.0
                    
                    similarity_matrix[i, j] = similarity
        
        return similarity_matrix
    
    def _handle_tie(self, outputs: List[torch.Tensor], similarity_scores: torch.Tensor) -> int:
        """
        处理平局情况
        
        Args:
            outputs: 输出列表
            similarity_scores: 相似度得分
            
        Returns:
            选中的输出索引
        """
        tie_breaking = self.config['tie_breaking']
        
        if tie_breaking == 'first_vote':
            return 0
        elif tie_breaking == 'random':
            return np.random.randint(0, len(outputs))
        else:  # detection_failure
            # 仍然返回相似度最高的，但标记为检测失败
            return torch.argmax(similarity_scores).item()
    
    def get_hardware_overhead(self, num_votes: int = None) -> Dict:
        """
        计算硬件开销
        
        Args:
            num_votes: 投票次数（None则使用统计值）
            
        Returns:
            硬件开销字典
        """
        if num_votes is None:
            num_votes = self.voting_statistics['total_votes']
        
        overhead = {
            'total_latency_ns': num_votes * self.config['voter_latency_ns'],
            'total_energy_pj': num_votes * self.config['voter_energy_pj'],
            'total_area_um2': self.config['voter_area_um2'],  # 面积不随投票次数变化
            'per_vote_latency_ns': self.config['voter_latency_ns'],
            'per_vote_energy_pj': self.config['voter_energy_pj']
        }
        
        return overhead
    
    def get_statistics(self) -> Dict:
        """获取表决统计信息"""
        stats = self.voting_statistics.copy()
        
        # 计算成功率
        if stats['total_votes'] > 0:
            stats['correction_rate'] = (
                stats['successful_corrections'] / stats['total_votes']
            )
            stats['failure_rate'] = (
                stats['detection_failures'] / stats['total_votes']
            )
        else:
            stats['correction_rate'] = 0.0
            stats['failure_rate'] = 0.0
        
        # 添加硬件开销
        stats['hardware_overhead'] = self.get_hardware_overhead()
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.voting_statistics = {
            'total_votes': 0,
            'successful_corrections': 0,
            'detection_failures': 0,
            'tie_breaker_used': 0,
            'corrections_by_layer': {},
            'failures_by_layer': {}
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("多数表决统计")
        print("=" * 60)
        print(f"总投票次数: {stats['total_votes']}")
        print(f"成功纠正次数: {stats['successful_corrections']}")
        print(f"检测失败次数: {stats['detection_failures']}")
        print(f"使用平局处理: {stats['tie_breaker_used']}")
        print(f"纠正成功率: {stats['correction_rate']:.2%}")
        print(f"检测失败率: {stats['failure_rate']:.2%}")
        
        print("\n硬件开销:")
        hw = stats['hardware_overhead']
        print(f"  总延迟: {hw['total_latency_ns']:.2f} ns")
        print(f"  总能耗: {hw['total_energy_pj']:.2f} pJ")
        print(f"  面积: {hw['total_area_um2']} μm²")
        
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


def test_voter():
    """测试多数表决器"""
    print("🧪 测试多数表决器\n")
    
    # 创建表决器
    voter = MajorityVoter({
        'enabled': True,
        'voting_strategy': 'simple_majority',
        'tie_breaking': 'detection_failure',
        'similarity_threshold': 0.95
    })
    
    # 测试场景1: 正常情况 - 3个相似输出，1个异常
    print("测试场景1: 3个正常输出 + 1个故障输出")
    normal_output = torch.randn(64, 32, 32)
    outputs_1 = [
        normal_output.clone(),
        normal_output.clone() + torch.randn_like(normal_output) * 0.01,  # 略有噪声
        normal_output.clone() + torch.randn_like(normal_output) * 0.01,
        normal_output.clone() * 10  # 故障输出
    ]
    
    corrected, correction_made, details = voter.vote(outputs_1, "test_layer_1")
    print(f"  纠正: {correction_made}")
    print(f"  选中索引: {details['majority_idx']}")
    print(f"  检测失败: {details['detection_failure']}")
    
    # 测试场景2: 平局情况 - 2对不同的输出
    print("\n测试场景2: 平局情况 (2对不同输出)")
    output_a = torch.randn(64, 32, 32)
    output_b = torch.randn(64, 32, 32)
    outputs_2 = [
        output_a.clone(),
        output_a.clone(),
        output_b.clone(),
        output_b.clone()
    ]
    
    corrected, correction_made, details = voter.vote(outputs_2, "test_layer_2")
    print(f"  纠正: {correction_made}")
    print(f"  检测失败: {details['detection_failure']}")
    
    # 打印统计
    voter.print_statistics()
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_voter()

