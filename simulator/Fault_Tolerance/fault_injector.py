#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
故障注入器
在OU计算输出中注入可配置的故障
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import random


class FaultModel(Enum):
    """故障模型类型"""
    BIT_FLIP = "bit_flip"                    # 单比特翻转
    OUTPUT_CORRUPTION = "output_corruption"   # 输出损坏
    STUCK_AT_ZERO = "stuck_at_zero"          # 固定为0
    STUCK_AT_ONE = "stuck_at_one"            # 固定为1
    RANDOM_VALUE = "random_value"            # 随机值
    AMPLIFICATION = "amplification"          # 放大错误


class FaultInjector:
    """故障注入器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化故障注入器
        
        Args:
            config: 故障注入配置
        """
        self.config = config or {
            'enabled': True,
            'fault_rate': 0.05,
            'fault_models': ['bit_flip'],
            'random_seed': None,  # 不固定随机种子，允许每次运行产生不同故障
            'bit_flip_positions': 'random',
        }
        
        # 设置随机种子
        if self.config.get('random_seed') is not None:
            random.seed(self.config['random_seed'])
            np.random.seed(self.config['random_seed'])
            torch.manual_seed(self.config['random_seed'])
        
        # 统计信息
        self.fault_statistics = {
            'total_outputs_processed': 0,
            'total_faults_injected': 0,
            'faults_by_model': {},
            'faults_by_layer': {}
        }
        
        print(f"💥 故障注入器已初始化")
        print(f"  故障率: {self.config['fault_rate']}")
        print(f"  故障模型: {self.config['fault_models']}")
    
    def inject_fault_to_outputs(self, 
                                outputs: List[torch.Tensor],
                                layer_name: str = "unknown") -> Tuple[List[torch.Tensor], List[bool]]:
        """
        对一组输出注入故障
        
        Args:
            outputs: 输出张量列表（冗余组内各OU的输出）
            layer_name: 层名称（用于统计）
            
        Returns:
            Tuple[List[torch.Tensor], List[bool]]: (带故障的输出列表, 故障标记列表)
        """
        if not self.config['enabled']:
            return outputs, [False] * len(outputs)
        
        faulty_outputs = []
        fault_flags = []
        
        for output in outputs:
            # 判断是否对这个输出注入故障
            if self._should_inject_fault():
                # 选择故障模型
                fault_model = self._select_fault_model()
                
                # 注入故障
                faulty_output = self._apply_fault_model(output, fault_model)
                
                # 更新统计
                self._update_statistics(layer_name, fault_model)
                
                faulty_outputs.append(faulty_output)
                fault_flags.append(True)
            else:
                faulty_outputs.append(output)
                fault_flags.append(False)
            
            self.fault_statistics['total_outputs_processed'] += 1
        
        return faulty_outputs, fault_flags
    
    def inject_fault_single(self, 
                           output: torch.Tensor,
                           layer_name: str = "unknown",
                           force_inject: bool = False) -> Tuple[torch.Tensor, bool]:
        """
        对单个输出注入故障
        
        Args:
            output: 输出张量
            layer_name: 层名称
            force_inject: 是否强制注入故障
            
        Returns:
            Tuple[torch.Tensor, bool]: (带故障的输出, 是否注入了故障)
        """
        if not self.config['enabled'] and not force_inject:
            return output, False
        
        if force_inject or self._should_inject_fault():
            fault_model = self._select_fault_model()
            faulty_output = self._apply_fault_model(output, fault_model)
            self._update_statistics(layer_name, fault_model)
            self.fault_statistics['total_outputs_processed'] += 1
            return faulty_output, True
        else:
            self.fault_statistics['total_outputs_processed'] += 1
            return output, False
    
    def _should_inject_fault(self) -> bool:
        """判断是否应该注入故障"""
        return random.random() < self.config['fault_rate']
    
    def _select_fault_model(self) -> FaultModel:
        """选择故障模型"""
        fault_models = self.config['fault_models']
        
        # 映射字符串到枚举
        model_map = {
            'bit_flip': FaultModel.BIT_FLIP,
            'output_corruption': FaultModel.OUTPUT_CORRUPTION,
            'stuck_at_zero': FaultModel.STUCK_AT_ZERO,
            'stuck_at_one': FaultModel.STUCK_AT_ONE,
            'random_value': FaultModel.RANDOM_VALUE,
            'amplification': FaultModel.AMPLIFICATION,
        }
        
        selected_model_str = random.choice(fault_models)
        return model_map.get(selected_model_str, FaultModel.BIT_FLIP)
    
    def _apply_fault_model(self, output: torch.Tensor, fault_model: FaultModel) -> torch.Tensor:
        """
        应用故障模型
        
        Args:
            output: 原始输出
            fault_model: 故障模型
            
        Returns:
            带故障的输出
        """
        faulty_output = output.clone()
        
        if fault_model == FaultModel.BIT_FLIP:
            faulty_output = self._bit_flip_fault(faulty_output)
        
        elif fault_model == FaultModel.OUTPUT_CORRUPTION:
            faulty_output = self._output_corruption_fault(faulty_output)
        
        elif fault_model == FaultModel.STUCK_AT_ZERO:
            faulty_output = self._stuck_at_fault(faulty_output, value=0.0)
        
        elif fault_model == FaultModel.STUCK_AT_ONE:
            faulty_output = self._stuck_at_fault(faulty_output, value=1.0)
        
        elif fault_model == FaultModel.RANDOM_VALUE:
            faulty_output = self._random_value_fault(faulty_output)
        
        elif fault_model == FaultModel.AMPLIFICATION:
            faulty_output = self._amplification_fault(faulty_output)
        
        return faulty_output
    
    def _bit_flip_fault(self, output: torch.Tensor) -> torch.Tensor:
        """
        单比特翻转故障
        在输出的浮点表示中翻转一个随机比特
        """
        # 简化实现：随机选择一些元素并翻转符号或添加扰动
        flat_output = output.flatten()
        num_elements = flat_output.numel()
        
        if num_elements == 0:
            return output
        
        # 选择要翻转的位置
        if self.config['bit_flip_positions'] == 'msb':
            # 翻转符号位（最高有效位）
            flip_indices = torch.randint(0, num_elements, (1,))
            flat_output[flip_indices] *= -1
        
        elif self.config['bit_flip_positions'] == 'lsb':
            # 在最低有效位添加小扰动
            flip_indices = torch.randint(0, num_elements, (1,))
            perturbation = torch.randn_like(flat_output[flip_indices]) * 1e-6
            flat_output[flip_indices] += perturbation
        
        else:  # random
            # 随机选择一个元素，并对其进行随机扰动
            flip_indices = torch.randint(0, num_elements, (1,))
            # 随机决定翻转类型
            if random.random() < 0.5:
                flat_output[flip_indices] *= -1  # 符号翻转
            else:
                # 数量级扰动
                flat_output[flip_indices] *= (random.random() * 2 + 0.5)
        
        return flat_output.reshape(output.shape)
    
    def _output_corruption_fault(self, output: torch.Tensor) -> torch.Tensor:
        """
        输出损坏故障
        向输出添加高斯噪声
        """
        noise_scale = torch.std(output) * 0.5  # 噪声强度为标准差的50%
        noise = torch.randn_like(output) * noise_scale
        return output + noise
    
    def _stuck_at_fault(self, output: torch.Tensor, value: float) -> torch.Tensor:
        """
        固定值故障
        将部分输出固定为特定值
        """
        mask = torch.rand_like(output) < 0.1  # 10%的元素受影响
        faulty_output = output.clone()
        faulty_output[mask] = value
        return faulty_output
    
    def _random_value_fault(self, output: torch.Tensor) -> torch.Tensor:
        """
        随机值故障
        将输出替换为随机值
        """
        # 随机选择一些元素替换为随机值
        mask = torch.rand_like(output) < 0.05  # 5%的元素受影响
        random_values = torch.randn_like(output) * torch.std(output)
        faulty_output = output.clone()
        faulty_output[mask] = random_values[mask]
        return faulty_output
    
    def _amplification_fault(self, output: torch.Tensor) -> torch.Tensor:
        """
        放大错误故障
        将输出值放大到异常范围
        """
        amplification_factor = random.uniform(5, 20)
        return output * amplification_factor
    
    def _update_statistics(self, layer_name: str, fault_model: FaultModel):
        """更新统计信息"""
        self.fault_statistics['total_faults_injected'] += 1
        
        # 按故障模型统计
        model_name = fault_model.value
        if model_name not in self.fault_statistics['faults_by_model']:
            self.fault_statistics['faults_by_model'][model_name] = 0
        self.fault_statistics['faults_by_model'][model_name] += 1
        
        # 按层统计
        if layer_name not in self.fault_statistics['faults_by_layer']:
            self.fault_statistics['faults_by_layer'][layer_name] = 0
        self.fault_statistics['faults_by_layer'][layer_name] += 1
    
    def get_statistics(self) -> Dict:
        """获取故障注入统计信息"""
        stats = self.fault_statistics.copy()
        
        # 计算故障率
        if stats['total_outputs_processed'] > 0:
            stats['actual_fault_rate'] = (
                stats['total_faults_injected'] / stats['total_outputs_processed']
            )
        else:
            stats['actual_fault_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """重置统计信息"""
        self.fault_statistics = {
            'total_outputs_processed': 0,
            'total_faults_injected': 0,
            'faults_by_model': {},
            'faults_by_layer': {}
        }
    
    def print_statistics(self):
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("故障注入统计")
        print("=" * 60)
        print(f"处理的输出总数: {stats['total_outputs_processed']}")
        print(f"注入的故障总数: {stats['total_faults_injected']}")
        print(f"实际故障率: {stats['actual_fault_rate']:.4%}")
        
        if stats['faults_by_model']:
            print("\n按故障模型分类:")
            for model, count in stats['faults_by_model'].items():
                percentage = count / stats['total_faults_injected'] * 100
                print(f"  {model}: {count} ({percentage:.1f}%)")
        
        if stats['faults_by_layer']:
            print("\n按层分类 (前10层):")
            sorted_layers = sorted(
                stats['faults_by_layer'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for layer, count in sorted_layers[:10]:
                print(f"  {layer}: {count}")
        
        print("=" * 60 + "\n")


def test_injector():
    """测试故障注入器"""
    print("🧪 测试故障注入器\n")
    
    # 创建配置
    config = {
        'enabled': True,
        'fault_rate': 0.1,  # 10% 故障率（用于测试）
        'fault_models': ['bit_flip', 'output_corruption'],
        'random_seed': 42,
    }
    
    # 创建注入器
    injector = FaultInjector(config)
    
    # 创建测试输出
    test_outputs = [
        torch.randn(64, 32, 32) for _ in range(4)
    ]
    
    print("原始输出统计:")
    for i, output in enumerate(test_outputs):
        print(f"  输出{i}: mean={output.mean():.4f}, std={output.std():.4f}")
    
    # 注入故障
    print("\n注入故障...")
    faulty_outputs, fault_flags = injector.inject_fault_to_outputs(
        test_outputs, layer_name="test_layer"
    )
    
    print("\n故障输出统计:")
    for i, (output, has_fault) in enumerate(zip(faulty_outputs, fault_flags)):
        status = "🚨 有故障" if has_fault else "✓ 正常"
        print(f"  输出{i}: mean={output.mean():.4f}, std={output.std():.4f} - {status}")
    
    # 打印统计
    injector.print_statistics()
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_injector()

