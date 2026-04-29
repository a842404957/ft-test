#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
容错仿真主流程
集成所有模块，执行端到端的容错机制仿真
"""

import csv
import json
import pickle
import time

import torch
import torch.nn as nn
import numpy as np
from typing import Any, Dict, List, Tuple, Optional
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from .config import FaultToleranceConfig, get_config
from .pattern_data_loader import PatternDataLoader
from .redundancy_group_parser import RedundancyGroupParser, RedundancyGroup
from .fault_injector import FaultInjector
from .majority_voter import MajorityVoter
from .nearest_pattern_corrector import NearestPatternCorrector
from .metrics_collector import MetricsCollector
from .report_generator import ReportGenerator


def resolve_excluded_critical_layers(layer_names: List[str], configured_exclusions=None) -> List[str]:
    if not layer_names:
        return []

    if configured_exclusions in (None, '', 'auto'):
        configured_exclusions = ['__first__', '__last__']
    elif not isinstance(configured_exclusions, (list, tuple, set)):
        configured_exclusions = [configured_exclusions]

    resolved_layers = []
    for item in configured_exclusions:
        if item in ('__first__', 'first'):
            candidate = layer_names[0]
        elif item in ('__last__', 'last'):
            candidate = layer_names[-1]
        else:
            candidate = item
        if candidate in layer_names and candidate not in resolved_layers:
            resolved_layers.append(candidate)
    return resolved_layers


class FaultToleranceSimulator:
    """容错仿真器主类"""
    
    def __init__(self, 
                 model: nn.Module,
                 model_name: str = 'Vgg16',
                 translate_name: str = 'ft_group_cluster_translate',
                 config_file: str = None,
                 data_dir: str = './'):
        """
        初始化容错仿真器
        
        Args:
            model: PyTorch模型
            model_name: 模型名称
            translate_name: 转换方法名称
            config_file: 配置文件路径
            data_dir: 数据文件目录
        """
        print("=" * 70)
        print("🛡️ FT-Oriented 容错机制仿真器")
        print("=" * 70)
        
        self.model = model
        self.model_name = model_name
        self.translate_name = translate_name
        self.data_dir = Path(data_dir)

        # 加载配置
        self.config = FaultToleranceConfig(config_file)

        # 🔧 使用配置文件中的设备设置
        config_device = self.config.get('simulation', 'device', 'cuda')
        if config_device == 'cuda' and torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif config_device == 'cpu':
            self.device = torch.device('cpu')
        else:
            # 回退到模型设备或自动检测
            self.device = next(model.parameters()).device
            print(f"  ⚠️ 配置设备 {config_device} 不可用，使用模型设备: {self.device}")

        # 确保模型在正确的设备上
        model = model.to(self.device)
        
        # 初始化各个模块
        print("\n📦 初始化模块...")
        
        # 1. 数据加载器
        self.data_loader = PatternDataLoader(
            model_name=model_name,
            translate_name=translate_name,
            data_dir=data_dir
        )
        
        if not self.data_loader.load_all_data():
            raise RuntimeError("❌ 数据加载失败，无法继续")
        
        # 2. 冗余组解析器
        self.redundancy_parser = RedundancyGroupParser(
            self.data_loader,
            config=self.config.get('redundancy_group')
        )
        
        if not self.redundancy_parser.parse_all_layers():
            raise RuntimeError("❌ 冗余组解析失败，无法继续")

        # Optional V1.3.13 Level1 candidate cache. The default path does not build
        # this cache, preserving V1.3.12 behavior unless selection is requested.
        self.level1_critical_layer_config = self._load_level1_critical_layer_config()
        self.level1_repair_cache: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
        self.level1_repair_cache_by_ou: Dict[str, Dict[Tuple[int, int], Tuple[Any, ...]]] = {}
        self.level1_cache_summary: Dict[str, Any] = {}
        if self._level1_selection_cache_enabled():
            self._compile_level1_repair_cache()
        
        # 3. 故障注入器
        self.fault_injector = FaultInjector(
            config=self.config.get('fault_injection')
        )
        
        # 4. 多数表决器
        self.majority_voter = MajorityVoter(
            config=self.config.get('majority_voter')
        )
        
        # 4.5 最近邻模式纠错器 (Level 2容错)
        self.nearest_pattern_corrector = NearestPatternCorrector(
            config=self.config.config.get('hierarchical_fault_tolerance', {}).get('level2', {})
        )
        
        # 构建相似度索引（如果启用）
        hierarchical_config = self.config.config.get('hierarchical_fault_tolerance', {})
        if hierarchical_config.get('enabled', True):
            if hierarchical_config.get('level2', {}).get('enabled', True):
                print("\n🔍 构建权重模式相似度索引...")
                layer_names = self.data_loader.get_all_layer_names()
                self.nearest_pattern_corrector.build_similarity_index(
                    model=self.model,
                    layer_names=layer_names
                )
        
        # 5. 指标收集器
        self.metrics_collector = MetricsCollector(
            config=self.config.get('metrics')
        )
        
        # 6. 报告生成器
        report_dir = self.config.get('report', 'output_dir')
        self.report_generator = ReportGenerator(output_dir=report_dir)
        
        # 仿真状态
        self.simulation_results = {}
        self.detailed_fault_mask = {}
        self.fault_detail_stats = {}
        
        print("\n✅ 所有模块初始化完成\n")
    
    def run_simulation(self, 
                      test_loader: torch.utils.data.DataLoader,
                      num_samples: int = -1) -> Dict:
        """
        运行完整的容错仿真
        
        Args:
            test_loader: 测试数据加载器
            num_samples: 测试样本数（-1表示全部）
            
        Returns:
            仿真结果字典
        """
        print("=" * 70)
        print("🚀 开始容错仿真")
        print("=" * 70)
        
        self.metrics_collector.start_timing()
        
        # 1. 基线性能（无故障）
        print("\n📊 第1阶段: 评估基线性能（无故障）")
        baseline_acc = self._evaluate_baseline(test_loader, num_samples)
        print(f"  基线准确率: {baseline_acc:.6%}")
        
        # 2. 故障性能（有故障，无容错）
        print("\n📊 第2阶段: 评估故障性能（无容错）")
        faulty_acc, ou_fault_mask = self._evaluate_with_faults(test_loader, num_samples, enable_ft=False, return_fault_mask=True)
        print(f"  故障准确率: {faulty_acc:.6%}")
        print(f"  准确率下降: {(baseline_acc - faulty_acc):.6%}")
        
        # 3. 容错性能（有故障，有容错）- 使用相同的故障
        print("\n📊 第3阶段: 评估容错性能（有容错机制）")
        ft_acc = self._evaluate_with_faults(test_loader, num_samples, enable_ft=True, fault_mask=ou_fault_mask)
        print(f"  容错准确率: {ft_acc:.6%}")
        print(f"  准确率恢复: {(ft_acc - faulty_acc):.6%}")
        
        
        # 4. 收集指标
        print("\n📊 第4阶段: 收集评估指标")
        self._collect_all_metrics(baseline_acc, faulty_acc, ft_acc)
        
        # 5. 生成报告
        print("\n📊 第5阶段: 生成评估报告")
        self._generate_reports()
        
        self.metrics_collector.end_timing()
        
        # 打印最终摘要
        self.metrics_collector.print_summary()
        
        print("\n" + "=" * 70)
        print("✅ 容错仿真完成")
        print("=" * 70)
        
        return self.metrics_collector.get_all_metrics()
    
    def _evaluate_baseline(self, 
                          test_loader: torch.utils.data.DataLoader,
                          num_samples: int = -1) -> float:
        """评估基线性能（无故障）"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                if num_samples > 0 and total >= num_samples:
                    break
                if num_samples > 0 and total + target.size(0) > num_samples:
                    remaining = num_samples - total
                    data = data[:remaining]
                    target = target[:remaining]
                
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        accuracy = correct / total if total > 0 else 0.0
        return accuracy
    
    def _evaluate_with_faults(self,
                             test_loader: torch.utils.data.DataLoader,
                             num_samples: int = -1,
                             enable_ft: bool = True,
                             fault_mask: Dict[str, List[Tuple[int, int]]] = None,
                             return_fault_mask: bool = False):
        """
        评估故障下的性能
        使用Forward Hooks实现实时容错
        
        Args:
            test_loader: 测试数据加载器
            num_samples: 测试样本数
            enable_ft: 是否启用容错机制
            fault_mask: 故障mask（如果提供，则使用此mask而不重新注入）
            return_fault_mask: 是否返回故障mask
            
        Returns:
            准确率（如果return_fault_mask=True，则返回(准确率, fault_mask)）
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # 重置统计
        self.fault_injector.reset_statistics()
        self.majority_voter.reset_statistics()
        self.nearest_pattern_corrector.reset_statistics()
        
        # 保存原始权重
        original_weights = self._save_model_weights()
        
        try:
            # 1. 注入OU级别的故障到权重
            if fault_mask is None:
                # 重新注入故障
                ou_fault_mask = self._inject_ou_level_faults()
            else:
                # 使用提供的故障mask
                ou_fault_mask = fault_mask
                self._apply_fault_mask(ou_fault_mask)
            
            total_faults = sum(len(faults) for faults in ou_fault_mask.values())
            print(f"  注入故障: {total_faults} 个OU受影响")
            
            # 2. 如果启用容错，直接在权重层面完成纠正
            hooks = []  # 始终初始化hooks列表
            hierarchical_stats = None
            if enable_ft:
                hierarchical_stats = self._apply_weight_level_correction(ou_fault_mask, original_weights)
                correctable_faults = hierarchical_stats['total_correctable']
                repair_mode = hierarchical_stats.get('repair_mode', 'normal')
                if repair_mode == 'oracle':
                    print(f"  已在权重层面执行 Oracle restore: {correctable_faults}/{total_faults} 个故障")
                else:
                    print(f"  已在权重层面纠正 {correctable_faults}/{total_faults} 个故障")
                self._hierarchical_stats = hierarchical_stats
                
                # 🔍 关键验证：检查Level 2纠正的权重是否真的被修改且保持
                level2_ous = hierarchical_stats.get('level2_corrected_ous', {})
                if level2_ous:
                    print(f"\n  🔍 关键验证：推理前检查Level 2纠正的权重...")
                    for layer_name, corrected_ous in level2_ous.items():
                        if corrected_ous and len(corrected_ous) > 0:
                            check_ou = corrected_ous[0]
                            module = self._get_module_by_name(layer_name)
                            if module is not None:
                                current_mean = module.weight.data[check_ou].mean().item()
                                faulty_mean = -original_weights[layer_name][check_ou].mean().item()  # 故障是*-1
                                print(f"     {layer_name}[{check_ou}]: 当前均值={current_mean:.6f}, 故障均值={faulty_mean:.6f}")
                                if abs(current_mean - faulty_mean) < 1e-6:
                                    print(f"     ⚠️  警告：权重没有被修改！仍然是故障状态！")
                                else:
                                    print(f"     ✅ 权重已修改，不再是故障状态")
                            break  # 只检查一层
            
            # 3. 运行推理
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(test_loader):
                    if num_samples > 0 and total >= num_samples:
                        break
                    if num_samples > 0 and total + target.size(0) > num_samples:
                        remaining = num_samples - total
                        data = data[:remaining]
                        target = target[:remaining]
                    
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                    
                    # 打印进度
                    if self.config.get('simulation', 'verbose') and (batch_idx + 1) % 10 == 0:
                        current_acc = correct / total if total > 0 else 0
                        print(f"  批次 {batch_idx + 1}: 当前准确率 {current_acc:.2%}", end='\r')
        
        finally:
            # 移除hooks
            for hook in hooks:
                hook.remove()
            
            # 恢复原始权重
            self._restore_model_weights(original_weights)
        
        if self.config.get('simulation', 'verbose'):
            print()  # 换行
        
        accuracy = correct / total if total > 0 else 0.0
        
        if return_fault_mask:
            return accuracy, ou_fault_mask
        else:
            return accuracy
    
    def _save_model_weights(self) -> Dict:
        """保存模型权重的副本"""
        saved_weights = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                saved_weights[name] = param.data.clone()
        return saved_weights
    
    def _restore_model_weights(self, saved_weights: Dict):
        """恢复模型权重"""
        for name, param in self.model.named_parameters():
            if name in saved_weights:
                param.data.copy_(saved_weights[name])

    def _build_ou_list(self, weight: torch.Tensor) -> List[Tuple[int, int]]:
        """构建OU列表（按输出通道优先排序）"""
        if weight.dim() < 2:
            return []
        out_channels = weight.shape[0]
        in_channels = weight.shape[1]
        return [(out_ch, in_ch) for out_ch in range(out_channels) for in_ch in range(in_channels)]

    @staticmethod
    def _flat_ou_index_to_tuple(flat_index: int, in_channels: int) -> Tuple[int, int]:
        return (int(flat_index) // int(in_channels), int(flat_index) % int(in_channels))

    def _sample_faulty_ous(self, weight: torch.Tensor, fault_rate: float) -> Tuple[List[Tuple[int, int]], int]:
        """按flat OU索引采样，避免为Vgg16/Res50大FC层物化完整OU tuple列表。"""
        if weight.dim() < 2:
            return [], 0
        out_channels = int(weight.shape[0])
        in_channels = int(weight.shape[1])
        total_ous = out_channels * in_channels
        if total_ous <= 0 or fault_rate <= 0:
            return [], total_ous
        fault_flags = np.random.rand(total_ous) < float(fault_rate)
        faulty_indices = np.flatnonzero(fault_flags)
        faulty_ous = [self._flat_ou_index_to_tuple(int(idx), in_channels) for idx in faulty_indices]
        return faulty_ous, total_ous

    @staticmethod
    def _new_fault_detail_stats() -> Dict[str, Any]:
        return {
            'affected_weight_count': 0,
            'fault_model_counts': {},
            'layer_details': {},
        }

    @staticmethod
    def _normalize_fault_model_probs(fault_models: List[str], probs: Optional[List[float]]) -> List[float]:
        if not fault_models:
            return []
        if probs is None or len(probs) != len(fault_models):
            return [1.0 / len(fault_models)] * len(fault_models)
        probs = [max(float(p), 0.0) for p in probs]
        total = sum(probs)
        if total <= 0:
            return [1.0 / len(fault_models)] * len(fault_models)
        return [p / total for p in probs]

    @staticmethod
    def _sample_fault_model(fault_models: List[str], probs: List[float]) -> str:
        if not fault_models:
            return 'bit_flip'
        return str(np.random.choice(fault_models, p=probs))

    @staticmethod
    def _record_fault_detail(stats: Dict[str, Any], layer_name: str, model_name: str):
        stats['affected_weight_count'] = int(stats.get('affected_weight_count', 0)) + 1
        stats.setdefault('fault_model_counts', {})
        stats['fault_model_counts'][model_name] = int(stats['fault_model_counts'].get(model_name, 0)) + 1
        layer_stats = stats.setdefault('layer_details', {}).setdefault(layer_name, {
            'faulty_ou_count': 0,
            'affected_weight_count': 0,
            'fault_model_counts': {},
        })
        layer_stats['affected_weight_count'] = int(layer_stats.get('affected_weight_count', 0)) + 1
        layer_stats.setdefault('fault_model_counts', {})
        layer_stats['fault_model_counts'][model_name] = int(layer_stats['fault_model_counts'].get(model_name, 0)) + 1

    @staticmethod
    def _fault_model_count(stats: Dict[str, Any], model_name: str) -> int:
        return int(stats.get('fault_model_counts', {}).get(model_name, 0))

    def _resolve_stuck_at_one_value(self,
                                    mode: str,
                                    configured_value: Optional[float],
                                    layer_weight: torch.Tensor,
                                    global_absmax: float) -> float:
        if configured_value is not None:
            return float(configured_value)
        mode = (mode or 'layer_absmax').strip().lower()
        if mode == 'constant_one':
            return 1.0
        if mode == 'global_absmax':
            return float(global_absmax)
        # 默认用健康层权重的绝对最大值，避免不同层量级被 constant 1 放大。
        return float(layer_weight.detach().abs().max().item())

    @staticmethod
    def _apply_fault_entry(flat_weights: torch.Tensor, entry: Any, num_weights: int):
        if isinstance(entry, dict):
            idx = int(entry.get('index', -1))
            if 0 <= idx < num_weights:
                flat_weights[idx] = float(entry.get('replacement_value', flat_weights[idx].item()))
            return
        idx = int(entry)
        if 0 <= idx < num_weights:
            flat_weights[idx] *= -1

    def _sync_fault_detail_stats_to_injector(self):
        self.fault_injector.fault_statistics['faults_by_model'] = dict(
            self.fault_detail_stats.get('fault_model_counts', {})
        )
        self.fault_injector.fault_statistics['fault_detail_stats'] = self.fault_detail_stats

    def _get_mask_groups(self, layer_name: str, weight: torch.Tensor) -> Dict[bytes, List[Tuple[int, int]]]:
        """按mask形状分组OU（用于Level 2快速候选筛选）"""
        if not hasattr(self, '_mask_group_index'):
            self._mask_group_index = {}
        if layer_name in self._mask_group_index:
            return self._mask_group_index[layer_name]

        mask = self.data_loader.get_layer_mask(layer_name)
        groups: Dict[bytes, List[Tuple[int, int]]] = {}

        if mask is None or mask.dim() < 2:
            key = b'all'
            groups[key] = self._build_ou_list(weight)
            self._mask_group_index[layer_name] = groups
            return groups

        if mask.shape[:2] != weight.shape[:2]:
            key = b'all'
            groups[key] = self._build_ou_list(weight)
            self._mask_group_index[layer_name] = groups
            return groups

        out_channels = weight.shape[0]
        in_channels = weight.shape[1]
        for out_ch in range(out_channels):
            for in_ch in range(in_channels):
                mask_slice = mask[out_ch, in_ch]
                mask_key = mask_slice.flatten().to(torch.uint8).cpu().numpy().tobytes()
                groups.setdefault(mask_key, []).append((out_ch, in_ch))

        self._mask_group_index[layer_name] = groups
        return groups

    @staticmethod
    def compute_member_to_member_scale(alpha_i: float, alpha_j: float, eps: float = 1e-8) -> float:
        """从group中的两个member multiplier推导member-to-member缩放比例。"""
        return alpha_i / (alpha_j + eps)

    @staticmethod
    def _extract_ou_weight(weight_tensor: torch.Tensor, ou: Tuple[int, int]) -> torch.Tensor:
        out_ch, in_ch = ou
        if weight_tensor.dim() == 4:
            return weight_tensor[out_ch, in_ch, :, :]
        return weight_tensor[out_ch, in_ch]

    @staticmethod
    def _compute_ou_similarity(faulty_weight: torch.Tensor,
                               candidate_weight: torch.Tensor,
                               mask_slice: Optional[torch.Tensor] = None) -> float:
        if mask_slice is not None and faulty_weight.dim() > 0 and mask_slice.shape == faulty_weight.shape:
            mask_bool = mask_slice.bool()
            faulty_vec = faulty_weight[mask_bool].flatten()
            candidate_vec = candidate_weight[mask_bool].flatten()
        else:
            faulty_vec = faulty_weight.flatten()
            candidate_vec = candidate_weight.flatten()

        if faulty_vec.numel() == 0 or candidate_vec.numel() == 0:
            return 0.0

        norm_f = torch.norm(faulty_vec, p=2)
        norm_c = torch.norm(candidate_vec, p=2)
        if norm_f <= 1e-8 or norm_c <= 1e-8:
            return 0.0

        cosine = torch.dot(faulty_vec.float(), candidate_vec.float()) / (norm_f * norm_c)
        return ((cosine + 1.0) / 2.0).item()

    @staticmethod
    def _l2_error(weight: torch.Tensor, reference: torch.Tensor) -> float:
        diff = (weight - reference).float()
        if diff.numel() == 1:
            return abs(float(diff.item()))
        return torch.norm(diff, p=2).item()

    @staticmethod
    def _new_repair_quality_bucket() -> Dict[str, float]:
        return {
            'attempted': 0,
            'effective_improved': 0,
            'exact_restored': 0,
            'before_error_sum': 0.0,
            'after_error_sum': 0.0,
        }

    @staticmethod
    def _finalize_repair_quality_bucket(bucket: Dict[str, float]) -> Dict[str, float]:
        attempted = int(bucket.get('attempted', 0))
        before_error_sum = float(bucket.get('before_error_sum', 0.0))
        after_error_sum = float(bucket.get('after_error_sum', 0.0))
        return {
            'attempted': attempted,
            'effective_improved': int(bucket.get('effective_improved', 0)),
            'exact_restored': int(bucket.get('exact_restored', 0)),
            'avg_before_error': before_error_sum / max(attempted, 1),
            'avg_after_error': after_error_sum / max(attempted, 1),
            'improved_rate': float(bucket.get('effective_improved', 0)) / max(attempted, 1),
        }

    @staticmethod
    def _record_repair_quality_errors(bucket: Dict[str, float],
                                      before_error: float,
                                      after_error: float,
                                      secondary_bucket: Optional[Dict[str, float]] = None):
        for target_bucket in [bucket, secondary_bucket]:
            if target_bucket is None:
                continue
            target_bucket['attempted'] += 1
            target_bucket['before_error_sum'] += float(before_error)
            target_bucket['after_error_sum'] += float(after_error)
            if after_error + 1e-8 < before_error:
                target_bucket['effective_improved'] += 1
            if after_error < 1e-6:
                target_bucket['exact_restored'] += 1

    @staticmethod
    def _record_repair_quality(bucket: Dict[str, float],
                               before_weight: torch.Tensor,
                               after_weight: torch.Tensor,
                               original_weight: torch.Tensor,
                               secondary_bucket: Optional[Dict[str, float]] = None):
        before_error = FaultToleranceSimulator._l2_error(before_weight, original_weight)
        after_error = FaultToleranceSimulator._l2_error(after_weight, original_weight)
        FaultToleranceSimulator._record_repair_quality_errors(
            bucket,
            before_error,
            after_error,
            secondary_bucket=secondary_bucket,
        )

    @staticmethod
    def _is_invalid_scale_factor(scale_factor: float, eps: float = 1e-6) -> bool:
        if not np.isfinite(scale_factor):
            return True
        return abs(float(scale_factor)) <= eps

    def _level1_config(self) -> Dict[str, Any]:
        hierarchical = self.config.config.get('hierarchical_fault_tolerance', {})
        return hierarchical.get('level1', {}) or {}

    def _load_level1_critical_layer_config(self) -> Dict[str, Dict[str, Any]]:
        cfg = self._level1_config()
        config_path = cfg.get('critical_layer_config') or cfg.get('critical_layers_config')
        if not config_path:
            return {}

        path = Path(str(config_path))
        if not path.is_absolute():
            candidates = [Path.cwd() / path, project_root / path, self.data_dir / path]
            path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        if not path.exists():
            print(f"  ⚠️ Level1 critical layer config not found: {config_path}")
            return {}
        try:
            with open(path, 'r', encoding='utf-8') as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"  ⚠️ Failed to load Level1 critical layer config {path}: {exc}")
            return {}
        if not isinstance(payload, dict):
            return {}
        if 'layers' in payload and isinstance(payload['layers'], dict):
            payload = payload['layers']
        normalized = {}
        for layer_name, layer_cfg in payload.items():
            if isinstance(layer_cfg, dict):
                normalized[str(layer_name)] = dict(layer_cfg)
        return normalized

    def _effective_level1_policy(self, layer_name: str) -> Dict[str, Any]:
        cfg = dict(self._level1_config())
        layer_cfg = self.level1_critical_layer_config.get(layer_name, {})
        mode = str(layer_cfg.get('selection', cfg.get('selection', 'default')) or 'default').strip().lower()
        topk = int(layer_cfg.get('topk', cfg.get('topk', 3)) or 3)
        max_expected_error = layer_cfg.get('max_expected_error', cfg.get('max_expected_error', None))
        min_expected_improvement = layer_cfg.get(
            'min_expected_improvement',
            cfg.get('min_expected_improvement', 0.0),
        )
        return {
            'selection': mode if mode in ('default', 'best_pair', 'weighted_average') else 'default',
            'topk': max(1, topk),
            'max_expected_error': None if max_expected_error in (None, '') else float(max_expected_error),
            'min_expected_improvement': float(min_expected_improvement or 0.0),
        }

    def _level1_selection_cache_enabled(self) -> bool:
        cfg = self._level1_config()
        global_mode = str(cfg.get('selection', 'default') or 'default').strip().lower()
        configured_modes = [
            str(layer_cfg.get('selection', 'default') or 'default').strip().lower()
            for layer_cfg in self.level1_critical_layer_config.values()
        ]
        requested = global_mode != 'default' or any(mode != 'default' for mode in configured_modes)
        if not requested:
            return False
        if bool(cfg.get('cache_critical_layers_only', False)) and not self.level1_critical_layer_config:
            print("  ⚠️ Level1 cache critical-only requested but no critical layer config was loaded; using default Level1 selection")
            return False
        return True

    def _layer_should_build_level1_cache(self, layer_name: str) -> bool:
        cfg = self._level1_config()
        if bool(cfg.get('cache_critical_layers_only', False)):
            return layer_name in self.level1_critical_layer_config
        return self._effective_level1_policy(layer_name)['selection'] != 'default' or layer_name in self.level1_critical_layer_config

    def _block_for_ou(self, group: RedundancyGroup, ou: Tuple[int, int]) -> Tuple[Optional[Dict[str, Any]], int]:
        entry = group.ou_to_block.get(ou)
        if entry is not None and entry['member_index'] < len(group.block_members):
            return group.block_members[entry['member_index']], int(entry['offset'])
        return None, 0

    def _multiplier_for_ou(self, group: RedundancyGroup, ou: Tuple[int, int], block: Optional[Dict[str, Any]]) -> float:
        if block is not None:
            return float(block.get('multiplier', 1.0))
        try:
            return float(group.multipliers[group.ou_indices.index(ou)])
        except (ValueError, IndexError):
            return 1.0

    def _compatible_level1_candidates(self,
                                      group: RedundancyGroup,
                                      faulty_ou: Tuple[int, int],
                                      block_offset: int) -> List[Tuple[Optional[Dict[str, Any]], Tuple[int, int], int]]:
        candidates = []
        if group.block_members:
            for candidate_block in group.block_members:
                channel_span = int(candidate_block.get('channel_span', 1))
                if block_offset >= channel_span:
                    continue
                candidate_ou = (
                    int(candidate_block['out_ch']),
                    int(candidate_block['in_ch_start']) + block_offset,
                )
                if candidate_ou == faulty_ou:
                    continue
                candidates.append((candidate_block, candidate_ou, block_offset))
            return candidates

        for candidate_ou in group.ou_indices:
            if candidate_ou == faulty_ou:
                continue
            candidates.append((None, candidate_ou, 0))
        return candidates

    def _compute_expected_repair_error(self,
                                       target_weight: torch.Tensor,
                                       candidate_weight: torch.Tensor,
                                       scale_factor: float) -> float:
        if self._is_invalid_scale_factor(scale_factor):
            return float('inf')
        target_vec = target_weight.detach().float()
        candidate_vec = candidate_weight.detach().float() * float(scale_factor)
        denom = torch.norm(target_vec, p=2).item()
        if denom <= 1e-8:
            denom = 1e-8
        error = torch.norm((target_vec - candidate_vec).float(), p=2).item() / denom
        return float(error) if np.isfinite(error) else float('inf')

    def _prefilter_level1_candidates(self,
                                     target_weight: torch.Tensor,
                                     target_multiplier: float,
                                     candidates: List[Tuple[Optional[Dict[str, Any]], Tuple[int, int], int]],
                                     layer_weights: torch.Tensor,
                                     limit: int) -> List[Tuple[Optional[Dict[str, Any]], Tuple[int, int], int]]:
        if len(candidates) <= limit:
            return candidates
        target_norm = torch.norm(target_weight.detach().float(), p=2).item()
        scored = []
        for candidate_block, candidate_ou, candidate_block_offset in candidates:
            candidate_multiplier = self._multiplier_for_ou(
                RedundancyGroup(-1, '', -1),
                candidate_ou,
                candidate_block,
            ) if candidate_block is not None else 1.0
            if candidate_block is None or abs(candidate_multiplier) <= 1e-8:
                multiplier_penalty = 1.0
            else:
                scale = self.compute_member_to_member_scale(target_multiplier, candidate_multiplier)
                multiplier_penalty = 0.0 if not self._is_invalid_scale_factor(scale) else 10.0
            candidate_weight = self._extract_ou_weight(layer_weights, candidate_ou)
            candidate_norm = torch.norm(candidate_weight.detach().float(), p=2).item()
            norm_penalty = abs(np.log((candidate_norm + 1e-8) / (target_norm + 1e-8)))
            similarity = self._compute_ou_similarity(target_weight, candidate_weight)
            scored.append((multiplier_penalty + norm_penalty + (1.0 - similarity), candidate_block, candidate_ou, candidate_block_offset))
        scored.sort(key=lambda item: item[0])
        return [(block, ou, offset) for _, block, ou, offset in scored[:limit]]

    def _compile_level1_repair_cache(self):
        cfg = self._level1_config()
        max_group_size = int(cfg.get('cache_max_group_size', 4096) or 4096)
        global_topk = int(cfg.get('topk', 3) or 3)
        prefilter_limit = max(global_topk * 16, 64)
        cache_rows = []
        summary_rows = []

        print("\n🔧 编译 Level1 repair candidate cache...")
        for layer_name in self.data_loader.get_all_layer_names():
            if not self._layer_should_build_level1_cache(layer_name):
                continue
            module = self._get_module_by_name(layer_name)
            if module is None or not hasattr(module, 'weight'):
                continue
            layer_weights = module.weight.data
            groups = self.redundancy_parser.get_layer_groups(layer_name)
            policy = self._effective_level1_policy(layer_name)
            topk = policy['topk']
            layer_errors = []
            no_valid = 0
            cache_entries = 0
            ou_count = 0

            for group in groups:
                if group.size() < 2:
                    continue
                group_ous = list(dict.fromkeys(group.ou_indices))
                for faulty_ou in group_ous:
                    faulty_block, block_offset = self._block_for_ou(group, faulty_ou)
                    target_multiplier = self._multiplier_for_ou(group, faulty_ou, faulty_block)
                    if self._is_invalid_scale_factor(target_multiplier):
                        no_valid += 1
                        continue
                    target_weight = self._extract_ou_weight(layer_weights, faulty_ou)
                    candidates = self._compatible_level1_candidates(group, faulty_ou, block_offset)
                    if group.size() > max_group_size:
                        candidates = self._prefilter_level1_candidates(
                            target_weight,
                            target_multiplier,
                            candidates,
                            layer_weights,
                            prefilter_limit,
                        )

                    records = []
                    for candidate_block, candidate_ou, candidate_block_offset in candidates:
                        candidate_multiplier = self._multiplier_for_ou(group, candidate_ou, candidate_block)
                        if abs(candidate_multiplier) <= 1e-8:
                            continue
                        scale_factor = self.compute_member_to_member_scale(target_multiplier, candidate_multiplier)
                        if self._is_invalid_scale_factor(scale_factor):
                            continue
                        candidate_weight = self._extract_ou_weight(layer_weights, candidate_ou)
                        expected_error = self._compute_expected_repair_error(target_weight, candidate_weight, scale_factor)
                        if not np.isfinite(expected_error):
                            continue
                        records.append({
                            'layer': layer_name,
                            'group_id': int(group.group_id),
                            'faulty_ou': tuple(int(v) for v in faulty_ou),
                            'faulty_block_start': int(faulty_block.get('in_ch_start', faulty_ou[1])) if faulty_block else int(faulty_ou[1]),
                            'channel_span': int(faulty_block.get('channel_span', 1)) if faulty_block else 1,
                            'block_offset': int(block_offset),
                            'candidate_ou': tuple(int(v) for v in candidate_ou),
                            'candidate_block_start': int(candidate_block.get('in_ch_start', candidate_ou[1])) if candidate_block else int(candidate_ou[1]),
                            'candidate_block_offset': int(candidate_block_offset),
                            'scale_factor': float(scale_factor),
                            'expected_error': float(expected_error),
                            'group_size': int(group.size()),
                        })

                    records.sort(key=lambda item: item['expected_error'])
                    records = records[:topk]
                    if not records:
                        no_valid += 1
                        continue
                    for rank, record in enumerate(records, start=1):
                        record['candidate_rank'] = rank
                        cache_rows.append(record)
                    key = (
                        layer_name,
                        int(group.group_id),
                        int(faulty_ou[0]),
                        int(faulty_ou[1]),
                        records[0]['faulty_block_start'],
                        records[0]['channel_span'],
                        records[0]['block_offset'],
                    )
                    self.level1_repair_cache[key] = records
                    self.level1_repair_cache_by_ou.setdefault(layer_name, {})[faulty_ou] = key
                    layer_errors.extend([record['expected_error'] for record in records])
                    cache_entries += len(records)
                    ou_count += 1

            summary = {
                'layer': layer_name,
                'selection': policy['selection'],
                'topk': topk,
                'cached_ou_count': ou_count,
                'cache_entries': cache_entries,
                'no_valid_candidate_count': no_valid,
                'cache_coverage': ou_count / max(ou_count + no_valid, 1),
                'avg_expected_error': float(np.mean(layer_errors)) if layer_errors else 0.0,
                'p95_expected_error': float(np.percentile(layer_errors, 95)) if layer_errors else 0.0,
            }
            summary_rows.append(summary)
            print(
                f"  ✓ {layer_name}: cached_ous={ou_count} entries={cache_entries} "
                f"avg_expected={summary['avg_expected_error']:.4f}"
            )

        self.level1_cache_summary = {
            'rows': summary_rows,
            'total_cache_entries': len(cache_rows),
            'critical_layers_only': bool(cfg.get('cache_critical_layers_only', False)),
            'max_group_size': max_group_size,
        }
        self._write_level1_repair_cache_files(cache_rows, summary_rows)

    def _write_level1_repair_cache_files(self, cache_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]]):
        if not cache_rows and not summary_rows:
            return
        self.data_dir.mkdir(parents=True, exist_ok=True)
        stem = f"model_{self.model_name}_{self.translate_name}"
        cache_path = self.data_dir / f"{stem}_level1_repair_cache.pkl"
        summary_csv = self.data_dir / f"{stem}_repair_cache_summary.csv"
        summary_json = self.data_dir / f"{stem}_repair_cache_summary.json"
        with open(cache_path, 'wb') as handle:
            pickle.dump({
                'cache': self.level1_repair_cache,
                'cache_by_ou': self.level1_repair_cache_by_ou,
                'rows': cache_rows,
                'summary': self.level1_cache_summary,
            }, handle, pickle.HIGHEST_PROTOCOL)
        fieldnames = [
            'layer', 'selection', 'topk', 'cached_ou_count', 'cache_entries',
            'no_valid_candidate_count', 'cache_coverage', 'avg_expected_error',
            'p95_expected_error',
        ]
        with open(summary_csv, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_rows:
                writer.writerow({name: row.get(name, '') for name in fieldnames})
        summary_json.write_text(json.dumps(self.level1_cache_summary, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"  🧾 Level1 repair cache saved: {cache_path}")

    def _get_level1_cache_records(self, layer_name: str, faulty_ou: Tuple[int, int]) -> List[Dict[str, Any]]:
        key = self.level1_repair_cache_by_ou.get(layer_name, {}).get(faulty_ou)
        if key is None:
            return []
        return list(self.level1_repair_cache.get(key, []))

    @staticmethod
    def _new_level1_selection_stats() -> Dict[str, Any]:
        return {
            'level1_selection_mode': 'default',
            'best_pair_used': 0,
            'weighted_average_used': 0,
            'fallback_to_default': 0,
            'low_confidence_repair': 0,
            'fallback_reasons': {},
            'summary_count': 0,
            'summary_sums': {
                'default_expected_error': 0.0,
                'selected_expected_error': 0.0,
                'expected_error_improvement': 0.0,
                'actual_before_error': 0.0,
                'actual_after_error': 0.0,
                'actual_error_improvement': 0.0,
            },
            'summary_counts': {
                'default_expected_error': 0,
                'selected_expected_error': 0,
                'expected_error_improvement': 0,
                'actual_before_error': 0,
                'actual_after_error': 0,
                'actual_error_improvement': 0,
            },
            'max_sampled_events': 10000,
            'events': [],
        }

    @staticmethod
    def _increment_reason(container: Dict[str, int], reason: str):
        container[reason] = int(container.get(reason, 0)) + 1

    @staticmethod
    def _pearson_corr(pairs: List[Tuple[float, float]]) -> Optional[float]:
        clean = [(float(x), float(y)) for x, y in pairs if np.isfinite(x) and np.isfinite(y)]
        if len(clean) < 2:
            return None
        xs = np.asarray([item[0] for item in clean], dtype=np.float64)
        ys = np.asarray([item[1] for item in clean], dtype=np.float64)
        if np.std(xs) <= 1e-12 or np.std(ys) <= 1e-12:
            return None
        return float(np.corrcoef(xs, ys)[0, 1])

    def _record_level1_selection_event(self,
                                       stats: Dict[str, Any],
                                       layer_name: str,
                                       mode: str,
                                       default_expected_error: float,
                                       selected_expected_error: float,
                                       actual_before_error: float,
                                       actual_after_error: float,
                                       fallback_reason: str = ''):
        if mode == 'best_pair':
            stats['best_pair_used'] += 1
        elif mode == 'weighted_average':
            stats['weighted_average_used'] += 1
        if fallback_reason:
            stats['fallback_to_default'] += 1
            self._increment_reason(stats['fallback_reasons'], fallback_reason)
        event = {
            'layer': layer_name,
            'selection_mode': mode,
            'fallback_reason': fallback_reason,
            'default_expected_error': float(default_expected_error) if np.isfinite(default_expected_error) else None,
            'selected_expected_error': float(selected_expected_error) if np.isfinite(selected_expected_error) else None,
            'expected_error_improvement': (
                float(default_expected_error - selected_expected_error)
                if np.isfinite(default_expected_error) and np.isfinite(selected_expected_error)
                else None
            ),
            'actual_before_error': float(actual_before_error),
            'actual_after_error': float(actual_after_error),
            'actual_error_improvement': float(actual_before_error - actual_after_error),
        }
        stats['summary_count'] = int(stats.get('summary_count', 0)) + 1
        for field, value in event.items():
            if field not in stats['summary_sums'] or value is None:
                continue
            stats['summary_sums'][field] += float(value)
            stats['summary_counts'][field] += 1

        # Keep a bounded sample for expected/actual correlation. Storing one
        # dict per repaired OU is too expensive for Vgg16-scale FC layers.
        if len(stats['events']) < int(stats.get('max_sampled_events', 10000)):
            stats['events'].append(event)

    def _finalize_level1_selection_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        events = stats.get('events', [])

        def avg(field: str) -> Optional[float]:
            count = int(stats.get('summary_counts', {}).get(field, 0))
            if count > 0:
                return float(stats.get('summary_sums', {}).get(field, 0.0)) / count
            values = [event.get(field) for event in events if event.get(field) is not None]
            return float(np.mean(values)) if values else None

        by_layer_pairs: Dict[str, List[Tuple[float, float]]] = {}
        global_pairs = []
        for event in events:
            expected = event.get('selected_expected_error')
            actual = event.get('actual_after_error')
            if expected is None or actual is None:
                continue
            global_pairs.append((expected, actual))
            by_layer_pairs.setdefault(event['layer'], []).append((expected, actual))

        by_layer_corr = {
            layer: self._pearson_corr(pairs)
            for layer, pairs in sorted(by_layer_pairs.items())
        }
        mode_counts = {}
        for event in events:
            mode = event.get('selection_mode', 'default')
            mode_counts[mode] = int(mode_counts.get(mode, 0)) + 1

        effective_mode = stats.get('level1_selection_mode', 'default')
        if mode_counts:
            effective_mode = next(iter(mode_counts.keys())) if len(mode_counts) == 1 else 'mixed'
        return {
            'level1_selection_mode': effective_mode,
            'best_pair_used': int(stats.get('best_pair_used', 0)),
            'weighted_average_used': int(stats.get('weighted_average_used', 0)),
            'fallback_to_default': int(stats.get('fallback_to_default', 0)),
            'fallback_reason': dict(stats.get('fallback_reasons', {})),
            'low_confidence_repair': int(stats.get('low_confidence_repair', 0)),
            'default_expected_error': avg('default_expected_error'),
            'selected_expected_error': avg('selected_expected_error'),
            'expected_error_improvement': avg('expected_error_improvement'),
            'actual_before_error': avg('actual_before_error'),
            'actual_after_error': avg('actual_after_error'),
            'actual_error_improvement': avg('actual_error_improvement'),
            'expected_actual_corr': self._pearson_corr(global_pairs),
            'expected_actual_corr_by_layer': by_layer_corr,
            'mode_counts': mode_counts,
            'event_count': int(stats.get('summary_count', len(events))),
            'sampled_event_count': len(events),
        }

    def _select_level1_repair_choice(self,
                                     layer_name: str,
                                     faulty_ou: Tuple[int, int],
                                     faulty_set: set,
                                     original_layer_weights: torch.Tensor,
                                     default_choice: Dict[str, Any],
                                     policy: Dict[str, Any]) -> Dict[str, Any]:
        mode = policy['selection']
        if mode == 'default':
            return default_choice

        records = [
            record for record in self._get_level1_cache_records(layer_name, faulty_ou)
            if tuple(record.get('candidate_ou', ())) not in faulty_set
        ][:policy['topk']]
        if not records:
            choice = dict(default_choice)
            choice['fallback_reason'] = 'no_healthy_cache_candidate'
            return choice

        max_expected_error = policy.get('max_expected_error')
        if max_expected_error is not None:
            filtered = [record for record in records if float(record['expected_error']) <= max_expected_error]
            if not filtered:
                choice = dict(default_choice)
                choice['fallback_reason'] = 'max_expected_error'
                choice['low_confidence'] = True
                return choice
            records = filtered

        if mode == 'best_pair':
            selected = min(records, key=lambda record: float(record['expected_error']))
            candidate_ou = tuple(selected['candidate_ou'])
            candidate_weight = self._extract_ou_weight(original_layer_weights, candidate_ou)
            corrected_weight = candidate_weight * float(selected['scale_factor'])
            selected_choice = {
                'replacement_ou': candidate_ou,
                'replacement_origin': 'best_pair',
                'scale_factor': float(selected['scale_factor']),
                'corrected_weight': corrected_weight,
                'selected_expected_error': float(selected['expected_error']),
                'mode': 'best_pair',
                'candidate_records': [selected],
            }
        elif mode == 'weighted_average':
            weighted_terms = []
            weights = []
            for record in records:
                candidate_ou = tuple(record['candidate_ou'])
                candidate_weight = self._extract_ou_weight(original_layer_weights, candidate_ou)
                weight = 1.0 / (float(record['expected_error']) + 1e-8)
                if not np.isfinite(weight):
                    continue
                weighted_terms.append(candidate_weight * float(record['scale_factor']) * weight)
                weights.append(weight)
            if not weighted_terms or sum(weights) <= 0:
                choice = dict(default_choice)
                choice['fallback_reason'] = 'weighted_average_invalid'
                return choice
            corrected_weight = torch.stack(weighted_terms, dim=0).sum(dim=0) / float(sum(weights))
            if not torch.isfinite(corrected_weight).all():
                choice = dict(default_choice)
                choice['fallback_reason'] = 'weighted_average_nonfinite'
                return choice
            selected_choice = {
                'replacement_ou': tuple(records[0]['candidate_ou']),
                'replacement_origin': 'weighted_average',
                'scale_factor': float(records[0]['scale_factor']),
                'corrected_weight': corrected_weight,
                'selected_expected_error': float(np.average(
                    [float(record['expected_error']) for record in records],
                    weights=weights,
                )),
                'mode': 'weighted_average',
                'candidate_records': records,
            }
        else:
            return default_choice

        default_expected = float(default_choice.get('default_expected_error', float('inf')))
        selected_expected = float(selected_choice['selected_expected_error'])
        min_improvement = float(policy.get('min_expected_improvement', 0.0) or 0.0)
        if np.isfinite(default_expected) and default_expected - selected_expected < min_improvement:
            choice = dict(default_choice)
            choice['fallback_reason'] = 'insufficient_expected_improvement'
            choice['candidate_records'] = selected_choice.get('candidate_records', [])
            return choice
        selected_choice['default_expected_error'] = default_expected
        selected_choice['fallback_reason'] = ''
        return selected_choice
    
    def _inject_ou_level_faults(self) -> Dict[str, List[Tuple[int, int]]]:
        """
        在OU（输出通道）级别注入故障
        
        Returns:
            Dict[layer_name, List[(out_ch, in_ch)]]: 每层中故障OU的坐标列表
        """
        ou_fault_mask = {}
        fault_config = self.config.get('fault_injection') or {}
        if not fault_config.get('enabled', True):
            print("\n  ⚠️ 故障注入已禁用")
            return ou_fault_mask

        fault_rate = fault_config.get('fault_rate', 0.0)
        layer_names = self.data_loader.get_all_layer_names()

        # 🔧 从配置文件读取排除的关键层
        excluded_layers = resolve_excluded_critical_layers(
            layer_names,
            fault_config.get('exclude_critical_layers', ['__first__', '__last__'])
        )
        target_layers = fault_config.get('target_layers', 'all')

        # 初始化详细故障位置记录
        self.detailed_fault_mask = {}
        self.fault_detail_stats = self._new_fault_detail_stats()
        fault_models = [str(model) for model in fault_config.get('fault_models', ['bit_flip'])]
        fault_model_probs = self._normalize_fault_model_probs(fault_models, fault_config.get('fault_model_probs'))
        use_structured_weight_faults = any(
            model in ('stuck_at_zero', 'stuck_at_one') for model in fault_models
        )
        fault_weight_ratio = float(fault_config.get('fault_weight_ratio', 1.0))
        stuck_at_one_mode = fault_config.get('stuck_at_one_value_mode', 'layer_absmax')
        stuck_at_one_value = fault_config.get('stuck_at_one_value')
        stuck_at_zero_value = float(fault_config.get('stuck_at_zero_value', 0.0))

        global_absmax = 1.0
        if stuck_at_one_mode == 'global_absmax':
            global_values = []
            for candidate_layer in layer_names:
                module = self._get_module_by_name(candidate_layer)
                if module is not None and hasattr(module, 'weight'):
                    global_values.append(float(module.weight.data.detach().abs().max().item()))
            if global_values:
                global_absmax = max(global_values)

        # 🔍 添加故障分布统计
        print(f"\n  🔍 故障注入详细分布（故障率={fault_rate}）:")
        if excluded_layers:
            print(f"  ⚠️  已排除关键层: {excluded_layers}")

        for layer_name in layer_names:
            if target_layers != 'all':
                if isinstance(target_layers, (list, tuple, set)):
                    if layer_name not in target_layers:
                        continue
                else:
                    if layer_name != target_layers:
                        continue
            if layer_name in excluded_layers:
                print(f"    {layer_name}: 跳过（关键层）")
                continue
            module = self._get_module_by_name(layer_name)
            if module is None or not hasattr(module, 'weight'):
                continue

            weight = module.weight.data
            if weight.dim() < 2:
                continue

            faulty_ous, total_ous = self._sample_faulty_ous(weight, fault_rate)
            if total_ous == 0 or fault_rate <= 0:
                ou_fault_mask[layer_name] = []
                continue

            self.detailed_fault_mask[layer_name] = {}

            # 注入故障到选定的OU
            for out_ch, in_ch in faulty_ous:
                if weight.dim() == 4:
                    ou_weights = weight[out_ch, in_ch, :, :].clone()
                    num_weights = ou_weights.numel()
                    if num_weights == 0:
                        continue

                    flat_weights = ou_weights.flatten()
                    if use_structured_weight_faults:
                        num_faulted_weights = max(1, int(num_weights * fault_weight_ratio))
                        num_faulted_weights = min(num_weights, num_faulted_weights)
                        flat_indices = torch.randperm(num_weights, device=weight.device)[:num_faulted_weights]
                        one_value = self._resolve_stuck_at_one_value(
                            mode=stuck_at_one_mode,
                            configured_value=stuck_at_one_value,
                            layer_weight=weight,
                            global_absmax=global_absmax,
                        )
                        entries = []
                        for raw_idx in flat_indices.cpu().tolist():
                            fault_model = self._sample_fault_model(fault_models, fault_model_probs)
                            if fault_model == 'stuck_at_zero':
                                replacement_value = stuck_at_zero_value
                            elif fault_model == 'stuck_at_one':
                                replacement_value = one_value
                            else:
                                fault_model = 'bit_flip'
                                replacement_value = -float(flat_weights[raw_idx].item())
                            flat_weights[raw_idx] = float(replacement_value)
                            entries.append({
                                'index': int(raw_idx),
                                'fault_model': fault_model,
                                'replacement_value': float(replacement_value),
                            })
                            self._record_fault_detail(self.fault_detail_stats, layer_name, fault_model)
                        self.detailed_fault_mask[layer_name][(out_ch, in_ch)] = entries
                        weight[out_ch, in_ch, :, :] = flat_weights.reshape(ou_weights.shape)
                    else:
                        num_weights_to_fault = min(8, num_weights)
                        if num_weights_to_fault == 0:
                            continue
                        bit_flip_ratio = fault_config.get('bit_flip_ratio', 0.25)
                        num_flipped = max(1, int(num_weights_to_fault * bit_flip_ratio))
                        flat_indices = torch.randperm(num_weights_to_fault, device=weight.device)[:num_flipped]
                        flat_weights[flat_indices] *= -1

                        self.detailed_fault_mask[layer_name][(out_ch, in_ch)] = flat_indices.cpu().tolist()
                        weight[out_ch, in_ch, :, :] = flat_weights.reshape(ou_weights.shape)
                        for _ in flat_indices.cpu().tolist():
                            self._record_fault_detail(self.fault_detail_stats, layer_name, 'bit_flip')
                elif weight.dim() == 2:
                    if use_structured_weight_faults:
                        fault_model = self._sample_fault_model(fault_models, fault_model_probs)
                        one_value = self._resolve_stuck_at_one_value(
                            mode=stuck_at_one_mode,
                            configured_value=stuck_at_one_value,
                            layer_weight=weight,
                            global_absmax=global_absmax,
                        )
                        if fault_model == 'stuck_at_zero':
                            replacement_value = stuck_at_zero_value
                        elif fault_model == 'stuck_at_one':
                            replacement_value = one_value
                        else:
                            fault_model = 'bit_flip'
                            replacement_value = -float(weight[out_ch, in_ch].item())
                        weight[out_ch, in_ch] = float(replacement_value)
                        self.detailed_fault_mask[layer_name][(out_ch, in_ch)] = [{
                            'index': 0,
                            'fault_model': fault_model,
                            'replacement_value': float(replacement_value),
                        }]
                        self._record_fault_detail(self.fault_detail_stats, layer_name, fault_model)
                    else:
                        weight[out_ch, in_ch] *= -1
                        self.detailed_fault_mask[layer_name][(out_ch, in_ch)] = [0]
                        self._record_fault_detail(self.fault_detail_stats, layer_name, 'bit_flip')

            # 记录故障
            ou_fault_mask[layer_name] = faulty_ous

            # 更新统计
            num_faulty_ous = len(faulty_ous)
            self.fault_injector.fault_statistics['total_faults_injected'] += num_faulty_ous
            self.fault_injector.fault_statistics['faults_by_layer'][layer_name] = num_faulty_ous
            self.fault_detail_stats.setdefault('layer_details', {}).setdefault(layer_name, {
                'faulty_ou_count': 0,
                'affected_weight_count': 0,
                'fault_model_counts': {},
            })['faulty_ou_count'] = num_faulty_ous

            total_flipped = sum(len(indices) for indices in self.detailed_fault_mask[layer_name].values())
            layer_stats = self.fault_detail_stats['layer_details'].get(layer_name, {})
            zero_count = layer_stats.get('fault_model_counts', {}).get('stuck_at_zero', 0)
            one_count = layer_stats.get('fault_model_counts', {}).get('stuck_at_one', 0)
            if use_structured_weight_faults:
                print(f"    {layer_name}: {num_faulty_ous} 个故障OU, {total_flipped} 个权重stuck-at "
                      f"(zero={zero_count}, one={one_count}, 总共 {total_ous} 个OU)")
            else:
                print(f"    {layer_name}: {num_faulty_ous} 个故障OU, {total_flipped} 个权重翻转 (总共 {total_ous} 个OU)")
        
        self._sync_fault_detail_stats_to_injector()
        return ou_fault_mask
    
    def _apply_fault_mask(self, fault_mask: Dict[str, List[Tuple[int, int]]]):
        """
        应用给定的故障mask到模型权重
        
        Args:
            fault_mask: 每层的故障OU坐标列表
        """
        self.fault_detail_stats = self._new_fault_detail_stats()
        for layer_name, faulty_ou_indices in fault_mask.items():
            module = self._get_module_by_name(layer_name)
            if module is None or not hasattr(module, 'weight'):
                continue
            
            weight = module.weight.data

            if weight.dim() < 2:
                continue

            all_ous = None
            self.fault_detail_stats.setdefault('layer_details', {}).setdefault(layer_name, {
                'faulty_ou_count': len(faulty_ou_indices),
                'affected_weight_count': 0,
                'fault_model_counts': {},
            })['faulty_ou_count'] = len(faulty_ou_indices)

            # 应用故障到指定的OU
            for ou_entry in faulty_ou_indices:
                if isinstance(ou_entry, (list, tuple)) and len(ou_entry) == 2:
                    out_ch, in_ch = ou_entry
                else:
                    if all_ous is None:
                        all_ous = self._build_ou_list(weight)
                    if ou_entry >= len(all_ous):
                        continue
                    out_ch, in_ch = all_ous[ou_entry]

                if weight.dim() == 4:
                    if hasattr(self, 'detailed_fault_mask') and layer_name in self.detailed_fault_mask:
                        fault_entries = self.detailed_fault_mask[layer_name].get((out_ch, in_ch))
                    else:
                        fault_entries = None

                    if fault_entries:
                        ou_weights = weight[out_ch, in_ch, :, :].clone()
                        flat_weights = ou_weights.flatten()
                        num_weights_to_fault = min(8, flat_weights.numel())
                        for entry in fault_entries:
                            if isinstance(entry, dict):
                                model_name = str(entry.get('fault_model', 'bit_flip'))
                                self._apply_fault_entry(flat_weights, entry, flat_weights.numel())
                            else:
                                model_name = 'bit_flip'
                                self._apply_fault_entry(flat_weights, entry, num_weights_to_fault)
                            self._record_fault_detail(self.fault_detail_stats, layer_name, model_name)
                        weight[out_ch, in_ch, :, :] = flat_weights.reshape(ou_weights.shape)
                    else:
                        weight[out_ch, in_ch, :, :] *= -1
                        self._record_fault_detail(self.fault_detail_stats, layer_name, 'bit_flip')
                elif weight.dim() == 2:
                    if hasattr(self, 'detailed_fault_mask') and layer_name in self.detailed_fault_mask:
                        fault_entries = self.detailed_fault_mask[layer_name].get((out_ch, in_ch))
                    else:
                        fault_entries = None
                    if fault_entries and isinstance(fault_entries[0], dict):
                        entry = fault_entries[0]
                        weight[out_ch, in_ch] = float(entry.get('replacement_value', weight[out_ch, in_ch].item()))
                        self._record_fault_detail(
                            self.fault_detail_stats,
                            layer_name,
                            str(entry.get('fault_model', 'bit_flip')),
                        )
                    else:
                        weight[out_ch, in_ch] *= -1
                        self._record_fault_detail(self.fault_detail_stats, layer_name, 'bit_flip')
            
            # 更新统计
            num_faulty_ous = len(faulty_ou_indices)
            self.fault_injector.fault_statistics['total_faults_injected'] += num_faulty_ous
            self.fault_injector.fault_statistics['faults_by_layer'][layer_name] = num_faulty_ous
        self._sync_fault_detail_stats_to_injector()
    
    def _apply_weight_level_correction(self,
                                      ou_fault_mask: Dict[str, List[Tuple[int, int]]],
                                      original_weights: Dict[str, torch.Tensor]) -> Dict:
        """在权重层面完成三级容错，返回实际纠正统计
        
        三级容错策略：
        - Level 1: 冗余组直接替换（100%恢复） - 用冗余组内健康OU的权重直接替换故障OU
        - Level 2: 相似模式替换（95%恢复） - 用最相似OU的权重替换故障OU（允许小误差）
        - Level 3: 自适应屏蔽（置0） - 将无法纠正的故障OU权重置0，最小化影响
        
        Args:
            ou_fault_mask: 故障mask，记录每层中哪些OU发生了故障
            original_weights: 原始健康权重，用于获取替换权重
            
        Returns:
            容错统计信息字典
        """
        hierarchical_config = self.config.config.get('hierarchical_fault_tolerance', {})
        repair_mode = str(hierarchical_config.get('repair_mode', 'normal')).strip().lower() or 'normal'

        level1_count = 0
        level2_count = 0
        level3_count = 0
        oracle_count = 0
        level1_prototype_repairs = 0
        level1_member_repairs = 0
        level1_exact_repairs = 0
        level1_scaled_repairs = 0
        level1_failed_singleton = 0
        level1_zero_scale_failed = 0
        level1_selection_stats = self._new_level1_selection_stats()
        level1_selection_stats['level1_selection_mode'] = str(
            hierarchical_config.get('level1', {}).get('selection', 'default') or 'default'
        ).strip().lower()

        repair_quality_buckets = {
            'level1': self._new_repair_quality_bucket(),
            'level2': self._new_repair_quality_bucket(),
            'level3': self._new_repair_quality_bucket(),
        }
        if repair_mode == 'oracle':
            repair_quality_buckets['oracle'] = self._new_repair_quality_bucket()
        layer_repair_quality_buckets: Dict[str, Dict[str, Dict[str, float]]] = {}

        def layer_quality_bucket(layer: str, level: str) -> Dict[str, float]:
            return layer_repair_quality_buckets.setdefault(layer, {}).setdefault(
                level,
                self._new_repair_quality_bucket(),
            )

        level1_corrected_ous: Dict[str, List[Tuple[int, int]]] = {}
        level2_corrected_ous: Dict[str, List[Tuple[int, int]]] = {}
        level3_corrected_ous: Dict[str, List[Tuple[int, int]]] = {}
        oracle_corrected_ous: Dict[str, List[Tuple[int, int]]] = {}
        uncorrected_by_layer: Dict[str, List[Tuple[int, int]]] = {}

        for layer_name, faulty_ou_indices in ou_fault_mask.items():
            if faulty_ou_indices is None or len(faulty_ou_indices) == 0:
                continue

            module = self._get_module_by_name(layer_name)
            if module is None or not hasattr(module, 'weight'):
                continue

            layer_weights = module.weight.data
            original_layer_weights = original_weights.get(layer_name)
            if original_layer_weights is None:
                original_layer_weights = layer_weights.clone()

            groups = self.redundancy_parser.get_layer_groups(layer_name)

            faulty_ous = []
            if faulty_ou_indices and isinstance(faulty_ou_indices[0], (list, tuple)):
                faulty_ous = [tuple(ou) for ou in faulty_ou_indices]
            else:
                all_ous = self._build_ou_list(layer_weights)
                for idx in faulty_ou_indices:
                    if idx < len(all_ous):
                        faulty_ous.append(all_ous[idx])

            faulty_set = set(faulty_ous)

            layer_level1 = []
            layer_level2 = []
            layer_level3 = []
            layer_oracle = []
            layer_start_time = time.time()
            print(f"  [FT correction] {layer_name}: faulty_ous={len(faulty_ous)} groups={len(groups)}")
            layer_mask = self.data_loader.get_layer_mask(layer_name)
            if layer_mask is not None and layer_mask.shape[:2] == layer_weights.shape[:2]:
                layer_mask = layer_mask.to(layer_weights.device)
            else:
                layer_mask = None

            if repair_mode == 'oracle':
                for faulty_ou in faulty_ous:
                    before_weight = self._extract_ou_weight(layer_weights, faulty_ou).clone()
                    original_weight = self._extract_ou_weight(original_layer_weights, faulty_ou).clone()

                    out_ch, in_ch = faulty_ou
                    if layer_weights.dim() == 4:
                        layer_weights[out_ch, in_ch, :, :] = original_weight
                    elif layer_weights.dim() == 2:
                        layer_weights[out_ch, in_ch] = original_weight
                    else:
                        continue

                    after_weight = self._extract_ou_weight(layer_weights, faulty_ou).clone()
                    self._record_repair_quality(
                        repair_quality_buckets['oracle'],
                        before_weight=before_weight,
                        after_weight=after_weight,
                        original_weight=original_weight,
                        secondary_bucket=layer_quality_bucket(layer_name, 'oracle'),
                    )
                    layer_oracle.append(faulty_ou)
                    oracle_count += 1

                oracle_corrected_ous[layer_name] = layer_oracle
                level1_corrected_ous[layer_name] = layer_level1
                level2_corrected_ous[layer_name] = layer_level2
                level3_corrected_ous[layer_name] = layer_level3
                continue

            # ---------- Level 1: 冗余组缩放替换（100%恢复）----------
            # 策略：用冗余组内健康OU的权重，按倍数关系缩放后替换故障OU
            if hierarchical_config.get('level1', {}).get('enabled', True) and groups:
                for faulty_ou in faulty_ous:
                    containing_group = None

                    containing_group = self.redundancy_parser.get_group_for_ou(layer_name, faulty_ou)

                    if containing_group is None:
                        continue
                    if containing_group.size() < 2:
                        level1_failed_singleton += 1
                        continue

                    # 找到健康的OU
                    out_ch, in_ch = faulty_ou
                    faulty_original_weight = self._extract_ou_weight(original_layer_weights, faulty_ou).clone()
                    faulty_current_weight = self._extract_ou_weight(layer_weights, faulty_ou).clone()
                    faulty_mask_slice = layer_mask[out_ch, in_ch] if layer_mask is not None else None
                    faulty_block_entry = containing_group.ou_to_block.get(faulty_ou)
                    faulty_block = None
                    block_offset = 0
                    if faulty_block_entry is not None and faulty_block_entry['member_index'] < len(containing_group.block_members):
                        faulty_block = containing_group.block_members[faulty_block_entry['member_index']]
                        block_offset = int(faulty_block_entry['offset'])

                    healthy_candidates = []
                    if faulty_block is not None:
                        for candidate_block in containing_group.block_members:
                            if block_offset >= int(candidate_block.get('channel_span', 1)):
                                continue
                            candidate_ou = (int(candidate_block['out_ch']), int(candidate_block['in_ch_start']) + block_offset)
                            if candidate_ou == faulty_ou or candidate_ou in faulty_set:
                                continue
                            healthy_candidates.append((candidate_block, candidate_ou))
                    else:
                        healthy_ous = [ou for ou in containing_group.ou_indices if ou != faulty_ou and ou not in faulty_set]
                        for candidate_ou in healthy_ous:
                            healthy_candidates.append((
                                {'out_ch': candidate_ou[0], 'in_ch_start': candidate_ou[1], 'channel_span': 1, 'multiplier': 1.0, 'role': 'member'},
                                candidate_ou,
                            ))

                    if not healthy_candidates:
                        continue

                    replacement_ou = None
                    replacement_block = None
                    replacement_origin = 'member'
                    if containing_group.prototype_block is not None:
                        prototype_candidate_ou = (
                            int(containing_group.prototype_block['out_ch']),
                            int(containing_group.prototype_block['in_ch_start']) + block_offset,
                        )
                        if prototype_candidate_ou != faulty_ou and prototype_candidate_ou not in faulty_set:
                            replacement_ou = prototype_candidate_ou
                            replacement_block = containing_group.prototype_block
                            replacement_origin = 'prototype'
                    elif containing_group.prototype_ou is not None:
                        for candidate_block, candidate_ou in healthy_candidates:
                            if candidate_ou == containing_group.prototype_ou:
                                replacement_ou = candidate_ou
                                replacement_block = candidate_block
                                replacement_origin = 'prototype'
                                break

                    if replacement_ou is None:
                        best_similarity = -1.0
                        for candidate_block, candidate_ou in healthy_candidates:
                            candidate_weight = self._extract_ou_weight(original_layer_weights, candidate_ou)
                            similarity = self._compute_ou_similarity(
                                faulty_weight=faulty_original_weight,
                                candidate_weight=candidate_weight,
                                mask_slice=faulty_mask_slice,
                            )
                            if similarity > best_similarity:
                                best_similarity = similarity
                                replacement_ou = candidate_ou
                                replacement_block = candidate_block
                                replacement_origin = 'prototype' if candidate_block.get('role') == 'prototype' else 'member'

                    if replacement_ou is None:
                        continue

                    if faulty_block is not None:
                        faulty_multiplier = float(faulty_block.get('multiplier', 1.0))
                    else:
                        fault_idx = containing_group.ou_indices.index(faulty_ou)
                        faulty_multiplier = containing_group.multipliers[fault_idx]
                    if replacement_block is not None:
                        replacement_multiplier = float(replacement_block.get('multiplier', 1.0))
                    else:
                        replacement_idx = containing_group.ou_indices.index(replacement_ou)
                        replacement_multiplier = containing_group.multipliers[replacement_idx]

                    # 计算缩放因子
                    if abs(replacement_multiplier) > 1e-8:
                        scale_factor = self.compute_member_to_member_scale(faulty_multiplier, replacement_multiplier)
                    else:
                        scale_factor = float('nan')

                    if layer_weights.dim() not in (2, 4):
                        continue

                    replacement_weight = self._extract_ou_weight(original_layer_weights, replacement_ou)
                    policy = self._effective_level1_policy(layer_name)
                    if policy.get('selection') == 'default':
                        default_expected_error = float('nan')
                    else:
                        default_expected_error = self._compute_expected_repair_error(
                            faulty_original_weight,
                            replacement_weight,
                            scale_factor,
                        )
                    default_choice = {
                        'replacement_ou': replacement_ou,
                        'replacement_origin': replacement_origin,
                        'scale_factor': scale_factor,
                        'corrected_weight': replacement_weight * scale_factor if not self._is_invalid_scale_factor(scale_factor) else faulty_current_weight,
                        'default_expected_error': default_expected_error,
                        'selected_expected_error': default_expected_error,
                        'mode': 'default',
                        'fallback_reason': '',
                    }
                    selected_choice = self._select_level1_repair_choice(
                        layer_name=layer_name,
                        faulty_ou=faulty_ou,
                        faulty_set=faulty_set,
                        original_layer_weights=original_layer_weights,
                        default_choice=default_choice,
                        policy=policy,
                    )
                    replacement_ou = selected_choice.get('replacement_ou', replacement_ou)
                    replacement_origin = selected_choice.get('replacement_origin', replacement_origin)
                    scale_factor = float(selected_choice.get('scale_factor', scale_factor))
                    corrected_weight = selected_choice.get('corrected_weight', default_choice['corrected_weight'])
                    if selected_choice.get('low_confidence'):
                        level1_selection_stats['low_confidence_repair'] += 1

                    if self._is_invalid_scale_factor(scale_factor):
                        level1_zero_scale_failed += 1
                        self._record_repair_quality(
                            repair_quality_buckets['level1'],
                            before_weight=faulty_current_weight,
                            after_weight=faulty_current_weight,
                            original_weight=faulty_original_weight,
                            secondary_bucket=layer_quality_bucket(layer_name, 'level1'),
                        )
                        if level1_zero_scale_failed <= 3:
                            print(f"  ⚠️ Level 1 zero-scale skipped: {layer_name}[{faulty_ou}] "
                                  f"(faulty_multiplier={faulty_multiplier:.4f}, replacement_multiplier={replacement_multiplier:.4f})")
                        continue

                    # 应用替换
                    if layer_weights.dim() == 4:
                        layer_weights[out_ch, in_ch, :, :] = corrected_weight
                    elif layer_weights.dim() == 2:
                        layer_weights[out_ch, in_ch] = corrected_weight
                    corrected_weight_after = self._extract_ou_weight(layer_weights, faulty_ou).clone()
                    actual_before_error = self._l2_error(faulty_current_weight, faulty_original_weight)
                    actual_after_error = self._l2_error(corrected_weight_after, faulty_original_weight)
                    self._record_level1_selection_event(
                        level1_selection_stats,
                        layer_name=layer_name,
                        mode=selected_choice.get('mode', 'default'),
                        default_expected_error=float(selected_choice.get('default_expected_error', default_expected_error)),
                            selected_expected_error=float(selected_choice.get('selected_expected_error', default_expected_error)),
                        actual_before_error=actual_before_error,
                        actual_after_error=actual_after_error,
                        fallback_reason=selected_choice.get('fallback_reason', ''),
                    )
                    self._record_repair_quality_errors(
                        repair_quality_buckets['level1'],
                        before_error=actual_before_error,
                        after_error=actual_after_error,
                        secondary_bucket=layer_quality_bucket(layer_name, 'level1'),
                    )

                    if level1_count < 3:
                        print(f"  ✅ Level 1: {layer_name}[{faulty_ou}] ← OU[{replacement_ou}] "
                              f"(组大小={containing_group.size()}, 来源={replacement_origin}, block_offset={block_offset}, 缩放因子={scale_factor:.4f})")

                    layer_level1.append(faulty_ou)
                    level1_count += 1
                    if replacement_origin == 'prototype':
                        level1_prototype_repairs += 1
                    else:
                        level1_member_repairs += 1
                    if containing_group.repair_mode == 'exact' and abs(scale_factor - 1.0) <= 1e-6:
                        level1_exact_repairs += 1
                    else:
                        level1_scaled_repairs += 1

            level1_corrected_ous[layer_name] = layer_level1
            if repair_mode != 'oracle':
                print(
                    f"  [FT correction] {layer_name} level1_done: "
                    f"level1={len(layer_level1)} elapsed={time.time() - layer_start_time:.1f}s"
                )

            # ---------- Level 2: 相似模式替换（95%恢复，允许误差）----------
            # 策略：用相似OU的权重替换故障OU（可能不完全相同，约95%准确）
            # 只处理Level 1未纠正的故障
            layer_level1_set = set(layer_level1)
            level2_enabled = hierarchical_config.get('level2', {}).get('enabled', True)
            level3_enabled = hierarchical_config.get('level3', {}).get('enabled', True)
            remaining_for_level2 = (
                [ou for ou in faulty_ous if ou not in layer_level1_set]
                if level2_enabled or level3_enabled
                else []
            )

            if level2_enabled:
                level2_cfg = hierarchical_config.get('level2', {})
                similarity_threshold = level2_cfg.get('similarity_threshold', 0.85)
                k_nearest = level2_cfg.get('k_nearest', 3)
                use_weighted_average = level2_cfg.get('use_weighted_average', True)
                max_candidates = level2_cfg.get('max_candidates', 2048)

                mask = self.data_loader.get_layer_mask(layer_name)
                mask_groups = self._get_mask_groups(layer_name, layer_weights)
                mask_for_calc = None
                if mask is not None and mask.shape[:2] == layer_weights.shape[:2]:
                    mask_for_calc = mask.to(layer_weights.device)

                for faulty_ou in remaining_for_level2:
                    self.nearest_pattern_corrector.correction_statistics['total_corrections'] += 1

                    out_ch, in_ch = faulty_ou
                    if layer_weights.dim() == 4:
                        faulty_weight = layer_weights[out_ch, in_ch, :, :]
                        original_weight = original_layer_weights[out_ch, in_ch, :, :].clone()
                    elif layer_weights.dim() == 2:
                        faulty_weight = layer_weights[out_ch, in_ch]
                        original_weight = original_layer_weights[out_ch, in_ch].clone()
                    else:
                        self.nearest_pattern_corrector.correction_statistics['failed_corrections'] += 1
                        continue
                    before_weight = faulty_weight.clone()

                    if mask is not None and mask.shape[:2] == layer_weights.shape[:2]:
                        mask_slice = mask[out_ch, in_ch]
                        mask_key = mask_slice.flatten().to(torch.uint8).cpu().numpy().tobytes()
                        candidates = list(mask_groups.get(mask_key, []))
                    else:
                        candidates = list(mask_groups.get(b'all', []))

                    candidates = [ou for ou in candidates if ou not in faulty_set]
                    if not candidates:
                        self.nearest_pattern_corrector.correction_statistics['failed_corrections'] += 1
                        continue

                    if max_candidates and len(candidates) > max_candidates:
                        sampled_idx = np.random.choice(len(candidates), max_candidates, replace=False)
                        candidates = [candidates[idx] for idx in sampled_idx]

                    def extract_vector(weight_tensor, ou):
                        o_ch, i_ch = ou
                        if weight_tensor.dim() == 4:
                            vec = weight_tensor[o_ch, i_ch, :, :]
                            if mask_for_calc is not None:
                                mask_local = mask_for_calc[o_ch, i_ch].bool()
                                vec = vec[mask_local]
                            else:
                                vec = vec.flatten()
                        else:
                            vec = weight_tensor[o_ch, i_ch].reshape(1)
                        return vec.flatten()

                    faulty_vec = extract_vector(original_layer_weights, faulty_ou)
                    if faulty_vec.numel() == 0:
                        self.nearest_pattern_corrector.correction_statistics['failed_corrections'] += 1
                        continue

                    similarities = []
                    for candidate in candidates:
                        candidate_vec = extract_vector(original_layer_weights, candidate)
                        if candidate_vec.numel() == 0:
                            continue
                        norm_f = torch.norm(faulty_vec)
                        norm_c = torch.norm(candidate_vec)
                        if norm_f > 0 and norm_c > 0:
                            cosine = torch.dot(faulty_vec, candidate_vec) / (norm_f * norm_c)
                            similarity = (cosine + 1) / 2
                        else:
                            similarity = torch.tensor(0.0, device=faulty_vec.device)
                        similarities.append((candidate, similarity.item()))

                    if not similarities:
                        self.nearest_pattern_corrector.correction_statistics['failed_corrections'] += 1
                        continue

                    similarities.sort(key=lambda x: x[1], reverse=True)
                    top_candidates = [(ou, sim) for ou, sim in similarities[:k_nearest] if sim >= similarity_threshold]

                    if not top_candidates:
                        self.nearest_pattern_corrector.correction_statistics['failed_corrections'] += 1
                        continue

                    if use_weighted_average and len(top_candidates) > 1:
                        sim_tensor = torch.tensor([sim for _, sim in top_candidates], device=layer_weights.device)
                        sim_tensor = sim_tensor / sim_tensor.sum()
                        corrected_weight = None
                        for (ou, _), weight in zip(top_candidates, sim_tensor):
                            if layer_weights.dim() == 4:
                                candidate_weight = original_layer_weights[ou[0], ou[1], :, :]
                            else:
                                candidate_weight = original_layer_weights[ou[0], ou[1]]
                            corrected_weight = candidate_weight * weight if corrected_weight is None else corrected_weight + candidate_weight * weight
                        avg_similarity = sum(sim for _, sim in top_candidates) / len(top_candidates)
                    else:
                        best_ou, best_sim = top_candidates[0]
                        avg_similarity = best_sim
                        if layer_weights.dim() == 4:
                            corrected_weight = original_layer_weights[best_ou[0], best_ou[1], :, :]
                        else:
                            corrected_weight = original_layer_weights[best_ou[0], best_ou[1]]

                    if layer_weights.dim() == 4:
                        layer_weights[out_ch, in_ch, :, :] = corrected_weight
                    else:
                        layer_weights[out_ch, in_ch] = corrected_weight
                    after_weight = self._extract_ou_weight(layer_weights, faulty_ou).clone()
                    self._record_repair_quality(
                        repair_quality_buckets['level2'],
                        before_weight=before_weight,
                        after_weight=after_weight,
                        original_weight=original_weight,
                        secondary_bucket=layer_quality_bucket(layer_name, 'level2'),
                    )

                    level2_count += 1
                    layer_level2.append(faulty_ou)
                    self.nearest_pattern_corrector.correction_statistics['successful_corrections'] += 1
                    self.nearest_pattern_corrector.correction_statistics['similarity_distribution'].append(avg_similarity)
                    self.nearest_pattern_corrector.correction_statistics.setdefault('corrections_by_layer', {})
                    self.nearest_pattern_corrector.correction_statistics['corrections_by_layer'][layer_name] = (
                        self.nearest_pattern_corrector.correction_statistics['corrections_by_layer'].get(layer_name, 0) + 1
                    )

            level2_corrected_ous[layer_name] = layer_level2

            # ---------- Level 3: 自适应屏蔽（置0，最小化影响）----------
            # 策略：无法通过Level 1纠正的故障，将OU权重置0，避免错误传播
            layer_level2_set = set(layer_level2)
            remaining_for_level3 = (
                [ou for ou in faulty_ous if ou not in layer_level1_set and ou not in layer_level2_set]
                if level3_enabled
                else []
            )
            if level3_enabled:
                for faulty_ou_idx in remaining_for_level3:
                    before_weight = self._extract_ou_weight(layer_weights, faulty_ou_idx).clone()
                    original_weight = self._extract_ou_weight(original_layer_weights, faulty_ou_idx).clone()
                    if layer_weights.dim() == 2:
                        out_ch, in_ch = faulty_ou_idx
                        layer_weights[out_ch, in_ch] = 0
                        if level3_count < 3:
                            print(f"  ✅ Level 3: {layer_name}[{faulty_ou_idx}] 权重置0")
                        layer_level3.append(faulty_ou_idx)
                        level3_count += 1
                        after_weight = self._extract_ou_weight(layer_weights, faulty_ou_idx).clone()
                        self._record_repair_quality(
                            repair_quality_buckets['level3'],
                            before_weight=before_weight,
                            after_weight=after_weight,
                            original_weight=original_weight,
                            secondary_bucket=layer_quality_bucket(layer_name, 'level3'),
                        )
                        continue
                    # ✅ Level 3细粒度容错：只置零故障权重，保留健康权重
                    if hasattr(self, 'detailed_fault_mask') and layer_name in self.detailed_fault_mask:
                        if faulty_ou_idx in self.detailed_fault_mask[layer_name]:
                            # 有详细故障记录，只置零故障权重
                            fault_entries = self.detailed_fault_mask[layer_name][faulty_ou_idx]

                            # 获取该OU的权重
                            out_ch, in_ch = faulty_ou_idx
                            ou_weights = layer_weights[out_ch, in_ch, :, :]
                            flat_weights = ou_weights.flatten()

                            for entry in fault_entries:
                                if isinstance(entry, dict):
                                    fault_index = int(entry.get('index', -1))
                                    if 0 <= fault_index < flat_weights.numel():
                                        flat_weights[fault_index] = 0
                                else:
                                    fault_index = int(entry)
                                    if 0 <= fault_index < min(8, flat_weights.numel()):
                                        flat_weights[fault_index] = 0
                            layer_weights[out_ch, in_ch, :, :] = flat_weights.reshape(ou_weights.shape)

                            if level3_count < 3:
                                print(f"  ✅ Level 3: {layer_name}[{faulty_ou_idx}] {len(fault_entries)} 个故障权重置0")
                        else:
                            # 没有故障记录，可能是整个OU故障（兼容性处理）
                            out_ch, in_ch = faulty_ou_idx
                            layer_weights[out_ch, in_ch, :, :] = torch.zeros_like(layer_weights[out_ch, in_ch, :, :])
                            if level3_count < 3:
                                print(f"  ⚠️ Level 3: {layer_name}[{faulty_ou_idx}] 无故障记录，整个OU置0")
                    else:
                        # 没有详细故障记录，使用传统方法（兼容性处理）
                        out_ch, in_ch = faulty_ou_idx
                        layer_weights[out_ch, in_ch, :, :] = torch.zeros_like(layer_weights[out_ch, in_ch, :, :])
                        if level3_count < 3:
                            print(f"  ⚠️ Level 3: {layer_name}[{faulty_ou_idx}] 使用传统方法，整个OU置0")

                    layer_level3.append(faulty_ou_idx)
                    level3_count += 1
                    after_weight = self._extract_ou_weight(layer_weights, faulty_ou_idx).clone()
                    self._record_repair_quality(
                        repair_quality_buckets['level3'],
                        before_weight=before_weight,
                        after_weight=after_weight,
                        original_weight=original_weight,
                        secondary_bucket=layer_quality_bucket(layer_name, 'level3'),
                    )
            else:
                # 如果Level 3未启用，这些故障无法纠正
                for idx in remaining_for_level3:
                    uncorrected_by_layer.setdefault(layer_name, []).append(idx)

            level3_corrected_ous[layer_name] = layer_level3

            # 剩余未纠正的故障
            layer_level3_set = set(layer_level3)
            residual_uncorrected = [
                ou for ou in faulty_ous
                if ou not in layer_level1_set
                and ou not in layer_level2_set
                and ou not in layer_level3_set
            ]
            if residual_uncorrected:
                uncorrected_by_layer.setdefault(layer_name, []).extend(residual_uncorrected)

        # 统计输出
        total_corrected = level1_count + level2_count + level3_count + oracle_count
        total_faults = sum(len(indices) for indices in ou_fault_mask.values())
        uncorrectable_count = total_faults - total_corrected

        print(f"  分层容错覆盖:")
        if repair_mode == 'oracle':
            print(f"    Oracle Restore: {oracle_count}")
        else:
            print(f"    Level 1 (冗余组缩放替换-100%): {level1_count}")
            print(f"    Level 2 (相似模式替换-95%): {level2_count}")
            print(f"    Level 3 (自适应屏蔽-置0): {level3_count}")
            print(f"    Level 1 zero-scale failed: {level1_zero_scale_failed}")
        print(f"    无法纠正: {uncorrectable_count}")
        
        # 🔍 显示各级容错的详细分布
        if repair_mode == 'oracle' and oracle_corrected_ous:
            oracle_distribution = {k: len(v) for k, v in oracle_corrected_ous.items() if v}
            if oracle_distribution:
                print(f"  📊 Oracle恢复分布: {oracle_distribution}")
        if level1_corrected_ous:
            level1_distribution = {k: len(v) for k, v in level1_corrected_ous.items() if v}
            if level1_distribution:
                print(f"  📊 Level 1纠正分布: {level1_distribution}")

        if level2_corrected_ous:
            level2_distribution = {k: len(v) for k, v in level2_corrected_ous.items() if v}
            if level2_distribution:
                print(f"  📊 Level 2纠正分布: {level2_distribution}")
        if level3_corrected_ous:
            level3_distribution = {k: len(v) for k, v in level3_corrected_ous.items() if v}
            if level3_distribution:
                print(f"  📊 Level 3纠正分布: {level3_distribution}")

        # 更新多数表决器统计，用于后续硬件开销计算
        self.majority_voter.voting_statistics['successful_corrections'] = total_corrected
        self.majority_voter.voting_statistics['total_votes'] = total_corrected
        self.majority_voter.voting_statistics['detection_failures'] = uncorrectable_count
        self.majority_voter.voting_statistics['corrections_by_layer'] = {
            layer: len(level1_corrected_ous.get(layer, [])) +
                   len(level2_corrected_ous.get(layer, [])) +
                   len(level3_corrected_ous.get(layer, [])) +
                   len(oracle_corrected_ous.get(layer, []))
            for layer in ou_fault_mask.keys()
            if (len(level1_corrected_ous.get(layer, [])) +
                len(level2_corrected_ous.get(layer, [])) +
                len(level3_corrected_ous.get(layer, [])) +
                len(oracle_corrected_ous.get(layer, []))) > 0
        }

        repair_quality = {
            level_name: self._finalize_repair_quality_bucket(bucket)
            for level_name, bucket in repair_quality_buckets.items()
            if bucket.get('attempted', 0) > 0
        }
        layer_repair_quality = {
            layer_name: {
                level_name: self._finalize_repair_quality_bucket(bucket)
                for level_name, bucket in layer_buckets.items()
                if bucket.get('attempted', 0) > 0
            }
            for layer_name, layer_buckets in layer_repair_quality_buckets.items()
        }
        level1_selection = self._finalize_level1_selection_stats(level1_selection_stats)

        return {
            'level1_count': level1_count,
            'level2_count': level2_count,
            'level3_count': level3_count,
            'oracle_count': oracle_count,
            'level1_prototype_repairs': level1_prototype_repairs,
            'level1_member_repairs': level1_member_repairs,
            'level1_exact_repairs': level1_exact_repairs,
            'level1_scaled_repairs': level1_scaled_repairs,
            'level1_failed_singleton': level1_failed_singleton,
            'level1_zero_scale_failed': level1_zero_scale_failed,
            'uncorrectable_count': uncorrectable_count,
            'total_correctable': total_corrected,
            'repair_mode': repair_mode,
            'repair_quality': repair_quality,
            'level1_selection': level1_selection,
            'level1_corrected_ous': level1_corrected_ous,
            'level2_corrected_ous': level2_corrected_ous,
            'level3_corrected_ous': level3_corrected_ous,
            'oracle_corrected_ous': oracle_corrected_ous,
            'uncorrected_by_layer': uncorrected_by_layer,
            'layer_repair_quality': layer_repair_quality,
        }
    
    
    
    def _get_module_by_name(self, layer_name: str) -> Optional[nn.Module]:
        """根据层名称获取模块"""
        # 移除 .weight 后缀
        module_name = layer_name.replace('.weight', '')
        
        # 尝试获取模块
        try:
            module = self.model
            for part in module_name.split('.'):
                module = getattr(module, part)
            return module
        except AttributeError:
            return None
    
    def _collect_all_metrics(self, 
                            baseline_acc: float,
                            faulty_acc: float,
                            ft_acc: float):
        """收集所有评估指标"""
        # 故障注入统计
        fault_stats = self.fault_injector.get_statistics()
        self.metrics_collector.update_fault_injection_stats(
            total_faults=fault_stats['total_faults_injected'],
            faults_by_layer=fault_stats['faults_by_layer'],
            faults_by_model=fault_stats.get('faults_by_model', {}),
            fault_detail_stats=fault_stats.get('fault_detail_stats', {}),
        )
        
        # 表决统计
        voting_stats = self.majority_voter.get_statistics()
        self.metrics_collector.update_voting_stats(
            successful_corrections=voting_stats['successful_corrections'],
            detection_failures=voting_stats['detection_failures'],
            corrections_by_layer=voting_stats['corrections_by_layer']
        )
        
        # 三级容错统计
        if hasattr(self, '_hierarchical_stats') and self._hierarchical_stats:
            # 获取Level 2的相似度统计
            level2_stats = self.nearest_pattern_corrector.get_statistics()
            level2_similarity = level2_stats.get('average_similarity', 0.0)
            
            self.metrics_collector.update_hierarchical_correction_stats(
                level1_count=self._hierarchical_stats['level1_count'],
                level2_count=self._hierarchical_stats['level2_count'],
                level3_count=self._hierarchical_stats['level3_count'],
                level2_similarity_avg=level2_similarity,
                level1_failed_singleton=self._hierarchical_stats.get('level1_failed_singleton', 0),
                level1_zero_scale_failed=self._hierarchical_stats.get('level1_zero_scale_failed', 0),
                repair_mode=self._hierarchical_stats.get('repair_mode', 'normal'),
                level1_selection=self._hierarchical_stats.get('level1_selection', {}),
            )
            self.metrics_collector.update_repair_quality_stats(
                self._hierarchical_stats.get('repair_quality', {})
            )
            self.metrics_collector.update_layer_repair_quality_stats(
                self._hierarchical_stats.get('layer_repair_quality', {})
            )
        
        # 硬件开销（包含三级容错的额外开销）
        hw_overhead = voting_stats['hardware_overhead']
        
        # 添加Level 2和Level 3的硬件开销
        level2_overhead = self.nearest_pattern_corrector.get_hardware_overhead()
        total_voter_latency = hw_overhead['total_latency_ns'] + level2_overhead['total_latency_ns']
        total_voter_energy = hw_overhead['total_energy_pj'] + level2_overhead['total_energy_pj']
        
        # Level 3开销（简化计算）
        if hasattr(self, '_hierarchical_stats') and self._hierarchical_stats:
            level3_config = self.config.config.get('hierarchical_fault_tolerance', {}).get('level3', {})
            level3_count = self._hierarchical_stats.get('level3_count', 0)
            total_voter_latency += level3_count * level3_config.get('fallback_latency_ns', 20)
            total_voter_energy += level3_count * level3_config.get('fallback_energy_pj', 100)
        
        self.metrics_collector.update_hardware_overhead(
            computation_latency=1000.0,  # 简化：使用固定值
            computation_energy=5000.0,
            voter_latency=total_voter_latency,
            voter_energy=total_voter_energy
        )
        
        # 准确率
        self.metrics_collector.update_accuracy(
            baseline_acc=baseline_acc,
            faulty_acc=faulty_acc,
            ft_acc=ft_acc
        )
        if hasattr(self, '_hierarchical_stats') and self._hierarchical_stats.get('repair_mode') == 'oracle':
            if ft_acc + 0.002 < baseline_acc:
                print("  ⚠️ Oracle restore 未接近 baseline accuracy，故障/恢复流程可能仍有问题")
        
        # 计算冗余开销
        self.metrics_collector.compute_redundancy_overhead(
            baseline_latency=500.0,  # 简化：使用估计值
            baseline_energy=2000.0
        )
    
    def _generate_reports(self):
        """生成所有报告"""
        metrics = self.metrics_collector.get_all_metrics()
        config_dict = self.config.config
        
        # 生成所有格式的报告
        files = self.report_generator.generate_all_reports(metrics, config_dict)
        
        print(f"\n📝 报告已生成:")
        for format_type, filepath in files.items():
            print(f"  - {format_type}: {filepath}")
    
    def print_configuration(self):
        """打印配置信息"""
        self.config.print_config()


def main():
    """主函数 - 示例用法"""
    print("🧪 容错仿真器示例\n")
    
    # 加载模型
    from model import Vgg16
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Vgg16(num_classes=10)
    
    # 加载训练好的模型参数
    model_file = 'model_Vgg16_ft_group_cluster_translate_after_translate_parameters.pth'
    
    try:
        model.load_state_dict(torch.load(model_file, map_location=device))
        model = model.to(device)
        model.eval()
        print(f"✅ 模型加载成功: {model_file}\n")
    except FileNotFoundError:
        print(f"❌ 模型文件未找到: {model_file}")
        print("  请先运行 main.py 进行模型训练\n")
        return
    
    # 加载测试数据
    from torchvision import transforms, datasets
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_dataset = datasets.CIFAR10('./cifar10_data', train=False, 
                                   transform=transform_test, download=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, 
                                             shuffle=False)
    
    # 创建仿真器
    simulator = FaultToleranceSimulator(
        model=model,
        model_name='Vgg16',
        translate_name='ft_group_cluster_translate',
        data_dir='./'
    )
    
    # 打印配置
    # simulator.print_configuration()
    
    # 运行仿真（使用少量样本进行快速测试）
    results = simulator.run_simulation(
        test_loader=test_loader,
        num_samples=1000  # 使用1000个样本进行测试
    )
    
    print("\n✅ 示例运行完成")


if __name__ == "__main__":
    main()
