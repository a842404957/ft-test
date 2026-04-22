#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估指标收集器
收集和计算容错机制的各项评估指标
"""

import time
import json
import numpy as np
from typing import Dict, List, Any, Optional
from collections import defaultdict


class MetricsCollector:
    """评估指标收集器"""
    
    def __init__(self, config: Dict = None):
        """
        初始化指标收集器
        
        Args:
            config: 配置参数
        """
        self.config = config or {
            'track_layer_wise': True,
            'compute_confidence_interval': True,
            'confidence_level': 0.95
        }
        
        # 可靠性指标
        self.reliability_metrics = {
            'total_faults_injected': 0,
            'faults_corrected': 0,
            'faults_missed': 0,
            'detection_failures': 0,
            'fault_correction_rate': 0.0,
            'layer_wise_faults': {},
            'layer_wise_corrections': {},
            # 三级容错统计
            'hierarchical_correction': {
                'level1_corrections': 0,  # 冗余组纠正
                'level2_corrections': 0,  # 相似模式纠正
                'level3_corrections': 0,  # 自适应屏蔽纠正
                'level1_rate': 0.0,
                'level2_rate': 0.0,
                'level3_rate': 0.0,
                'level2_similarity_avg': 0.0,  # Level 2平均相似度
                'level1_zero_scale_failed': 0,
                'repair_mode': 'normal',
            }
        }
        self.reliability_metrics['repair_quality'] = {}
        
        # 硬件开销指标
        self.hardware_metrics = {
            'total_latency_ns': 0.0,
            'total_energy_pj': 0.0,
            'computation_latency_ns': 0.0,
            'voter_latency_ns': 0.0,
            'computation_energy_pj': 0.0,
            'voter_energy_pj': 0.0,
            'redundancy_overhead_ratio': 0.0,
            'layer_wise_latency': {},
            'layer_wise_energy': {}
        }
        
        # 准确率指标
        self.accuracy_metrics = {
            'baseline_accuracy': 0.0,
            'faulty_accuracy': 0.0,
            'ft_accuracy': 0.0,  # 容错后的准确率
            'accuracy_recovery_rate': 0.0
        }
        
        # 时间统计
        self.timing = {
            'start_time': None,
            'end_time': None,
            'total_time_seconds': 0.0
        }
        
        print("📊 评估指标收集器已初始化")
    
    def start_timing(self):
        """开始计时"""
        self.timing['start_time'] = time.time()
    
    def end_timing(self):
        """结束计时"""
        self.timing['end_time'] = time.time()
        if self.timing['start_time']:
            self.timing['total_time_seconds'] = (
                self.timing['end_time'] - self.timing['start_time']
            )
    
    def update_fault_injection_stats(self, total_faults: int, 
                                     faults_by_layer: Dict[str, int]):
        """更新故障注入统计"""
        self.reliability_metrics['total_faults_injected'] = total_faults
        self.reliability_metrics['layer_wise_faults'] = faults_by_layer.copy()
    
    def update_voting_stats(self, successful_corrections: int,
                          detection_failures: int,
                          corrections_by_layer: Dict[str, int]):
        """更新表决统计"""
        self.reliability_metrics['faults_corrected'] = successful_corrections
        self.reliability_metrics['detection_failures'] = detection_failures
        self.reliability_metrics['faults_missed'] = (
            self.reliability_metrics['total_faults_injected'] - successful_corrections
        )
        
        # 计算纠正率
        if self.reliability_metrics['total_faults_injected'] > 0:
            self.reliability_metrics['fault_correction_rate'] = (
                successful_corrections / self.reliability_metrics['total_faults_injected']
            )
        
        self.reliability_metrics['layer_wise_corrections'] = corrections_by_layer.copy()
    
    def update_hierarchical_correction_stats(self,
                                            level1_count: int,
                                            level2_count: int,
                                            level3_count: int,
                                            level2_similarity_avg: float = 0.0,
                                            level1_zero_scale_failed: int = 0,
                                            repair_mode: str = 'normal'):
        """更新三级容错统计"""
        self.reliability_metrics['hierarchical_correction']['level1_corrections'] = level1_count
        self.reliability_metrics['hierarchical_correction']['level2_corrections'] = level2_count
        self.reliability_metrics['hierarchical_correction']['level3_corrections'] = level3_count
        self.reliability_metrics['hierarchical_correction']['level2_similarity_avg'] = level2_similarity_avg
        self.reliability_metrics['hierarchical_correction']['level1_zero_scale_failed'] = level1_zero_scale_failed
        self.reliability_metrics['hierarchical_correction']['repair_mode'] = repair_mode
        
        # 计算各级纠正率
        total_corrected = level1_count + level2_count + level3_count
        if total_corrected > 0:
            self.reliability_metrics['hierarchical_correction']['level1_rate'] = level1_count / total_corrected
            self.reliability_metrics['hierarchical_correction']['level2_rate'] = level2_count / total_corrected
            self.reliability_metrics['hierarchical_correction']['level3_rate'] = level3_count / total_corrected

    def update_repair_quality_stats(self, repair_quality: Dict[str, Any]):
        """更新逐级修复质量统计。"""
        self.reliability_metrics['repair_quality'] = json.loads(json.dumps(repair_quality))
    
    def update_hardware_overhead(self, 
                                computation_latency: float,
                                computation_energy: float,
                                voter_latency: float,
                                voter_energy: float,
                                layer_wise_latency: Optional[Dict] = None,
                                layer_wise_energy: Optional[Dict] = None):
        """更新硬件开销"""
        self.hardware_metrics['computation_latency_ns'] = computation_latency
        self.hardware_metrics['computation_energy_pj'] = computation_energy
        self.hardware_metrics['voter_latency_ns'] = voter_latency
        self.hardware_metrics['voter_energy_pj'] = voter_energy
        
        self.hardware_metrics['total_latency_ns'] = computation_latency + voter_latency
        self.hardware_metrics['total_energy_pj'] = computation_energy + voter_energy
        
        if layer_wise_latency:
            self.hardware_metrics['layer_wise_latency'] = layer_wise_latency.copy()
        if layer_wise_energy:
            self.hardware_metrics['layer_wise_energy'] = layer_wise_energy.copy()
    
    def update_accuracy(self, baseline_acc: float, 
                       faulty_acc: float, 
                       ft_acc: float):
        """更新准确率指标"""
        self.accuracy_metrics['baseline_accuracy'] = baseline_acc
        self.accuracy_metrics['faulty_accuracy'] = faulty_acc
        self.accuracy_metrics['ft_accuracy'] = ft_acc
        
        # 计算准确率恢复率
        if baseline_acc > 0:
            degradation = baseline_acc - faulty_acc
            if degradation > 0:
                recovery = (ft_acc - faulty_acc) / degradation
                self.accuracy_metrics['accuracy_recovery_rate'] = min(recovery, 1.0)
            else:
                self.accuracy_metrics['accuracy_recovery_rate'] = 1.0
    
    def compute_redundancy_overhead(self, baseline_latency: float,
                                   baseline_energy: float):
        """计算冗余开销"""
        if baseline_latency > 0:
            latency_overhead = (
                self.hardware_metrics['total_latency_ns'] / baseline_latency
            )
        else:
            latency_overhead = 0.0
        
        if baseline_energy > 0:
            energy_overhead = (
                self.hardware_metrics['total_energy_pj'] / baseline_energy
            )
        else:
            energy_overhead = 0.0
        
        self.hardware_metrics['latency_overhead_ratio'] = latency_overhead
        self.hardware_metrics['energy_overhead_ratio'] = energy_overhead
        self.hardware_metrics['redundancy_overhead_ratio'] = (
            latency_overhead + energy_overhead
        ) / 2
    
    def get_all_metrics(self) -> Dict:
        """获取所有指标"""
        all_metrics = {
            'reliability': self.reliability_metrics.copy(),
            'hardware_overhead': self.hardware_metrics.copy(),
            'accuracy': self.accuracy_metrics.copy(),
            'timing': self.timing.copy()
        }
        
        return all_metrics
    
    def get_summary_metrics(self) -> Dict:
        """获取摘要指标（用于快速展示）"""
        summary = {
            # 可靠性
            'Total Faults Injected': self.reliability_metrics['total_faults_injected'],
            'Faults Corrected': self.reliability_metrics['faults_corrected'],
            'Faults Missed': self.reliability_metrics['faults_missed'],
            'Fault Correction Rate': f"{self.reliability_metrics['fault_correction_rate']:.2%}",
            
            # 硬件开销
            'Total Latency (ns)': f"{self.hardware_metrics['total_latency_ns']:.2f}",
            'Total Energy (pJ)': f"{self.hardware_metrics['total_energy_pj']:.2f}",
            'Voter Overhead (Latency)': f"{self.hardware_metrics['voter_latency_ns']:.2f} ns",
            'Voter Overhead (Energy)': f"{self.hardware_metrics['voter_energy_pj']:.2f} pJ",
            
            # 准确率
            'Baseline Accuracy': f"{self.accuracy_metrics['baseline_accuracy']:.2%}",
            'Faulty Accuracy': f"{self.accuracy_metrics['faulty_accuracy']:.2%}",
            'FT Accuracy': f"{self.accuracy_metrics['ft_accuracy']:.2%}",
            'Accuracy Recovery Rate': f"{self.accuracy_metrics['accuracy_recovery_rate']:.2%}",
            
            # 时间
            'Total Simulation Time': f"{self.timing['total_time_seconds']:.2f} s"
        }
        
        return summary
    
    def print_summary(self):
        """打印摘要信息"""
        print("\n" + "=" * 70)
        print("容错机制评估报告")
        print("=" * 70)
        
        print("\n📊 可靠性指标:")
        print(f"  总故障注入数:      {self.reliability_metrics['total_faults_injected']}")
        print(f"  成功纠正数:        {self.reliability_metrics['faults_corrected']}")
        print(f"  未纠正数:          {self.reliability_metrics['faults_missed']}")
        print(f"  检测失败数:        {self.reliability_metrics['detection_failures']}")
        print(f"  故障纠正率:        {self.reliability_metrics['fault_correction_rate']:.2%}")
        
        # 三级容错统计
        hierarchical = self.reliability_metrics['hierarchical_correction']
        if hierarchical['level1_corrections'] + hierarchical['level2_corrections'] + hierarchical['level3_corrections'] > 0:
            print("\n🔧 三级容错策略统计:")
            print(f"  Level 1 (冗余组):     {hierarchical['level1_corrections']:4d} ({hierarchical['level1_rate']:.1%})")
            if hierarchical.get('level1_zero_scale_failed', 0) > 0:
                print(f"    - zero-scale failed: {hierarchical['level1_zero_scale_failed']}")
            print(f"  Level 2 (相似模式):   {hierarchical['level2_corrections']:4d} ({hierarchical['level2_rate']:.1%})")
            if hierarchical['level2_similarity_avg'] > 0:
                print(f"    - 平均相似度:      {hierarchical['level2_similarity_avg']:.4f}")
            print(f"  Level 3 (自适应屏蔽): {hierarchical['level3_corrections']:4d} ({hierarchical['level3_rate']:.1%})")
            if hierarchical.get('repair_mode', 'normal') != 'normal':
                print(f"  Repair Mode:          {hierarchical['repair_mode']}")

        repair_quality = self.reliability_metrics.get('repair_quality', {})
        if repair_quality:
            print("\n🧪 修复质量统计:")
            for level_name, stats in repair_quality.items():
                print(
                    f"  {level_name}: attempted={stats.get('attempted', 0)} "
                    f"improved={stats.get('effective_improved', 0)} "
                    f"exact={stats.get('exact_restored', 0)} "
                    f"improved_rate={stats.get('improved_rate', 0.0):.2%}"
                )
        
        print("\n⚙️ 硬件开销指标:")
        print(f"  总延迟:            {self.hardware_metrics['total_latency_ns']:.2f} ns")
        print(f"    - 计算延迟:      {self.hardware_metrics['computation_latency_ns']:.2f} ns")
        print(f"    - 表决器延迟:    {self.hardware_metrics['voter_latency_ns']:.2f} ns")
        print(f"  总能耗:            {self.hardware_metrics['total_energy_pj']:.2f} pJ")
        print(f"    - 计算能耗:      {self.hardware_metrics['computation_energy_pj']:.2f} pJ")
        print(f"    - 表决器能耗:    {self.hardware_metrics['voter_energy_pj']:.2f} pJ")
        
        if 'latency_overhead_ratio' in self.hardware_metrics:
            print(f"  延迟开销比:        {self.hardware_metrics['latency_overhead_ratio']:.2f}x")
            print(f"  能耗开销比:        {self.hardware_metrics['energy_overhead_ratio']:.2f}x")
        
        print("\n🎯 准确率指标:")
        print(f"  基线准确率:        {self.accuracy_metrics['baseline_accuracy']:.6%}")
        print(f"  故障准确率:        {self.accuracy_metrics['faulty_accuracy']:.6%}")
        print(f"  容错准确率:        {self.accuracy_metrics['ft_accuracy']:.6%}")
        print(f"  准确率恢复率:      {self.accuracy_metrics['accuracy_recovery_rate']:.6%}")
        
        print("\n⏱️ 时间统计:")
        print(f"  仿真用时:          {self.timing['total_time_seconds']:.2f} 秒")
        
        print("=" * 70 + "\n")
    
    def print_layer_wise_stats(self, top_n: int = 10):
        """打印逐层统计"""
        print("\n" + "=" * 70)
        print("逐层统计 (Top {})".format(top_n))
        print("=" * 70)
        
        # 故障最多的层
        if self.reliability_metrics['layer_wise_faults']:
            print("\n故障数最多的层:")
            sorted_faults = sorted(
                self.reliability_metrics['layer_wise_faults'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for layer, count in sorted_faults[:top_n]:
                corrections = self.reliability_metrics['layer_wise_corrections'].get(layer, 0)
                rate = corrections / count if count > 0 else 0
                print(f"  {layer:20s}: {count:4d} 故障, {corrections:4d} 纠正 ({rate:.1%})")
        
        # 延迟最高的层
        if self.hardware_metrics['layer_wise_latency']:
            print("\n延迟最高的层:")
            sorted_latency = sorted(
                self.hardware_metrics['layer_wise_latency'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for layer, latency in sorted_latency[:top_n]:
                print(f"  {layer:20s}: {latency:.2f} ns")
        
        # 能耗最高的层
        if self.hardware_metrics['layer_wise_energy']:
            print("\n能耗最高的层:")
            sorted_energy = sorted(
                self.hardware_metrics['layer_wise_energy'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for layer, energy in sorted_energy[:top_n]:
                print(f"  {layer:20s}: {energy:.2f} pJ")
        
        print("=" * 70 + "\n")
    
    def save_to_json(self, filename: str):
        """保存指标到JSON文件"""
        metrics = self.get_all_metrics()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            print(f"✅ 指标已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存指标失败: {e}")
    
    def save_summary_to_csv(self, filename: str):
        """保存摘要到CSV文件"""
        import csv
        
        summary = self.get_summary_metrics()
        
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                for metric, value in summary.items():
                    writer.writerow([metric, value])
            print(f"✅ 摘要已保存到: {filename}")
        except Exception as e:
            print(f"❌ 保存摘要失败: {e}")


def test_metrics_collector():
    """测试指标收集器"""
    print("🧪 测试评估指标收集器\n")
    
    # 创建收集器
    collector = MetricsCollector()
    
    # 模拟数据收集
    collector.start_timing()
    
    # 故障注入统计
    collector.update_fault_injection_stats(
        total_faults=100,
        faults_by_layer={
            'conv1.weight': 15,
            'conv2.weight': 20,
            'fc1.weight': 10
        }
    )
    
    # 表决统计
    collector.update_voting_stats(
        successful_corrections=85,
        detection_failures=5,
        corrections_by_layer={
            'conv1.weight': 12,
            'conv2.weight': 18,
            'fc1.weight': 9
        }
    )
    
    # 硬件开销
    collector.update_hardware_overhead(
        computation_latency=1000.0,
        computation_energy=5000.0,
        voter_latency=100.0,
        voter_energy=500.0,
        layer_wise_latency={
            'conv1.weight': 300.0,
            'conv2.weight': 400.0,
            'fc1.weight': 200.0
        }
    )
    
    # 准确率
    collector.update_accuracy(
        baseline_acc=0.92,
        faulty_acc=0.75,
        ft_acc=0.89
    )
    
    # 计算冗余开销
    collector.compute_redundancy_overhead(
        baseline_latency=500.0,
        baseline_energy=2000.0
    )
    
    import time
    time.sleep(0.1)  # 模拟仿真时间
    collector.end_timing()
    
    # 打印报告
    collector.print_summary()
    collector.print_layer_wise_stats(top_n=3)
    
    # 保存结果
    collector.save_to_json('test_metrics.json')
    collector.save_summary_to_csv('test_summary.csv')
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_metrics_collector()
