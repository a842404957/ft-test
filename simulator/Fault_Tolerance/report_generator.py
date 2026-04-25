#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器
生成容错机制评估报告（支持多种格式）
"""

import os
import json
import csv
from typing import Dict, List, Any
from pathlib import Path
from datetime import datetime


class ReportGenerator:
    """报告生成器"""
    
    def __init__(self, output_dir: str = './fault_tolerance_results'):
        """
        初始化报告生成器
        
        Args:
            output_dir: 输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"📝 报告生成器已初始化")
        print(f"  输出目录: {self.output_dir}")
    
    def generate_all_reports(self, metrics: Dict, config: Dict = None) -> Dict[str, str]:
        """
        生成所有格式的报告
        
        Args:
            metrics: 评估指标
            config: 配置信息
            
        Returns:
            生成的报告文件路径字典
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        generated_files = {}
        
        # JSON报告
        json_file = self.generate_json_report(metrics, timestamp)
        generated_files['json'] = json_file
        
        # CSV报告
        csv_file = self.generate_csv_report(metrics, timestamp)
        generated_files['csv'] = csv_file
        
        # Markdown报告
        md_file = self.generate_markdown_report(metrics, config, timestamp)
        generated_files['markdown'] = md_file
        
        print(f"\n✅ 所有报告已生成")
        
        return generated_files
    
    def generate_json_report(self, metrics: Dict, timestamp: str = None) -> str:
        """生成JSON格式报告"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = self.output_dir / f'fault_tolerance_report_{timestamp}.json'
        
        # 添加元数据
        report = {
            'metadata': {
                'timestamp': timestamp,
                'report_version': '1.0',
                'generator': 'PRAP-PIM Fault Tolerance Framework'
            },
            'metrics': metrics
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ JSON报告: {filename}")
        return str(filename)
    
    def generate_csv_report(self, metrics: Dict, timestamp: str = None) -> str:
        """生成CSV格式报告"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = self.output_dir / f'fault_tolerance_summary_{timestamp}.csv'
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # 写入表头
            writer.writerow(['Category', 'Metric', 'Value'])
            
            # 可靠性指标
            if 'reliability' in metrics:
                rel = metrics['reliability']
                writer.writerow(['Reliability', 'Total Faults Injected', rel.get('total_faults_injected', 0)])
                writer.writerow(['Reliability', 'Faults Corrected', rel.get('faults_corrected', 0)])
                writer.writerow(['Reliability', 'Faults Missed', rel.get('faults_missed', 0)])
                writer.writerow(['Reliability', 'Fault Correction Rate', f"{rel.get('fault_correction_rate', 0):.2%}"])
                fault_detail = rel.get('fault_detail_stats', {})
                fault_model_counts = fault_detail.get('fault_model_counts', rel.get('faults_by_model', {}))
                writer.writerow(['Reliability', 'Affected Weight Count', fault_detail.get('affected_weight_count', 0)])
                writer.writerow(['Reliability', 'Stuck At Zero Count', fault_model_counts.get('stuck_at_zero', 0)])
                writer.writerow(['Reliability', 'Stuck At One Count', fault_model_counts.get('stuck_at_one', 0)])
                for model_name, count in sorted(fault_model_counts.items()):
                    writer.writerow(['FaultModel', model_name, count])
                hierarchical = rel.get('hierarchical_correction', {})
                if hierarchical:
                    writer.writerow(['Reliability', 'Level1 Corrections', hierarchical.get('level1_corrections', 0)])
                    writer.writerow(['Reliability', 'Level2 Corrections', hierarchical.get('level2_corrections', 0)])
                    writer.writerow(['Reliability', 'Level3 Corrections', hierarchical.get('level3_corrections', 0)])
                    writer.writerow(['Reliability', 'Level1 Failed Singleton', hierarchical.get('level1_failed_singleton', 0)])
                    writer.writerow(['Reliability', 'Level1 Zero Scale Failed', hierarchical.get('level1_zero_scale_failed', 0)])
                    writer.writerow(['Reliability', 'Repair Mode', hierarchical.get('repair_mode', 'normal')])
                    level1_selection = hierarchical.get('level1_selection', {}) or {}
                    if level1_selection:
                        writer.writerow(['Level1Selection', 'level1_selection_mode', level1_selection.get('level1_selection_mode', 'default')])
                        writer.writerow(['Level1Selection', 'best_pair_used', level1_selection.get('best_pair_used', 0)])
                        writer.writerow(['Level1Selection', 'weighted_average_used', level1_selection.get('weighted_average_used', 0)])
                        writer.writerow(['Level1Selection', 'fallback_to_default', level1_selection.get('fallback_to_default', 0)])
                        writer.writerow(['Level1Selection', 'fallback_reason', json.dumps(level1_selection.get('fallback_reason', {}), ensure_ascii=False)])
                        writer.writerow(['Level1Selection', 'low_confidence_repair', level1_selection.get('low_confidence_repair', 0)])
                        for metric_name in [
                            'default_expected_error',
                            'selected_expected_error',
                            'expected_error_improvement',
                            'actual_before_error',
                            'actual_after_error',
                            'actual_error_improvement',
                            'expected_actual_corr',
                        ]:
                            value = level1_selection.get(metric_name)
                            writer.writerow(['Level1Selection', metric_name, '' if value is None else f"{value:.6f}"])
                repair_quality = rel.get('repair_quality', {})
                for level_name, stats in repair_quality.items():
                    writer.writerow(['RepairQuality', f'{level_name}.attempted', stats.get('attempted', 0)])
                    writer.writerow(['RepairQuality', f'{level_name}.effective_improved', stats.get('effective_improved', 0)])
                    writer.writerow(['RepairQuality', f'{level_name}.exact_restored', stats.get('exact_restored', 0)])
                    writer.writerow(['RepairQuality', f'{level_name}.avg_before_error', f"{stats.get('avg_before_error', 0):.6f}"])
                    writer.writerow(['RepairQuality', f'{level_name}.avg_after_error', f"{stats.get('avg_after_error', 0):.6f}"])
                    writer.writerow(['RepairQuality', f'{level_name}.improved_rate', f"{stats.get('improved_rate', 0):.2%}"])
                layer_repair_quality = rel.get('layer_repair_quality', {})
                for layer_name, levels in sorted(layer_repair_quality.items()):
                    for level_name, stats in sorted(levels.items()):
                        prefix = f'{layer_name}.{level_name}'
                        writer.writerow(['LayerRepairQuality', f'{prefix}.attempted', stats.get('attempted', 0)])
                        writer.writerow(['LayerRepairQuality', f'{prefix}.effective_improved', stats.get('effective_improved', 0)])
                        writer.writerow(['LayerRepairQuality', f'{prefix}.avg_before_error', f"{stats.get('avg_before_error', 0):.6f}"])
                        writer.writerow(['LayerRepairQuality', f'{prefix}.avg_after_error', f"{stats.get('avg_after_error', 0):.6f}"])
                        writer.writerow(['LayerRepairQuality', f'{prefix}.improved_rate', f"{stats.get('improved_rate', 0):.2%}"])
            
            # 硬件开销指标
            if 'hardware_overhead' in metrics:
                hw = metrics['hardware_overhead']
                writer.writerow(['Hardware', 'Total Latency (ns)', f"{hw.get('total_latency_ns', 0):.2f}"])
                writer.writerow(['Hardware', 'Total Energy (pJ)', f"{hw.get('total_energy_pj', 0):.2f}"])
                writer.writerow(['Hardware', 'Voter Latency (ns)', f"{hw.get('voter_latency_ns', 0):.2f}"])
                writer.writerow(['Hardware', 'Voter Energy (pJ)', f"{hw.get('voter_energy_pj', 0):.2f}"])
            
            # 准确率指标
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                writer.writerow(['Accuracy', 'Baseline Accuracy', f"{acc.get('baseline_accuracy', 0):.2%}"])
                writer.writerow(['Accuracy', 'Faulty Accuracy', f"{acc.get('faulty_accuracy', 0):.2%}"])
                writer.writerow(['Accuracy', 'FT Accuracy', f"{acc.get('ft_accuracy', 0):.2%}"])
                writer.writerow(['Accuracy', 'Recovery Rate', f"{acc.get('accuracy_recovery_rate', 0):.2%}"])
        
        print(f"  ✓ CSV报告: {filename}")
        return str(filename)
    
    def generate_markdown_report(self, metrics: Dict, config: Dict = None, 
                                timestamp: str = None) -> str:
        """生成Markdown格式报告"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = self.output_dir / f'fault_tolerance_report_{timestamp}.md'
        
        with open(filename, 'w', encoding='utf-8') as f:
            # 标题
            f.write("# PRAP-PIM 容错机制评估报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 配置信息
            if config:
                f.write("## 1. 配置信息\n\n")
                f.write("### 故障注入配置\n\n")
                if 'fault_injection' in config:
                    fi_config = config['fault_injection']
                    f.write(f"- **故障率**: {fi_config.get('fault_rate', 0):.2%}\n")
                    f.write(f"- **故障模型**: {', '.join(fi_config.get('fault_models', []))}\n")
                    f.write(f"- **随机种子**: {fi_config.get('random_seed', 'None')}\n\n")
                
                f.write("### 多数表决器配置\n\n")
                if 'majority_voter' in config:
                    mv_config = config['majority_voter']
                    f.write(f"- **表决策略**: {mv_config.get('voting_strategy', 'N/A')}\n")
                    f.write(f"- **平局处理**: {mv_config.get('tie_breaking', 'N/A')}\n")
                    f.write(f"- **表决器延迟**: {mv_config.get('voter_latency_ns', 0)} ns\n")
                    f.write(f"- **表决器能耗**: {mv_config.get('voter_energy_pj', 0)} pJ\n\n")
                
                f.write("---\n\n")
            
            # 可靠性指标
            f.write("## 2. 可靠性指标\n\n")
            if 'reliability' in metrics:
                rel = metrics['reliability']
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 总故障注入数 | {rel.get('total_faults_injected', 0)} |\n")
                f.write(f"| 成功纠正数 | {rel.get('faults_corrected', 0)} |\n")
                f.write(f"| 未纠正数 | {rel.get('faults_missed', 0)} |\n")
                f.write(f"| 检测失败数 | {rel.get('detection_failures', 0)} |\n")
                f.write(f"| **故障纠正率** | **{rel.get('fault_correction_rate', 0):.2%}** |\n\n")
                fault_detail = rel.get('fault_detail_stats', {})
                fault_model_counts = fault_detail.get('fault_model_counts', rel.get('faults_by_model', {}))
                if fault_detail or fault_model_counts:
                    f.write("### 2.0 故障模型分布\n\n")
                    f.write("| 指标 | 值 |\n")
                    f.write("|------|----|\n")
                    f.write(f"| 受影响权重数 | {fault_detail.get('affected_weight_count', 0)} |\n")
                    f.write(f"| stuck_at_zero | {fault_model_counts.get('stuck_at_zero', 0)} |\n")
                    f.write(f"| stuck_at_one | {fault_model_counts.get('stuck_at_one', 0)} |\n")
                    for model_name, count in sorted(fault_model_counts.items()):
                        if model_name not in ('stuck_at_zero', 'stuck_at_one'):
                            f.write(f"| {model_name} | {count} |\n")
                    f.write("\n")
                hierarchical = rel.get('hierarchical_correction', {})
                if hierarchical:
                    f.write("### 2.1 三级容错统计\n\n")
                    f.write("| 指标 | 值 |\n")
                    f.write("|------|----|\n")
                    f.write(f"| Level 1纠正数 | {hierarchical.get('level1_corrections', 0)} |\n")
                    f.write(f"| Level 2纠正数 | {hierarchical.get('level2_corrections', 0)} |\n")
                    f.write(f"| Level 3纠正数 | {hierarchical.get('level3_corrections', 0)} |\n")
                    f.write(f"| Level 1 failed singleton | {hierarchical.get('level1_failed_singleton', 0)} |\n")
                    f.write(f"| Level 1 zero-scale failed | {hierarchical.get('level1_zero_scale_failed', 0)} |\n")
                    f.write(f"| 修复模式 | {hierarchical.get('repair_mode', 'normal')} |\n\n")
                    level1_selection = hierarchical.get('level1_selection', {}) or {}
                    if level1_selection:
                        f.write("### 2.1.1 Level1 Selection 统计\n\n")
                        f.write("| 指标 | 值 |\n")
                        f.write("|------|----|\n")
                        f.write(f"| level1_selection_mode | {level1_selection.get('level1_selection_mode', 'default')} |\n")
                        f.write(f"| best_pair_used | {level1_selection.get('best_pair_used', 0)} |\n")
                        f.write(f"| weighted_average_used | {level1_selection.get('weighted_average_used', 0)} |\n")
                        f.write(f"| fallback_to_default | {level1_selection.get('fallback_to_default', 0)} |\n")
                        f.write(f"| fallback_reason | `{json.dumps(level1_selection.get('fallback_reason', {}), ensure_ascii=False)}` |\n")
                        f.write(f"| low_confidence_repair | {level1_selection.get('low_confidence_repair', 0)} |\n")
                        for metric_name in [
                            'default_expected_error',
                            'selected_expected_error',
                            'expected_error_improvement',
                            'actual_before_error',
                            'actual_after_error',
                            'actual_error_improvement',
                            'expected_actual_corr',
                        ]:
                            value = level1_selection.get(metric_name)
                            f.write(f"| {metric_name} | {'' if value is None else f'{value:.6f}'} |\n")
                        f.write("\n")
                repair_quality = rel.get('repair_quality', {})
                if repair_quality:
                    f.write("### 2.2 修复质量统计\n\n")
                    f.write("| Level | Attempted | Improved | Exact Restored | Avg Before Error | Avg After Error | Improved Rate |\n")
                    f.write("|------|-----------|----------|----------------|------------------|-----------------|---------------|\n")
                    for level_name, stats in repair_quality.items():
                        f.write(
                            f"| {level_name} | {stats.get('attempted', 0)} | {stats.get('effective_improved', 0)} | "
                            f"{stats.get('exact_restored', 0)} | {stats.get('avg_before_error', 0):.6f} | "
                            f"{stats.get('avg_after_error', 0):.6f} | {stats.get('improved_rate', 0):.2%} |\n"
                        )
                    f.write("\n")
                layer_repair_quality = rel.get('layer_repair_quality', {})
                if layer_repair_quality:
                    f.write("### 2.2.1 逐层修复质量 Top Residual\n\n")
                    f.write("| Layer | Level | Attempted | Improved Rate | Avg Before Error | Avg After Error |\n")
                    f.write("|------|-------|-----------|---------------|------------------|-----------------|\n")
                    layer_rows = []
                    for layer_name, levels in layer_repair_quality.items():
                        for level_name, stats in levels.items():
                            layer_rows.append((stats.get('avg_after_error', 0), layer_name, level_name, stats))
                    for _, layer_name, level_name, stats in sorted(layer_rows, reverse=True)[:10]:
                        f.write(
                            f"| {layer_name} | {level_name} | {stats.get('attempted', 0)} | "
                            f"{stats.get('improved_rate', 0):.2%} | {stats.get('avg_before_error', 0):.6f} | "
                            f"{stats.get('avg_after_error', 0):.6f} |\n"
                        )
                    f.write("\n")
                
                # 逐层故障统计
                if 'layer_wise_faults' in rel and rel['layer_wise_faults']:
                    f.write("### 2.3 逐层故障统计\n\n")
                    f.write("| 层名称 | 故障数 | 纠正数 | 纠正率 |\n")
                    f.write("|--------|--------|--------|--------|\n")
                    
                    sorted_layers = sorted(
                        rel['layer_wise_faults'].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    
                    for layer, faults in sorted_layers[:10]:
                        corrections = rel.get('layer_wise_corrections', {}).get(layer, 0)
                        rate = corrections / faults if faults > 0 else 0
                        f.write(f"| {layer} | {faults} | {corrections} | {rate:.1%} |\n")
                    
                    f.write("\n")
            
            f.write("---\n\n")
            
            # 硬件开销指标
            f.write("## 3. 硬件开销指标\n\n")
            if 'hardware_overhead' in metrics:
                hw = metrics['hardware_overhead']
                f.write("### 3.1 总体开销\n\n")
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 总延迟 | {hw.get('total_latency_ns', 0):.2f} ns |\n")
                f.write(f"| - 计算延迟 | {hw.get('computation_latency_ns', 0):.2f} ns |\n")
                f.write(f"| - 表决器延迟 | {hw.get('voter_latency_ns', 0):.2f} ns |\n")
                f.write(f"| 总能耗 | {hw.get('total_energy_pj', 0):.2f} pJ |\n")
                f.write(f"| - 计算能耗 | {hw.get('computation_energy_pj', 0):.2f} pJ |\n")
                f.write(f"| - 表决器能耗 | {hw.get('voter_energy_pj', 0):.2f} pJ |\n\n")
                
                if 'latency_overhead_ratio' in hw:
                    f.write("### 3.2 开销比\n\n")
                    f.write("| 指标 | 值 |\n")
                    f.write("|------|----|\n")
                    f.write(f"| 延迟开销比 | {hw.get('latency_overhead_ratio', 0):.2f}x |\n")
                    f.write(f"| 能耗开销比 | {hw.get('energy_overhead_ratio', 0):.2f}x |\n\n")
            
            f.write("---\n\n")
            
            # 准确率指标
            f.write("## 4. 准确率指标\n\n")
            if 'accuracy' in metrics:
                acc = metrics['accuracy']
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 基线准确率 | {acc.get('baseline_accuracy', 0):.2%} |\n")
                f.write(f"| 故障准确率 | {acc.get('faulty_accuracy', 0):.2%} |\n")
                f.write(f"| **容错准确率** | **{acc.get('ft_accuracy', 0):.2%}** |\n")
                f.write(f"| 准确率恢复率 | {acc.get('accuracy_recovery_rate', 0):.2%} |\n\n")
                
                # 准确率变化可视化
                baseline = acc.get('baseline_accuracy', 0) * 100
                faulty = acc.get('faulty_accuracy', 0) * 100
                ft = acc.get('ft_accuracy', 0) * 100
                
                f.write("### 4.1 准确率对比\n\n")
                f.write("```\n")
                f.write(f"基线准确率:   {'█' * int(baseline/2)} {baseline:.1f}%\n")
                f.write(f"故障准确率:   {'█' * int(faulty/2)} {faulty:.1f}%\n")
                f.write(f"容错准确率:   {'█' * int(ft/2)} {ft:.1f}%\n")
                f.write("```\n\n")
            
            f.write("---\n\n")
            
            # 时间统计
            f.write("## 5. 时间统计\n\n")
            if 'timing' in metrics:
                timing = metrics['timing']
                f.write("| 指标 | 值 |\n")
                f.write("|------|----|\n")
                f.write(f"| 仿真用时 | {timing.get('total_time_seconds', 0):.2f} 秒 |\n\n")
            
            # 结论
            f.write("---\n\n")
            f.write("## 6. 结论\n\n")
            
            if 'reliability' in metrics and 'accuracy' in metrics:
                rel = metrics['reliability']
                acc = metrics['accuracy']
                
                correction_rate = rel.get('fault_correction_rate', 0)
                recovery_rate = acc.get('accuracy_recovery_rate', 0)
                
                if correction_rate > 0.9 and recovery_rate > 0.8:
                    f.write("✅ **容错机制表现优秀**\n\n")
                    f.write("- 故障纠正率超过90%\n")
                    f.write("- 准确率恢复率超过80%\n")
                elif correction_rate > 0.7 and recovery_rate > 0.5:
                    f.write("⚠️ **容错机制表现良好**\n\n")
                    f.write("- 故障纠正率在70%以上\n")
                    f.write("- 仍有优化空间\n")
                else:
                    f.write("❌ **容错机制需要改进**\n\n")
                    f.write("- 故障纠正率较低\n")
                    f.write("- 建议优化冗余策略\n")
            
            f.write("\n---\n\n")
            f.write(f"*报告生成器版本: 1.0*\n")
        
        print(f"  ✓ Markdown报告: {filename}")
        return str(filename)
    
    def generate_comparison_report(self, results_list: List[Dict], 
                                  labels: List[str]) -> str:
        """
        生成对比报告（用于比较不同配置的结果）
        
        Args:
            results_list: 结果列表
            labels: 标签列表
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f'comparison_report_{timestamp}.md'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# 容错机制对比报告\n\n")
            f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")
            
            # 构建对比表
            f.write("## 性能对比\n\n")
            f.write("| 配置 | 故障纠正率 | 容错准确率 | 总延迟 (ns) | 总能耗 (pJ) |\n")
            f.write("|------|-----------|-----------|------------|------------|\n")
            
            for label, results in zip(labels, results_list):
                rel = results.get('reliability', {})
                acc = results.get('accuracy', {})
                hw = results.get('hardware_overhead', {})
                
                f.write(f"| {label} | ")
                f.write(f"{rel.get('fault_correction_rate', 0):.1%} | ")
                f.write(f"{acc.get('ft_accuracy', 0):.1%} | ")
                f.write(f"{hw.get('total_latency_ns', 0):.2f} | ")
                f.write(f"{hw.get('total_energy_pj', 0):.2f} |\n")
            
            f.write("\n")
        
        print(f"  ✓ 对比报告: {filename}")
        return str(filename)


def test_report_generator():
    """测试报告生成器"""
    print("🧪 测试报告生成器\n")
    
    # 创建模拟数据
    metrics = {
        'reliability': {
            'total_faults_injected': 100,
            'faults_corrected': 85,
            'faults_missed': 15,
            'detection_failures': 5,
            'fault_correction_rate': 0.85,
            'layer_wise_faults': {
                'conv1.weight': 20,
                'conv2.weight': 30,
                'fc1.weight': 15
            },
            'layer_wise_corrections': {
                'conv1.weight': 18,
                'conv2.weight': 25,
                'fc1.weight': 12
            }
        },
        'hardware_overhead': {
            'total_latency_ns': 1100.0,
            'computation_latency_ns': 1000.0,
            'voter_latency_ns': 100.0,
            'total_energy_pj': 5500.0,
            'computation_energy_pj': 5000.0,
            'voter_energy_pj': 500.0,
            'latency_overhead_ratio': 2.2,
            'energy_overhead_ratio': 2.75
        },
        'accuracy': {
            'baseline_accuracy': 0.92,
            'faulty_accuracy': 0.75,
            'ft_accuracy': 0.89,
            'accuracy_recovery_rate': 0.82
        },
        'timing': {
            'total_time_seconds': 125.5
        }
    }
    
    config = {
        'fault_injection': {
            'fault_rate': 0.01,
            'fault_models': ['bit_flip', 'output_corruption'],
            'random_seed': 42
        },
        'majority_voter': {
            'voting_strategy': 'simple_majority',
            'tie_breaking': 'detection_failure',
            'voter_latency_ns': 10,
            'voter_energy_pj': 50
        }
    }
    
    # 创建报告生成器
    generator = ReportGenerator(output_dir='./test_reports')
    
    # 生成所有报告
    files = generator.generate_all_reports(metrics, config)
    
    print("\n✅ 测试完成")
    print(f"生成的文件:")
    for format_type, filepath in files.items():
        print(f"  - {format_type}: {filepath}")


if __name__ == "__main__":
    test_report_generator()
