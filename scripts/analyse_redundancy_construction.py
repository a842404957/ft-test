#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
from simulator.Fault_Tolerance.redundancy_group_parser import RedundancyGroupParser


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _mean_or_zero(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _dominant_keep_count(distribution: Dict[str, Any]) -> int:
    if not distribution:
        return 0
    dominant_key, _ = max(
        distribution.items(),
        key=lambda item: (int(item[1]), -int(item[0])),
    )
    return int(dominant_key)


def _format_multiplier_key(multiplier: float) -> str:
    rounded = round(float(multiplier), 6)
    if abs(rounded) <= 1e-8:
        rounded = 0.0
    return f'{rounded:.6f}'


def _mask_density(mask_tensor) -> float:
    if mask_tensor is None:
        return 0.0
    return float(mask_tensor.float().sum().item() / max(mask_tensor.numel(), 1))


def _infer_pattern_value_number(mask_tensor) -> float:
    if mask_tensor is None or mask_tensor.dim() < 2:
        return 0.0
    reshaped = mask_tensor.reshape(mask_tensor.shape[0], mask_tensor.shape[1], -1)
    active_counts = reshaped.float().sum(dim=2)
    return float(active_counts.mean().item())


def _infer_channel_number_from_group_info(layer_group_info: Dict[str, Any], layer_mask) -> int:
    block_count = int(layer_group_info.get('block_count', 0))
    ou_count = int(layer_group_info.get('ou_count', 0))
    if block_count > 0 and ou_count >= block_count and ou_count % block_count == 0:
        return int(ou_count // block_count)
    for group in layer_group_info.get('groups', []):
        for member in group.get('members', []):
            return int(member.get('channel_span', 1))
    if layer_mask is not None and layer_mask.dim() >= 2:
        return 1
    return 0


def _compute_scale_stats_from_group_members(groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    scale_distribution: Dict[str, int] = {}
    multipliers: List[float] = []
    for group in groups:
        for member in group.get('members', []):
            multiplier = float(member.get('multiplier', 1.0))
            scale_distribution[_format_multiplier_key(multiplier)] = scale_distribution.get(_format_multiplier_key(multiplier), 0) + 1
            multipliers.append(multiplier)
    zero_multiplier = sum(1 for value in multipliers if abs(value) <= 1e-8)
    return {
        'scale_distribution': scale_distribution,
        'zero_multiplier_ratio': _safe_ratio(zero_multiplier, len(multipliers)),
        'nonzero_multiplier_ratio': _safe_ratio(len(multipliers) - zero_multiplier, len(multipliers)),
    }


def _load_projection_metrics(model_name: str, translate_name: str, data_dir: str) -> Dict[str, Any]:
    metrics_path = Path(data_dir) / f'model_{model_name}_{translate_name}_projection_metrics.json'
    if not metrics_path.exists():
        return {
            'projection_strength': 0.0,
            'projected_accuracy': None,
            'projected_accuracy_drop': None,
            'layer_projection_deltas': [],
        }
    with open(metrics_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    return {
        'projection_strength': float(payload.get('projection_strength', 0.0)),
        'projected_accuracy': payload.get('projected_accuracy'),
        'projected_accuracy_drop': payload.get('projected_accuracy_drop'),
        'layer_projection_deltas': payload.get('layer_projection_deltas', []),
    }


def _layer_diag_from_group_info(layer_name: str, layer_group_info: Dict[str, Any], layer_mask) -> Dict[str, Any]:
    groups = layer_group_info.get('groups', [])
    group_sizes = [
        int(group.get('group_size', group.get('member_count', len(group.get('members', [])))))
        for group in groups
    ]
    singleton_groups = sum(1 for size in group_sizes if size < 2)
    scale_stats = _compute_scale_stats_from_group_members(groups)
    seed_info = layer_group_info.get('seed_info', {}) or {}

    diagnostics = {
        'layer': layer_name,
        'data_source': 'group_information',
        'grouping_mode': layer_group_info.get('grouping_mode', 'ftscore'),
        'selected_mask_strategy': seed_info.get('selected_strategy', ''),
        'mask_density': _mask_density(layer_mask),
        'pattern_value_number': _infer_pattern_value_number(layer_mask),
        'channel_number': _infer_channel_number_from_group_info(layer_group_info, layer_mask),
        'total_ous': int(layer_group_info.get('ou_count', 0)),
        'total_groups': int(layer_group_info.get('group_count', len(groups))),
        'singleton_groups': int(layer_group_info.get('singleton_group_count', singleton_groups)),
        'singleton_ratio': float(layer_group_info.get('singleton_ratio', _safe_ratio(singleton_groups, len(groups)))),
        'repairable_ou_ratio': _safe_ratio(int(layer_group_info.get('repairable_ou_count', 0)), int(layer_group_info.get('ou_count', 0))),
        'avg_group_size': float(layer_group_info.get('avg_group_size', _mean_or_zero(group_sizes))),
        'max_group_size': int(layer_group_info.get('max_group_size', max(group_sizes) if group_sizes else 0)),
        'exact_group_ratio': float(layer_group_info.get('exact_group_ratio', 0.0)),
        'scaled_group_ratio': float(layer_group_info.get('scaled_group_ratio', 0.0)),
        'zero_multiplier_ratio': float(layer_group_info.get('zero_multiplier_ratio', scale_stats['zero_multiplier_ratio'])),
        'nonzero_multiplier_ratio': float(layer_group_info.get('nonzero_multiplier_ratio', scale_stats['nonzero_multiplier_ratio'])),
        'scale_distribution': layer_group_info.get('scale_distribution', scale_stats['scale_distribution']),
        'selected_candidate_coverage': float(seed_info.get('estimated_coverage', 0.0)),
        'selected_candidate_distortion': float(seed_info.get('pruning_distortion', 0.0)),
        'selected_candidate_singleton_ratio': float(seed_info.get('estimated_singleton_ratio', 0.0)),
        'selected_candidate_zero_multiplier_ratio': float(seed_info.get('estimated_zero_multiplier_ratio', 0.0)),
        'selected_candidate_avg_group_size': float(seed_info.get('estimated_avg_group_size', 0.0)),
        'target_ratio': float(layer_group_info.get('target_ratio', 0.0)),
        'candidate_count': int(seed_info.get('candidate_count', 0)),
        'candidate_summaries': seed_info.get('candidate_summaries', []),
        'prototype_budget_ratio': float(layer_group_info.get('prototype_budget_ratio', 0.0)),
        'prototype_budget': int(layer_group_info.get('prototype_budget', 0)),
        'prototype_count': int(layer_group_info.get('prototype_count', layer_group_info.get('group_count', len(groups)))),
        'bucket_mode': str(layer_group_info.get('bucket_mode', '')),
        'mask_family': layer_group_info.get('mask_family', []),
        'mask_keep_ratios': layer_group_info.get('mask_keep_ratios', []),
        'target_coverage': float(layer_group_info.get('target_coverage', 0.0)),
        'achieved_coverage': float(layer_group_info.get('achieved_coverage', layer_group_info.get('coverage_ratio', 0.0))),
        'coverage_gap': float(layer_group_info.get('coverage_gap', 0.0)),
        'assignment_error_mean': float(layer_group_info.get('assignment_error_mean', 0.0)),
        'assignment_error_p50': float(layer_group_info.get('assignment_error_p50', 0.0)),
        'assignment_error_p95': float(layer_group_info.get('assignment_error_p95', 0.0)),
        'direction_error_mean': float(layer_group_info.get('direction_error_mean', 0.0)),
        'direction_error_p50': float(layer_group_info.get('direction_error_p50', 0.0)),
        'direction_error_p95': float(layer_group_info.get('direction_error_p95', 0.0)),
        'raw_error_mean': float(layer_group_info.get('raw_error_mean', 0.0)),
        'raw_error_p50': float(layer_group_info.get('raw_error_p50', 0.0)),
        'raw_error_p95': float(layer_group_info.get('raw_error_p95', 0.0)),
        'max_scale_error': float(layer_group_info.get('max_scale_error', 0.0)),
        'max_singleton_error': float(layer_group_info.get('max_singleton_error', 0.0)),
        'mask_codebook_size': int(layer_group_info.get('mask_codebook_size', 0)),
        'mask_keep_count_distribution': layer_group_info.get('mask_keep_count_distribution', {}),
        'dominant_mask_keep_count': int(layer_group_info.get('dominant_mask_keep_count', _dominant_keep_count(layer_group_info.get('mask_keep_count_distribution', {})))),
        'mask_codebook_entropy': float(layer_group_info.get('mask_codebook_entropy', 0.0)),
        'mask_codebook_id_distribution': layer_group_info.get('mask_codebook_id_distribution', {}),
        'mask_codebook_source': str(layer_group_info.get('mask_codebook_source', '')),
        'mask_codebook_assign': str(layer_group_info.get('mask_codebook_assign', '')),
        'prototype_space': str(layer_group_info.get('prototype_space', '')),
        'projection_cap': float(layer_group_info.get('projection_cap', 1.0)),
        'projection_lambda': float(layer_group_info.get('projection_lambda', 1.0)),
        'forced_assignment_count': int(layer_group_info.get('forced_assignment_count', 0)),
        'singleton_due_to_high_error': int(layer_group_info.get('singleton_due_to_high_error', 0)),
        'force_prototype_assignment': int(layer_group_info.get('force_prototype_assignment', 0)),
        'relaxed': int(layer_group_info.get('relaxed', 0)),
        'relax_steps': int(layer_group_info.get('relax_steps', 0)),
    }
    diagnostics['issue_tags'] = _infer_issue_tags(diagnostics)
    return diagnostics


def _layer_diag_from_parser(layer_name: str, groups, loader: PatternDataLoader) -> Dict[str, Any]:
    map_table = loader.get_layer_map(layer_name)
    layer_mask = loader.get_layer_mask(layer_name)
    total_ous = 0
    if map_table is not None and len(map_table.shape) >= 2:
        total_ous = int(map_table.shape[0] * map_table.shape[1])

    group_sizes = [group.size() for group in groups]
    singleton_groups = sum(1 for size in group_sizes if size < 2)
    repairable_ous = sum(size for size in group_sizes if size >= 2)
    exact_groups = 0
    scaled_groups = 0
    scale_distribution: Dict[str, int] = {}
    multipliers: List[float] = []
    channel_numbers: List[int] = []

    for group in groups:
        if group.size() >= 2:
            if group.repair_mode == 'exact' and all(abs(multiplier - 1.0) <= 1e-6 for multiplier in group.multipliers):
                exact_groups += 1
            else:
                scaled_groups += 1
        for multiplier in group.multipliers:
            multipliers.append(float(multiplier))
            key = _format_multiplier_key(multiplier)
            scale_distribution[key] = scale_distribution.get(key, 0) + 1
        if group.block_members:
            channel_numbers.extend(int(member.get('channel_span', 1)) for member in group.block_members)

    zero_multiplier = sum(1 for value in multipliers if abs(value) <= 1e-8)
    diagnostics = {
        'layer': layer_name,
        'data_source': 'map_information_fallback',
        'grouping_mode': 'legacy_prap_fallback',
        'selected_mask_strategy': 'legacy_prap_fallback',
        'mask_density': _mask_density(layer_mask),
        'pattern_value_number': _infer_pattern_value_number(layer_mask),
        'channel_number': int(round(_mean_or_zero(channel_numbers))) if channel_numbers else 1,
        'total_ous': total_ous,
        'total_groups': len(groups),
        'singleton_groups': singleton_groups,
        'singleton_ratio': _safe_ratio(singleton_groups, len(groups)),
        'repairable_ou_ratio': _safe_ratio(repairable_ous, total_ous),
        'avg_group_size': _mean_or_zero(group_sizes),
        'max_group_size': max(group_sizes) if group_sizes else 0,
        'exact_group_ratio': _safe_ratio(exact_groups, len(groups)),
        'scaled_group_ratio': _safe_ratio(scaled_groups, len(groups)),
        'zero_multiplier_ratio': _safe_ratio(zero_multiplier, len(multipliers)),
        'nonzero_multiplier_ratio': _safe_ratio(len(multipliers) - zero_multiplier, len(multipliers)),
        'scale_distribution': scale_distribution,
        'selected_candidate_coverage': loader.get_layer_coverage_ratio(layer_name),
        'selected_candidate_distortion': 0.0,
        'selected_candidate_singleton_ratio': _safe_ratio(singleton_groups, len(groups)),
        'selected_candidate_zero_multiplier_ratio': _safe_ratio(zero_multiplier, len(multipliers)),
        'selected_candidate_avg_group_size': _mean_or_zero(group_sizes),
        'target_ratio': 0.0,
        'candidate_count': 0,
        'candidate_summaries': [],
        'prototype_budget_ratio': 0.0,
        'prototype_budget': 0,
        'prototype_count': len(groups),
        'bucket_mode': '',
        'mask_family': [],
        'mask_keep_ratios': [],
        'target_coverage': 0.0,
        'achieved_coverage': _safe_ratio(repairable_ous, total_ous),
        'coverage_gap': 0.0,
        'assignment_error_mean': 0.0,
        'assignment_error_p50': 0.0,
        'assignment_error_p95': 0.0,
        'direction_error_mean': 0.0,
        'direction_error_p50': 0.0,
        'direction_error_p95': 0.0,
        'raw_error_mean': 0.0,
        'raw_error_p50': 0.0,
        'raw_error_p95': 0.0,
        'max_scale_error': 0.0,
        'max_singleton_error': 0.0,
        'mask_codebook_size': 0,
        'mask_keep_count_distribution': {},
        'dominant_mask_keep_count': 0,
        'mask_codebook_entropy': 0.0,
        'mask_codebook_id_distribution': {},
        'mask_codebook_source': '',
        'mask_codebook_assign': '',
        'prototype_space': '',
        'projection_cap': 1.0,
        'projection_lambda': 1.0,
        'forced_assignment_count': 0,
        'singleton_due_to_high_error': 0,
        'force_prototype_assignment': 0,
        'relaxed': 0,
        'relax_steps': 0,
    }
    diagnostics['issue_tags'] = _infer_issue_tags(diagnostics)
    return diagnostics


def _infer_issue_tags(row: Dict[str, Any]) -> List[str]:
    tags = []
    strategy = str(row.get('selected_mask_strategy', ''))
    mask_density = float(row.get('mask_density', 0.0))
    singleton_ratio = float(row.get('singleton_ratio', 0.0))
    repairable_ou_ratio = float(row.get('repairable_ou_ratio', 0.0))
    zero_multiplier_ratio = float(row.get('zero_multiplier_ratio', 0.0))
    target_ratio = float(row.get('target_ratio', 0.0))

    if mask_density >= 0.9 and singleton_ratio >= 0.8:
        tags.append('dense_mask_high_singleton')
    if strategy in ('dense_mask', 'shape_seed') or 'shape_seed' in strategy or strategy.startswith('shared_topk_1.0000'):
        tags.append('weak_pruning_strategy')
    if zero_multiplier_ratio >= 0.05:
        tags.append('zero_multiplier_issue')
    if repairable_ou_ratio <= 0.1 and singleton_ratio >= 0.9:
        tags.append('low_redundancy_high_singleton')
    if target_ratio > 0 and repairable_ou_ratio + 1e-8 < target_ratio:
        tags.append('below_target_coverage')
    return tags


def build_redundancy_construction_report(model_name: str, translate_name: str, data_dir: str) -> Dict[str, Any]:
    loader = PatternDataLoader(model_name=model_name, translate_name=translate_name, data_dir=data_dir)
    if not loader.load_all_data():
        raise RuntimeError('failed to load redundancy construction artifacts')

    layer_rows: List[Dict[str, Any]] = []
    if loader.group_information:
        for layer_name in loader.get_all_layer_names():
            layer_group_info = loader.get_layer_group_info(layer_name)
            if layer_group_info is None:
                continue
            layer_rows.append(_layer_diag_from_group_info(layer_name, layer_group_info, loader.get_layer_mask(layer_name)))
    else:
        parser = RedundancyGroupParser(loader)
        if not parser.parse_all_layers():
            raise RuntimeError('failed to parse redundancy groups from fallback map information')
        for layer_name in loader.get_all_layer_names():
            layer_rows.append(_layer_diag_from_parser(layer_name, parser.get_layer_groups(layer_name), loader))

    projection_metrics = _load_projection_metrics(model_name, translate_name, data_dir)
    delta_by_layer = {
        str(item.get('layer')): float(item.get('relative_weight_delta', 0.0))
        for item in projection_metrics.get('layer_projection_deltas', []) or []
    }
    for row in layer_rows:
        row['projection_strength'] = float(projection_metrics.get('projection_strength', 0.0))
        row['projected_accuracy'] = projection_metrics.get('projected_accuracy')
        row['projected_accuracy_drop'] = projection_metrics.get('projected_accuracy_drop')
        row['projection_layer_delta'] = float(delta_by_layer.get(row['layer'], 0.0))

    sorted_singleton = sorted(layer_rows, key=lambda row: (row['singleton_ratio'], row['singleton_groups']), reverse=True)
    singleton_topk = sorted_singleton[: min(8, len(sorted_singleton))]
    correlation = 0.0
    if len(layer_rows) >= 2:
        df = pd.DataFrame(layer_rows)
        if df['mask_density'].nunique() > 1 and df['repairable_ou_ratio'].nunique() > 1:
            correlation = float(df['mask_density'].corr(df['repairable_ou_ratio']))

    global_summary = {
        'layer_count': len(layer_rows),
        'group_count': int(sum(row['total_groups'] for row in layer_rows)),
        'total_ous': int(sum(row['total_ous'] for row in layer_rows)),
        'avg_group_size': _mean_or_zero([row['avg_group_size'] for row in layer_rows]),
        'avg_repairable_ou_ratio': _mean_or_zero([row['repairable_ou_ratio'] for row in layer_rows]),
        'avg_singleton_ratio': _mean_or_zero([row['singleton_ratio'] for row in layer_rows]),
        'avg_assignment_error_mean': _mean_or_zero([row['assignment_error_mean'] for row in layer_rows]),
        'avg_assignment_error_p50': _mean_or_zero([row['assignment_error_p50'] for row in layer_rows]),
        'avg_assignment_error_p95': _mean_or_zero([row['assignment_error_p95'] for row in layer_rows]),
        'avg_projection_layer_delta': _mean_or_zero([row['projection_layer_delta'] for row in layer_rows]),
        'mask_density_vs_repairable_ratio_corr': correlation,
        'projection_strength': float(projection_metrics.get('projection_strength', 0.0)),
        'projected_accuracy': projection_metrics.get('projected_accuracy'),
        'projected_accuracy_drop': projection_metrics.get('projected_accuracy_drop'),
        'singleton_topk_layers': singleton_topk,
    }

    return {
        'model_name': model_name,
        'translate_name': translate_name,
        'data_dir': str(Path(data_dir).resolve()),
        'global': global_summary,
        'layers': layer_rows,
    }


def _build_report(model_name: str, translate_name: str, data_dir: str) -> Dict[str, Any]:
    return build_redundancy_construction_report(model_name, translate_name, data_dir)


def _write_markdown(report: Dict[str, Any], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('# Redundancy Construction Diagnostics\n\n')
        handle.write(f"- model: `{report['model_name']}`\n")
        handle.write(f"- translate: `{report['translate_name']}`\n")
        handle.write(f"- data_dir: `{report['data_dir']}`\n\n")
        global_summary = report['global']
        handle.write('## Global Summary\n\n')
        handle.write(f"- layer_count: {global_summary['layer_count']}\n")
        handle.write(f"- group_count: {global_summary['group_count']}\n")
        handle.write(f"- total_ous: {global_summary['total_ous']}\n")
        handle.write(f"- avg_group_size: {global_summary['avg_group_size']:.4f}\n")
        handle.write(f"- avg_repairable_ou_ratio: {global_summary['avg_repairable_ou_ratio']:.4f}\n")
        handle.write(f"- avg_singleton_ratio: {global_summary['avg_singleton_ratio']:.4f}\n")
        handle.write(f"- avg_assignment_error_mean: {global_summary['avg_assignment_error_mean']:.4f}\n")
        handle.write(f"- avg_assignment_error_p50: {global_summary['avg_assignment_error_p50']:.4f}\n")
        handle.write(f"- avg_assignment_error_p95: {global_summary['avg_assignment_error_p95']:.4f}\n")
        handle.write(f"- avg_projection_layer_delta: {global_summary['avg_projection_layer_delta']:.4f}\n")
        handle.write(f"- mask_density_vs_repairable_ratio_corr: {global_summary['mask_density_vs_repairable_ratio_corr']:.4f}\n\n")
        if global_summary.get('projected_accuracy') is not None:
            handle.write(f"- projection_strength: {global_summary['projection_strength']:.4f}\n")
            handle.write(f"- projected_accuracy: {global_summary['projected_accuracy']:.4f}\n")
            handle.write(f"- projected_accuracy_drop: {global_summary['projected_accuracy_drop']:.4f}\n\n")

        handle.write('## Singleton Top-K Layers\n\n')
        handle.write('| layer | strategy | mask_density | singleton_ratio | repairable_ou_ratio | assignment_error_p95 | zero_multiplier_ratio | issue_tags |\n')
        handle.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for row in global_summary['singleton_topk_layers']:
            handle.write(
                f"| {row['layer']} | {row['selected_mask_strategy']} | {row['mask_density']:.4f} | "
                f"{row['singleton_ratio']:.4f} | {row['repairable_ou_ratio']:.4f} | {row['assignment_error_p95']:.4f} | "
                f"{row['zero_multiplier_ratio']:.4f} | {', '.join(row['issue_tags'])} |\n"
            )
        handle.write('\n')

        handle.write('## Layer Diagnostics\n\n')
        handle.write('| layer | grouping_mode | strategy | keep_count | projection_cap | mask_density | repairable_ou_ratio | singleton_ratio | avg_group_size | assignment_error_p95 | projection_layer_delta | projected_accuracy_drop |\n')
        handle.write('| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for row in report['layers']:
            projected_drop = '' if row['projected_accuracy_drop'] is None else f"{row['projected_accuracy_drop']:.4f}"
            handle.write(
                f"| {row['layer']} | {row['grouping_mode']} | {row['selected_mask_strategy']} | {row['dominant_mask_keep_count']} | {row['projection_cap']:.4f} | {row['mask_density']:.4f} | "
                f"{row['repairable_ou_ratio']:.4f} | {row['singleton_ratio']:.4f} | "
                f"{row['avg_group_size']:.4f} | {row['assignment_error_p95']:.4f} | {row['projection_layer_delta']:.4f} | "
                f"{projected_drop} |\n"
            )


def _comparison_rows(ft_report: Dict[str, Any], prap_report: Dict[str, Any]) -> List[Dict[str, Any]]:
    prap_by_layer = {row['layer']: row for row in prap_report['layers']}
    rows = []
    for ft_row in ft_report['layers']:
        prap_row = prap_by_layer.get(ft_row['layer'])
        if prap_row is None:
            continue
        rows.append({
            'layer': ft_row['layer'],
            'ft_repairable_ou_ratio': ft_row['repairable_ou_ratio'],
            'prap_repairable_ou_ratio': prap_row['repairable_ou_ratio'],
            'repairable_ou_ratio_delta': ft_row['repairable_ou_ratio'] - prap_row['repairable_ou_ratio'],
            'ft_avg_group_size': ft_row['avg_group_size'],
            'prap_avg_group_size': prap_row['avg_group_size'],
            'avg_group_size_delta': ft_row['avg_group_size'] - prap_row['avg_group_size'],
            'ft_singleton_ratio': ft_row['singleton_ratio'],
            'prap_singleton_ratio': prap_row['singleton_ratio'],
            'singleton_ratio_delta': ft_row['singleton_ratio'] - prap_row['singleton_ratio'],
            'ft_scale_distribution': json.dumps(ft_row['scale_distribution'], ensure_ascii=False),
            'prap_scale_distribution': json.dumps(prap_row['scale_distribution'], ensure_ascii=False),
        })
    return rows


def _write_comparison_markdown(rows: List[Dict[str, Any]], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as handle:
        handle.write('# PRAP vs FT Redundancy Summary\n\n')
        handle.write('| layer | ft_repairable_ou_ratio | prap_repairable_ou_ratio | delta | ft_avg_group_size | prap_avg_group_size | ft_singleton_ratio | prap_singleton_ratio |\n')
        handle.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for row in rows:
            handle.write(
                f"| {row['layer']} | {row['ft_repairable_ou_ratio']:.4f} | {row['prap_repairable_ou_ratio']:.4f} | "
                f"{row['repairable_ou_ratio_delta']:.4f} | {row['ft_avg_group_size']:.4f} | {row['prap_avg_group_size']:.4f} | "
                f"{row['ft_singleton_ratio']:.4f} | {row['prap_singleton_ratio']:.4f} |\n"
            )


def main():
    parser = argparse.ArgumentParser(description='Analyse redundancy construction diagnostics for FT-oriented grouping')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--translate', type=str, default='ft_group_cluster_translate', help='translate method name')
    parser.add_argument('--data-dir', type=str, default='.', help='artifact directory')
    parser.add_argument('--output-dir', type=str, default='.', help='directory to save csv/json/md outputs')
    parser.add_argument('--prap-translate', type=str, default='', help='optional PRAP translate label for statistical comparison')
    parser.add_argument('--prap-data-dir', type=str, default='', help='optional artifact directory for PRAP baseline; defaults to --data-dir')
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_redundancy_construction_report(args.model, args.translate, args.data_dir)
    csv_path = output_dir / f'model_{args.model}_{args.translate}_redundancy_diagnostics.csv'
    json_path = output_dir / f'model_{args.model}_{args.translate}_redundancy_diagnostics.json'
    md_path = output_dir / f'model_{args.model}_{args.translate}_redundancy_diagnostics.md'

    csv_rows = []
    for row in report['layers']:
        csv_row = dict(row)
        csv_row['scale_distribution'] = json.dumps(csv_row['scale_distribution'], ensure_ascii=False)
        csv_row['candidate_summaries'] = json.dumps(csv_row['candidate_summaries'], ensure_ascii=False)
        csv_row['mask_keep_count_distribution'] = json.dumps(csv_row.get('mask_keep_count_distribution', {}), ensure_ascii=False)
        csv_row['mask_codebook_id_distribution'] = json.dumps(csv_row.get('mask_codebook_id_distribution', {}), ensure_ascii=False)
        csv_row['mask_family'] = json.dumps(csv_row.get('mask_family', []), ensure_ascii=False)
        csv_row['mask_keep_ratios'] = json.dumps(csv_row.get('mask_keep_ratios', []), ensure_ascii=False)
        csv_row['issue_tags'] = ','.join(csv_row['issue_tags'])
        csv_rows.append(csv_row)
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_markdown(report, md_path)
    budget_csv_path = output_dir / f'model_{args.model}_{args.translate}_budgeted_diagnostics.csv'
    budget_json_path = output_dir / f'model_{args.model}_{args.translate}_budgeted_diagnostics.json'
    budget_md_path = output_dir / f'model_{args.model}_{args.translate}_budgeted_diagnostics.md'
    pd.DataFrame(csv_rows).to_csv(budget_csv_path, index=False)
    budget_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    _write_markdown(report, budget_md_path)

    print(f'saved csv -> {csv_path}')
    print(f'saved json -> {json_path}')
    print(f'saved md -> {md_path}')
    print(f'saved budgeted csv -> {budget_csv_path}')
    print(f'saved budgeted json -> {budget_json_path}')
    print(f'saved budgeted md -> {budget_md_path}')
    print('top singleton layers:')
    for row in report['global']['singleton_topk_layers'][:5]:
        print('  - {} strategy={} mask_density={:.4f} singleton_ratio={:.4f} repairable_ou_ratio={:.4f} tags={}'.format(
            row['layer'],
            row['selected_mask_strategy'],
            row['mask_density'],
            row['singleton_ratio'],
            row['repairable_ou_ratio'],
            ','.join(row['issue_tags']),
        ))

    if args.prap_translate:
        prap_report = build_redundancy_construction_report(args.model, args.prap_translate, args.prap_data_dir or args.data_dir)
        comparison_rows = _comparison_rows(report, prap_report)
        comparison_json = output_dir / 'prap_vs_ft_redundancy_summary.json'
        comparison_csv = output_dir / 'prap_vs_ft_redundancy_summary.csv'
        comparison_md = output_dir / 'prap_vs_ft_redundancy_summary.md'
        comparison_json.write_text(json.dumps(comparison_rows, indent=2, ensure_ascii=False), encoding='utf-8')
        pd.DataFrame(comparison_rows).to_csv(comparison_csv, index=False)
        _write_comparison_markdown(comparison_rows, comparison_md)
        print(f'saved prap-vs-ft json -> {comparison_json}')
        print(f'saved prap-vs-ft csv -> {comparison_csv}')
        print(f'saved prap-vs-ft md -> {comparison_md}')


if __name__ == '__main__':
    main()
