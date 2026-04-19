#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd

from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
from simulator.Fault_Tolerance.redundancy_group_parser import RedundancyGroupParser


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _layer_stats_from_group_info(layer_name, layer_group_info):
    groups = layer_group_info.get('groups', [])
    total_ou_count = int(layer_group_info.get('ou_count', 0))
    repairable_ou_count = int(layer_group_info.get('repairable_ou_count', 0))
    repairable_block_count = int(layer_group_info.get('repairable_block_count', 0))
    total_block_count = int(layer_group_info.get('block_count', total_ou_count))

    exact_repairable_ou = 0
    scaled_repairable_ou = 0
    singleton_groups = 0
    member_counts = []
    total_members = 0

    for group in groups:
        member_count = int(group.get('member_count', group.get('group_size', len(group.get('members', [])))))
        covered_ou_count = int(group.get('covered_ou_count', sum(member.get('channel_span', 1) for member in group.get('members', []))))
        member_counts.append(member_count)
        total_members += member_count
        if member_count < 2:
            singleton_groups += 1
            continue
        if group.get('repair_mode', 'scaled') == 'exact':
            exact_repairable_ou += covered_ou_count
        else:
            scaled_repairable_ou += covered_ou_count

    return {
        'layer_name': layer_name,
        'group_count': len(groups),
        'total_ou_count': total_ou_count,
        'total_block_count': total_block_count,
        'repairable_ou_count': repairable_ou_count,
        'repairable_block_count': repairable_block_count,
        'group_coverage_ratio': _safe_ratio(repairable_ou_count, total_ou_count),
        'exact_repairable_ratio': _safe_ratio(exact_repairable_ou, total_ou_count),
        'scaled_repairable_ratio': _safe_ratio(scaled_repairable_ou, total_ou_count),
        'singleton_ratio': _safe_ratio(singleton_groups, len(groups)),
        'avg_group_size': _safe_ratio(sum(member_counts), len(member_counts)),
        'prototype_member_ratio': _safe_ratio(len(groups), total_members),
        'level1_potential_recovery_ratio': _safe_ratio(exact_repairable_ou + scaled_repairable_ou, total_ou_count),
        'exact_group_count': int(layer_group_info.get('exact_group_count', 0)),
        'scaled_group_count': int(layer_group_info.get('scaled_group_count', 0)),
        'data_source': 'group_information',
    }


def _layer_stats_from_parser(layer_name, parser, data_loader):
    groups = parser.get_layer_groups(layer_name)
    map_table = data_loader.get_layer_map(layer_name)
    total_ou_count = 0
    if map_table is not None and len(map_table.shape) >= 2:
        total_ou_count = int(map_table.shape[0] * map_table.shape[1])

    exact_repairable_ou = 0
    scaled_repairable_ou = 0
    singleton_groups = 0
    group_sizes = []
    total_members = 0

    for group in groups:
        group_size = group.size()
        group_sizes.append(group_size)
        total_members += group_size
        if group_size < 2:
            singleton_groups += 1
            continue
        if group.repair_mode == 'exact' and all(abs(multiplier - 1.0) <= 1e-6 for multiplier in group.multipliers):
            exact_repairable_ou += group_size
        else:
            scaled_repairable_ou += group_size

    repairable_ou_count = exact_repairable_ou + scaled_repairable_ou
    return {
        'layer_name': layer_name,
        'group_count': len(groups),
        'total_ou_count': total_ou_count,
        'total_block_count': total_ou_count,
        'repairable_ou_count': repairable_ou_count,
        'repairable_block_count': repairable_ou_count,
        'group_coverage_ratio': _safe_ratio(repairable_ou_count, total_ou_count),
        'exact_repairable_ratio': _safe_ratio(exact_repairable_ou, total_ou_count),
        'scaled_repairable_ratio': _safe_ratio(scaled_repairable_ou, total_ou_count),
        'singleton_ratio': _safe_ratio(singleton_groups, len(groups)),
        'avg_group_size': _safe_ratio(sum(group_sizes), len(group_sizes)),
        'prototype_member_ratio': _safe_ratio(len(groups), total_members),
        'level1_potential_recovery_ratio': _safe_ratio(repairable_ou_count, total_ou_count),
        'exact_group_count': 0,
        'scaled_group_count': 0,
        'data_source': 'map_information_fallback',
    }


def _aggregate_global_stats(layer_stats):
    if not layer_stats:
        return {}

    total_ou_count = sum(layer['total_ou_count'] for layer in layer_stats)
    total_block_count = sum(layer['total_block_count'] for layer in layer_stats)
    repairable_ou_count = sum(layer['repairable_ou_count'] for layer in layer_stats)
    repairable_block_count = sum(layer['repairable_block_count'] for layer in layer_stats)
    exact_repairable_ou = sum(layer['exact_repairable_ratio'] * layer['total_ou_count'] for layer in layer_stats)
    scaled_repairable_ou = sum(layer['scaled_repairable_ratio'] * layer['total_ou_count'] for layer in layer_stats)
    total_groups = sum(layer['group_count'] for layer in layer_stats)
    singleton_groups = sum(layer['singleton_ratio'] * layer['group_count'] for layer in layer_stats)
    exact_group_count = sum(layer['exact_group_count'] for layer in layer_stats)
    scaled_group_count = sum(layer['scaled_group_count'] for layer in layer_stats)

    return {
        'group_coverage_ratio': _safe_ratio(repairable_ou_count, total_ou_count),
        'block_coverage_ratio': _safe_ratio(repairable_block_count, total_block_count),
        'exact_repairable_ratio': _safe_ratio(exact_repairable_ou, total_ou_count),
        'scaled_repairable_ratio': _safe_ratio(scaled_repairable_ou, total_ou_count),
        'singleton_ratio': _safe_ratio(singleton_groups, total_groups),
        'avg_group_size': _safe_ratio(sum(layer['avg_group_size'] * layer['group_count'] for layer in layer_stats), total_groups),
        'prototype_member_ratio': _safe_ratio(sum(layer['prototype_member_ratio'] * layer['group_count'] for layer in layer_stats), total_groups),
        'level1_potential_recovery_ratio': _safe_ratio(repairable_ou_count, total_ou_count),
        'group_count': total_groups,
        'total_ou_count': total_ou_count,
        'total_block_count': total_block_count,
        'exact_group_count': int(exact_group_count),
        'scaled_group_count': int(scaled_group_count),
    }


def analyse(model_name, translate_name, data_dir):
    loader = PatternDataLoader(model_name=model_name, translate_name=translate_name, data_dir=data_dir)
    if not loader.load_all_data():
        raise RuntimeError('failed to load FT artifacts')

    layer_stats = []
    if loader.group_information:
        for layer_name in loader.get_all_layer_names():
            layer_group_info = loader.get_layer_group_info(layer_name)
            if layer_group_info is None:
                continue
            layer_stats.append(_layer_stats_from_group_info(layer_name, layer_group_info))
    else:
        parser = RedundancyGroupParser(loader)
        if not parser.parse_all_layers():
            raise RuntimeError('failed to parse groups from fallback map information')
        for layer_name in loader.get_all_layer_names():
            layer_stats.append(_layer_stats_from_parser(layer_name, parser, loader))

    global_stats = _aggregate_global_stats(layer_stats)
    return {
        'model_name': model_name,
        'translate_name': translate_name,
        'data_dir': str(Path(data_dir).resolve()),
        'global': global_stats,
        'layers': layer_stats,
    }


def print_summary(report):
    global_stats = report['global']
    print('=' * 72)
    print('FT-Oriented Analysis Summary')
    print('=' * 72)
    print('model = {}'.format(report['model_name']))
    print('translate = {}'.format(report['translate_name']))
    print('group_coverage_ratio = {:.4f}'.format(global_stats.get('group_coverage_ratio', 0.0)))
    print('exact_repairable_ratio = {:.4f}'.format(global_stats.get('exact_repairable_ratio', 0.0)))
    print('scaled_repairable_ratio = {:.4f}'.format(global_stats.get('scaled_repairable_ratio', 0.0)))
    print('singleton_ratio = {:.4f}'.format(global_stats.get('singleton_ratio', 0.0)))
    print('avg_group_size = {:.4f}'.format(global_stats.get('avg_group_size', 0.0)))
    print('prototype_member_ratio = {:.4f}'.format(global_stats.get('prototype_member_ratio', 0.0)))
    print('level1_potential_recovery_ratio = {:.4f}'.format(global_stats.get('level1_potential_recovery_ratio', 0.0)))
    print('group_count = {}'.format(int(global_stats.get('group_count', 0))))
    print('total_ou_count = {}'.format(int(global_stats.get('total_ou_count', 0))))
    print('=' * 72)
    for layer in report['layers']:
        print('{}: coverage={:.4f} exact={:.4f} scaled={:.4f} singleton={:.4f} avg_group_size={:.2f}'.format(
            layer['layer_name'],
            layer['group_coverage_ratio'],
            layer['exact_repairable_ratio'],
            layer['scaled_repairable_ratio'],
            layer['singleton_ratio'],
            layer['avg_group_size'],
        ))


def main():
    parser = argparse.ArgumentParser(description='Analyse FT-oriented grouping artifacts')
    parser.add_argument('--model', type=str, required=True, help='model name')
    parser.add_argument('--translate', type=str, default='ft_group_cluster_translate', help='translate method name')
    parser.add_argument('--data-dir', type=str, default='.', help='artifact directory')
    parser.add_argument('--output-json', type=str, default='', help='optional json output path')
    parser.add_argument('--output-csv', type=str, default='', help='optional csv output path for layer stats')
    args = parser.parse_args()

    report = analyse(args.model, args.translate, args.data_dir)
    print_summary(report)

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print('saved json -> {}'.format(output_json))

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(report['layers']).to_csv(output_csv, index=False)
        print('saved csv -> {}'.format(output_csv))


if __name__ == '__main__':
    main()
