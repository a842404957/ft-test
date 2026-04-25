#!/usr/bin/env python3

import argparse
import csv
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from run_hierarchical_fault_tolerance import (  # noqa: E402
    _build_runtime_config,
    load_model,
    load_test_data,
    resolve_levels_overrides,
    resolve_runtime_model_name,
)
from simulator.Fault_Tolerance.fault_tolerance_simulation import FaultToleranceSimulator  # noqa: E402


DEFAULT_TARGET_LAYERS = [
    'conv11.weight',
    'conv12.weight',
    'conv13.weight',
    'conv14.weight',
    'conv15.weight',
    'conv16.weight',
    'shortcut3.weight',
]


def parse_layer_list(text: str) -> List[str]:
    if not text:
        return list(DEFAULT_TARGET_LAYERS)
    return [chunk.strip() for chunk in text.split(',') if chunk.strip()]


def _ou_to_list(ou: Tuple[int, int]) -> List[int]:
    return [int(ou[0]), int(ou[1])]


def _list_to_ou(value: Iterable[int]) -> Tuple[int, int]:
    out_ch, in_ch = list(value)
    return int(out_ch), int(in_ch)


def serialize_fault_bundle(fault_mask: Dict[str, List[Tuple[int, int]]],
                           detailed_fault_mask: Dict[str, Dict[Tuple[int, int], List[Any]]],
                           fault_detail_stats: Dict[str, Any],
                           seed: int = None) -> Dict[str, Any]:
    detailed_layers = {}
    for layer_name, entries in (detailed_fault_mask or {}).items():
        detailed_layers[layer_name] = [
            {'ou': _ou_to_list(ou), 'entries': copy.deepcopy(values)}
            for ou, values in sorted(entries.items(), key=lambda item: item[0])
        ]
    return {
        'seed': seed,
        'fault_mask': {
            layer_name: [_ou_to_list(ou) for ou in fault_ous]
            for layer_name, fault_ous in (fault_mask or {}).items()
        },
        'detailed_fault_mask': detailed_layers,
        'fault_detail_stats': copy.deepcopy(fault_detail_stats or {}),
    }


def deserialize_fault_bundle(payload: Dict[str, Any]) -> Dict[str, Any]:
    detailed = {}
    for layer_name, entries in (payload.get('detailed_fault_mask') or {}).items():
        detailed[layer_name] = {
            _list_to_ou(item['ou']): copy.deepcopy(item.get('entries', []))
            for item in entries
        }
    return {
        'seed': payload.get('seed'),
        'fault_mask': {
            layer_name: [_list_to_ou(ou) for ou in fault_ous]
            for layer_name, fault_ous in (payload.get('fault_mask') or {}).items()
        },
        'detailed_fault_mask': detailed,
        'fault_detail_stats': copy.deepcopy(payload.get('fault_detail_stats') or {}),
    }


def save_fault_bundle(path: Path,
                      fault_mask: Dict[str, List[Tuple[int, int]]],
                      detailed_fault_mask: Dict[str, Dict[Tuple[int, int], List[Any]]],
                      fault_detail_stats: Dict[str, Any],
                      seed: int = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = serialize_fault_bundle(fault_mask, detailed_fault_mask, fault_detail_stats, seed=seed)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    return path


def load_fault_bundle(path: Path) -> Dict[str, Any]:
    return deserialize_fault_bundle(json.loads(Path(path).read_text(encoding='utf-8')))


def filter_fault_bundle(bundle: Dict[str, Any], excluded_layer: str) -> Dict[str, Any]:
    fault_mask = {
        layer_name: list(fault_ous)
        for layer_name, fault_ous in bundle.get('fault_mask', {}).items()
        if layer_name != excluded_layer
    }
    detailed = {
        layer_name: copy.deepcopy(entries)
        for layer_name, entries in bundle.get('detailed_fault_mask', {}).items()
        if layer_name != excluded_layer
    }
    stats = _recompute_fault_detail_stats(fault_mask, detailed)
    return {
        'seed': bundle.get('seed'),
        'fault_mask': fault_mask,
        'detailed_fault_mask': detailed,
        'fault_detail_stats': stats,
    }


def count_total_faults(fault_mask: Dict[str, List[Tuple[int, int]]]) -> int:
    return sum(len(fault_ous or []) for fault_ous in (fault_mask or {}).values())


def _recompute_fault_detail_stats(fault_mask: Dict[str, List[Tuple[int, int]]],
                                  detailed_fault_mask: Dict[str, Dict[Tuple[int, int], List[Any]]]) -> Dict[str, Any]:
    stats = {'affected_weight_count': 0, 'fault_model_counts': {}, 'layer_details': {}}
    for layer_name, fault_ous in (fault_mask or {}).items():
        layer_detail = stats['layer_details'].setdefault(layer_name, {
            'faulty_ou_count': len(fault_ous or []),
            'affected_weight_count': 0,
            'fault_model_counts': {},
        })
        layer_detail['faulty_ou_count'] = len(fault_ous or [])
        for ou in fault_ous or []:
            entries = (detailed_fault_mask or {}).get(layer_name, {}).get(tuple(ou), [])
            if not entries:
                entries = [0]
            for entry in entries:
                model_name = str(entry.get('fault_model', 'bit_flip')) if isinstance(entry, dict) else 'bit_flip'
                stats['affected_weight_count'] += 1
                stats['fault_model_counts'][model_name] = int(stats['fault_model_counts'].get(model_name, 0)) + 1
                layer_detail['affected_weight_count'] += 1
                layer_detail['fault_model_counts'][model_name] = int(layer_detail['fault_model_counts'].get(model_name, 0)) + 1
    return stats


def restore_layer_or_faulted_ous(model: torch.nn.Module,
                                 original_weights: Dict[str, torch.Tensor],
                                 layer_name: str,
                                 fault_mask: Dict[str, List[Tuple[int, int]]],
                                 restore_mode: str = 'faulted_ous') -> int:
    module_name = layer_name.replace('.weight', '')
    module = model
    for part in module_name.split('.'):
        module = getattr(module, part)
    if layer_name not in original_weights:
        return 0
    original_layer = original_weights[layer_name].to(module.weight.data.device)
    if restore_mode == 'whole_layer':
        module.weight.data.copy_(original_layer)
        return int(module.weight.data.numel())
    restored = 0
    for out_ch, in_ch in fault_mask.get(layer_name, []):
        if module.weight.data.dim() == 4:
            module.weight.data[out_ch, in_ch, :, :] = original_layer[out_ch, in_ch, :, :]
            restored += int(module.weight.data[out_ch, in_ch, :, :].numel())
        elif module.weight.data.dim() == 2:
            module.weight.data[out_ch, in_ch] = original_layer[out_ch, in_ch]
            restored += 1
    return restored


def _evaluate_current_model(simulator: FaultToleranceSimulator, test_loader, num_samples: int) -> float:
    return simulator._evaluate_baseline(test_loader, num_samples)


def _apply_fault_bundle(simulator: FaultToleranceSimulator, bundle: Dict[str, Any]):
    simulator.fault_injector.reset_statistics()
    simulator.majority_voter.reset_statistics()
    simulator.nearest_pattern_corrector.reset_statistics()
    simulator.detailed_fault_mask = copy.deepcopy(bundle.get('detailed_fault_mask') or {})
    simulator.fault_detail_stats = copy.deepcopy(bundle.get('fault_detail_stats') or {})
    simulator._apply_fault_mask(copy.deepcopy(bundle.get('fault_mask') or {}))


def _generate_fault_bundle(simulator: FaultToleranceSimulator, seed: int = None) -> Dict[str, Any]:
    original_weights = simulator._save_model_weights()
    try:
        simulator.fault_injector.reset_statistics()
        fault_mask = simulator._inject_ou_level_faults()
        return {
            'seed': seed,
            'fault_mask': copy.deepcopy(fault_mask),
            'detailed_fault_mask': copy.deepcopy(simulator.detailed_fault_mask),
            'fault_detail_stats': copy.deepcopy(simulator.fault_detail_stats),
        }
    finally:
        simulator._restore_model_weights(original_weights)


def _residual_error_sum(state: Dict[str, torch.Tensor],
                        original_weights: Dict[str, torch.Tensor],
                        layer_name: str,
                        fault_ous: List[Tuple[int, int]]) -> float:
    if layer_name not in state or layer_name not in original_weights:
        return 0.0
    total = 0.0
    layer_state = state[layer_name]
    original = original_weights[layer_name].to(layer_state.device)
    for out_ch, in_ch in fault_ous or []:
        if layer_state.dim() == 4:
            diff = layer_state[out_ch, in_ch, :, :] - original[out_ch, in_ch, :, :]
        elif layer_state.dim() == 2:
            diff = layer_state[out_ch, in_ch] - original[out_ch, in_ch]
        else:
            continue
        total += float(torch.norm(diff.float(), p=2).item())
    return total


def _build_states(simulator: FaultToleranceSimulator, test_loader, num_samples: int, bundle: Dict[str, Any]):
    original_weights = simulator._save_model_weights()
    try:
        _apply_fault_bundle(simulator, bundle)
        faulty_accuracy = _evaluate_current_model(simulator, test_loader, num_samples)
        faulty_state = simulator._save_model_weights()

        simulator._restore_model_weights(original_weights)
        _apply_fault_bundle(simulator, bundle)
        repair_stats = simulator._apply_weight_level_correction(copy.deepcopy(bundle['fault_mask']), original_weights)
        repaired_accuracy = _evaluate_current_model(simulator, test_loader, num_samples)
        repaired_state = simulator._save_model_weights()
    finally:
        simulator._restore_model_weights(original_weights)
    return {
        'original_weights': original_weights,
        'faulty_state': faulty_state,
        'faulty_accuracy': faulty_accuracy,
        'repaired_state': repaired_state,
        'level1_accuracy': repaired_accuracy,
        'repair_stats': repair_stats,
    }


def _layer_quality(repair_stats: Dict[str, Any], layer_name: str) -> Dict[str, Any]:
    return (repair_stats.get('layer_repair_quality', {}).get(layer_name, {}).get('level1', {}) or {})


def build_layer_oracle_rows(simulator: FaultToleranceSimulator,
                            test_loader,
                            num_samples: int,
                            bundle: Dict[str, Any],
                            target_layers: List[str],
                            states: Dict[str, Any] = None,
                            baseline_accuracy: float = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if baseline_accuracy is None:
        baseline_accuracy = _evaluate_current_model(simulator, test_loader, num_samples)
    if states is None:
        states = _build_states(simulator, test_loader, num_samples, bundle)
    original_weights = states['original_weights']
    repaired_state = states['repaired_state']
    repair_stats = states['repair_stats']
    rows = []
    for layer_name in target_layers:
        for restore_mode in ('faulted_ous', 'whole_layer'):
            simulator._restore_model_weights(repaired_state)
            restored_weight_count = restore_layer_or_faulted_ous(
                simulator.model,
                original_weights,
                layer_name,
                bundle['fault_mask'],
                restore_mode=restore_mode,
            )
            oracle_accuracy = _evaluate_current_model(simulator, test_loader, num_samples)
            quality = _layer_quality(repair_stats, layer_name)
            fault_count = len(bundle['fault_mask'].get(layer_name, []))
            corrected = len(repair_stats.get('level1_corrected_ous', {}).get(layer_name, []))
            uncorrected = len(repair_stats.get('uncorrected_by_layer', {}).get(layer_name, []))
            rows.append({
                'layer': layer_name,
                'restore_mode': restore_mode,
                'fault_count': fault_count,
                'level1_corrected': corrected,
                'uncorrected': uncorrected,
                'residual_error_sum': _residual_error_sum(repaired_state, original_weights, layer_name, bundle['fault_mask'].get(layer_name, [])),
                'repair_improved_rate': quality.get('improved_rate', ''),
                'level1_accuracy': states['level1_accuracy'],
                'level1_plus_layer_oracle_accuracy': oracle_accuracy,
                'marginal_accuracy_gain': oracle_accuracy - states['level1_accuracy'],
                'restored_weight_count': restored_weight_count,
            })
    simulator._restore_model_weights(original_weights)
    rows.sort(key=lambda row: (row['restore_mode'] != 'faulted_ous', -float(row['marginal_accuracy_gain'])))
    for restore_mode in ('faulted_ous', 'whole_layer'):
        ranked = [row for row in rows if row['restore_mode'] == restore_mode]
        for rank, row in enumerate(ranked, start=1):
            row['layer_rank_by_gain'] = rank
    summary = {
        'baseline_accuracy': baseline_accuracy,
        'faulty_accuracy': states['faulty_accuracy'],
        'level1_accuracy': states['level1_accuracy'],
        'total_faults': count_total_faults(bundle['fault_mask']),
    }
    return rows, summary


def build_fault_exclusion_rows(simulator: FaultToleranceSimulator,
                               test_loader,
                               num_samples: int,
                               bundle: Dict[str, Any],
                               target_layers: List[str],
                               original_faulty_accuracy: float = None,
                               original_level1_accuracy: float = None,
                               states: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    original_total_faults = count_total_faults(bundle['fault_mask'])
    rows = []
    if states is None:
        states = _build_states(simulator, test_loader, num_samples, bundle)
    healthy_weights = states['original_weights']
    faulty_state = states['faulty_state']
    repaired_state = states['repaired_state']
    if original_faulty_accuracy is None:
        original_faulty_accuracy = states['faulty_accuracy']
    if original_level1_accuracy is None:
        original_level1_accuracy = states['level1_accuracy']
    repair_stats = states.get('repair_stats', {})
    for layer_name in target_layers:
        filtered = filter_fault_bundle(bundle, layer_name)
        try:
            simulator._restore_model_weights(faulty_state)
            restore_layer_or_faulted_ous(
                simulator.model,
                healthy_weights,
                layer_name,
                bundle['fault_mask'],
                restore_mode='whole_layer',
            )
            faulty_without = _evaluate_current_model(simulator, test_loader, num_samples)

            simulator._restore_model_weights(repaired_state)
            restore_layer_or_faulted_ous(
                simulator.model,
                healthy_weights,
                layer_name,
                bundle['fault_mask'],
                restore_mode='whole_layer',
            )
            ft_without = _evaluate_current_model(simulator, test_loader, num_samples)
        finally:
            simulator._restore_model_weights(healthy_weights)
        remaining_total_faults = count_total_faults(filtered['fault_mask'])
        removed_faults = original_total_faults - remaining_total_faults
        rows.append({
            'excluded_layer': layer_name,
            'original_total_faults': original_total_faults,
            'remaining_total_faults': remaining_total_faults,
            'removed_faults': removed_faults,
            'removed_fault_ratio': (removed_faults / original_total_faults) if original_total_faults else 0.0,
            'faulty_accuracy_without_layer_faults': faulty_without,
            'ft_accuracy_without_layer_faults': ft_without,
            'marginal_fault_damage': faulty_without - original_faulty_accuracy,
            'marginal_ft_gain': ft_without - original_level1_accuracy,
            'level1_corrected_without_layer_faults': int(repair_stats.get('level1_count', 0)) - len(
                repair_stats.get('level1_corrected_ous', {}).get(layer_name, [])
            ),
        })
    rows.sort(key=lambda row: -float(row['marginal_fault_damage']))
    return rows


def _write_table(rows: List[Dict[str, Any]], output_dir: Path, stem: str, title: str, summary: Dict[str, Any]) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f'{stem}.csv'
    json_path = output_dir / f'{stem}.json'
    md_path = output_dir / f'{stem}.md'
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    json_path.write_text(json.dumps({'summary': summary, 'rows': rows}, indent=2, ensure_ascii=False), encoding='utf-8')
    lines = [f'# {title}', '']
    for key, value in summary.items():
        lines.append(f'- {key}: `{value}`')
    lines.append('')
    if rows:
        lines.append('| ' + ' | '.join(fieldnames) + ' |')
        lines.append('| ' + ' | '.join(['---'] * len(fieldnames)) + ' |')
        for row in rows:
            lines.append('| ' + ' | '.join(str(row.get(name, '')) for name in fieldnames) + ' |')
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'csv': str(csv_path), 'json': str(json_path), 'md': str(md_path)}


def _build_simulator(args):
    runtime_model_name, num_classes = resolve_runtime_model_name(args.model, args.config)
    final_config, _ = _build_runtime_config(
        base_config_file=args.config,
        config_overrides=resolve_levels_overrides('level1'),
        report_output_dir=str(Path(args.output_dir).resolve()),
        runtime_model_name=runtime_model_name,
        fault_seed=args.fault_seed,
    )
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False, encoding='utf-8') as handle:
        json.dump(final_config, handle, indent=2, ensure_ascii=False)
        runtime_config_path = Path(handle.name)
    device = final_config.get('simulation', {}).get('device', 'cuda')
    model = load_model(
        runtime_model_name,
        args.translate,
        device=device,
        config_file=str(runtime_config_path),
        data_dir=args.artifact_dir,
        num_classes=num_classes,
    )
    if model is None:
        raise RuntimeError('failed to load model')
    test_loader = load_test_data(config_file=str(runtime_config_path))
    simulator = FaultToleranceSimulator(
        model=model,
        model_name=runtime_model_name,
        translate_name=args.translate,
        config_file=str(runtime_config_path),
        data_dir=args.artifact_dir,
    )
    return simulator, test_loader, runtime_config_path


def analyse_layer_impact_seed(args):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    target_layers = parse_layer_list(args.target_layers)
    simulator, test_loader, runtime_config_path = _build_simulator(args)
    try:
        if args.reuse_fault_mask:
            bundle = load_fault_bundle(args.fault_mask_file)
        else:
            bundle = _generate_fault_bundle(simulator, seed=args.fault_seed)
        if args.export_fault_mask:
            fault_mask_path = Path(args.fault_mask_file) if args.fault_mask_file else output_dir / f'seed{args.fault_seed}_fault_mask.json'
            save_fault_bundle(
                fault_mask_path,
                bundle['fault_mask'],
                bundle['detailed_fault_mask'],
                bundle.get('fault_detail_stats', {}),
                seed=args.fault_seed,
            )

        baseline_accuracy = _evaluate_current_model(simulator, test_loader, args.samples)
        states = _build_states(simulator, test_loader, args.samples, bundle)

        results = {}
        layer_rows = []
        layer_summary = {}
        if args.mode in ('layer_oracle', 'both'):
            layer_rows, layer_summary = build_layer_oracle_rows(
                simulator,
                test_loader,
                args.samples,
                bundle,
                target_layers,
                states=states,
                baseline_accuracy=baseline_accuracy,
            )
            results['layer_oracle'] = _write_table(
                layer_rows,
                output_dir,
                'seed_layer_impact',
                'Seed Layer Impact Oracle Ablation',
                layer_summary,
            )

        if args.mode in ('fault_exclusion', 'both'):
            if not layer_summary:
                _, layer_summary = build_layer_oracle_rows(
                    simulator,
                    test_loader,
                    args.samples,
                    bundle,
                    [],
                    states=states,
                    baseline_accuracy=baseline_accuracy,
                )
            exclusion_rows = build_fault_exclusion_rows(
                simulator,
                test_loader,
                args.samples,
                bundle,
                target_layers,
                original_faulty_accuracy=layer_summary['faulty_accuracy'],
                original_level1_accuracy=layer_summary['level1_accuracy'],
                states=states,
            )
            results['fault_exclusion'] = _write_table(
                exclusion_rows,
                output_dir,
                'seed_fault_exclusion',
                'Seed Fault Exclusion Ablation',
                {
                    **layer_summary,
                    'note': 'Fault-exclusion removes faults from the selected layer; it does not resample to keep total fault count constant.',
                },
            )
        return {'results': results, 'layer_rows': layer_rows, 'summary': layer_summary}
    finally:
        try:
            runtime_config_path.unlink(missing_ok=True)
        except Exception:
            pass


def build_parser():
    parser = argparse.ArgumentParser(description='Analyse layer impact for one fault seed using explicit fault-mask reuse.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--translate', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--fault-seed', type=int, default=43)
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--target-layers', default=','.join(DEFAULT_TARGET_LAYERS))
    parser.add_argument('--mode', choices=['layer_oracle', 'fault_exclusion', 'both'], default='both')
    parser.add_argument('--export-fault-mask', action='store_true')
    parser.add_argument('--fault-mask-file', default='')
    parser.add_argument('--reuse-fault-mask', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    result = analyse_layer_impact_seed(args)
    for group in result['results'].values():
        for label, path in group.items():
            print(f'{label}={path}')


if __name__ == '__main__':
    main()
