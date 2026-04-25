#!/usr/bin/env python3

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_ft_results import read_simulation_summary


def parse_seed_list(seed_text: str):
    seeds = []
    for chunk in str(seed_text).split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start, end = chunk.split('-', 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(chunk))
    return seeds


def latest_file(directory: Path, pattern: str):
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def default_output_dir(model: str, translate: str, tag: str):
    return REPO_ROOT / 'results' / 'ft_runs' / model / translate / tag / 'seed_sweep'


def format_pct(value):
    if value == '' or value is None:
        return ''
    return f'{float(value):.2%}'


def summarize_gate(rows):
    valid_rows = [row for row in rows if row.get('repair_mode') != 'oracle']
    recoveries = [float(row['recovery_rate']) for row in valid_rows if row.get('recovery_rate') != '']
    improved_rates = [float(row['repair_improved_rate']) for row in valid_rows if row.get('repair_improved_rate') != '']
    comparable_rows = [
        row for row in valid_rows
        if row.get('ft_accuracy') not in ('', None) and row.get('faulty_accuracy') not in ('', None)
    ]
    ft_not_worse = all(float(row['ft_accuracy']) >= float(row['faulty_accuracy']) for row in comparable_rows)
    gate = {
        'seed_count': len(valid_rows),
        'recovery_mean': statistics.mean(recoveries) if recoveries else 0.0,
        'recovery_std': statistics.stdev(recoveries) if len(recoveries) > 1 else 0.0,
        'recovery_min': min(recoveries) if recoveries else 0.0,
        'recovery_max': max(recoveries) if recoveries else 0.0,
        'worst_seed': '',
        'ft_accuracy_not_worse_than_faulty': ft_not_worse if comparable_rows else False,
        'avg_recovery_gate': False,
        'no_seed_below_80_gate': False,
        'repair_improved_gate': False,
        'passed': False,
    }
    if valid_rows and recoveries:
        worst_row = min(valid_rows, key=lambda row: float(row.get('recovery_rate') or 0.0))
        gate['worst_seed'] = worst_row.get('seed', '')
    gate['avg_recovery_gate'] = gate['recovery_mean'] >= 0.85
    gate['no_seed_below_80_gate'] = gate['recovery_min'] >= 0.80 if recoveries else False
    gate['repair_improved_gate'] = min(improved_rates) >= 0.90 if improved_rates else False
    gate['passed'] = (
        gate['avg_recovery_gate']
        and gate['no_seed_below_80_gate']
        and gate['ft_accuracy_not_worse_than_faulty']
        and gate['repair_improved_gate']
    )
    return gate


def write_summary(rows, gate, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'seed', 'output_dir', 'summary_csv', 'report_json', 'report_md',
        'baseline_accuracy', 'faulty_accuracy', 'ft_accuracy', 'accuracy_drop',
        'recovery_rate', 'correction_rate', 'repair_improved_rate',
        'total_faults', 'level1_corrections', 'level1_zero_scale_failed',
        'level1_failed_singleton', 'affected_weight_count',
        'stuck_at_zero_count', 'stuck_at_one_count', 'level1_selection_mode',
        'best_pair_used', 'weighted_average_used', 'fallback_to_default',
        'selected_expected_error', 'actual_after_error',
        'oracle_recovery', 'repair_mode',
    ]
    csv_path = output_dir / 'seed_sweep_summary.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})

    json_path = output_dir / 'seed_sweep_summary.json'
    json_path.write_text(json.dumps({'rows': rows, 'gate': gate}, indent=2, ensure_ascii=False), encoding='utf-8')

    md_path = output_dir / 'seed_sweep_summary.md'
    lines = [
        '# Fault Seed Sweep Summary',
        '',
        f"- passed: `{gate['passed']}`",
        f"- recovery mean/std/min/max: `{gate['recovery_mean']:.4f}` / `{gate['recovery_std']:.4f}` / `{gate['recovery_min']:.4f}` / `{gate['recovery_max']:.4f}`",
        f"- worst_seed: `{gate['worst_seed']}`",
        '',
        '| seed | baseline | faulty | ft | recovery | correction | repair improved | total faults | level1 | stuck0 | stuck1 |',
        '| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for row in rows:
        lines.append(
            f"| {row.get('seed', '')} | {format_pct(row.get('baseline_accuracy'))} | "
            f"{format_pct(row.get('faulty_accuracy'))} | {format_pct(row.get('ft_accuracy'))} | "
            f"{format_pct(row.get('recovery_rate'))} | {format_pct(row.get('correction_rate'))} | "
            f"{format_pct(row.get('repair_improved_rate'))} | {row.get('total_faults', '')} | "
            f"{row.get('level1_corrections', '')} | {row.get('stuck_at_zero_count', '')} | {row.get('stuck_at_one_count', '')} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'csv': str(csv_path), 'json': str(json_path), 'md': str(md_path)}


def run_seed_sweep(args, command_runner=subprocess.run):
    seeds = parse_seed_list(args.seeds)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir(args.model, args.translate, args.tag).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        seed_dir = output_dir / f'seed_{seed}'
        seed_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(REPO_ROOT / 'run_hierarchical_fault_tolerance.py'),
            '--mode', 'single',
            '--model', args.model,
            '--translate', args.translate,
            '--config', args.config,
            '--repair-mode', args.repair_mode,
            '--levels', args.levels,
            '--samples', str(args.samples),
            '--fault-seed', str(seed),
            '--artifact-dir', args.artifact_dir,
            '--output-dir', str(seed_dir),
        ]
        optional_args = [
            ('--level1-selection', getattr(args, 'level1_selection', None)),
            ('--level1-topk', getattr(args, 'level1_topk', None)),
            ('--level1-critical-layer-config', getattr(args, 'level1_critical_layer_config', None)),
            ('--level1-max-expected-error', getattr(args, 'level1_max_expected_error', None)),
            ('--level1-min-expected-improvement', getattr(args, 'level1_min_expected_improvement', None)),
            ('--level1-cache-max-group-size', getattr(args, 'level1_cache_max_group_size', None)),
        ]
        for flag, value in optional_args:
            if value not in (None, ''):
                command.extend([flag, str(value)])
        if getattr(args, 'level1_cache_critical_layers_only', False):
            command.append('--level1-cache-critical-layers-only')
        if args.dry_run:
            print(' '.join(command))
        else:
            command_runner(command, cwd=str(REPO_ROOT), check=True)

        summary_csv = latest_file(seed_dir, 'fault_tolerance_summary_*.csv')
        report_json = latest_file(seed_dir, 'fault_tolerance_report_*.json')
        report_md = latest_file(seed_dir, 'fault_tolerance_report_*.md')
        metrics = read_simulation_summary(summary_csv) if summary_csv else {}
        baseline = metrics.get('baseline_accuracy', '')
        faulty = metrics.get('faulty_accuracy', '')
        accuracy_drop = float(baseline) - float(faulty) if baseline != '' and faulty != '' else ''
        repair_mode = metrics.get('repair_mode', args.repair_mode)
        row = {
            'seed': seed,
            'output_dir': str(seed_dir),
            'summary_csv': str(summary_csv) if summary_csv else '',
            'report_json': str(report_json) if report_json else '',
            'report_md': str(report_md) if report_md else '',
            'baseline_accuracy': baseline,
            'faulty_accuracy': faulty,
            'ft_accuracy': metrics.get('ft_accuracy', ''),
            'accuracy_drop': accuracy_drop,
            'recovery_rate': metrics.get('accuracy_recovery_rate', ''),
            'correction_rate': metrics.get('fault_correction_rate', ''),
            'repair_improved_rate': metrics.get('repair_improved_rate', ''),
            'total_faults': metrics.get('total_faults_injected', ''),
            'level1_corrections': metrics.get('level1_corrections', ''),
            'level1_zero_scale_failed': metrics.get('level1_zero_scale_failed', ''),
            'level1_failed_singleton': metrics.get('level1_failed_singleton', ''),
            'affected_weight_count': metrics.get('affected_weight_count', ''),
            'stuck_at_zero_count': metrics.get('stuck_at_zero_count', ''),
            'stuck_at_one_count': metrics.get('stuck_at_one_count', ''),
            'level1_selection_mode': metrics.get('level1_selection_mode', ''),
            'best_pair_used': metrics.get('best_pair_used', ''),
            'weighted_average_used': metrics.get('weighted_average_used', ''),
            'fallback_to_default': metrics.get('fallback_to_default', ''),
            'selected_expected_error': metrics.get('selected_expected_error', ''),
            'actual_after_error': metrics.get('actual_after_error', ''),
            'oracle_recovery': metrics.get('accuracy_recovery_rate', '') if repair_mode == 'oracle' else '',
            'repair_mode': repair_mode,
        }
        rows.append(row)

    gate = summarize_gate(rows)
    paths = write_summary(rows, gate, output_dir)
    print(f"seed_sweep_summary_csv={paths['csv']}")
    print(f"seed_sweep_summary_json={paths['json']}")
    print(f"seed_sweep_summary_md={paths['md']}")
    print(f"gate_passed={gate['passed']}")
    return {'rows': rows, 'gate': gate, **paths}


def build_parser():
    parser = argparse.ArgumentParser(description='Run fault-tolerance simulation across fault seeds.')
    parser.add_argument('--model', required=True)
    parser.add_argument('--translate', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--repair-mode', default='normal', choices=['normal', 'oracle'])
    parser.add_argument('--levels', default='level1', choices=['level1', 'level1_level2', 'all'])
    parser.add_argument('--samples', type=int, default=-1)
    parser.add_argument('--seeds', required=True, help='Comma list or range, e.g. 42,43,44 or 42-51')
    parser.add_argument('--artifact-dir', required=True)
    parser.add_argument('--output-dir', default='')
    parser.add_argument('--tag', default='res18_codebook_adapt')
    parser.add_argument('--level1-selection', default=None, choices=['default', 'best_pair', 'weighted_average'])
    parser.add_argument('--level1-topk', type=int, default=None)
    parser.add_argument('--level1-critical-layer-config', default='')
    parser.add_argument('--level1-max-expected-error', type=float, default=None)
    parser.add_argument('--level1-min-expected-improvement', type=float, default=None)
    parser.add_argument('--level1-cache-max-group-size', type=int, default=None)
    parser.add_argument('--level1-cache-critical-layers-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser


def main():
    args = build_parser().parse_args()
    run_seed_sweep(args)


if __name__ == '__main__':
    main()
