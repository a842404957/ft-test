#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from fault_tolerance_analyse import analyse


ARTIFACT_SUFFIXES = [
    'mask.pkl',
    'map_information.pkl',
    'multiple_relationship_information.pkl',
    'group_information.pkl',
    'coverage_ratio_information.pkl',
    'reuse_ratio_information.pkl',
    'after_translate_parameters.pth',
    'refresh_log.csv',
]


def load_config(config_path: Path):
    if config_path.exists():
        return json.loads(config_path.read_text(encoding='utf-8'))
    return {}


def resolve_report_dir(config_path: Path, override_report_dir: str):
    if override_report_dir:
        return Path(override_report_dir).resolve()
    config = load_config(config_path)
    report_dir = config.get('report', {}).get('output_dir', './fault_tolerance_results')
    return Path(report_dir).resolve()


def collect_artifact_paths(repo_root: Path, model: str, translate: str):
    artifact_paths = {}
    for suffix in ARTIFACT_SUFFIXES:
        path = repo_root / f'model_{model}_{translate}_{suffix}'
        artifact_paths[suffix] = str(path.resolve()) if path.exists() else ''
    return artifact_paths


def copy_latest_reports(report_dir: Path, run_dir: Path):
    copied_paths = []
    if not report_dir.exists():
        return copied_paths

    patterns = [
        'fault_tolerance_report_*.json',
        'fault_tolerance_report_*.md',
        'fault_tolerance_summary_*.csv',
        'comparison_report_*.md',
        'comparison_summary.json',
        'comparison_summary.csv',
        'comparison_summary.md',
    ]
    for pattern in patterns:
        candidates = sorted(report_dir.glob(pattern), key=lambda path: path.stat().st_mtime)
        if not candidates:
            continue
        source = candidates[-1]
        target = run_dir / source.name
        shutil.copy2(source, target)
        copied_paths.append(str(target.resolve()))
    return copied_paths


def write_summary(run_dir: Path, report: dict, metadata: dict):
    global_stats = report.get('global', {})

    metadata_path = run_dir / 'run_metadata.json'
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

    summary_csv = run_dir / 'summary.csv'
    with open(summary_csv, 'w', newline='', encoding='utf-8') as handle:
        fieldnames = [
            'timestamp',
            'model',
            'translate',
            'config',
            'samples',
            'group_coverage_ratio',
            'exact_repairable_ratio',
            'scaled_repairable_ratio',
            'singleton_ratio',
            'avg_group_size',
            'level1_potential_recovery_ratio',
            'report_dir',
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({
            'timestamp': metadata['timestamp'],
            'model': metadata['model'],
            'translate': metadata['translate'],
            'config': metadata['config'],
            'samples': metadata['samples'],
            'group_coverage_ratio': global_stats.get('group_coverage_ratio', 0.0),
            'exact_repairable_ratio': global_stats.get('exact_repairable_ratio', 0.0),
            'scaled_repairable_ratio': global_stats.get('scaled_repairable_ratio', 0.0),
            'singleton_ratio': global_stats.get('singleton_ratio', 0.0),
            'avg_group_size': global_stats.get('avg_group_size', 0.0),
            'level1_potential_recovery_ratio': global_stats.get('level1_potential_recovery_ratio', 0.0),
            'report_dir': metadata['report_dir'],
        })

    summary_md = run_dir / 'summary.md'
    lines = [
        '# FT Run Summary',
        '',
        f"- model: `{metadata['model']}`",
        f"- translate: `{metadata['translate']}`",
        f"- config: `{metadata['config']}`",
        f"- samples: `{metadata['samples']}`",
        f"- report_dir: `{metadata['report_dir']}`",
        '',
        '## Analysis',
        '',
        f"- group_coverage_ratio: `{global_stats.get('group_coverage_ratio', 0.0):.6f}`",
        f"- exact_repairable_ratio: `{global_stats.get('exact_repairable_ratio', 0.0):.6f}`",
        f"- scaled_repairable_ratio: `{global_stats.get('scaled_repairable_ratio', 0.0):.6f}`",
        f"- singleton_ratio: `{global_stats.get('singleton_ratio', 0.0):.6f}`",
        f"- avg_group_size: `{global_stats.get('avg_group_size', 0.0):.6f}`",
        f"- level1_potential_recovery_ratio: `{global_stats.get('level1_potential_recovery_ratio', 0.0):.6f}`",
        '',
        '## Artifacts',
        '',
    ]
    for name, path in metadata['artifact_paths'].items():
        lines.append(f"- {name}: `{path or 'missing'}`")
    lines.extend(['', '## Copied Reports', ''])
    for path in metadata['copied_report_paths']:
        lines.append(f"- `{path}`")
    summary_md.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Collect FT run outputs into a unified results directory')
    parser.add_argument('--model', required=True, help='model name')
    parser.add_argument('--translate', default='ft_group_cluster_translate', help='translate method name')
    parser.add_argument('--config', default='fault_tolerance_config_high_fault_rate.json', help='config file path')
    parser.add_argument('--samples', type=int, default=256, help='simulation sample count')
    parser.add_argument('--results-root', default='results/ft_runs', help='root directory for collected summaries')
    parser.add_argument('--tag', default='', help='optional run tag; defaults to timestamp')
    parser.add_argument('--data-dir', default='.', help='artifact directory for fault_tolerance_analyse')
    parser.add_argument('--report-dir', default='', help='optional report output directory override')
    args = parser.parse_args()

    repo_root = Path.cwd()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = args.tag or timestamp
    run_dir = (repo_root / args.results_root / f'{args.model}_{args.translate}_{run_tag}').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    report = analyse(args.model, args.translate, args.data_dir)
    analysis_json = run_dir / 'analysis.json'
    analysis_csv = run_dir / 'analysis_layers.csv'
    analysis_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    with open(analysis_csv, 'w', newline='', encoding='utf-8') as handle:
        if report['layers']:
            writer = csv.DictWriter(handle, fieldnames=list(report['layers'][0].keys()))
            writer.writeheader()
            writer.writerows(report['layers'])

    config_path = (repo_root / args.config).resolve()
    report_dir = resolve_report_dir(config_path, args.report_dir)
    copied_report_paths = copy_latest_reports(report_dir, run_dir)
    artifact_paths = collect_artifact_paths(repo_root, args.model, args.translate)

    metadata = {
        'timestamp': timestamp,
        'model': args.model,
        'translate': args.translate,
        'config': str(config_path),
        'samples': args.samples,
        'data_dir': str(Path(args.data_dir).resolve()),
        'report_dir': str(report_dir),
        'artifact_paths': artifact_paths,
        'copied_report_paths': copied_report_paths,
        'analysis_json': str(analysis_json.resolve()),
        'analysis_csv': str(analysis_csv.resolve()),
    }
    write_summary(run_dir, report, metadata)

    print(f'collected_results_dir={run_dir}')
    print(f'analysis_json={analysis_json}')
    print(f'analysis_csv={analysis_csv}')


if __name__ == '__main__':
    main()
