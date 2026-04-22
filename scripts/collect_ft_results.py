#!/usr/bin/env python3

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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
    'training_profile.csv',
    'regularization_layers.csv',
    'mask_sweep_report.csv',
    'mask_sweep_report.json',
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


def build_run_dir(results_root: Path, model: str, translate: str, tag: str) -> Path:
    return (results_root / model / translate / tag).resolve()


def collect_artifact_paths(artifact_dir: Path, model: str, translate: str):
    artifact_paths = {}
    for suffix in ARTIFACT_SUFFIXES:
        path = artifact_dir / f'model_{model}_{translate}_{suffix}'
        artifact_paths[suffix] = str(path.resolve()) if path.exists() else ''
    return artifact_paths


def copy_artifacts(artifact_dir: Path, run_dir: Path, model: str, translate: str):
    copied_paths = {}
    artifacts_dir = run_dir / 'artifacts'
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ARTIFACT_SUFFIXES:
        source = artifact_dir / f'model_{model}_{translate}_{suffix}'
        if not source.exists():
            copied_paths[suffix] = ''
            continue
        target = artifacts_dir / source.name
        if source.resolve() != target.resolve():
            shutil.copy2(source, target)
        copied_paths[suffix] = str(target.resolve())
    return copied_paths


def copy_latest_reports(report_dir: Path, run_dir: Path):
    copied_paths = []
    if not report_dir.exists():
        return copied_paths

    reports_dir = run_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)

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
        target = reports_dir / source.name
        shutil.copy2(source, target)
        copied_paths.append(str(target.resolve()))
    return copied_paths


def classify_report_paths(copied_paths):
    manifest = {
        'simulation_report_json': '',
        'simulation_report_md': '',
        'simulation_summary_csv': '',
        'comparison_report_md': '',
        'comparison_summary_json': '',
        'comparison_summary_csv': '',
        'comparison_summary_md': '',
    }
    for raw_path in copied_paths:
        name = Path(raw_path).name
        if name.startswith('fault_tolerance_report_') and name.endswith('.json'):
            manifest['simulation_report_json'] = raw_path
        elif name.startswith('fault_tolerance_report_') and name.endswith('.md'):
            manifest['simulation_report_md'] = raw_path
        elif name.startswith('fault_tolerance_summary_') and name.endswith('.csv'):
            manifest['simulation_summary_csv'] = raw_path
        elif name == 'comparison_report.md' or name.startswith('comparison_report_'):
            manifest['comparison_report_md'] = raw_path
        elif name == 'comparison_summary.json':
            manifest['comparison_summary_json'] = raw_path
        elif name == 'comparison_summary.csv':
            manifest['comparison_summary_csv'] = raw_path
        elif name == 'comparison_summary.md':
            manifest['comparison_summary_md'] = raw_path
    return manifest


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
            'analysis_json',
            'analysis_csv',
            'training_profile_csv',
            'regularization_layers_csv',
            'simulation_summary_csv',
            'comparison_summary_csv',
            'artifact_dir',
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
            'analysis_json': metadata['analysis_json'],
            'analysis_csv': metadata['analysis_csv'],
            'training_profile_csv': metadata['copied_artifact_paths'].get('training_profile.csv', ''),
            'regularization_layers_csv': metadata['copied_artifact_paths'].get('regularization_layers.csv', ''),
            'simulation_summary_csv': metadata['report_manifest'].get('simulation_summary_csv', ''),
            'comparison_summary_csv': metadata['report_manifest'].get('comparison_summary_csv', ''),
            'artifact_dir': metadata['artifact_dir'],
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
        f"- analysis_json: `{metadata['analysis_json']}`",
        f"- analysis_csv: `{metadata['analysis_csv']}`",
        f"- artifact_dir: `{metadata['artifact_dir']}`",
        f"- group_coverage_ratio: `{global_stats.get('group_coverage_ratio', 0.0):.6f}`",
        f"- exact_repairable_ratio: `{global_stats.get('exact_repairable_ratio', 0.0):.6f}`",
        f"- scaled_repairable_ratio: `{global_stats.get('scaled_repairable_ratio', 0.0):.6f}`",
        f"- singleton_ratio: `{global_stats.get('singleton_ratio', 0.0):.6f}`",
        f"- avg_group_size: `{global_stats.get('avg_group_size', 0.0):.6f}`",
        f"- level1_potential_recovery_ratio: `{global_stats.get('level1_potential_recovery_ratio', 0.0):.6f}`",
        '',
        '## Simulation Reports',
        '',
    ]
    for name, path in metadata['report_manifest'].items():
        lines.append(f"- {name}: `{path or 'missing'}`")
    lines.extend([
        '',
        '## Artifacts',
        '',
    ])
    for name, path in metadata['artifact_paths'].items():
        lines.append(f"- {name}: `{path or 'missing'}`")
    lines.extend(['', '## Copied Artifacts', ''])
    for name, path in metadata['copied_artifact_paths'].items():
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
    parser.add_argument('--results-root', default='results/ft_runs', help='root directory; final path is <results-root>/<model>/<translate>/<tag>/')
    parser.add_argument('--tag', default='', help='optional run tag; defaults to timestamp and becomes the last path component')
    parser.add_argument('--data-dir', default='.', help='deprecated alias for artifact directory; kept for compatibility')
    parser.add_argument('--artifact-dir', default='', help='artifact directory for analysis and artifact collection')
    parser.add_argument('--report-dir', default='', help='optional report output directory override')
    args = parser.parse_args()

    repo_root = Path.cwd().resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_tag = args.tag or timestamp
    run_dir = build_run_dir(Path(args.results_root).resolve() if Path(args.results_root).is_absolute() else (repo_root / args.results_root), args.model, args.translate, run_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.artifact_dir or args.data_dir).resolve()

    report = analyse(args.model, args.translate, str(artifact_dir))
    analysis_dir = run_dir / 'analysis'
    analysis_dir.mkdir(parents=True, exist_ok=True)
    analysis_json = analysis_dir / 'ft_report.json'
    analysis_csv = analysis_dir / 'ft_layers.csv'
    analysis_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')

    with open(analysis_csv, 'w', newline='', encoding='utf-8') as handle:
        if report['layers']:
            writer = csv.DictWriter(handle, fieldnames=list(report['layers'][0].keys()))
            writer.writeheader()
            writer.writerows(report['layers'])

    config_path = (repo_root / args.config).resolve()
    report_dir = resolve_report_dir(config_path, args.report_dir)
    copied_report_paths = copy_latest_reports(report_dir, run_dir)
    report_manifest = classify_report_paths(copied_report_paths)
    artifact_paths = collect_artifact_paths(artifact_dir, args.model, args.translate)
    copied_artifact_paths = copy_artifacts(artifact_dir, run_dir, args.model, args.translate)

    metadata = {
        'timestamp': timestamp,
        'model': args.model,
        'translate': args.translate,
        'config': str(config_path),
        'samples': args.samples,
        'data_dir': str(artifact_dir),
        'artifact_dir': str(artifact_dir),
        'report_dir': str(report_dir),
        'artifact_paths': artifact_paths,
        'copied_artifact_paths': copied_artifact_paths,
        'copied_report_paths': copied_report_paths,
        'report_manifest': report_manifest,
        'analysis_json': str(analysis_json.resolve()),
        'analysis_csv': str(analysis_csv.resolve()),
    }
    write_summary(run_dir, report, metadata)

    print(f'collected_results_dir={run_dir}')
    print(f'analysis_json={analysis_json}')
    print(f'analysis_csv={analysis_csv}')


if __name__ == '__main__':
    main()
