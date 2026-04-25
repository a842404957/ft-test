#!/usr/bin/env python3

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def latest_file(directory: Path, pattern: str):
    candidates = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime)
    return candidates[-1] if candidates else None


def infer_seed(path: Path, report: dict):
    text = str(path)
    match = re.search(r'seed[_-]?(\d+)', text)
    if match:
        return int(match.group(1))
    config = report.get('config', {})
    try:
        return int(config.get('fault_injection', {}).get('random_seed'))
    except Exception:
        return ''


def load_report(report_json: Path):
    payload = json.loads(report_json.read_text(encoding='utf-8'))
    metrics = payload.get('metrics', payload)
    return payload, metrics


def discover_reports(evidence_root: Path, report_dirs_text: str):
    if report_dirs_text:
        dirs = [Path(item).resolve() for item in report_dirs_text.split(',') if item.strip()]
    else:
        dirs = sorted({path.parent for path in Path(evidence_root).resolve().rglob('fault_tolerance_report_*.json')})
    reports = []
    for directory in dirs:
        report_json = latest_file(directory, 'fault_tolerance_report_*.json')
        if report_json:
            reports.append(report_json)
    return reports


def flatten_reports(report_paths):
    rows = []
    for report_path in report_paths:
        payload, metrics = load_report(report_path)
        seed = infer_seed(report_path, payload)
        rel = metrics.get('reliability', {})
        layer_faults = rel.get('layer_wise_faults', {})
        layer_corrections = rel.get('layer_wise_corrections', {})
        layer_repair_quality = rel.get('layer_repair_quality', {})
        fault_detail_layers = rel.get('fault_detail_stats', {}).get('layer_details', {})

        all_layers = sorted(set(layer_faults) | set(layer_corrections) | set(layer_repair_quality) | set(fault_detail_layers))
        for layer in all_layers:
            faults = int(layer_faults.get(layer, 0) or 0)
            corrections = int(layer_corrections.get(layer, 0) or 0)
            level_quality = layer_repair_quality.get(layer, {}).get('level1') or {}
            detail = fault_detail_layers.get(layer, {})
            fault_model_counts = detail.get('fault_model_counts', {})
            avg_after = float(level_quality.get('avg_after_error', 0.0) or 0.0)
            attempted = int(level_quality.get('attempted', 0) or 0)
            rows.append({
                'seed': seed,
                'report_json': str(report_path),
                'layer': layer,
                'fault_count': faults,
                'level1_corrected_count': corrections,
                'uncorrected_count': max(faults - corrections, 0),
                'repair_attempted': attempted,
                'repair_improved_rate': level_quality.get('improved_rate', ''),
                'avg_before_error': level_quality.get('avg_before_error', ''),
                'avg_after_error': level_quality.get('avg_after_error', ''),
                'residual_error_score': avg_after * attempted,
                'affected_weight_count': detail.get('affected_weight_count', ''),
                'stuck_at_zero_count': fault_model_counts.get('stuck_at_zero', ''),
                'stuck_at_one_count': fault_model_counts.get('stuck_at_one', ''),
                'diagnostic_note': '' if layer_repair_quality else 'layer_repair_quality_missing_rerun_required',
            })
    return rows


def write_outputs(rows, output_dir: Path, focus_seed: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'seed', 'layer', 'fault_count', 'level1_corrected_count', 'uncorrected_count',
        'repair_attempted', 'repair_improved_rate', 'avg_before_error', 'avg_after_error',
        'residual_error_score', 'affected_weight_count', 'stuck_at_zero_count',
        'stuck_at_one_count', 'diagnostic_note', 'report_json',
    ]
    csv_path = output_dir / f'seed{focus_seed}_diagnosis.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, '') for name in fieldnames})

    focus_rows = [row for row in rows if str(row.get('seed')) == str(focus_seed)]
    top_residual = sorted(focus_rows, key=lambda row: float(row.get('residual_error_score') or 0.0), reverse=True)[:10]
    top_uncorrected = sorted(focus_rows, key=lambda row: int(row.get('uncorrected_count') or 0), reverse=True)[:10]
    summary = {
        'focus_seed': focus_seed,
        'row_count': len(rows),
        'top_residual_layers': top_residual,
        'top_uncorrected_layers': top_uncorrected,
    }
    json_path = output_dir / f'seed{focus_seed}_diagnosis.json'
    json_path.write_text(json.dumps({'rows': rows, 'summary': summary}, indent=2, ensure_ascii=False), encoding='utf-8')

    md_path = output_dir / f'seed{focus_seed}_diagnosis.md'
    lines = [
        f'# Seed {focus_seed} Fault Diagnosis',
        '',
        '## Top Residual Layers',
        '',
        '| layer | faults | corrected | uncorrected | improved rate | avg after | residual score | note |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in top_residual:
        improved = row.get('repair_improved_rate', '')
        if isinstance(improved, float):
            improved = f'{improved:.2%}'
        lines.append(
            f"| {row['layer']} | {row['fault_count']} | {row['level1_corrected_count']} | "
            f"{row['uncorrected_count']} | {improved} | {row.get('avg_after_error', '')} | "
            f"{float(row.get('residual_error_score') or 0.0):.6f} | {row.get('diagnostic_note', '')} |"
        )
    lines.extend([
        '',
        '## Top Uncorrected Layers',
        '',
        '| layer | faults | corrected | uncorrected | note |',
        '| --- | ---: | ---: | ---: | --- |',
    ])
    for row in top_uncorrected:
        lines.append(
            f"| {row['layer']} | {row['fault_count']} | {row['level1_corrected_count']} | "
            f"{row['uncorrected_count']} | {row.get('diagnostic_note', '')} |"
        )
    md_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'csv': str(csv_path), 'json': str(json_path), 'md': str(md_path), 'summary': summary}


def analyse_fault_seed_failure(args):
    report_paths = discover_reports(Path(args.evidence_root), args.report_dirs)
    rows = flatten_reports(report_paths)
    paths = write_outputs(rows, Path(args.output_dir).resolve(), args.focus_seed)
    print(f"seed_diagnosis_csv={paths['csv']}")
    print(f"seed_diagnosis_json={paths['json']}")
    print(f"seed_diagnosis_md={paths['md']}")
    return {'rows': rows, **paths}


def build_parser():
    parser = argparse.ArgumentParser(description='Analyse per-layer differences for bad fault seeds.')
    parser.add_argument('--evidence-root', default='.', help='Root containing simulation report directories')
    parser.add_argument('--report-dirs', default='', help='Comma-separated report directories; overrides evidence-root scan')
    parser.add_argument('--focus-seed', type=int, default=43)
    parser.add_argument('--output-dir', required=True)
    return parser


def main():
    args = build_parser().parse_args()
    analyse_fault_seed_failure(args)


if __name__ == '__main__':
    main()
