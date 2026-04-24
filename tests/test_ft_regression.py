import csv
import json
import os
import pickle
import tempfile
import unittest
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from cut import (
    _compile_ft_regularization_state,
    _projection_strength_for_epoch,
    _resolve_budgeted_layer_config,
    _sample_regularization_state_links,
    apply_ft_group_projection,
    assign_ou_to_mask_codebook,
    build_redundancy_mask_codebook,
    budgeted_assign_members,
    budgeted_select_prototypes,
    extract_ou_patterns,
    ft_codebook_budgeted_select_mask_candidate,
    ft_codebook_budgeted_translate,
    ft_budgeted_select_mask_candidate,
    ft_budgeted_group_translate,
    ft_group_score_mask,
    ft_group_translate_train,
)
from fault_tolerance_analyse import analyse
from main import build_budgeted_grouping_config, prepare_ft_artifacts, resolve_ft_training_config, save_ft_build_only_projection
from run_hierarchical_fault_tolerance import _build_runtime_config, resolve_runtime_model_name, resolve_levels_overrides
from scripts.analyse_redundancy_construction import build_redundancy_construction_report
from simulator.Fault_Tolerance.fault_tolerance_simulation import FaultToleranceSimulator
from simulator.Fault_Tolerance.fault_tolerance_simulation import resolve_excluded_critical_layers
from simulator.Fault_Tolerance.config import FaultToleranceConfig
from simulator.Fault_Tolerance.pattern_data_loader import PatternDataLoader
from simulator.Fault_Tolerance.redundancy_group_parser import RedundancyGroupParser


class TinyConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, kernel_size=1, bias=False)


class TinyFCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 8, bias=False)


class TinyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 2, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x.mean(dim=(2, 3))


def _write_pkl(path: Path, payload):
    with open(path, 'wb') as handle:
        pickle.dump(payload, handle, pickle.HIGHEST_PROTOCOL)


def _build_ft_group_artifacts(temp_dir: Path, model_name: str = 'Toy'):
    mask = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
    group_information = {
        'conv.weight': {
            'layer_name': 'conv.weight',
            'group_count': 1,
            'block_count': 4,
            'ou_count': 8,
            'repairable_block_count': 2,
            'repairable_ou_count': 4,
            'coverage_ratio': 0.5,
            'block_coverage_ratio': 0.5,
            'singleton_group_count': 0,
            'singleton_ratio': 0.0,
            'avg_group_size': 2.0,
            'exact_group_count': 0,
            'scaled_group_count': 1,
            'exact_group_ratio': 0.0,
            'scaled_group_ratio': 1.0,
            'groups': [
                {
                    'group_id': 0,
                    'member_count': 2,
                    'group_size': 2,
                    'covered_ou_count': 4,
                    'repair_mode': 'scaled',
                    'prototype': {
                        'out_ch': 0,
                        'in_ch_start': 0,
                        'channel_span': 2,
                        'multiplier': 1.0,
                        'role': 'prototype',
                        'mask_signature': 'shared',
                    },
                    'members': [
                        {
                            'out_ch': 0,
                            'in_ch_start': 0,
                            'channel_span': 2,
                            'multiplier': 1.0,
                            'role': 'prototype',
                            'mask_signature': 'shared',
                            'similarity': 1.0,
                        },
                        {
                            'out_ch': 1,
                            'in_ch_start': 0,
                            'channel_span': 2,
                            'multiplier': 2.0,
                            'role': 'member',
                            'mask_signature': 'shared',
                            'similarity': 1.0,
                        },
                    ],
                }
            ],
        }
    }
    reuse_ratio = {'conv.weight': 0.5}

    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_group_information.pkl', group_information)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_mask.pkl', mask)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_coverage_ratio_information.pkl', reuse_ratio)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_reuse_ratio_information.pkl', reuse_ratio)


def _build_prap_fallback_artifacts(temp_dir: Path, model_name: str = 'ToyOld'):
    map_information = {'fc.weight': torch.full((2, 2, 2), -1, dtype=torch.int64)}
    map_information['fc.weight'][0, 0] = torch.tensor([1, 0])
    multiple_relationship = {'fc.weight': torch.ones((2, 2), dtype=torch.float32)}
    reuse_ratio = {'fc.weight': 0.5}
    mask = {'fc.weight': torch.ones((2, 2), dtype=torch.float32)}

    _write_pkl(temp_dir / f'model_{model_name}_shape_and_value_similar_map_information.pkl', map_information)
    _write_pkl(temp_dir / f'model_{model_name}_shape_and_value_multiple_relationship_information.pkl', multiple_relationship)
    _write_pkl(temp_dir / f'model_{model_name}_shape_and_value_reuse_ratio_information.pkl', reuse_ratio)
    _write_pkl(temp_dir / f'model_{model_name}_pattern_mask.pkl', mask)


def _build_zero_scale_ft_artifacts(temp_dir: Path, model_name: str = 'ToyZero'):
    mask = {'conv.weight': torch.ones((2, 2, 1, 1), dtype=torch.float32)}
    group_information = {
        'conv.weight': {
            'layer_name': 'conv.weight',
            'group_count': 1,
            'block_count': 2,
            'ou_count': 4,
            'repairable_block_count': 2,
            'repairable_ou_count': 4,
            'coverage_ratio': 1.0,
            'block_coverage_ratio': 1.0,
            'singleton_group_count': 0,
            'singleton_ratio': 0.0,
            'avg_group_size': 2.0,
            'exact_group_count': 0,
            'scaled_group_count': 1,
            'exact_group_ratio': 0.0,
            'scaled_group_ratio': 1.0,
            'groups': [
                {
                    'group_id': 0,
                    'member_count': 2,
                    'group_size': 2,
                    'covered_ou_count': 2,
                    'repair_mode': 'scaled',
                    'prototype': {
                        'out_ch': 0,
                        'in_ch_start': 0,
                        'channel_span': 1,
                        'multiplier': 1.0,
                        'role': 'prototype',
                        'mask_signature': 'shared',
                    },
                    'members': [
                        {
                            'out_ch': 0,
                            'in_ch_start': 0,
                            'channel_span': 1,
                            'multiplier': 1.0,
                            'role': 'prototype',
                            'mask_signature': 'shared',
                            'similarity': 1.0,
                        },
                        {
                            'out_ch': 1,
                            'in_ch_start': 0,
                            'channel_span': 1,
                            'multiplier': 0.0,
                            'role': 'member',
                            'mask_signature': 'shared',
                            'similarity': 1.0,
                        },
                    ],
                }
            ],
        }
    }
    coverage_ratio = {'conv.weight': 1.0}

    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_group_information.pkl', group_information)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_mask.pkl', mask)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_coverage_ratio_information.pkl', coverage_ratio)
    _write_pkl(temp_dir / f'model_{model_name}_ft_group_cluster_translate_reuse_ratio_information.pkl', coverage_ratio)


def _build_codebook_group_artifacts(temp_dir: Path, model_name: str = 'ToyCodebookTrain'):
    mask = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
    group_information = {
        'conv.weight': {
            'layer_name': 'conv.weight',
            'group_count': 1,
            'block_count': 4,
            'ou_count': 8,
            'repairable_block_count': 2,
            'repairable_ou_count': 4,
            'coverage_ratio': 0.5,
            'block_coverage_ratio': 0.5,
            'singleton_group_count': 0,
            'singleton_ratio': 0.0,
            'avg_group_size': 2.0,
            'exact_group_count': 0,
            'scaled_group_count': 1,
            'exact_group_ratio': 0.0,
            'scaled_group_ratio': 1.0,
            'grouping_mode': 'codebook_budgeted',
            'projection_cap': 0.1,
            'projection_lambda': 0.5,
            'mask_codebook_size': 2,
            'mask_keep_count_distribution': {'2': 2},
            'dominant_mask_keep_count': 2,
            'prototype_space': 'normalized_direction',
            'assignment_error_mean': 0.05,
            'assignment_error_p50': 0.05,
            'assignment_error_p95': 0.08,
            'groups': [
                {
                    'group_id': 0,
                    'member_count': 2,
                    'group_size': 2,
                    'covered_ou_count': 4,
                    'repair_mode': 'scaled',
                    'prototype': {
                        'out_ch': 0,
                        'in_ch_start': 0,
                        'channel_span': 2,
                        'multiplier': 1.0,
                        'role': 'prototype',
                        'mask_signature': 'shared',
                    },
                    'members': [
                        {
                            'out_ch': 0,
                            'in_ch_start': 0,
                            'channel_span': 2,
                            'multiplier': 1.0,
                            'role': 'prototype',
                            'mask_signature': 'shared',
                            'similarity': 1.0,
                        },
                        {
                            'out_ch': 1,
                            'in_ch_start': 0,
                            'channel_span': 2,
                            'multiplier': 1.0,
                            'role': 'member',
                            'mask_signature': 'shared',
                            'similarity': 0.99,
                        },
                    ],
                }
            ],
        }
    }
    coverage_ratio = {'conv.weight': 0.5}
    _write_pkl(temp_dir / f'model_{model_name}_ft_codebook_budgeted_translate_group_information.pkl', group_information)
    _write_pkl(temp_dir / f'model_{model_name}_ft_codebook_budgeted_translate_mask.pkl', mask)
    _write_pkl(temp_dir / f'model_{model_name}_ft_codebook_budgeted_translate_coverage_ratio_information.pkl', coverage_ratio)
    _write_pkl(temp_dir / f'model_{model_name}_ft_codebook_budgeted_translate_reuse_ratio_information.pkl', coverage_ratio)


def _make_budget_pattern(out_ch: int, values, mask_signature: bytes = b'shared'):
    masked = torch.tensor(values, dtype=torch.float32)
    return {
        'out_ch': out_ch,
        'in_ch_start': 0,
        'channel_span': masked.numel(),
        'raw': masked.clone(),
        'masked': masked,
        'norm1': float(masked.abs().sum().item()),
        'mask_signature': mask_signature,
        'mask_nonzero_count': int(masked.numel()),
        'mask_channel_counts': (int(masked.numel()),),
    }


@contextmanager
def _working_directory(path: Path):
    previous_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous_cwd)


def _read_csv_rows(path: Path):
    with open(path, 'r', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def _run_tiny_ft_train(temp_dir: Path, model_name: str = 'ToyTrain', translate_name: str = 'ft_group_cluster_translate',
                       ft_reg_interval: int = 2, group_refresh_epoch=None, translate_epoch=None,
                       ft_reg_min_coverage: float = 0.0, ft_reg_min_groups: int = 1,
                       ft_reg_boost_after_refresh: bool = False, output_dir: str = '.',
                       ft_grouping_mode: str = 'ftscore',
                       ft_projection_ramp_start: float = 0.0,
                       ft_projection_ramp_end: float = 0.0,
                       ft_projection_ramp_epochs=None,
                       ft_projection_loss_lambda: float = 0.0,
                       ft_projection_reg_max_links: int = 0,
                       ft_codebook_freeze_grouping: bool = False,
                       ft_codebook_use_legacy_regularization: bool = True):
    if group_refresh_epoch is None:
        group_refresh_epoch = [2]
    if translate_epoch is None:
        translate_epoch = [3]

    model = TinyTrainModel()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_schedule': scheduler.state_dict(),
        'epoch': 0,
    }

    torch.save(checkpoint, temp_dir / f'model_{model_name}_original_parameter_epoch0_ckpt.pth')
    if translate_name == 'ft_codebook_budgeted_translate':
        _build_codebook_group_artifacts(temp_dir, model_name=model_name)
    else:
        _build_ft_group_artifacts(temp_dir, model_name=model_name)

    inputs = torch.randn(8, 4, 4, 4)
    targets = torch.randint(0, 2, (8,))
    train_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)
    test_loader = DataLoader(TensorDataset(inputs, targets), batch_size=2, shuffle=False)

    with open(temp_dir / f'model_{model_name}_{translate_name}_group_information.pkl', 'rb') as handle:
        group_information = pickle.load(handle)
    with open(temp_dir / f'model_{model_name}_{translate_name}_mask.pkl', 'rb') as handle:
        mask = pickle.load(handle)

    map_information = {'conv.weight': torch.full((4, 2, 2), -1, dtype=torch.int64)}
    multiple_relationship_information = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
    reuse_ratio_information = {'conv.weight': 0.5}

    with _working_directory(temp_dir):
        ft_group_translate_train(
            model=model,
            model_name=model_name,
            translate_name=translate_name,
            weight_name=['conv.weight'],
            in_channel=[4],
            out_channel=[2],
            kernel_size=[1],
            channel_number=[2],
            ft_layer_enabled=[True],
            mask=mask,
            group_information=group_information,
            map_information=map_information,
            multiple_relationship_information=multiple_relationship_information,
            reuse_ratio_information=reuse_ratio_information,
            ft_mask_lambda=1e-3,
            ft_proto_lambda=1e-3,
            ft_balance_lambda=1e-4,
            ft_sep_lambda=5e-5,
            device='cpu',
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            test_loader=test_loader,
            max_epoches=3,
            translate_epoch=translate_epoch,
            checkpoint_epoch=0,
            group_refresh_epoch=group_refresh_epoch,
            min_group_size=2,
            target_group_size=[2],
            sim_threshold=[0.85],
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            pattern_value_number=[1],
            pattern_shape_number=8,
            OU_size=2,
            ft_reg_interval=ft_reg_interval,
            ft_reg_min_coverage=ft_reg_min_coverage,
            ft_reg_min_groups=ft_reg_min_groups,
            ft_reg_boost_after_refresh=ft_reg_boost_after_refresh,
            ft_grouping_mode=ft_grouping_mode,
            ft_projection_ramp_start=ft_projection_ramp_start,
            ft_projection_ramp_end=ft_projection_ramp_end,
            ft_projection_ramp_epochs=ft_projection_ramp_epochs or translate_epoch,
            ft_projection_loss_lambda=ft_projection_loss_lambda,
            ft_projection_reg_max_links=ft_projection_reg_max_links,
            ft_codebook_freeze_grouping=ft_codebook_freeze_grouping,
            ft_codebook_use_legacy_regularization=ft_codebook_use_legacy_regularization,
            output_dir=output_dir,
        )


class TestFTRegression(unittest.TestCase):
    def test_extract_ou_patterns_respects_block_span(self):
        model = TinyConvModel()
        with torch.no_grad():
            model.conv.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1))

        patterns = extract_ou_patterns(
            model=model,
            weight_name='conv.weight',
            in_channel=4,
            out_channel=2,
            kernel_size=1,
            channel_number=2,
            mask=torch.ones_like(model.conv.weight),
        )

        self.assertEqual(len(patterns), 4)
        self.assertEqual({entry['channel_span'] for entry in patterns}, {2})
        self.assertEqual({entry['in_ch_start'] for entry in patterns}, {0, 2})
        self.assertEqual({entry['block_kind'] for entry in patterns}, {'conv1x1'})
        self.assertTrue(all(isinstance(entry['mask_signature'], bytes) for entry in patterns))

    def test_ft_score_mask_deduplicates_dense_fc_candidates(self):
        model = TinyFCModel()
        with torch.no_grad():
            model.fc.weight.copy_(torch.arange(64, dtype=torch.float32).reshape(8, 8))

        _, seed_info = ft_group_score_mask(
            model=model,
            weight_name='fc.weight',
            in_channel=8,
            out_channel=8,
            kernel_size=1,
            channel_number=8,
            pattern_value_number=1,
            pattern_shape_number=8,
            OU_size=8,
            target_group_size=4,
            sim_threshold=0.85,
        )

        self.assertLess(seed_info['candidate_count'], 6)

    def test_compile_ft_regularization_state_skips_low_value_layers(self):
        group_information = {
            'high.weight': {
                'coverage_ratio': 0.5,
                'groups': [
                    {
                        'group_size': 2,
                        'prototype': {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0},
                        'members': [
                            {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'prototype'},
                            {'out_ch': 1, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 2.0, 'role': 'member'},
                        ],
                    }
                ],
            },
            'low.weight': {
                'coverage_ratio': 0.01,
                'groups': [
                    {
                        'group_size': 2,
                        'prototype': {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0},
                        'members': [
                            {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'prototype'},
                            {'out_ch': 1, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'member'},
                        ],
                    }
                ],
            },
        }
        state = _compile_ft_regularization_state(
            weight_name=['high.weight', 'low.weight'],
            ft_layer_enabled=[True, True],
            group_information=group_information,
            min_coverage=0.1,
            min_repairable_groups=1,
        )

        self.assertIn('high.weight', state['layers'])
        self.assertNotIn('low.weight', state['layers'])
        self.assertEqual(state['summary']['layer_count'], 1)
        self.assertEqual(state['summary']['skipped_low_coverage_layers'], 1)

    def test_cost_preset_resolution_and_override(self):
        args = Namespace(
            ft_cost_preset='balanced',
            ft_low_cost=False,
            ft_end_epoch=None,
            ft_reg_interval=None,
            ft_reg_min_coverage=None,
            ft_reg_min_groups=None,
            ft_reg_boost_after_refresh=False,
            base_checkpoint_epoch=150,
            translate_epochs='200',
            refresh_epochs='190,200',
        )
        config = resolve_ft_training_config(args, argv=['--ft-cost-preset', 'balanced'])
        self.assertEqual(config['selected_preset'], 'balanced')
        self.assertEqual(config['ft_end_epoch'], 180)
        self.assertEqual(config['ft_reg_interval'], 5)
        self.assertEqual(config['ft_group_refresh_epoch'], [160, 170, 180])

        args.ft_cost_preset = 'none'
        args.ft_low_cost = True
        args.ft_reg_interval = 7
        config = resolve_ft_training_config(args, argv=['--ft-low-cost', '--ft-reg-interval', '7'])
        self.assertEqual(config['selected_preset'], 'fast')
        self.assertEqual(config['ft_reg_interval'], 7)
        self.assertEqual(config['ft_end_epoch'], 160)

        args.ft_cost_preset = 'fast'
        args.ft_low_cost = False
        args.ft_reg_interval = None
        args.refresh_epochs = '158'
        config = resolve_ft_training_config(args, argv=['--ft-cost-preset', 'fast', '--refresh-epochs', '158'])
        self.assertEqual(config['ft_group_refresh_epoch'], [158])

    def test_runtime_model_name_prefers_cli_over_config(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {
                    'name': 'Vgg16',
                    'num_classes': 10,
                }
            }), encoding='utf-8')
            model_name, num_classes = resolve_runtime_model_name('Res18', str(config_path))
            self.assertEqual(model_name, 'Res18')
            self.assertEqual(num_classes, 10)

    def test_runtime_config_uses_runtime_model_name(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {
                    'name': 'Vgg16',
                    'num_classes': 10,
                }
            }), encoding='utf-8')
            runtime_config, _ = _build_runtime_config(
                base_config_file=str(config_path),
                runtime_model_name='Res18',
            )
            self.assertEqual(runtime_config['model']['name'], 'Res18')

    def test_resolve_excluded_critical_layers_uses_first_and_last(self):
        layer_names = ['conv1.weight', 'layer1.0.conv1.weight', 'fc.weight']
        resolved = resolve_excluded_critical_layers(layer_names, ['__first__', '__last__'])
        self.assertEqual(resolved, ['conv1.weight', 'fc.weight'])

    def test_training_profile_and_regularization_reports_are_written(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _run_tiny_ft_train(temp_dir, ft_reg_interval=2, group_refresh_epoch=[2], translate_epoch=[3])

            profile_path = temp_dir / 'model_ToyTrain_ft_group_cluster_translate_training_profile.csv'
            report_path = temp_dir / 'model_ToyTrain_ft_group_cluster_translate_regularization_layers.csv'
            self.assertTrue(profile_path.exists())
            self.assertTrue(report_path.exists())

            profile_rows = _read_csv_rows(profile_path)
            report_rows = _read_csv_rows(report_path)
            self.assertGreaterEqual(len(profile_rows), 3)
            train_rows = [row for row in profile_rows if int(row['batch_count']) > 0]
            self.assertTrue(train_rows)
            self.assertIn('reg_batch_count', train_rows[0])
            self.assertIn('epoch_time_sec', train_rows[0])
            self.assertIn('projection_time_sec', train_rows[-1])
            self.assertIn('mask_loss_reg_batch_avg', train_rows[0])
            self.assertIn('proto_loss_reg_batch_avg', train_rows[0])
            self.assertIn('sep_loss_reg_batch_avg', train_rows[0])
            self.assertLess(int(train_rows[0]['reg_batch_count']), int(train_rows[0]['batch_count']))

            self.assertEqual(len(report_rows), 1)
            self.assertEqual(report_rows[0]['layer'], 'conv.weight')
            self.assertIn('skip_reason', report_rows[0])
            self.assertEqual(report_rows[0]['reg_enabled'], '1')
            self.assertEqual(report_rows[0]['mask_reg_enabled'], '1')
            self.assertEqual(report_rows[0]['group_reg_enabled'], '1')

    def test_custom_output_dir_writes_artifacts_and_reports(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_dir = temp_dir / 'artifacts'
            _run_tiny_ft_train(temp_dir, output_dir=str(artifact_dir))

            self.assertTrue((artifact_dir / 'model_ToyTrain_ft_group_cluster_translate_training_profile.csv').exists())
            self.assertTrue((artifact_dir / 'model_ToyTrain_ft_group_cluster_translate_regularization_layers.csv').exists())
            self.assertTrue((artifact_dir / 'model_ToyTrain_ft_group_cluster_translate_after_translate_parameters.pth').exists())

    def test_layer_report_distinguishes_mask_and_group_regularization(self):
        group_information = {
            'high.weight': {
                'coverage_ratio': 0.5,
                'groups': [
                    {
                        'group_size': 2,
                        'prototype': {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0},
                        'members': [
                            {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'prototype'},
                            {'out_ch': 1, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 2.0, 'role': 'member'},
                        ],
                    }
                ],
            },
            'low.weight': {
                'coverage_ratio': 0.01,
                'groups': [
                    {
                        'group_size': 2,
                        'prototype': {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0},
                        'members': [
                            {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'prototype'},
                            {'out_ch': 1, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0, 'role': 'member'},
                        ],
                    }
                ],
            },
        }
        state = _compile_ft_regularization_state(
            weight_name=['high.weight', 'low.weight'],
            ft_layer_enabled=[True, True],
            group_information=group_information,
            min_coverage=0.1,
            min_repairable_groups=1,
        )
        rows = {row['layer']: row for row in state['layer_rows']}
        self.assertEqual(rows['high.weight']['mask_reg_enabled'], 1)
        self.assertEqual(rows['high.weight']['group_reg_enabled'], 1)
        self.assertEqual(rows['low.weight']['mask_reg_enabled'], 1)
        self.assertEqual(rows['low.weight']['group_reg_enabled'], 0)
        self.assertEqual(rows['low.weight']['skip_reason'], 'low_coverage')

    def test_refresh_rebuilds_regularization_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            with mock.patch('cut._refresh_acceptance_score', return_value=0.0), \
                    mock.patch('cut._compile_ft_regularization_state', wraps=_compile_ft_regularization_state) as compile_mock:
                _run_tiny_ft_train(temp_dir, ft_reg_interval=2, group_refresh_epoch=[2], translate_epoch=[3])
            self.assertGreaterEqual(compile_mock.call_count, 2)

    def test_reg_boost_after_refresh_increases_reg_batch_count(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _run_tiny_ft_train(
                temp_dir,
                ft_reg_interval=4,
                group_refresh_epoch=[2],
                translate_epoch=[3],
                ft_reg_boost_after_refresh=True,
            )
            profile_rows = _read_csv_rows(temp_dir / 'model_ToyTrain_ft_group_cluster_translate_training_profile.csv')
            by_epoch = {int(row['epoch']): row for row in profile_rows}
            self.assertEqual(by_epoch[2]['effective_reg_interval'], '2')
            self.assertEqual(by_epoch[3]['effective_reg_interval'], '2')

    def test_group_information_parser_and_level1_scale_are_block_aware(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='Toy')

            loader = PatternDataLoader(model_name='Toy', translate_name='ft_group_cluster_translate', data_dir=str(temp_dir))
            self.assertTrue(loader.load_all_data())

            parser = RedundancyGroupParser(loader)
            self.assertTrue(parser.parse_all_layers())
            groups = parser.get_layer_groups('conv.weight')
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].block_members[0]['channel_span'], 2)
            self.assertEqual(groups[0].ou_to_block[(1, 1)]['offset'], 1)

            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {'name': 'Toy', 'num_classes': 10},
                'simulation': {'device': 'cpu', 'verbose': False, 'batch_size': 1},
                'hierarchical_fault_tolerance': {
                    'enabled': True,
                    'level1': {'enabled': True},
                    'level2': {'enabled': False},
                    'level3': {'enabled': False},
                },
                'report': {'output_dir': str(temp_dir / 'reports')},
            }), encoding='utf-8')

            model = TinyConvModel()
            with torch.no_grad():
                model.conv.weight.zero_()
                model.conv.weight[0, 0, 0, 0] = 1.0
                model.conv.weight[0, 1, 0, 0] = 2.0
                model.conv.weight[1, 0, 0, 0] = 2.0
                model.conv.weight[1, 1, 0, 0] = 4.0

            simulator = FaultToleranceSimulator(
                model=model,
                model_name='Toy',
                translate_name='ft_group_cluster_translate',
                config_file=str(config_path),
                data_dir=str(temp_dir),
            )
            original_weights = simulator._save_model_weights()

            with torch.no_grad():
                simulator.model.conv.weight[1, 1, 0, 0] = -123.0

            stats = simulator._apply_weight_level_correction({'conv.weight': [(1, 1)]}, original_weights)
            corrected_value = simulator.model.conv.weight[1, 1, 0, 0].item()

            self.assertEqual(stats['level1_count'], 1)
            self.assertEqual(stats['level1_scaled_repairs'], 1)
            self.assertAlmostEqual(
                simulator.compute_member_to_member_scale(alpha_i=2.0, alpha_j=1.0),
                2.0,
                places=6,
            )
            self.assertAlmostEqual(corrected_value, 4.0, places=6)

    def test_fault_tolerance_analyse_returns_expected_structure(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='Toy')

            report = analyse('Toy', 'ft_group_cluster_translate', str(temp_dir))
            self.assertIn('global', report)
            self.assertIn('layers', report)
            self.assertEqual(report['translate_name'], 'ft_group_cluster_translate')
            self.assertGreaterEqual(report['global']['group_coverage_ratio'], 0.0)
            self.assertEqual(report['layers'][0]['data_source'], 'group_information')

    def test_coverage_ratio_file_is_preferred_but_reuse_alias_still_works(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='ToyCoverage')

            loader = PatternDataLoader(model_name='ToyCoverage', translate_name='ft_group_cluster_translate', data_dir=str(temp_dir))
            self.assertTrue(loader.load_all_data())
            self.assertIsNotNone(loader.coverage_ratio_information)
            self.assertAlmostEqual(loader.get_layer_coverage_ratio('conv.weight'), 0.5, places=6)
            self.assertAlmostEqual(loader.get_layer_reuse_ratio('conv.weight'), 0.5, places=6)

    def test_prap_fallback_still_loads_and_parses(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_prap_fallback_artifacts(temp_dir, model_name='ToyOld')

            loader = PatternDataLoader(model_name='ToyOld', translate_name='ft_group_cluster_translate', data_dir=str(temp_dir))
            self.assertTrue(loader.load_all_data())
            self.assertIsNone(loader.group_information)
            self.assertIsNotNone(loader.get_layer_map('fc.weight'))

            parser = RedundancyGroupParser(loader)
            self.assertTrue(parser.parse_all_layers())
            groups = parser.get_layer_groups('fc.weight')
            self.assertEqual(len(groups), 1)
            self.assertEqual(groups[0].size(), 2)

    def test_oracle_restore_restores_weights(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='ToyOracle')

            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {'name': 'ToyOracle', 'num_classes': 10},
                'simulation': {'device': 'cpu', 'verbose': False, 'batch_size': 1},
                'hierarchical_fault_tolerance': {
                    'enabled': True,
                    'repair_mode': 'oracle',
                    'level1': {'enabled': True},
                    'level2': {'enabled': False},
                    'level3': {'enabled': False},
                },
                'report': {'output_dir': str(temp_dir / 'reports')},
            }), encoding='utf-8')

            model = TinyConvModel()
            with torch.no_grad():
                model.conv.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1))

            simulator = FaultToleranceSimulator(
                model=model,
                model_name='ToyOracle',
                translate_name='ft_group_cluster_translate',
                config_file=str(config_path),
                data_dir=str(temp_dir),
            )
            original_weights = simulator._save_model_weights()
            with torch.no_grad():
                simulator.model.conv.weight[1, 1, 0, 0] = -123.0

            stats = simulator._apply_weight_level_correction({'conv.weight': [(1, 1)]}, original_weights)
            self.assertEqual(stats['repair_mode'], 'oracle')
            self.assertEqual(stats['oracle_count'], 1)
            self.assertAlmostEqual(simulator.model.conv.weight[1, 1, 0, 0].item(), original_weights['conv.weight'][1, 1, 0, 0].item(), places=6)
            self.assertEqual(stats['repair_quality']['oracle']['exact_restored'], 1)

    def test_zero_scale_level1_is_not_counted_as_success(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_zero_scale_ft_artifacts(temp_dir, model_name='ToyZero')

            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {'name': 'ToyZero', 'num_classes': 10},
                'simulation': {'device': 'cpu', 'verbose': False, 'batch_size': 1},
                'hierarchical_fault_tolerance': {
                    'enabled': True,
                    'repair_mode': 'normal',
                    'level1': {'enabled': True},
                    'level2': {'enabled': False},
                    'level3': {'enabled': False},
                },
                'report': {'output_dir': str(temp_dir / 'reports')},
            }), encoding='utf-8')

            model = TinyConvModel()
            with torch.no_grad():
                model.conv.weight.zero_()
                model.conv.weight[0, 0, 0, 0] = 2.0
                model.conv.weight[1, 0, 0, 0] = -99.0

            simulator = FaultToleranceSimulator(
                model=model,
                model_name='ToyZero',
                translate_name='ft_group_cluster_translate',
                config_file=str(config_path),
                data_dir=str(temp_dir),
            )
            original_weights = simulator._save_model_weights()

            stats = simulator._apply_weight_level_correction({'conv.weight': [(1, 0)]}, original_weights)
            self.assertEqual(stats['level1_count'], 0)
            self.assertEqual(stats['level1_zero_scale_failed'], 1)
            self.assertEqual(stats['repair_quality']['level1']['attempted'], 1)
            self.assertEqual(stats['repair_quality']['level1']['effective_improved'], 0)

    def test_repair_quality_metrics_fields_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='ToyQuality')

            config_path = temp_dir / 'config.json'
            config_path.write_text(json.dumps({
                'model': {'name': 'ToyQuality', 'num_classes': 10},
                'simulation': {'device': 'cpu', 'verbose': False, 'batch_size': 1},
                'hierarchical_fault_tolerance': {
                    'enabled': True,
                    'repair_mode': 'normal',
                    'level1': {'enabled': True},
                    'level2': {'enabled': False},
                    'level3': {'enabled': True},
                },
                'report': {'output_dir': str(temp_dir / 'reports')},
            }), encoding='utf-8')

            model = TinyConvModel()
            with torch.no_grad():
                model.conv.weight.zero_()
                model.conv.weight[0, 0, 0, 0] = 1.0
                model.conv.weight[0, 1, 0, 0] = 2.0
                model.conv.weight[1, 0, 0, 0] = 2.0
                model.conv.weight[1, 1, 0, 0] = 4.0

            simulator = FaultToleranceSimulator(
                model=model,
                model_name='ToyQuality',
                translate_name='ft_group_cluster_translate',
                config_file=str(config_path),
                data_dir=str(temp_dir),
            )
            original_weights = simulator._save_model_weights()
            with torch.no_grad():
                simulator.model.conv.weight[1, 1, 0, 0] = -111.0
            stats = simulator._apply_weight_level_correction({'conv.weight': [(1, 1)]}, original_weights)
            level1_quality = stats['repair_quality']['level1']
            self.assertIn('attempted', level1_quality)
            self.assertIn('effective_improved', level1_quality)
            self.assertIn('exact_restored', level1_quality)
            self.assertIn('avg_before_error', level1_quality)
            self.assertIn('avg_after_error', level1_quality)
            self.assertIn('improved_rate', level1_quality)

    def test_stress_configs_load(self):
        stress3 = FaultToleranceConfig('fault_tolerance_config_stress_3pct.json')
        stress5 = FaultToleranceConfig('fault_tolerance_config_stress_5pct.json')
        late = FaultToleranceConfig('fault_tolerance_config_target_late_layers.json')
        self.assertAlmostEqual(stress3.get('fault_injection', 'fault_rate'), 0.03, places=6)
        self.assertAlmostEqual(stress5.get('fault_injection', 'fault_rate'), 0.05, places=6)
        self.assertEqual(stress3.get('fault_injection', 'bit_flip_ratio'), 1.0)
        self.assertIn('conv10.weight', late.get('fault_injection', 'target_layers'))

    def test_level1_only_mode_resolves(self):
        overrides = resolve_levels_overrides('level1')
        self.assertTrue(overrides['level1.enabled'])
        self.assertFalse(overrides['level2.enabled'])
        self.assertFalse(overrides['level3.enabled'])

    def test_redundancy_construction_analysis_script_on_synthetic_data(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _build_ft_group_artifacts(temp_dir, model_name='ToyDiag')
            report = build_redundancy_construction_report('ToyDiag', 'ft_group_cluster_translate', str(temp_dir))
            self.assertIn('global', report)
            self.assertTrue(report['layers'])
            layer = report['layers'][0]
            self.assertIn('selected_mask_strategy', layer)
            self.assertIn('mask_density', layer)
            self.assertIn('repairable_ou_ratio', layer)
            self.assertIn('candidate_summaries', layer)

    def test_sparse_mask_candidate_can_reduce_singleton_ratio(self):
        model = TinyFCModel()
        with torch.no_grad():
            weight = torch.tensor([
                [4.0, 4.0, 1.0, -1.0, 1.5, -1.5, 2.0, -2.0],
                [8.0, 8.0, -1.0, 1.0, -1.5, 1.5, -2.0, 2.0],
                [2.0, 2.0, 0.9, -0.9, 1.3, -1.3, 1.8, -1.8],
                [6.0, 6.0, -0.9, 0.9, -1.3, 1.3, -1.8, 1.8],
                [3.0, 3.0, 1.1, -1.1, 1.4, -1.4, 2.1, -2.1],
                [9.0, 9.0, -1.1, 1.1, -1.4, 1.4, -2.1, 2.1],
                [5.0, 5.0, 0.8, -0.8, 1.2, -1.2, 1.7, -1.7],
                [10.0, 10.0, -0.8, 0.8, -1.2, 1.2, -1.7, 1.7],
            ], dtype=torch.float32)
            model.fc.weight.copy_(weight)

        _, seed_info = ft_group_score_mask(
            model=model,
            weight_name='fc.weight',
            in_channel=8,
            out_channel=8,
            kernel_size=1,
            channel_number=8,
            pattern_value_number=1,
            pattern_shape_number=8,
            OU_size=8,
            target_group_size=4,
            sim_threshold=0.98,
            mask_density_sweep=True,
            mask_keep_ratios=[1.0, 0.25],
            prefer_sparser_mask=True,
            singleton_penalty=0.35,
            zero_scale_penalty=0.1,
        )
        candidates = {item['strategy']: item for item in seed_info['candidate_summaries']}
        dense_like = None
        sparse_like = None
        for item in candidates.values():
            if abs(item['mask_density'] - 1.0) <= 1e-6:
                dense_like = item
            if item['mask_density'] < 1.0:
                sparse_like = item if sparse_like is None else min(sparse_like, item, key=lambda row: row['estimated_singleton_ratio'])
        self.assertIsNotNone(dense_like)
        self.assertIsNotNone(sparse_like)
        self.assertLess(sparse_like['estimated_singleton_ratio'], dense_like['estimated_singleton_ratio'])

    def test_budgeted_prototype_selection_is_deterministic(self):
        patterns = [
            _make_budget_pattern(0, [1.0, 0.0]),
            _make_budget_pattern(1, [0.9, 0.1]),
            _make_budget_pattern(2, [0.0, 1.0]),
            _make_budget_pattern(3, [0.1, 0.9]),
        ]
        selected_first = budgeted_select_prototypes(patterns, prototype_budget=2)
        selected_second = budgeted_select_prototypes(patterns, prototype_budget=2)
        self.assertEqual(selected_first, selected_second)
        self.assertEqual(len(selected_first), 2)
        self.assertEqual(len(set(selected_first)), 2)

    def test_budgeted_assignment_respects_scale_candidates(self):
        patterns = [
            _make_budget_pattern(0, [1.0, 2.0]),
            _make_budget_pattern(1, [2.0, 4.0]),
            _make_budget_pattern(2, [0.5, 1.0]),
        ]
        groups = budgeted_assign_members(
            pattern_list=patterns,
            prototype_indices=[0],
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            max_scale_error=0.05,
            exact_threshold=0.999,
        )
        main_group = next(group for group in groups if group['prototype']['out_ch'] == 0)
        member_multipliers = {
            member['pattern']['out_ch']: member['multiplier']
            for member in main_group['members']
            if member['role'] != 'prototype'
        }
        self.assertAlmostEqual(member_multipliers[1], 2.0, places=6)
        self.assertAlmostEqual(member_multipliers[2], 0.5, places=6)

    def test_budgeted_grouping_reduces_singletons_on_synthetic_data(self):
        model = TinyFCModel()
        with torch.no_grad():
            weight = torch.tensor([
                [1.0, 2.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0],
                [2.0, 4.0, 0.0, 0.0, 2.0, 4.0, 0.0, 0.0],
                [1.1, 2.2, 0.0, 0.0, 1.1, 2.2, 0.0, 0.0],
                [5.0, 1.0, 0.0, 0.0, 5.0, 1.0, 0.0, 0.0],
                [10.0, 2.0, 0.0, 0.0, 10.0, 2.0, 0.0, 0.0],
                [5.2, 1.1, 0.0, 0.0, 5.2, 1.1, 0.0, 0.0],
                [0.2, 0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.0],
                [0.4, 0.4, 0.0, 0.0, 0.4, 0.4, 0.0, 0.0],
            ], dtype=torch.float32)
            model.fc.weight.copy_(weight)

        dense_mask = torch.ones_like(model.fc.weight)
        _, _, _, layer_group_info = ft_budgeted_group_translate(
            model=model,
            in_channel=8,
            out_channel=8,
            weight_name='fc.weight',
            kernel_size=1,
            channel_number=8,
            mask=dense_mask,
            min_group_size=2,
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            budget_config={
                'target_coverage': 0.6,
                'max_singleton': 0.5,
                'min_avg_group_size': 2.0,
                'prototype_budget_ratio': 0.25,
                'prototype_budget_min': 1,
                'prototype_budget_max': 4,
                'relax_threshold': 0.85,
                'max_scale_error': 0.25,
                'bucket_mode': 'nonzero_count',
            },
        )
        self.assertLess(layer_group_info['singleton_ratio'], 1.0)
        self.assertGreaterEqual(layer_group_info['avg_group_size'], 2.0)

    def test_budgeted_mask_family_selection_can_pick_sparse_candidate(self):
        model = TinyFCModel()
        with torch.no_grad():
            weight = torch.tensor([
                [4.0, 4.0, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
                [8.0, 8.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
                [2.0, 2.0, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1],
                [6.0, 6.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1],
            ], dtype=torch.float32)
            model.fc.weight[:4].copy_(weight)

        _, seed_info, outputs = ft_budgeted_select_mask_candidate(
            model=model,
            weight_name='fc.weight',
            in_channel=8,
            out_channel=8,
            kernel_size=1,
            channel_number=8,
            pattern_value_number=1,
            pattern_shape_number=8,
            OU_size=8,
            min_group_size=2,
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            budget_config={
                'target_coverage': 0.6,
                'max_singleton': 0.5,
                'min_avg_group_size': 2.0,
                'prototype_budget_ratio': 0.25,
                'prototype_budget_min': 1,
                'prototype_budget_max': 4,
                'relax_threshold': 0.85,
                'max_scale_error': 0.25,
                'bucket_mode': 'nonzero_count',
                'mask_family': ['shared_topk'],
                'mask_keep_ratios': [0.25],
            },
        )
        self.assertEqual(seed_info['selected_strategy'], 'shared_topk_0.2500_budgeted')
        self.assertEqual(outputs[3]['grouping_mode'], 'budgeted')
        self.assertGreater(outputs[3]['coverage_ratio'], 0.0)

    def test_budgeted_build_only_artifact_protocol_works(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_dir = temp_dir / 'artifacts'
            model = TinyTrainModel()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'epoch': 0,
            }
            torch.save(checkpoint, temp_dir / 'model_ToyBudget_original_parameter_epoch0_ckpt.pth')

            mask = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
            map_information = {'conv.weight': torch.zeros((4, 2, 2), dtype=torch.long)}
            multiple_relationship_information = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
            reuse_ratio_information = {'conv.weight': 0.0}

            with _working_directory(temp_dir):
                mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                    model_original=model,
                    model_name='ToyBudget',
                    translate_name='ft_budgeted_group_translate',
                    weight_name=['conv.weight'],
                    layer_in_channel=[4],
                    layer_out_channel=[2],
                    kernel_size=[1],
                    channel_number=[2],
                    pattern_value_number=[1],
                    mask=mask,
                    map_information=map_information,
                    multiple_relationship_information=multiple_relationship_information,
                    reuse_ratio_information=reuse_ratio_information,
                    ft_layer_enabled=[True],
                    ft_group_target_ratio=[0.75],
                    ft_target_group_size=[2],
                    ft_similarity_threshold=[0.85],
                    checkpoint_epoch=0,
                    force_rebuild=True,
                    artifact_dir=str(artifact_dir),
                    ft_grouping_mode='budgeted',
                    ft_budget_config={
                        'grouping_mode': 'budgeted',
                        'target_coverage': 0.6,
                        'max_singleton': 0.5,
                        'min_avg_group_size': 2.0,
                        'prototype_budget_ratio': 0.25,
                        'prototype_budget_min': 1,
                        'prototype_budget_max': 4,
                        'relax_threshold': 0.85,
                        'max_scale_error': 0.25,
                        'bucket_mode': 'nonzero_count',
                        'mask_family': ['shape_seed', 'shared_topk'],
                        'mask_keep_ratios': [0.6667, 0.4444],
                        'layer_overrides': {},
                    },
                )
            self.assertTrue((artifact_dir / 'model_ToyBudget_ft_budgeted_group_translate_group_information.pkl').exists())
            self.assertTrue((artifact_dir / 'model_ToyBudget_ft_budgeted_group_translate_mask.pkl').exists())
            self.assertEqual(group_information['conv.weight']['grouping_mode'], 'budgeted')
            self.assertIn('prototype_budget', group_information['conv.weight'])
            self.assertIn('assignment_error_p95', group_information['conv.weight'])
            self.assertEqual(group_information['conv.weight']['bucket_mode'], 'nonzero_count')

    def test_redundancy_diagnostics_include_budgeted_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            model = TinyTrainModel()
            with torch.no_grad():
                model.conv.weight[0, 0, 0, 0] = 1.0
                model.conv.weight[0, 1, 0, 0] = 2.0
                model.conv.weight[1, 0, 0, 0] = 2.0
                model.conv.weight[1, 1, 0, 0] = 4.0
            mask = torch.ones_like(model.conv.weight)
            _, _, coverage_ratio, group_info = ft_budgeted_group_translate(
                model=model,
                in_channel=4,
                out_channel=2,
                weight_name='conv.weight',
                kernel_size=1,
                channel_number=2,
                mask=mask,
                min_group_size=2,
                exact_threshold=0.98,
                scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
                budget_config={
                    'target_coverage': 0.6,
                    'max_singleton': 0.5,
                    'min_avg_group_size': 2.0,
                    'prototype_budget_ratio': 0.25,
                    'prototype_budget_min': 1,
                    'prototype_budget_max': 4,
                    'relax_threshold': 0.85,
                    'max_scale_error': 0.25,
                    'bucket_mode': 'shape_family',
                    'mask_family': ['shape_seed', 'shared_topk'],
                    'mask_keep_ratios': [0.6667, 0.4444],
                },
            )
            _write_pkl(temp_dir / 'model_ToyBudgetDiag_ft_budgeted_group_translate_group_information.pkl', {'conv.weight': group_info})
            _write_pkl(temp_dir / 'model_ToyBudgetDiag_ft_budgeted_group_translate_mask.pkl', {'conv.weight': mask})
            _write_pkl(temp_dir / 'model_ToyBudgetDiag_ft_budgeted_group_translate_coverage_ratio_information.pkl', {'conv.weight': coverage_ratio})
            _write_pkl(temp_dir / 'model_ToyBudgetDiag_ft_budgeted_group_translate_reuse_ratio_information.pkl', {'conv.weight': coverage_ratio})

            report = build_redundancy_construction_report('ToyBudgetDiag', 'ft_budgeted_group_translate', str(temp_dir))
            layer = report['layers'][0]
            self.assertEqual(layer['grouping_mode'], 'budgeted')
            self.assertIn('prototype_budget_ratio', layer)
            self.assertIn('assignment_error_p95', layer)
            self.assertIn('target_coverage', layer)
            self.assertIn('coverage_gap', layer)
            self.assertEqual(layer['bucket_mode'], 'shape_family')

    def test_mask_codebook_is_deterministic(self):
        patterns = [
            _make_budget_pattern(0, [4.0, 3.0, 0.1, 0.0]),
            _make_budget_pattern(1, [4.1, 2.9, 0.0, 0.1]),
            _make_budget_pattern(2, [0.1, 0.0, 3.0, 4.0]),
            _make_budget_pattern(3, [0.0, 0.1, 2.9, 4.1]),
        ]
        first = build_redundancy_mask_codebook(patterns, codebook_size=2, keep_counts=[2, 1], codebook_source='mixed')
        second = build_redundancy_mask_codebook(patterns, codebook_size=2, keep_counts=[2, 1], codebook_source='mixed')
        self.assertEqual(
            [(entry['keep_count'], entry['mask'].tolist()) for entry in first],
            [(entry['keep_count'], entry['mask'].tolist()) for entry in second],
        )

    def test_codebook_assignment_reduces_mask_signature_fragmentation(self):
        patterns = [
            _make_budget_pattern(0, [5.0, 4.0, 0.2, 0.1]),
            _make_budget_pattern(1, [4.9, 4.1, 0.1, 0.2]),
            _make_budget_pattern(2, [0.2, 0.1, 5.0, 4.0]),
            _make_budget_pattern(3, [0.1, 0.2, 4.9, 4.1]),
        ]
        codebook = build_redundancy_mask_codebook(patterns, codebook_size=2, keep_counts=[2], codebook_source='mixed')
        assigned_patterns, _, stats = assign_ou_to_mask_codebook(patterns, codebook, assign_mode='mixed')
        self.assertLessEqual(stats['mask_codebook_size'], 2)
        self.assertLess(len({pattern['mask_codebook_id'] for pattern in assigned_patterns}), len(assigned_patterns))

    def test_codebook_budgeted_grouping_reduces_singletons_on_synthetic_data(self):
        model = TinyFCModel()
        with torch.no_grad():
            model.fc.weight.copy_(torch.tensor([
                [4.0, 4.0, 0.1, 0.0, 4.0, 4.0, 0.1, 0.0],
                [8.0, 8.0, 0.2, 0.0, 8.0, 8.0, 0.2, 0.0],
                [3.8, 4.1, 0.0, 0.1, 3.8, 4.1, 0.0, 0.1],
                [0.1, 0.0, 5.0, 5.0, 0.1, 0.0, 5.0, 5.0],
                [0.2, 0.0, 10.0, 10.0, 0.2, 0.0, 10.0, 10.0],
                [0.0, 0.1, 4.8, 5.2, 0.0, 0.1, 4.8, 5.2],
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
            ], dtype=torch.float32))

        layer_mask, _, _, coverage_ratio, group_info = ft_codebook_budgeted_translate(
            model=model,
            in_channel=8,
            out_channel=8,
            weight_name='fc.weight',
            kernel_size=1,
            channel_number=8,
            mask=None,
            min_group_size=2,
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            budget_config={
                'target_coverage': 0.6,
                'max_singleton': 0.5,
                'min_avg_group_size': 2.0,
                'prototype_budget_ratio': 0.25,
                'prototype_budget_min': 1,
                'prototype_budget_max': 4,
                'relax_threshold': 0.85,
                'max_scale_error': 0.25,
                'max_singleton_error': 1.5,
                'mask_codebook_size': 2,
                'mask_codebook_keep_counts': [2],
                'mask_codebook_source': 'mixed',
                'mask_codebook_assign': 'mixed',
                'force_prototype_assignment': True,
                'normalize_prototype_vectors': True,
                'prototype_space': 'normalized_direction',
            },
        )
        self.assertEqual(group_info['grouping_mode'], 'codebook_budgeted')
        self.assertGreater(coverage_ratio, 0.0)
        self.assertLess(group_info['singleton_ratio'], 1.0)
        self.assertGreaterEqual(group_info['mask_codebook_size'], 1)
        self.assertEqual(layer_mask.shape, model.fc.weight.shape)

    def test_codebook_grouping_avoids_zero_scale_valid_members(self):
        model = TinyFCModel()
        with torch.no_grad():
            model.fc.weight.copy_(torch.tensor([
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [2.0, 2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0, 3.0, 3.0, 0.0, 0.0],
                [4.0, 4.0, 0.0, 0.0, 4.0, 4.0, 0.0, 0.0],
                [5.0, 5.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0],
                [6.0, 6.0, 0.0, 0.0, 6.0, 6.0, 0.0, 0.0],
                [7.0, 7.0, 0.0, 0.0, 7.0, 7.0, 0.0, 0.0],
                [8.0, 8.0, 0.0, 0.0, 8.0, 8.0, 0.0, 0.0],
            ], dtype=torch.float32))
        _, _, _, _, group_info = ft_codebook_budgeted_translate(
            model=model,
            in_channel=8,
            out_channel=8,
            weight_name='fc.weight',
            kernel_size=1,
            channel_number=8,
            mask=None,
            min_group_size=2,
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            budget_config={
                'target_coverage': 0.6,
                'max_singleton': 0.5,
                'min_avg_group_size': 2.0,
                'prototype_budget_ratio': 0.25,
                'prototype_budget_min': 1,
                'prototype_budget_max': 4,
                'relax_threshold': 0.85,
                'max_scale_error': 0.25,
                'max_singleton_error': 1.5,
                'mask_codebook_size': 2,
                'mask_codebook_keep_counts': [2],
                'mask_codebook_source': 'mixed',
                'mask_codebook_assign': 'mixed',
                'force_prototype_assignment': True,
                'normalize_prototype_vectors': True,
                'prototype_space': 'normalized_direction',
            },
        )
        for group in group_info['groups']:
            for member in group['members']:
                if member['role'] == 'prototype':
                    continue
                self.assertGreater(abs(float(member['multiplier'])), 1e-8)

    def test_projection_strength_zero_keeps_model_unchanged(self):
        model = TinyTrainModel()
        with torch.no_grad():
            model.conv.weight.copy_(torch.arange(8, dtype=torch.float32).reshape(2, 4, 1, 1))
        original = model.conv.weight.detach().clone()
        mask = {'conv.weight': torch.ones_like(model.conv.weight)}
        group_information = {
            'conv.weight': {
                'groups': [{
                    'prototype': {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 2, 'multiplier': 1.0},
                    'members': [
                        {'out_ch': 0, 'in_ch_start': 0, 'channel_span': 2, 'multiplier': 1.0, 'role': 'prototype'},
                        {'out_ch': 1, 'in_ch_start': 0, 'channel_span': 2, 'multiplier': 2.0, 'role': 'member'},
                    ],
                }],
            },
        }
        apply_ft_group_projection(model, ['conv.weight'], mask, group_information, projection_strength=0.0)
        self.assertTrue(torch.allclose(model.conv.weight.detach(), original))

    def test_codebook_projection_artifacts_exist(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_dir = temp_dir / 'artifacts'
            model = TinyTrainModel()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'epoch': 0,
            }
            torch.save(checkpoint, temp_dir / 'model_ToyCodebook_original_parameter_epoch0_ckpt.pth')
            with _working_directory(temp_dir):
                layer_mask, _, _, _, group_info = ft_codebook_budgeted_translate(
                    model=model,
                    in_channel=4,
                    out_channel=2,
                    weight_name='conv.weight',
                    kernel_size=1,
                    channel_number=2,
                    mask=None,
                    min_group_size=2,
                    exact_threshold=0.98,
                    scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
                    budget_config={
                        'target_coverage': 0.6,
                        'max_singleton': 0.5,
                        'min_avg_group_size': 2.0,
                        'prototype_budget_ratio': 0.25,
                        'prototype_budget_min': 1,
                        'prototype_budget_max': 4,
                        'relax_threshold': 0.85,
                        'max_scale_error': 0.25,
                        'max_singleton_error': 1.5,
                        'mask_codebook_size': 2,
                        'mask_codebook_keep_counts': [1],
                        'mask_codebook_source': 'mixed',
                        'mask_codebook_assign': 'mixed',
                        'force_prototype_assignment': True,
                        'normalize_prototype_vectors': True,
                        'prototype_space': 'normalized_direction',
                    },
                )
                save_ft_build_only_projection(
                    model=model,
                    model_name='ToyCodebook',
                    translate_name='ft_codebook_budgeted_translate',
                    weight_name=['conv.weight'],
                    mask={'conv.weight': layer_mask},
                    group_information={'conv.weight': group_info},
                    checkpoint_epoch=0,
                    artifact_dir=str(artifact_dir),
                    projection_strength=0.5,
                    evaluate_projected=False,
                )
            self.assertTrue((artifact_dir / 'model_ToyCodebook_ft_codebook_budgeted_translate_projected_parameters.pth').exists())
            self.assertTrue((artifact_dir / 'model_ToyCodebook_ft_codebook_budgeted_translate_after_translate_parameters.pth').exists())
            self.assertTrue((artifact_dir / 'model_ToyCodebook_ft_codebook_budgeted_translate_projection_metrics.json').exists())

    def test_diagnostics_include_projected_accuracy_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            model = TinyTrainModel()
            layer_mask, _, _, coverage_ratio, group_info = ft_codebook_budgeted_translate(
                model=model,
                in_channel=4,
                out_channel=2,
                weight_name='conv.weight',
                kernel_size=1,
                channel_number=2,
                mask=None,
                min_group_size=2,
                exact_threshold=0.98,
                scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
                budget_config={
                    'target_coverage': 0.6,
                    'max_singleton': 0.5,
                    'min_avg_group_size': 2.0,
                    'prototype_budget_ratio': 0.25,
                    'prototype_budget_min': 1,
                    'prototype_budget_max': 4,
                    'relax_threshold': 0.85,
                    'max_scale_error': 0.25,
                    'max_singleton_error': 1.5,
                    'mask_codebook_size': 2,
                    'mask_codebook_keep_counts': [1],
                    'mask_codebook_source': 'mixed',
                    'mask_codebook_assign': 'mixed',
                    'force_prototype_assignment': True,
                    'normalize_prototype_vectors': True,
                    'prototype_space': 'normalized_direction',
                },
            )
            _write_pkl(temp_dir / 'model_ToyCodebookDiag_ft_codebook_budgeted_translate_group_information.pkl', {'conv.weight': group_info})
            _write_pkl(temp_dir / 'model_ToyCodebookDiag_ft_codebook_budgeted_translate_mask.pkl', {'conv.weight': layer_mask})
            _write_pkl(temp_dir / 'model_ToyCodebookDiag_ft_codebook_budgeted_translate_coverage_ratio_information.pkl', {'conv.weight': coverage_ratio})
            _write_pkl(temp_dir / 'model_ToyCodebookDiag_ft_codebook_budgeted_translate_reuse_ratio_information.pkl', {'conv.weight': coverage_ratio})
            with open(temp_dir / 'model_ToyCodebookDiag_ft_codebook_budgeted_translate_projection_metrics.json', 'w', encoding='utf-8') as handle:
                json.dump({
                    'projection_strength': 0.5,
                    'projected_accuracy': 0.91,
                    'projected_accuracy_drop': 0.03,
                }, handle)

            report = build_redundancy_construction_report('ToyCodebookDiag', 'ft_codebook_budgeted_translate', str(temp_dir))
            layer = report['layers'][0]
            self.assertEqual(layer['grouping_mode'], 'codebook_budgeted')
            self.assertIn('mask_codebook_size', layer)
            self.assertIn('assignment_error_p50', layer)
            self.assertIn('projected_accuracy', layer)
            self.assertIn('projected_accuracy_drop', layer)
            self.assertEqual(report['global']['projected_accuracy'], 0.91)

    def test_codebook_build_only_artifact_protocol_works(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_dir = temp_dir / 'artifacts'
            model = TinyTrainModel()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'epoch': 0,
            }
            torch.save(checkpoint, temp_dir / 'model_ToyCodebookBuild_original_parameter_epoch0_ckpt.pth')
            mask = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
            map_information = {'conv.weight': torch.zeros((4, 2, 2), dtype=torch.long)}
            multiple_relationship_information = {'conv.weight': torch.ones((2, 4, 1, 1), dtype=torch.float32)}
            reuse_ratio_information = {'conv.weight': 0.0}
            with _working_directory(temp_dir):
                mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                    model_original=model,
                    model_name='ToyCodebookBuild',
                    translate_name='ft_codebook_budgeted_translate',
                    weight_name=['conv.weight'],
                    layer_in_channel=[4],
                    layer_out_channel=[2],
                    kernel_size=[1],
                    channel_number=[2],
                    pattern_value_number=[1],
                    mask=mask,
                    map_information=map_information,
                    multiple_relationship_information=multiple_relationship_information,
                    reuse_ratio_information=reuse_ratio_information,
                    ft_layer_enabled=[True],
                    ft_group_target_ratio=[0.75],
                    ft_target_group_size=[2],
                    ft_similarity_threshold=[0.85],
                    checkpoint_epoch=0,
                    force_rebuild=True,
                    artifact_dir=str(artifact_dir),
                    ft_grouping_mode='codebook_budgeted',
                    ft_budget_config={
                        'grouping_mode': 'codebook_budgeted',
                        'target_coverage': 0.6,
                        'max_singleton': 0.5,
                        'min_avg_group_size': 2.0,
                        'prototype_budget_ratio': 0.25,
                        'prototype_budget_min': 1,
                        'prototype_budget_max': 4,
                        'relax_threshold': 0.85,
                        'max_scale_error': 0.25,
                        'max_singleton_error': 1.5,
                        'mask_codebook_size': 2,
                        'mask_codebook_keep_counts': [1],
                        'mask_codebook_source': 'mixed',
                        'mask_codebook_assign': 'mixed',
                        'force_prototype_assignment': True,
                        'normalize_prototype_vectors': True,
                        'prototype_space': 'normalized_direction',
                        'layer_overrides': {},
                    },
                )
            self.assertTrue((artifact_dir / 'model_ToyCodebookBuild_ft_codebook_budgeted_translate_group_information.pkl').exists())
            self.assertTrue((artifact_dir / 'model_ToyCodebookBuild_ft_codebook_budgeted_translate_mask.pkl').exists())
            self.assertEqual(group_information['conv.weight']['grouping_mode'], 'codebook_budgeted')
            self.assertIn('mask_codebook_size', group_information['conv.weight'])
            self.assertIn('forced_assignment_count', group_information['conv.weight'])

    def test_projection_sanity_report_updates_without_rebuild(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            artifact_dir = temp_dir / 'artifacts'
            artifact_dir.mkdir(parents=True, exist_ok=True)
            model = TinyTrainModel()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_schedule': scheduler.state_dict(),
                'epoch': 0,
            }
            torch.save(checkpoint, temp_dir / 'model_ToySanity_original_parameter_epoch0_ckpt.pth')
            _build_codebook_group_artifacts(temp_dir, model_name='ToySanity')
            with open(temp_dir / 'model_ToySanity_ft_codebook_budgeted_translate_group_information.pkl', 'rb') as handle:
                group_information = pickle.load(handle)
            with open(temp_dir / 'model_ToySanity_ft_codebook_budgeted_translate_mask.pkl', 'rb') as handle:
                mask = pickle.load(handle)

            with _working_directory(temp_dir):
                save_ft_build_only_projection(
                    model=model,
                    model_name='ToySanity',
                    translate_name='ft_codebook_budgeted_translate',
                    weight_name=['conv.weight'],
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=0,
                    artifact_dir=str(artifact_dir),
                    projection_strength=0.0,
                    evaluate_projected=False,
                )
                save_ft_build_only_projection(
                    model=model,
                    model_name='ToySanity',
                    translate_name='ft_codebook_budgeted_translate',
                    weight_name=['conv.weight'],
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=0,
                    artifact_dir=str(artifact_dir),
                    projection_strength=0.1,
                    evaluate_projected=False,
                )

            sanity_json = artifact_dir / 'model_ToySanity_ft_codebook_budgeted_translate_projection_sanity.json'
            self.assertTrue(sanity_json.exists())
            payload = json.loads(sanity_json.read_text(encoding='utf-8'))
            strengths = [round(float(item['projection_strength']), 4) for item in payload]
            self.assertEqual(strengths, [0.0, 0.1])

    def test_codebook_layer_config_overrides_keep_counts_and_projection_cap(self):
        config_payload = {
            'groups': [
                {
                    'layers': 'conv1-conv5',
                    'mask_codebook_keep_counts': [8, 4],
                    'projection_cap': 0.05,
                    'projection_lambda': 0.4,
                }
            ],
            'layers': {
                'conv14-conv17,shortcut2,shortcut3,fc': {
                    'mask_codebook_keep_counts': [2, 1],
                    'projection_cap': 0.3,
                }
            },
        }
        resolved = _resolve_budgeted_layer_config(
            {
                'layer_overrides': {
                    'conv1-conv5': {
                        'mask_codebook_keep_counts': [8, 4],
                        'projection_cap': 0.05,
                        'projection_lambda': 0.4,
                    },
                    'conv14-conv17,shortcut2,shortcut3,fc': {
                        'mask_codebook_keep_counts': [2, 1],
                        'projection_cap': 0.3,
                    },
                }
            },
            'conv3.weight',
        )
        self.assertEqual(resolved['mask_codebook_keep_counts'], [8, 4])
        self.assertAlmostEqual(resolved['projection_cap'], 0.05)
        self.assertAlmostEqual(resolved['projection_lambda'], 0.4)
        late_resolved = _resolve_budgeted_layer_config(
            {'layer_overrides': config_payload['layers']},
            'conv15.weight',
        )
        self.assertEqual(late_resolved['mask_codebook_keep_counts'], [2, 1])
        self.assertAlmostEqual(late_resolved['projection_cap'], 0.3)

    def test_projection_ramp_schedule_monotonic_and_logged(self):
        ramp_values = [
            _projection_strength_for_epoch(epoch, 0.0, 0.1, [1, 3])
            for epoch in [1, 2, 3, 4]
        ]
        self.assertLessEqual(ramp_values[0], ramp_values[1])
        self.assertLessEqual(ramp_values[1], ramp_values[2])
        self.assertEqual(ramp_values[2], ramp_values[3])

        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _run_tiny_ft_train(
                temp_dir=temp_dir,
                model_name='ToyAdapt',
                translate_name='ft_codebook_budgeted_translate',
                group_refresh_epoch=[],
                translate_epoch=[3],
                output_dir='artifacts',
                ft_grouping_mode='codebook_budgeted',
                ft_projection_ramp_start=0.0,
                ft_projection_ramp_end=0.1,
                ft_projection_ramp_epochs=[1, 3],
                ft_projection_loss_lambda=1e-4,
                ft_codebook_freeze_grouping=True,
                ft_codebook_use_legacy_regularization=False,
            )
            profile_path = temp_dir / 'artifacts' / 'model_ToyAdapt_ft_codebook_budgeted_translate_training_profile.csv'
            rows = _read_csv_rows(profile_path)
            strengths = [float(row['effective_projection_strength']) for row in rows]
            self.assertGreaterEqual(strengths[-1], strengths[0])
            self.assertTrue(all(int(row['legacy_regularization_enabled']) == 0 for row in rows))

    def test_projection_regularization_state_can_sample_links(self):
        groups = []
        for group_index in range(4):
            groups.append({
                'prototype': {'out_ch': group_index, 'in_ch_start': 0, 'channel_span': 1, 'multiplier': 1.0},
                'members': [
                    {'out_ch': group_index, 'in_ch_start': member_index + 1, 'channel_span': 1, 'multiplier': 1.0}
                    for member_index in range(8)
                ],
            })
        state = {
            'layers': {'conv.weight': groups},
            'summary': {'member_link_count': 32},
            'layer_rows': [],
        }
        sampled = _sample_regularization_state_links(state, max_links=5)
        sampled_links = sum(
            len(group['members'])
            for layer_groups in sampled['layers'].values()
            for group in layer_groups
        )
        self.assertLessEqual(sampled_links, 5)
        self.assertEqual(sampled['summary']['projection_original_member_link_count'], 32)
        self.assertEqual(sampled['summary']['projection_sampled_member_link_count'], sampled_links)

    def test_codebook_short_training_saves_after_translate_parameters(self):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            _run_tiny_ft_train(
                temp_dir=temp_dir,
                model_name='ToyAdaptSave',
                translate_name='ft_codebook_budgeted_translate',
                group_refresh_epoch=[],
                translate_epoch=[2],
                output_dir='artifacts',
                ft_grouping_mode='codebook_budgeted',
                ft_projection_ramp_start=0.0,
                ft_projection_ramp_end=0.05,
                ft_projection_ramp_epochs=[1, 2],
                ft_projection_loss_lambda=1e-4,
                ft_projection_reg_max_links=1,
                ft_codebook_freeze_grouping=True,
                ft_codebook_use_legacy_regularization=False,
            )
            self.assertTrue((temp_dir / 'artifacts' / 'model_ToyAdaptSave_ft_codebook_budgeted_translate_after_translate_parameters.pth').exists())
            rows = _read_csv_rows(temp_dir / 'artifacts' / 'model_ToyAdaptSave_ft_codebook_budgeted_translate_training_profile.csv')
            self.assertTrue(all(int(row['projection_sampled_member_links']) <= 1 for row in rows))


if __name__ == '__main__':
    unittest.main(verbosity=2)
