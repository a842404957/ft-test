import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from cut import _compile_ft_regularization_state, extract_ou_patterns, ft_group_score_mask
from fault_tolerance_analyse import analyse
from simulator.Fault_Tolerance.fault_tolerance_simulation import FaultToleranceSimulator
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


if __name__ == '__main__':
    unittest.main(verbosity=2)
