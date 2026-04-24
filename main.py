import os
import sys
import argparse
import json
import torch
import torch.utils.data
import pickle as pkl
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from model import Vgg16, Res18, Res50, WRN
from train_model import train, test
from cut import (
    pattern_translate,
    get_structure_mask,
    get_ORC_mask,
    get_shape_mask,
    pattern_value_identical_translate,
    pattern_value_similar_translate,
    structure_and_value_identical_translate,
    pattern_shape_and_value_similar_translate,
    apply_ft_group_projection,
    parameters_to_txt,
    ft_group_score_mask,
    ft_group_cluster_translate,
    ft_budgeted_group_translate,
    ft_budgeted_select_mask_candidate,
    ft_codebook_budgeted_translate,
    ft_codebook_budgeted_select_mask_candidate,
    ft_group_translate_train,
    write_mask_sweep_report,
)


model_name = 'Vgg16'  # select one from[Vgg16, Res18, Res50, WRN]
translate_name = 'ft_group_cluster_translate'  # 新默认方法：面向容错的OU分组剪枝/映射
ft_grouping_mode_default = 'ftscore'

lr = 0.1
epoches = 200
batch_size = 128
weight_decay_1 = 0.001
weight_decay_2 = 0.001

OU_size = 8
pattern_shape_number = 8
base_checkpoint_epoch = 150
translate_epoch = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
ft_translate_epoch = [200]

ft_min_group_size = 2
ft_target_group_size_default = 4
ft_similarity_threshold_default = 0.85
ft_exact_similarity_threshold = 0.98
ft_scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
ft_group_refresh_epoch = [190, 200]
ft_balance_lambda = 1e-4
ft_proto_lambda = 1e-3
ft_mask_lambda = 1e-3
ft_sep_lambda = 5e-5
ft_reg_interval_default = 1
ft_reg_min_coverage_default = 0.0
ft_reg_min_groups_default = 1
ft_low_cost_end_epoch = 160
ft_cost_presets = {
    'fast': {
        'end_epoch': 160,
        'reg_interval': 10,
        'reg_min_coverage': 0.1,
        'reg_min_groups': 64,
        'refresh_epochs': [156, 160],
    },
    'balanced': {
        'end_epoch': 180,
        'reg_interval': 5,
        'reg_min_coverage': 0.05,
        'reg_min_groups': 32,
        'refresh_epochs': [160, 170, 180],
    },
    'full': {
        'end_epoch': 200,
        'reg_interval': 5,
        'reg_min_coverage': 0.0,
        'reg_min_groups': 1,
        'refresh_epochs': [170, 185, 200],
    },
}
ft_budget_target_coverage_default = 0.6
ft_budget_max_singleton_default = 0.5
ft_budget_min_avg_group_size_default = 2.0
ft_prototype_budget_ratio_default = 0.25
ft_prototype_budget_min_default = 4
ft_prototype_budget_max_default = 256
ft_budget_relax_threshold_default = 0.85
ft_budget_max_scale_error_default = 0.25
ft_budget_bucket_mode_default = 'nonzero_count'
ft_budget_mask_family_default = 'shape_seed,shared_topk,per_out_topk'
ft_budget_mask_keep_ratios_default = '0.6667,0.4444'
ft_mask_codebook_size_default = 4
ft_mask_codebook_keep_counts_default = '4,2'
ft_mask_codebook_source_default = 'mixed'
ft_mask_codebook_assign_default = 'mixed'
ft_force_prototype_assignment_default = True
ft_max_singleton_error_default = 1.5
ft_projection_strength_default = 1.0
ft_evaluate_projected_default = True
ft_normalize_prototype_vectors_default = True
ft_prototype_space_default = 'normalized_direction'
ft_projection_ramp_start_default = 0.0
ft_projection_ramp_end_default = 0.1
ft_projection_ramp_epochs_default = '151,160'
ft_projection_loss_lambda_default = 1e-4
ft_projection_reg_max_links_default = 8192
ft_codebook_adapt_epochs_default = 10
ft_codebook_freeze_grouping_default = True
ft_codebook_refresh_epochs_default = ''
ft_codebook_adapt_reg_interval_default = 10
ft_codebook_use_legacy_regularization_default = False
ft_grouping_translate_names = {'ft_group_cluster_translate', 'ft_budgeted_group_translate', 'ft_codebook_budgeted_translate'}


def get_dataloader(data_name):
    print('...Preparing data...')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if data_name == 'cifar10':
        cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size = 50000, 10000, 10, (3, 32, 32)
        cifar10_train_dataset = datasets.CIFAR10('./cifar10_data', train=True, transform=transform_train, download=False)
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=True)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


def parse_args():
    parser = argparse.ArgumentParser(description='FT-oriented grouping training entry')
    parser.add_argument('--model', type=str, default=model_name, choices=['Vgg16', 'Res18', 'Res50', 'WRN'], help='model name')
    parser.add_argument('--translate', type=str, default=translate_name, help='translate method name')
    parser.add_argument('--ft-grouping-mode', type=str, default=ft_grouping_mode_default, choices=['ftscore', 'budgeted', 'codebook_budgeted'], help='FT grouping mode used by FT-oriented translate paths')
    parser.add_argument('--run-tag', type=str, default='', help='optional run tag; when set without --artifact-dir, artifacts are written to results/ft_runs/<model>/<translate>/<run-tag>/artifacts')
    parser.add_argument('--artifact-dir', type=str, default='', help='optional artifact directory for FT outputs; overrides the default run-tag-derived artifact location')
    parser.add_argument('--build-only', action='store_true', help='only build FT grouping artifacts and projected parameters; skip FT fine-tuning')
    parser.add_argument('--force-rebuild', action='store_true', help='ignore cached FT artifacts and rebuild them from the base checkpoint')
    parser.add_argument('--ft-cost-preset', type=str, default='none', choices=['none', 'fast', 'balanced', 'full'], help='cost preset for FT training schedule and regularization')
    parser.add_argument('--ft-low-cost', action='store_true', help='use a lower-cost FT training preset with sparse FT regularization and a shorter FT fine-tuning window')
    parser.add_argument('--ft-end-epoch', type=int, default=None, help='final epoch for FT fine-tuning; default keeps 200, ft-low-cost defaults to 160')
    parser.add_argument('--ft-reg-interval', type=int, default=None, help='apply FT regularization every N batches; default 1, ft-low-cost defaults to 10')
    parser.add_argument('--ft-reg-min-coverage', type=float, default=None, help='minimum layer coverage ratio required for FT regularization; default 0.0, ft-low-cost defaults to 0.1')
    parser.add_argument('--ft-reg-min-groups', type=int, default=None, help='minimum repairable group count required for FT regularization on a layer; default 1, ft-low-cost defaults to 64')
    parser.add_argument('--ft-reg-boost-after-refresh', action='store_true', help='halve the effective FT regularization interval during refresh epochs and the following epoch')
    parser.add_argument('--ft-mask-density-sweep', action='store_true', help='explicitly compare multiple pruning keep ratios when selecting FT masks')
    parser.add_argument('--ft-mask-keep-ratios', type=str, default='', help='comma-separated explicit mask keep ratios, e.g. 1.0,0.6667,0.4444')
    parser.add_argument('--ft-target-coverage', type=float, default=None, help='prefer the minimum-distortion candidate that reaches this repairable coverage target')
    parser.add_argument('--ft-prefer-sparser-mask', action='store_true', help='bias FT mask selection toward lower mask density when scores are close')
    parser.add_argument('--ft-score-singleton-penalty', type=float, default=0.18, help='penalty applied to singleton_ratio in FTScore_v2')
    parser.add_argument('--ft-score-zero-scale-penalty', type=float, default=0.12, help='penalty applied to zero_multiplier_ratio in FTScore_v2')
    parser.add_argument('--ft-budget-target-coverage', type=float, default=ft_budget_target_coverage_default, help='budgeted grouping target repairable coverage ratio')
    parser.add_argument('--ft-budget-max-singleton', type=float, default=ft_budget_max_singleton_default, help='budgeted grouping target upper bound for singleton ratio')
    parser.add_argument('--ft-budget-min-avg-group-size', type=float, default=ft_budget_min_avg_group_size_default, help='budgeted grouping target lower bound for average group size')
    parser.add_argument('--ft-prototype-budget-ratio', type=float, default=ft_prototype_budget_ratio_default, help='budgeted grouping prototype budget ratio per block bucket')
    parser.add_argument('--ft-prototype-budget-min', type=int, default=ft_prototype_budget_min_default, help='minimum prototype budget per block bucket')
    parser.add_argument('--ft-prototype-budget-max', type=int, default=ft_prototype_budget_max_default, help='maximum prototype budget per block bucket')
    parser.add_argument('--ft-budget-relax-threshold', type=float, default=ft_budget_relax_threshold_default, help='budgeted grouping relax factor; lower values reduce prototype budget and widen error tolerance')
    parser.add_argument('--ft-budget-max-scale-error', type=float, default=ft_budget_max_scale_error_default, help='maximum normalized scale assignment error for budgeted grouping')
    parser.add_argument('--ft-budget-bucket-mode', type=str, default=ft_budget_bucket_mode_default, choices=['exact_mask', 'nonzero_count', 'shape_family'], help='bucket granularity for budgeted grouping before prototype assignment')
    parser.add_argument('--ft-budget-mask-family', type=str, default=ft_budget_mask_family_default, help='comma-separated lightweight mask family used by budgeted build-only selection')
    parser.add_argument('--ft-budget-mask-keep-ratios', type=str, default=ft_budget_mask_keep_ratios_default, help='comma-separated keep ratios for budgeted sparse mask family candidates')
    parser.add_argument('--ft-mask-codebook-size', type=int, default=ft_mask_codebook_size_default, help='maximum mask codebook size per layer/block bucket for codebook-budgeted grouping')
    parser.add_argument('--ft-mask-codebook-keep-counts', type=str, default=ft_mask_codebook_keep_counts_default, help='comma-separated keep counts for mask codebook candidates, e.g. 4,2')
    parser.add_argument('--ft-mask-codebook-source', type=str, default=ft_mask_codebook_source_default, choices=['importance', 'frequency', 'mixed'], help='source used to build the redundancy-inducing mask codebook')
    parser.add_argument('--ft-mask-codebook-assign', type=str, default=ft_mask_codebook_assign_default, choices=['min_distortion', 'max_redundancy', 'mixed'], help='assignment objective used when mapping OUs to the mask codebook')
    parser.add_argument('--ft-force-prototype-assignment', action=argparse.BooleanOptionalAction, default=ft_force_prototype_assignment_default, help='force every OU into the nearest prototype group unless assignment error exceeds --ft-max-singleton-error')
    parser.add_argument('--ft-max-singleton-error', type=float, default=ft_max_singleton_error_default, help='hard upper bound on normalized assignment error before an OU is left as singleton in codebook-budgeted grouping')
    parser.add_argument('--ft-projection-strength', type=float, default=ft_projection_strength_default, help='blend factor for projected codebook-budgeted weights during build-only projection')
    parser.add_argument('--ft-evaluate-projected', action=argparse.BooleanOptionalAction, default=ft_evaluate_projected_default, help='evaluate projected build-only model immediately after writing artifacts')
    parser.add_argument('--ft-normalize-prototype-vectors', action=argparse.BooleanOptionalAction, default=ft_normalize_prototype_vectors_default, help='normalize prototype selection vectors before deterministic prototype search')
    parser.add_argument('--ft-prototype-space', type=str, default=ft_prototype_space_default, choices=['raw', 'normalized_direction', 'quantized_direction'], help='feature space used for codebook-budgeted prototype selection')
    parser.add_argument('--ft-budget-layer-config', type=str, default='', help='optional JSON file with per-layer budgeted grouping overrides')
    parser.add_argument('--ft-codebook-layer-config', type=str, default='', help='optional JSON file with per-layer codebook-budgeted overrides such as keep_counts, target_coverage, projection_cap, and projection_lambda')
    parser.add_argument('--ft-projection-ramp-start', type=float, default=ft_projection_ramp_start_default, help='starting projection strength for codebook-aware short adaptation training')
    parser.add_argument('--ft-projection-ramp-end', type=float, default=ft_projection_ramp_end_default, help='ending projection strength for codebook-aware short adaptation training')
    parser.add_argument('--ft-projection-ramp-epochs', type=str, default=ft_projection_ramp_epochs_default, help='comma-separated projection ramp start/end epochs, e.g. 151,160')
    parser.add_argument('--ft-projection-loss-lambda', type=float, default=ft_projection_loss_lambda_default, help='weight for projection consistency regularization during codebook-aware short training')
    parser.add_argument('--ft-projection-reg-max-links', type=int, default=ft_projection_reg_max_links_default, help='maximum sampled member links used by projection consistency regularization per reg batch; 0 disables sampling')
    parser.add_argument('--ft-codebook-adapt-epochs', type=int, default=ft_codebook_adapt_epochs_default, help='short adaptation length in epochs for codebook-aware training')
    parser.add_argument('--ft-codebook-freeze-grouping', action=argparse.BooleanOptionalAction, default=ft_codebook_freeze_grouping_default, help='freeze codebook grouping during short adaptation unless refresh epochs are explicitly enabled')
    parser.add_argument('--ft-codebook-refresh-epochs', type=str, default=ft_codebook_refresh_epochs_default, help='comma-separated refresh epochs for codebook-aware short adaptation; empty keeps grouping frozen by default')
    parser.add_argument('--ft-codebook-use-legacy-regularization', action=argparse.BooleanOptionalAction, default=ft_codebook_use_legacy_regularization_default, help='include legacy FT mask/prototype regularization during codebook-aware short adaptation; disabled by default so short adaptation focuses on CE + projection consistency')
    parser.add_argument('--base-checkpoint-epoch', type=int, default=base_checkpoint_epoch, help='checkpoint epoch used as FT build/training start point')
    parser.add_argument('--translate-epochs', type=str, default=','.join(str(epoch) for epoch in ft_translate_epoch), help='comma-separated evaluation/projection epochs during FT training')
    parser.add_argument('--refresh-epochs', type=str, default=','.join(str(epoch) for epoch in ft_group_refresh_epoch), help='comma-separated group refresh epochs; empty disables refresh')
    return parser.parse_args()


def parse_epoch_list(raw_value, allow_empty=False):
    if raw_value is None:
        return []
    normalized = str(raw_value).strip()
    if normalized == '':
        if allow_empty:
            return []
        raise ValueError('epoch list cannot be empty')

    tokens = [token.strip() for token in normalized.split(',') if token.strip()]
    if not tokens:
        if allow_empty:
            return []
        raise ValueError('epoch list cannot be empty')

    epoch_values = sorted({int(token) for token in tokens})
    if not allow_empty and not epoch_values:
        raise ValueError('epoch list cannot be empty')
    return epoch_values


def parse_float_list(raw_value):
    if raw_value is None:
        return []
    normalized = str(raw_value).strip()
    if normalized == '':
        return []
    return [float(token.strip()) for token in normalized.split(',') if token.strip()]


def normalize_schedule_epochs(epoch_values, checkpoint_epoch, end_epoch, ensure_final=False):
    filtered_epochs = sorted({epoch for epoch in epoch_values if checkpoint_epoch < epoch <= end_epoch})
    if ensure_final and end_epoch > checkpoint_epoch and end_epoch not in filtered_epochs:
        filtered_epochs.append(end_epoch)
    return filtered_epochs


def _flag_present(argv, flag_name):
    return any(token == flag_name or token.startswith(flag_name + '=') for token in argv)


def is_ft_grouping_translate(translate_name):
    return translate_name in ft_grouping_translate_names


def resolve_ft_grouping_mode(translate_name, requested_mode):
    if translate_name == 'ft_budgeted_group_translate':
        return 'budgeted'
    if translate_name == 'ft_codebook_budgeted_translate':
        return 'codebook_budgeted'
    return requested_mode


def _load_budget_layer_config(layer_config_path):
    if not layer_config_path:
        return {}
    with open(layer_config_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('layer config must be a JSON object')

    normalized = {}
    direct_layers = payload.get('layers', {})
    if isinstance(direct_layers, dict):
        for selector, override in direct_layers.items():
            if isinstance(override, dict):
                normalized[str(selector)] = dict(override)

    group_entries = payload.get('groups', [])
    if isinstance(group_entries, list):
        for index, entry in enumerate(group_entries):
            if not isinstance(entry, dict):
                continue
            selector = entry.get('layers') or entry.get('selector') or entry.get('match') or entry.get('name')
            if not selector:
                raise ValueError(f'layer config group entry #{index} must provide "layers", "selector", or "match"')
            override = {key: value for key, value in entry.items() if key not in {'layers', 'selector', 'match', 'name'}}
            normalized[str(selector)] = override

    for selector, override in payload.items():
        if selector in {'layers', 'groups'}:
            continue
        if isinstance(override, dict):
            selector_key = str(override.get('layers') or override.get('selector') or override.get('match') or selector)
            normalized[selector_key] = {
                key: value for key, value in override.items()
                if key not in {'layers', 'selector', 'match'}
            }
    return normalized


def _merge_layer_override_maps(*override_maps):
    merged = {}
    for override_map in override_maps:
        for selector, override in (override_map or {}).items():
            base_override = dict(merged.get(selector, {}))
            base_override.update(dict(override))
            merged[selector] = base_override
    return merged


def _parse_float_list(raw_value, *, option_name, allow_empty=False):
    normalized = str(raw_value).strip()
    if normalized == '':
        if allow_empty:
            return []
        raise ValueError(f'{option_name} cannot be empty')
    values = []
    for token in normalized.split(','):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values and not allow_empty:
        raise ValueError(f'{option_name} cannot be empty')
    return values


def build_budgeted_grouping_config(args, translate_name):
    grouping_mode = resolve_ft_grouping_mode(translate_name, args.ft_grouping_mode)
    if args.ft_budget_target_coverage < 0.0 or args.ft_budget_target_coverage > 1.0:
        raise ValueError('--ft-budget-target-coverage must be within [0, 1]')
    if args.ft_budget_max_singleton < 0.0 or args.ft_budget_max_singleton > 1.0:
        raise ValueError('--ft-budget-max-singleton must be within [0, 1]')
    if args.ft_budget_min_avg_group_size < 1.0:
        raise ValueError('--ft-budget-min-avg-group-size must be >= 1')
    if args.ft_prototype_budget_ratio <= 0.0 or args.ft_prototype_budget_ratio > 1.0:
        raise ValueError('--ft-prototype-budget-ratio must be within (0, 1]')
    if args.ft_prototype_budget_min <= 0:
        raise ValueError('--ft-prototype-budget-min must be positive')
    if args.ft_prototype_budget_max < args.ft_prototype_budget_min:
        raise ValueError('--ft-prototype-budget-max must be >= --ft-prototype-budget-min')
    if args.ft_budget_relax_threshold <= 0.0 or args.ft_budget_relax_threshold >= 1.0:
        raise ValueError('--ft-budget-relax-threshold must be within (0, 1)')
    if args.ft_budget_max_scale_error <= 0.0:
        raise ValueError('--ft-budget-max-scale-error must be positive')
    if args.ft_mask_codebook_size <= 0:
        raise ValueError('--ft-mask-codebook-size must be positive')
    mask_codebook_keep_counts = [int(value) for value in _parse_float_list(
        args.ft_mask_codebook_keep_counts,
        option_name='--ft-mask-codebook-keep-counts',
        allow_empty=False,
    )]
    if any(value <= 0 for value in mask_codebook_keep_counts):
        raise ValueError('--ft-mask-codebook-keep-counts values must be positive integers')
    if args.ft_max_singleton_error <= 0.0:
        raise ValueError('--ft-max-singleton-error must be positive')
    if args.ft_projection_strength < 0.0 or args.ft_projection_strength > 1.0:
        raise ValueError('--ft-projection-strength must be within [0, 1]')
    budget_mask_family = [item.strip() for item in str(args.ft_budget_mask_family).split(',') if item.strip()]
    if not budget_mask_family:
        raise ValueError('--ft-budget-mask-family must contain at least one candidate family')
    invalid_budget_mask_family = sorted(set(budget_mask_family) - {'shape_seed', 'shared_topk', 'per_out_topk'})
    if invalid_budget_mask_family:
        raise ValueError(f'unsupported --ft-budget-mask-family entries: {", ".join(invalid_budget_mask_family)}')
    budget_mask_keep_ratios = _parse_float_list(
        args.ft_budget_mask_keep_ratios,
        option_name='--ft-budget-mask-keep-ratios',
        allow_empty=True,
    )
    for keep_ratio in budget_mask_keep_ratios:
        if keep_ratio <= 0.0 or keep_ratio > 1.0:
            raise ValueError('--ft-budget-mask-keep-ratios values must be within (0, 1]')

    layer_overrides = _merge_layer_override_maps(
        _load_budget_layer_config(args.ft_budget_layer_config),
        _load_budget_layer_config(args.ft_codebook_layer_config),
    )
    layer_config_paths = []
    if args.ft_budget_layer_config:
        layer_config_paths.append(os.path.abspath(args.ft_budget_layer_config))
    if args.ft_codebook_layer_config:
        layer_config_paths.append(os.path.abspath(args.ft_codebook_layer_config))
    return {
        'grouping_mode': grouping_mode,
        'target_coverage': float(args.ft_budget_target_coverage),
        'max_singleton': float(args.ft_budget_max_singleton),
        'min_avg_group_size': float(args.ft_budget_min_avg_group_size),
        'prototype_budget_ratio': float(args.ft_prototype_budget_ratio),
        'prototype_budget_min': int(args.ft_prototype_budget_min),
        'prototype_budget_max': int(args.ft_prototype_budget_max),
        'relax_threshold': float(args.ft_budget_relax_threshold),
        'max_scale_error': float(args.ft_budget_max_scale_error),
        'bucket_mode': str(args.ft_budget_bucket_mode),
        'mask_family': budget_mask_family,
        'mask_keep_ratios': [float(ratio) for ratio in budget_mask_keep_ratios],
        'mask_codebook_size': int(args.ft_mask_codebook_size),
        'mask_codebook_keep_counts': mask_codebook_keep_counts,
        'mask_codebook_source': str(args.ft_mask_codebook_source),
        'mask_codebook_assign': str(args.ft_mask_codebook_assign),
        'force_prototype_assignment': bool(args.ft_force_prototype_assignment),
        'max_singleton_error': float(args.ft_max_singleton_error),
        'projection_strength': float(args.ft_projection_strength),
        'evaluate_projected': bool(args.ft_evaluate_projected),
        'normalize_prototype_vectors': bool(args.ft_normalize_prototype_vectors),
        'prototype_space': str(args.ft_prototype_space),
        'projection_cap': 1.0,
        'projection_lambda': 1.0,
        'layer_overrides': layer_overrides,
        'layer_config_path': ','.join(layer_config_paths),
    }


def build_codebook_adaptation_config(args, translate_name, checkpoint_epoch, ft_training_config, grouping_mode, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    refresh_explicit = _flag_present(argv, '--ft-codebook-refresh-epochs')
    reg_interval_explicit = _flag_present(argv, '--ft-reg-interval')

    if args.ft_projection_ramp_start < 0.0 or args.ft_projection_ramp_start > 1.0:
        raise ValueError('--ft-projection-ramp-start must be within [0, 1]')
    if args.ft_projection_ramp_end < 0.0 or args.ft_projection_ramp_end > 1.0:
        raise ValueError('--ft-projection-ramp-end must be within [0, 1]')
    if args.ft_projection_loss_lambda < 0.0:
        raise ValueError('--ft-projection-loss-lambda must be >= 0')
    if args.ft_projection_reg_max_links < 0:
        raise ValueError('--ft-projection-reg-max-links must be >= 0')
    if args.ft_codebook_adapt_epochs <= 0:
        raise ValueError('--ft-codebook-adapt-epochs must be positive')

    enabled = grouping_mode == 'codebook_budgeted'
    effective_end_epoch = ft_training_config['ft_end_epoch']
    effective_translate_epochs = list(ft_training_config['ft_translate_epoch'])
    effective_refresh_epochs = list(ft_training_config['ft_group_refresh_epoch'])
    effective_reg_interval = int(ft_training_config['ft_reg_interval'])

    if enabled:
        effective_end_epoch = min(ft_training_config['ft_end_epoch'], checkpoint_epoch + int(args.ft_codebook_adapt_epochs))
        if not reg_interval_explicit and args.ft_cost_preset == 'none' and not args.ft_low_cost:
            effective_reg_interval = max(effective_reg_interval, ft_codebook_adapt_reg_interval_default)
        effective_translate_epochs = normalize_schedule_epochs(
            effective_translate_epochs,
            checkpoint_epoch=checkpoint_epoch,
            end_epoch=effective_end_epoch,
            ensure_final=True,
        )
        if args.ft_codebook_freeze_grouping and not refresh_explicit:
            effective_refresh_epochs = []
        elif refresh_explicit:
            effective_refresh_epochs = normalize_schedule_epochs(
                parse_epoch_list(args.ft_codebook_refresh_epochs, allow_empty=True),
                checkpoint_epoch=checkpoint_epoch,
                end_epoch=effective_end_epoch,
                ensure_final=False,
            )
        else:
            effective_refresh_epochs = normalize_schedule_epochs(
                effective_refresh_epochs,
                checkpoint_epoch=checkpoint_epoch,
                end_epoch=effective_end_epoch,
                ensure_final=False,
            )

    ramp_epochs = normalize_schedule_epochs(
        parse_epoch_list(args.ft_projection_ramp_epochs, allow_empty=False),
        checkpoint_epoch=checkpoint_epoch,
        end_epoch=effective_end_epoch,
        ensure_final=True,
    )
    if not ramp_epochs:
        ramp_epochs = [effective_end_epoch]
    return {
        'enabled': enabled,
        'projection_ramp_start': float(args.ft_projection_ramp_start),
        'projection_ramp_end': float(args.ft_projection_ramp_end),
        'projection_ramp_epochs': ramp_epochs,
        'projection_loss_lambda': float(args.ft_projection_loss_lambda),
        'projection_reg_max_links': int(args.ft_projection_reg_max_links),
        'codebook_adapt_epochs': int(args.ft_codebook_adapt_epochs),
        'codebook_freeze_grouping': bool(args.ft_codebook_freeze_grouping),
        'codebook_use_legacy_regularization': bool(args.ft_codebook_use_legacy_regularization),
        'effective_end_epoch': int(effective_end_epoch),
        'effective_translate_epochs': effective_translate_epochs,
        'effective_refresh_epochs': effective_refresh_epochs,
        'effective_reg_interval': int(effective_reg_interval),
    }


def _projection_sanity_record(model_name, translate_name, projection_metrics, group_information):
    layer_rows = []
    for layer_name, layer_info in (group_information or {}).items():
        if not isinstance(layer_info, dict):
            continue
        layer_rows.append({
            'layer': layer_name,
            'assignment_error_p95': float(layer_info.get('assignment_error_p95', 0.0)),
            'singleton_ratio': float(layer_info.get('singleton_ratio', 0.0)),
            'repairable_ou_ratio': float(layer_info.get('coverage_ratio', 0.0)),
            'projection_cap': float(layer_info.get('projection_cap', 1.0)),
            'relative_weight_delta': 0.0,
        })
    delta_by_layer = {
        str(item.get('layer')): float(item.get('relative_weight_delta', 0.0))
        for item in projection_metrics.get('layer_projection_deltas', []) or []
    }
    for row in layer_rows:
        row['relative_weight_delta'] = delta_by_layer.get(row['layer'], 0.0)
    top_damaged_layers = sorted(
        layer_rows,
        key=lambda item: (
            item['relative_weight_delta'],
            item['assignment_error_p95'],
            item['singleton_ratio'],
        ),
        reverse=True,
    )[:5]
    return {
        'projection_strength': float(projection_metrics.get('projection_strength', 0.0)),
        'projected_accuracy': projection_metrics.get('projected_accuracy'),
        'projected_accuracy_drop': projection_metrics.get('projected_accuracy_drop'),
        'top_damaged_layers': top_damaged_layers,
        'model_name': model_name,
        'translate_name': translate_name,
    }


def _update_projection_sanity_report(model_name, translate_name, artifact_dir, projection_metrics, group_information):
    record = _projection_sanity_record(model_name, translate_name, projection_metrics, group_information)
    report_prefix = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_projection_sanity')
    json_path = report_prefix + '.json'
    csv_path = report_prefix + '.csv'
    md_path = report_prefix + '.md'

    existing_records = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            existing_records = loaded
    existing_by_strength = {
        round(float(item.get('projection_strength', 0.0)), 6): item
        for item in existing_records
    }
    existing_by_strength[round(record['projection_strength'], 6)] = record
    merged_records = [existing_by_strength[key] for key in sorted(existing_by_strength.keys())]

    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(merged_records, handle, indent=2, ensure_ascii=False)

    csv_rows = []
    for item in merged_records:
        top_layers = item.get('top_damaged_layers', [])
        csv_rows.append({
            'projection_strength': item.get('projection_strength'),
            'projected_accuracy': item.get('projected_accuracy'),
            'projected_accuracy_drop': item.get('projected_accuracy_drop'),
            'top_damaged_layers': ','.join(layer.get('layer', '') for layer in top_layers),
            'top_damaged_layer_deltas': json.dumps(top_layers, ensure_ascii=False),
        })
    import pandas as pd
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    with open(md_path, 'w', encoding='utf-8') as handle:
        handle.write('# Projection Sanity Sweep\n\n')
        handle.write('| projection_strength | projected_accuracy | projected_accuracy_drop | top_damaged_layers |\n')
        handle.write('| --- | --- | --- | --- |\n')
        for item in merged_records:
            projected_accuracy = item.get('projected_accuracy')
            projected_accuracy_drop = item.get('projected_accuracy_drop')
            handle.write(
                f"| {float(item.get('projection_strength', 0.0)):.4f} | "
                f"{'' if projected_accuracy is None else f'{float(projected_accuracy):.4f}'} | "
                f"{'' if projected_accuracy_drop is None else f'{float(projected_accuracy_drop):.4f}'} | "
                f"{', '.join(layer.get('layer', '') for layer in item.get('top_damaged_layers', []))} |\n"
            )

    return {
        'json_path': json_path,
        'csv_path': csv_path,
        'md_path': md_path,
    }


def resolve_ft_training_config(args, argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    refresh_explicit = _flag_present(argv, '--refresh-epochs')

    selected_preset = args.ft_cost_preset
    if selected_preset == 'none' and args.ft_low_cost:
        selected_preset = 'fast'
    preset_config = ft_cost_presets.get(selected_preset, {})

    if args.ft_end_epoch is not None:
        ft_end_epoch = args.ft_end_epoch
    elif preset_config:
        ft_end_epoch = preset_config['end_epoch']
    else:
        ft_end_epoch = epoches

    if args.ft_reg_interval is not None:
        ft_reg_interval = args.ft_reg_interval
    elif preset_config:
        ft_reg_interval = preset_config['reg_interval']
    else:
        ft_reg_interval = ft_reg_interval_default

    if args.ft_reg_min_coverage is not None:
        ft_reg_min_coverage = args.ft_reg_min_coverage
    elif preset_config:
        ft_reg_min_coverage = preset_config['reg_min_coverage']
    else:
        ft_reg_min_coverage = ft_reg_min_coverage_default

    if args.ft_reg_min_groups is not None:
        ft_reg_min_groups = args.ft_reg_min_groups
    elif preset_config:
        ft_reg_min_groups = preset_config['reg_min_groups']
    else:
        ft_reg_min_groups = ft_reg_min_groups_default

    if refresh_explicit:
        refresh_epoch_values = parse_epoch_list(args.refresh_epochs, allow_empty=True)
    elif preset_config:
        refresh_epoch_values = list(preset_config['refresh_epochs'])
    else:
        refresh_epoch_values = list(ft_group_refresh_epoch)

    if ft_end_epoch <= args.base_checkpoint_epoch:
        raise ValueError('--ft-end-epoch must be greater than --base-checkpoint-epoch')
    if ft_reg_interval <= 0:
        raise ValueError('--ft-reg-interval must be positive')
    if ft_reg_min_coverage < 0.0 or ft_reg_min_coverage > 1.0:
        raise ValueError('--ft-reg-min-coverage must be within [0, 1]')
    if ft_reg_min_groups <= 0:
        raise ValueError('--ft-reg-min-groups must be positive')

    translate_epoch_values = normalize_schedule_epochs(
        parse_epoch_list(args.translate_epochs, allow_empty=False),
        checkpoint_epoch=args.base_checkpoint_epoch,
        end_epoch=ft_end_epoch,
        ensure_final=True,
    )
    refresh_epoch_values = normalize_schedule_epochs(
        refresh_epoch_values,
        checkpoint_epoch=args.base_checkpoint_epoch,
        end_epoch=ft_end_epoch,
        ensure_final=bool(preset_config) and not refresh_explicit,
    )

    return {
        'selected_preset': selected_preset,
        'ft_low_cost': args.ft_low_cost,
        'ft_end_epoch': ft_end_epoch,
        'ft_reg_interval': ft_reg_interval,
        'ft_reg_min_coverage': ft_reg_min_coverage,
        'ft_reg_min_groups': ft_reg_min_groups,
        'ft_translate_epoch': translate_epoch_values,
        'ft_group_refresh_epoch': refresh_epoch_values,
        'ft_reg_boost_after_refresh': args.ft_reg_boost_after_refresh,
    }


def resolve_artifact_dir(args):
    if args.artifact_dir:
        artifact_dir = os.path.abspath(args.artifact_dir)
    elif args.run_tag:
        artifact_dir = os.path.abspath(os.path.join('results', 'ft_runs', args.model, args.translate, args.run_tag, 'artifacts'))
    else:
        artifact_dir = os.path.abspath('.')
    os.makedirs(artifact_dir, exist_ok=True)
    return artifact_dir


def prepare_ft_artifacts(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel,
                         kernel_size, channel_number, pattern_value_number, mask, map_information,
                         multiple_relationship_information, reuse_ratio_information, ft_layer_enabled,
                         ft_group_target_ratio, ft_target_group_size, ft_similarity_threshold,
                         checkpoint_epoch, force_rebuild=False, artifact_dir='.',
                         ft_grouping_mode='ftscore', ft_budget_config=None,
                         ft_mask_density_sweep=False, ft_mask_keep_ratios=None,
                         ft_target_coverage=None, ft_prefer_sparser_mask=False,
                         ft_score_singleton_penalty=0.18, ft_score_zero_scale_penalty=0.12):
    group_information = {layer: None for layer in weight_name}
    os.makedirs(artifact_dir, exist_ok=True)

    mask_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_mask.pkl')
    map_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_map_information.pkl')
    mult_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_multiple_relationship_information.pkl')
    coverage_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_coverage_ratio_information.pkl')
    reuse_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_reuse_ratio_information.pkl')
    group_file = os.path.join(artifact_dir, f'model_{model_name}_{translate_name}_group_information.pkl')

    cache_files = [mask_file, map_file, mult_file, group_file]
    need_regenerate = force_rebuild or any(not os.path.exists(file_path) for file_path in cache_files)
    if not need_regenerate and not (os.path.exists(coverage_file) or os.path.exists(reuse_file)):
        need_regenerate = True

    def _flush_artifacts():
        with open(mask_file, 'wb') as f_mask:
            pkl.dump(mask, f_mask, pkl.HIGHEST_PROTOCOL)
        with open(map_file, 'wb') as f_map:
            pkl.dump(map_information, f_map, pkl.HIGHEST_PROTOCOL)
        with open(mult_file, 'wb') as f_mult:
            pkl.dump(multiple_relationship_information, f_mult, pkl.HIGHEST_PROTOCOL)
        with open(coverage_file, 'wb') as f_coverage:
            pkl.dump(reuse_ratio_information, f_coverage, pkl.HIGHEST_PROTOCOL)
        with open(reuse_file, 'wb') as f_reuse:
            pkl.dump(reuse_ratio_information, f_reuse, pkl.HIGHEST_PROTOCOL)
        with open(group_file, 'wb') as f_group:
            pkl.dump(group_information, f_group, pkl.HIGHEST_PROTOCOL)

    if need_regenerate:
        if force_rebuild:
            print('[FT artifacts] force rebuild enabled; ignoring cached artifacts')
        checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth')
        model_original.load_state_dict(checkpoint['model'])

        for i in range(0, len(weight_name)):
            print('[FT artifacts] layer {} build start ({}/{})'.format(weight_name[i], i + 1, len(weight_name)))
            if not ft_layer_enabled[i]:
                group_information[weight_name[i]] = {
                    'layer_name': weight_name[i],
                    'group_count': 0,
                    'block_count': layer_in_channel[i],
                    'ou_count': layer_in_channel[i] * layer_out_channel[i],
                    'repairable_block_count': 0,
                    'repairable_ou_count': 0,
                    'coverage_ratio': 0.0,
                    'block_coverage_ratio': 0.0,
                    'singleton_group_count': 0,
                    'singleton_ratio': 0.0,
                    'avg_group_size': 0.0,
                    'exact_group_count': 0,
                    'scaled_group_count': 0,
                    'exact_group_ratio': 0.0,
                    'scaled_group_ratio': 0.0,
                    'groups': [],
                }
                _flush_artifacts()
                print('[FT artifacts] layer {} skipped (ft_layer_enabled=False)'.format(weight_name[i]))
                continue

            target_group_size = ft_target_group_size[i]
            similarity_threshold = ft_similarity_threshold[i]
            selected_group_outputs = None
            use_codebook_budgeted = ft_grouping_mode == 'codebook_budgeted'
            use_budgeted_mask_family = (
                ft_grouping_mode == 'budgeted'
                and not ft_mask_density_sweep
                and not ft_mask_keep_ratios
                and ft_target_coverage is None
                and not ft_prefer_sparser_mask
            )
            if use_codebook_budgeted:
                mask[weight_name[i]], group_seed_info, selected_group_outputs = ft_codebook_budgeted_select_mask_candidate(
                    model=model_original,
                    weight_name=weight_name[i],
                    in_channel=layer_in_channel[i],
                    out_channel=layer_out_channel[i],
                    kernel_size=kernel_size[i],
                    channel_number=channel_number[i],
                    pattern_value_number=pattern_value_number[i],
                    pattern_shape_number=pattern_shape_number,
                    OU_size=OU_size,
                    min_group_size=ft_min_group_size,
                    exact_threshold=ft_exact_similarity_threshold,
                    scale_candidates=ft_scale_candidates,
                    budget_config=ft_budget_config,
                )
            elif use_budgeted_mask_family:
                mask[weight_name[i]], group_seed_info, selected_group_outputs = ft_budgeted_select_mask_candidate(
                    model=model_original,
                    weight_name=weight_name[i],
                    in_channel=layer_in_channel[i],
                    out_channel=layer_out_channel[i],
                    kernel_size=kernel_size[i],
                    channel_number=channel_number[i],
                    pattern_value_number=pattern_value_number[i],
                    pattern_shape_number=pattern_shape_number,
                    OU_size=OU_size,
                    min_group_size=ft_min_group_size,
                    exact_threshold=ft_exact_similarity_threshold,
                    scale_candidates=ft_scale_candidates,
                    budget_config=ft_budget_config,
                )
            else:
                mask[weight_name[i]], group_seed_info = ft_group_score_mask(
                    model=model_original,
                    weight_name=weight_name[i],
                    in_channel=layer_in_channel[i],
                    out_channel=layer_out_channel[i],
                    kernel_size=kernel_size[i],
                    channel_number=channel_number[i],
                    pattern_value_number=pattern_value_number[i],
                    pattern_shape_number=pattern_shape_number,
                    OU_size=OU_size,
                    target_group_size=target_group_size,
                    sim_threshold=similarity_threshold,
                    mask_density_sweep=ft_mask_density_sweep,
                    mask_keep_ratios=ft_mask_keep_ratios,
                    target_coverage=ft_target_coverage,
                    prefer_sparser_mask=ft_prefer_sparser_mask,
                    singleton_penalty=ft_score_singleton_penalty,
                    zero_scale_penalty=ft_score_zero_scale_penalty,
                )
            print('[FTScore seed] {} strategy={} coverage={:.4f} avg_group_size={:.2f} fallback={}'.format(
                weight_name[i],
                group_seed_info.get('selected_strategy'),
                group_seed_info.get('estimated_coverage', 0.0),
                group_seed_info.get('estimated_avg_group_size', 0.0),
                group_seed_info.get('fallback_used', False),
            ))
            if selected_group_outputs is not None:
                map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]], group_information[weight_name[i]] = selected_group_outputs
            else:
                if ft_grouping_mode == 'budgeted':
                    grouping_fn = ft_budgeted_group_translate
                elif ft_grouping_mode == 'codebook_budgeted':
                    grouping_fn = ft_codebook_budgeted_translate
                else:
                    grouping_fn = ft_group_cluster_translate
                grouping_outputs = grouping_fn(
                    model=model_original,
                    in_channel=layer_in_channel[i],
                    out_channel=layer_out_channel[i],
                    weight_name=weight_name[i],
                    kernel_size=kernel_size[i],
                    channel_number=channel_number[i],
                    mask=mask[weight_name[i]],
                    min_group_size=ft_min_group_size,
                    target_group_size=target_group_size,
                    sim_threshold=similarity_threshold,
                    exact_threshold=ft_exact_similarity_threshold,
                    scale_candidates=ft_scale_candidates,
                    grouping_mode=ft_grouping_mode,
                    budget_config=ft_budget_config,
                )
                if ft_grouping_mode == 'codebook_budgeted':
                    mask[weight_name[i]], map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]], group_information[weight_name[i]] = grouping_outputs
                else:
                    map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]], group_information[weight_name[i]] = grouping_outputs
            group_information[weight_name[i]]['target_ratio'] = ft_group_target_ratio[i]
            group_information[weight_name[i]]['seed_info'] = group_seed_info
            group_information[weight_name[i]]['grouping_mode'] = ft_grouping_mode
            _flush_artifacts()
            print('[FT artifacts] layer {} build done mode={} coverage={:.4f} groups={} singleton={:.4f}'.format(
                weight_name[i],
                ft_grouping_mode,
                float(group_information[weight_name[i]].get('coverage_ratio', 0.0)),
                int(group_information[weight_name[i]].get('group_count', 0)),
                float(group_information[weight_name[i]].get('singleton_ratio', 0.0)),
            ))
    else:
        with open(mask_file, 'rb') as f_mask:
            mask = pkl.load(f_mask)
        with open(map_file, 'rb') as f_map:
            map_information = pkl.load(f_map)
        with open(mult_file, 'rb') as f_mult:
            multiple_relationship_information = pkl.load(f_mult)
        if os.path.exists(coverage_file):
            with open(coverage_file, 'rb') as f_coverage:
                reuse_ratio_information = pkl.load(f_coverage)
        elif os.path.exists(reuse_file):
            with open(reuse_file, 'rb') as f_reuse:
                reuse_ratio_information = pkl.load(f_reuse)
        else:
            raise FileNotFoundError(f'missing both {coverage_file} and {reuse_file}')
        if os.path.exists(group_file):
            with open(group_file, 'rb') as f_group:
                group_information = pkl.load(f_group)
        else:
            print(f'Warning: missing {group_file}, simulator will fallback to map_information.')

    write_mask_sweep_report(model_name, translate_name, group_information, output_dir=artifact_dir)

    return mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information


def save_ft_build_only_projection(model, model_name, translate_name, weight_name, mask, group_information, checkpoint_epoch,
                                  artifact_dir='.', projection_strength=1.0, evaluate_projected=False,
                                  device='cpu', test_loader=None, original_accuracy=None):
    os.makedirs(artifact_dir, exist_ok=True)
    checkpoint_path = 'model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    original_weights = {
        layer_name: model.state_dict()[layer_name].detach().clone()
        for layer_name in weight_name
        if layer_name in model.state_dict()
    }
    apply_ft_group_projection(model, weight_name, mask, group_information, projection_strength=projection_strength)
    projected_parameters_path = os.path.join(artifact_dir, 'model_' + model_name + '_' + translate_name + '_projected_parameters.pth')
    after_translate_path = os.path.join(artifact_dir, 'model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth')
    torch.save(model.state_dict(), projected_parameters_path)
    torch.save(model.state_dict(), after_translate_path)
    parameters_to_txt(model, model_name, translate_name, output_dir=artifact_dir)
    projection_metrics = {
        'projection_strength': float(projection_strength),
        'projected_parameters_path': projected_parameters_path,
        'after_translate_parameters_path': after_translate_path,
    }
    layer_projection_deltas = []
    for layer_name in weight_name:
        if layer_name not in original_weights or layer_name not in model.state_dict():
            continue
        original_layer = original_weights[layer_name].float()
        projected_layer = model.state_dict()[layer_name].detach().float()
        original_norm = float(torch.norm(original_layer.reshape(-1), p=2).item())
        delta_norm = float(torch.norm((projected_layer - original_layer).reshape(-1), p=2).item())
        layer_projection_deltas.append({
            'layer': layer_name,
            'relative_weight_delta': float(delta_norm / (original_norm + 1e-8)),
            'projection_cap': float((group_information.get(layer_name) or {}).get('projection_cap', 1.0)),
            'assignment_error_p95': float((group_information.get(layer_name) or {}).get('assignment_error_p95', 0.0)),
            'singleton_ratio': float((group_information.get(layer_name) or {}).get('singleton_ratio', 0.0)),
        })
    projection_metrics['layer_projection_deltas'] = layer_projection_deltas
    if evaluate_projected and test_loader is not None:
        projected_accuracy, projected_loss = test(model, device, test_loader)
        projection_metrics['projected_accuracy'] = float(projected_accuracy)
        projection_metrics['projected_loss'] = float(projected_loss)
        if original_accuracy is not None:
            projection_metrics['projected_accuracy_drop'] = float(original_accuracy - projected_accuracy)
        else:
            projection_metrics['projected_accuracy_drop'] = None
        print('[FT build-only] projected_accuracy={:.4f} projected_loss={:.4f} projected_accuracy_drop={}'.format(
            projected_accuracy,
            projected_loss,
            'n/a' if projection_metrics['projected_accuracy_drop'] is None else f"{projection_metrics['projected_accuracy_drop']:.4f}",
        ))
    metrics_path = os.path.join(artifact_dir, 'model_' + model_name + '_' + translate_name + '_projection_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as handle:
        json.dump(projection_metrics, handle, indent=2, ensure_ascii=False)
    sanity_report_paths = _update_projection_sanity_report(
        model_name=model_name,
        translate_name=translate_name,
        artifact_dir=artifact_dir,
        projection_metrics=projection_metrics,
        group_information=group_information,
    )
    print('[FT build-only] projection_sanity_report csv={} json={} md={}'.format(
        sanity_report_paths['csv_path'],
        sanity_report_paths['json_path'],
        sanity_report_paths['md_path'],
    ))
    print('[FT build-only] saved projected parameters from epoch {} -> {}'.format(checkpoint_epoch, artifact_dir))


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model
    translate_name = args.translate
    build_only = args.build_only
    force_rebuild = args.force_rebuild
    base_checkpoint_epoch = args.base_checkpoint_epoch
    ft_config = resolve_ft_training_config(args)
    ft_low_cost = ft_config['ft_low_cost']
    ft_end_epoch = ft_config['ft_end_epoch']
    ft_reg_interval = ft_config['ft_reg_interval']
    ft_reg_min_coverage = ft_config['ft_reg_min_coverage']
    ft_reg_min_groups = ft_config['ft_reg_min_groups']
    ft_translate_epoch = ft_config['ft_translate_epoch']
    ft_group_refresh_epoch = ft_config['ft_group_refresh_epoch']
    ft_reg_boost_after_refresh = ft_config['ft_reg_boost_after_refresh']
    ft_mask_keep_ratios = parse_float_list(args.ft_mask_keep_ratios)
    if args.ft_target_coverage is not None and not (0.0 <= args.ft_target_coverage <= 1.0):
        raise ValueError('--ft-target-coverage must be within [0, 1]')
    artifact_dir = resolve_artifact_dir(args)
    ft_budget_config = build_budgeted_grouping_config(args, translate_name)
    ft_codebook_adapt_config = build_codebook_adaptation_config(
        args=args,
        translate_name=translate_name,
        checkpoint_epoch=base_checkpoint_epoch,
        ft_training_config=ft_config,
        grouping_mode=ft_budget_config['grouping_mode'],
    )
    ft_end_epoch = ft_codebook_adapt_config['effective_end_epoch']
    ft_translate_epoch = ft_codebook_adapt_config['effective_translate_epochs']
    ft_group_refresh_epoch = ft_codebook_adapt_config['effective_refresh_epochs']
    ft_reg_interval = ft_codebook_adapt_config['effective_reg_interval']
    print('[FT schedule] preset={} low_cost={} end_epoch={} translate_epochs={} refresh_epochs={} reg_interval={} reg_min_coverage={:.3f} reg_min_groups={} reg_boost_after_refresh={} artifact_dir={}'.format(
        ft_config['selected_preset'],
        ft_low_cost,
        ft_end_epoch,
        ft_translate_epoch,
        ft_group_refresh_epoch,
        ft_reg_interval,
        ft_reg_min_coverage,
        ft_reg_min_groups,
        ft_reg_boost_after_refresh,
        artifact_dir,
    ))
    print('[FT score] mask_density_sweep={} keep_ratios={} target_coverage={} prefer_sparser_mask={} singleton_penalty={:.3f} zero_scale_penalty={:.3f}'.format(
        args.ft_mask_density_sweep,
        ft_mask_keep_ratios,
        args.ft_target_coverage,
        args.ft_prefer_sparser_mask,
        args.ft_score_singleton_penalty,
        args.ft_score_zero_scale_penalty,
    ))
    print('[FT grouping] mode={} target_coverage={:.3f} max_singleton={:.3f} min_avg_group_size={:.3f} prototype_budget_ratio={:.3f} prototype_budget=[{}, {}] relax_threshold={:.3f} max_scale_error={:.3f} bucket_mode={} mask_family={} keep_ratios={} codebook_size={} codebook_keep_counts={} codebook_source={} codebook_assign={} force_assignment={} max_singleton_error={:.3f} projection_strength={:.3f} evaluate_projected={} normalize_vectors={} prototype_space={} layer_config={}'.format(
        ft_budget_config['grouping_mode'],
        ft_budget_config['target_coverage'],
        ft_budget_config['max_singleton'],
        ft_budget_config['min_avg_group_size'],
        ft_budget_config['prototype_budget_ratio'],
        ft_budget_config['prototype_budget_min'],
        ft_budget_config['prototype_budget_max'],
        ft_budget_config['relax_threshold'],
        ft_budget_config['max_scale_error'],
        ft_budget_config['bucket_mode'],
        ft_budget_config['mask_family'],
        ft_budget_config['mask_keep_ratios'],
        ft_budget_config['mask_codebook_size'],
        ft_budget_config['mask_codebook_keep_counts'],
        ft_budget_config['mask_codebook_source'],
        ft_budget_config['mask_codebook_assign'],
        ft_budget_config['force_prototype_assignment'],
        ft_budget_config['max_singleton_error'],
        ft_budget_config['projection_strength'],
        ft_budget_config['evaluate_projected'],
        ft_budget_config['normalize_prototype_vectors'],
        ft_budget_config['prototype_space'],
        ft_budget_config['layer_config_path'] or '-',
    ))
    print('[FT adapt] enabled={} codebook_adapt_epochs={} projection_ramp_start={:.3f} projection_ramp_end={:.3f} projection_ramp_epochs={} projection_loss_lambda={:.6f} projection_reg_max_links={} freeze_grouping={} refresh_epochs={}'.format(
        ft_codebook_adapt_config['enabled'],
        ft_codebook_adapt_config['codebook_adapt_epochs'],
        ft_codebook_adapt_config['projection_ramp_start'],
        ft_codebook_adapt_config['projection_ramp_end'],
        ft_codebook_adapt_config['projection_ramp_epochs'],
        ft_codebook_adapt_config['projection_loss_lambda'],
        ft_codebook_adapt_config['projection_reg_max_links'],
        ft_codebook_adapt_config['codebook_freeze_grouping'],
        ft_group_refresh_epoch,
    ))
    print('[FT adapt objective] legacy_regularization={} effective_reg_interval={}'.format(
        ft_codebook_adapt_config['codebook_use_legacy_regularization'],
        ft_reg_interval,
    ))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = True  # 不改变给定的神经网络结构的情况下，大大提升其训练和预测的速度
    train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集


    # 创建并训练模型
    if model_name == 'Vgg16':
        model_original = Vgg16(num_classes)
        model_original = model_original.to(device)
        kernel_size = [3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       1, 1, 1]
        layer_in_channel = [3, 64, 64, 128, 128, 256, 256,
                            256, 512, 512, 512, 512, 512,
                            512, 4096, 4096]
        layer_out_channel = [64, 64, 128, 128, 256, 256, 256,
                             512, 512, 512, 512, 512, 512,
                             4096, 4096, num_classes]
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'fc1.weight', 'fc2.weight', 'fc3.weight']
        optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=0)  # 创建优化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

        if not os.path.exists('model_' + model_name + '_original_parameters.pth'):
            train(model_original, model_name, weight_decay_1, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch[0])  # 训练模型
        model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
        original_accuracy, _ = test(model_original, device, test_loader)  # 获得原始模型的准确率
        print(original_accuracy)

        pattern_value_number = [OU_size, OU_size, OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                OU_size, OU_size, OU_size, OU_size, OU_size,
                                1, 1, 1]
        channel_number = [1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          OU_size, OU_size, OU_size]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0]

        if 'shape' in translate_name:
            pattern_value_number = [8, 8, 8, 4, 4, 4, 4,
                                    4, 2, 2, 2, 2, 2,
                                    1, 1, 1]
            channel_number = [1, 1, 1, 2, 2, 2, 2,
                              2, 4, 4, 4, 4, 4,
                              4 * OU_size, 4 * OU_size, OU_size]

        if 'structure' in translate_name:
            best_keep_ratio = [1.0, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.25, 0.25, 1.0]

        if 'ORC' in translate_name:
            best_keep_ratio = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.05, 0.05, 1.0]

        if 'value_identical' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.75, 0.75, 0.75, 0.75, 0.75,
                               0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                               0.75, 0.75, 1.0]

        if 'value_similar' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.5, 0.5, 0.25, 0.25, 0.25,
                               0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                               0.03125, 0.03125, 1.0]

        # 创建剪枝矩阵
        value_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        mask = dict(zip(weight_name, value_list))
        for i in range(0, len(weight_name)):
            if 'fc' in weight_name[i]:
                mask[weight_name[i]] = torch.ones(layer_out_channel[i], layer_in_channel[i])

        # 创建weight-pattern重用映射矩阵
        layer_map_list = [torch.ones((layer_in_channel[i], layer_out_channel[i], 2)) for i in range(0, len(weight_name))]
        map_information = dict(zip(weight_name, layer_map_list))

        # 创建weight-pattern倍数关系矩阵
        layer_multiple_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))

        # 记录每一层weight-pattern的重用率
        layer_reuse_ratio_list = [torch.zeros(1) for i in range(0, len(weight_name))]
        reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))
        ft_layer_enabled = [True] * len(weight_name)
        ft_group_target_ratio = [0.75] * len(weight_name)
        ft_target_group_size = [ft_target_group_size_default] * len(weight_name)
        ft_similarity_threshold = [ft_similarity_threshold_default] * len(weight_name)
        group_information = {layer: None for layer in weight_name}

        if is_ft_grouping_translate(translate_name):
            mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                model_original=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                layer_in_channel=layer_in_channel,
                layer_out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                pattern_value_number=pattern_value_number,
                mask=mask,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_layer_enabled=ft_layer_enabled,
                ft_group_target_ratio=ft_group_target_ratio,
                ft_target_group_size=ft_target_group_size,
                ft_similarity_threshold=ft_similarity_threshold,
                checkpoint_epoch=base_checkpoint_epoch,
                force_rebuild=force_rebuild,
                artifact_dir=artifact_dir,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
            )
            if build_only:
                save_ft_build_only_projection(
                    model=model_original,
                    model_name=model_name,
                    translate_name=translate_name,
                    weight_name=weight_name,
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=base_checkpoint_epoch,
                    artifact_dir=artifact_dir,
                    projection_strength=ft_budget_config['projection_strength'],
                    evaluate_projected=ft_budget_config['evaluate_projected'],
                    device=device,
                    test_loader=test_loader,
                    original_accuracy=original_accuracy,
                )
                raise SystemExit(0)

        if 'structure_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_structure_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_structure_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'ORC_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_ORC_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_ORC_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'weight_pattern_shape' in translate_name:
            if not os.path.exists('model_' + model_name + '_pattern_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    mask[weight_name[i]] = get_shape_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], channel_number[i], pattern_value_number[i], pattern_shape_number, OU_size)  # 计算剪枝矩阵
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'value_identical' in translate_name:
            if not os.path.exists('model_' + model_name + '_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if 'value_similar' in translate_name:
            if not os.path.exists('model_' + model_name + '_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'structure_pruning_and_weight_pattern_value_identical_translate':
            if not os.path.exists('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = structure_and_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'weight_pattern_shape_and_value_similar_translate':
            if not os.path.exists('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_shape_and_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if is_ft_grouping_translate(translate_name):
            ft_group_translate_train(
                model=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                in_channel=layer_in_channel,
                out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                ft_layer_enabled=ft_layer_enabled,
                mask=mask,
                group_information=group_information,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_mask_lambda=ft_mask_lambda,
                ft_proto_lambda=ft_proto_lambda,
                ft_balance_lambda=ft_balance_lambda,
                ft_sep_lambda=ft_sep_lambda,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                test_loader=test_loader,
                max_epoches=ft_end_epoch,
                translate_epoch=ft_translate_epoch,
                checkpoint_epoch=base_checkpoint_epoch,
                group_refresh_epoch=ft_group_refresh_epoch,
                min_group_size=ft_min_group_size,
                target_group_size=ft_target_group_size,
                sim_threshold=ft_similarity_threshold,
                exact_threshold=ft_exact_similarity_threshold,
                scale_candidates=ft_scale_candidates,
                pattern_value_number=pattern_value_number,
                pattern_shape_number=pattern_shape_number,
                OU_size=OU_size,
                ft_reg_interval=ft_reg_interval,
                ft_reg_min_coverage=ft_reg_min_coverage,
                ft_reg_min_groups=ft_reg_min_groups,
                ft_reg_boost_after_refresh=ft_reg_boost_after_refresh,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
                ft_projection_ramp_start=ft_codebook_adapt_config['projection_ramp_start'],
                ft_projection_ramp_end=ft_codebook_adapt_config['projection_ramp_end'],
                ft_projection_ramp_epochs=ft_codebook_adapt_config['projection_ramp_epochs'],
                ft_projection_loss_lambda=ft_codebook_adapt_config['projection_loss_lambda'],
                ft_projection_reg_max_links=ft_codebook_adapt_config['projection_reg_max_links'],
                ft_codebook_freeze_grouping=ft_codebook_adapt_config['codebook_freeze_grouping'],
                ft_codebook_use_legacy_regularization=ft_codebook_adapt_config['codebook_use_legacy_regularization'],
                output_dir=artifact_dir,
            )
        else:
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)


    # 创建并训练模型
    if model_name == 'Res18':
        model_original = Res18(num_classes)
        model_original = model_original.to(device)
        kernel_size = [3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3,
                       1, 1, 1,
                       1]
        layer_in_channel = [3, 64, 64, 64, 64, 64,
                            128, 128, 128, 128, 256, 256,
                            256, 256, 512, 512, 512,
                            64, 128, 256,
                            512]
        layer_out_channel = [64, 64, 64, 64, 64, 128,
                             128, 128, 128, 256, 256, 256,
                             256, 512, 512, 512, 512,
                             128, 256, 512,
                             num_classes]
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight',
                       'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight',
                       'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']

        optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=0)  # 创建优化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

        if not os.path.exists('model_' + model_name + '_original_parameters.pth'):
            train(model_original, model_name, weight_decay_1, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch[0])  # 训练模型
        model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
        original_accuracy, _ = test(model_original, device, test_loader)  # 获得原始模型的准确率
        print(original_accuracy)

        pattern_value_number = [OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                OU_size, OU_size, OU_size, OU_size, OU_size,
                                1, 1, 1,
                                1]
        channel_number = [1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1,
                          OU_size, OU_size, OU_size,
                          OU_size]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0]

        if 'shape' in translate_name:
            pattern_value_number = [8, 4, 4, 4, 4, 4,
                                    4, 4, 4, 4, 4, 4,
                                    4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]
            channel_number = [1, 2, 2, 2, 2, 2,
                              2, 2, 2, 2, 2, 2,
                              2, 2, 4, 4, 4,
                              OU_size, OU_size, OU_size,
                              OU_size]

        if 'structure' in translate_name:
            best_keep_ratio = [1.0, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65,
                               1.0]

        if 'ORC' in translate_name:
            best_keep_ratio = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2,
                               1.0]

        if 'value_identical' in translate_name:
            best_keep_ratio = [1.0, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6,
                               1.0]

        if 'value_similar' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.75, 0.75, 0.75, 0.375,
                               0.375, 0.375, 0.375, 0.1875, 0.1875, 0.1875,
                               0.1875, 0.09375, 0.09375, 0.09375, 0.09375,
                               0.375, 0.1875, 0.09375,
                               1.0]

        # 创建剪枝矩阵
        value_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        mask = dict(zip(weight_name, value_list))
        for i in range(0, len(weight_name)):
            if 'fc' in weight_name[i]:
                mask[weight_name[i]] = torch.ones(layer_out_channel[i], layer_in_channel[i])

        # 创建weight-pattern重用映射矩阵
        layer_map_list = [torch.ones((layer_in_channel[i], layer_out_channel[i], 2)) for i in range(0, len(weight_name))]
        map_information = dict(zip(weight_name, layer_map_list))

        # 创建weight-pattern倍数关系矩阵
        layer_multiple_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))

        # 记录每一层weight-pattern的重用率
        layer_reuse_ratio_list = [torch.zeros(1) for i in range(0, len(weight_name))]
        reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))
        ft_layer_enabled = [True] * len(weight_name)
        ft_group_target_ratio = [0.75] * len(weight_name)
        ft_target_group_size = [ft_target_group_size_default] * len(weight_name)
        ft_similarity_threshold = [ft_similarity_threshold_default] * len(weight_name)
        group_information = {layer: None for layer in weight_name}

        if is_ft_grouping_translate(translate_name):
            mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                model_original=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                layer_in_channel=layer_in_channel,
                layer_out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                pattern_value_number=pattern_value_number,
                mask=mask,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_layer_enabled=ft_layer_enabled,
                ft_group_target_ratio=ft_group_target_ratio,
                ft_target_group_size=ft_target_group_size,
                ft_similarity_threshold=ft_similarity_threshold,
                checkpoint_epoch=base_checkpoint_epoch,
                force_rebuild=force_rebuild,
                artifact_dir=artifact_dir,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
            )
            if build_only:
                save_ft_build_only_projection(
                    model=model_original,
                    model_name=model_name,
                    translate_name=translate_name,
                    weight_name=weight_name,
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=base_checkpoint_epoch,
                    artifact_dir=artifact_dir,
                    projection_strength=ft_budget_config['projection_strength'],
                    evaluate_projected=ft_budget_config['evaluate_projected'],
                    device=device,
                    test_loader=test_loader,
                    original_accuracy=original_accuracy,
                )
                raise SystemExit(0)

        if 'structure_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_structure_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_structure_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'ORC_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_ORC_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_ORC_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'weight_pattern_shape' in translate_name:
            if not os.path.exists('model_' + model_name + '_pattern_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    mask[weight_name[i]] = get_shape_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], channel_number[i], pattern_value_number[i], pattern_shape_number, OU_size)  # 计算剪枝矩阵
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'value_identical' in translate_name:
            if not os.path.exists('model_' + model_name + '_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if 'value_similar' in translate_name:
            if not os.path.exists('model_' + model_name + '_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'structure_pruning_and_weight_pattern_value_identical_translate':
            if not os.path.exists('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = structure_and_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'weight_pattern_shape_and_value_similar_translate':
            if not os.path.exists('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_shape_and_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if is_ft_grouping_translate(translate_name):
            ft_group_translate_train(
                model=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                in_channel=layer_in_channel,
                out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                ft_layer_enabled=ft_layer_enabled,
                mask=mask,
                group_information=group_information,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_mask_lambda=ft_mask_lambda,
                ft_proto_lambda=ft_proto_lambda,
                ft_balance_lambda=ft_balance_lambda,
                ft_sep_lambda=ft_sep_lambda,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                test_loader=test_loader,
                max_epoches=ft_end_epoch,
                translate_epoch=ft_translate_epoch,
                checkpoint_epoch=base_checkpoint_epoch,
                group_refresh_epoch=ft_group_refresh_epoch,
                min_group_size=ft_min_group_size,
                target_group_size=ft_target_group_size,
                sim_threshold=ft_similarity_threshold,
                exact_threshold=ft_exact_similarity_threshold,
                scale_candidates=ft_scale_candidates,
                pattern_value_number=pattern_value_number,
                pattern_shape_number=pattern_shape_number,
                OU_size=OU_size,
                ft_reg_interval=ft_reg_interval,
                ft_reg_min_coverage=ft_reg_min_coverage,
                ft_reg_min_groups=ft_reg_min_groups,
                ft_reg_boost_after_refresh=ft_reg_boost_after_refresh,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
                ft_projection_ramp_start=ft_codebook_adapt_config['projection_ramp_start'],
                ft_projection_ramp_end=ft_codebook_adapt_config['projection_ramp_end'],
                ft_projection_ramp_epochs=ft_codebook_adapt_config['projection_ramp_epochs'],
                ft_projection_loss_lambda=ft_codebook_adapt_config['projection_loss_lambda'],
                ft_projection_reg_max_links=ft_codebook_adapt_config['projection_reg_max_links'],
                ft_codebook_freeze_grouping=ft_codebook_adapt_config['codebook_freeze_grouping'],
                ft_codebook_use_legacy_regularization=ft_codebook_adapt_config['codebook_use_legacy_regularization'],
                output_dir=artifact_dir,
            )
        else:
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)


    # 创建并训练模型
    if model_name == 'Res50':
        model_original = Res50(num_classes)
        model_original = model_original.to(device)
        kernel_size = [3, 1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1, 1, 3, 1, 1, 3, 1,
                       1, 3, 1,
                       1, 1, 1, 1,
                       1]
        layer_in_channel = [3, 64, 64, 64, 256, 64, 64, 256, 64, 64,
                            256, 128, 128, 512, 128, 128, 512, 128, 128,
                            512, 128, 128, 512, 256, 256, 1024, 256, 256,
                            1024, 256, 256, 1024, 256, 256, 1024, 256, 256,
                            1024, 256, 256, 1024, 512, 512, 2048, 512, 512,
                            2048, 512, 512,
                            64, 256, 512, 1024,
                            2048]
        layer_out_channel = [64, 64, 64, 256, 64, 64, 256, 64, 64, 256,
                             128, 128, 512, 128, 128, 512, 128, 128, 512,
                             128, 128, 512, 256, 256, 1024, 256, 256, 1024,
                             256, 256, 1024, 256, 256, 1024, 256, 256, 1024,
                             256, 256, 1024, 512, 512, 2048, 512, 512, 2048,
                             512, 512, 2048,
                             256, 512, 1024, 2048,
                             num_classes]
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight', 'conv8.weight', 'conv9.weight', 'conv10.weight',
                       'conv11.weight', 'conv12.weight', 'conv13.weight', 'conv14.weight', 'conv15.weight', 'conv16.weight', 'conv17.weight', 'conv18.weight', 'conv19.weight',
                       'conv20.weight', 'conv21.weight', 'conv22.weight', 'conv23.weight', 'conv24.weight', 'conv25.weight', 'conv26.weight', 'conv27.weight', 'conv28.weight',
                       'conv29.weight', 'conv30.weight', 'conv31.weight', 'conv32.weight', 'conv33.weight', 'conv34.weight', 'conv35.weight', 'conv36.weight', 'conv37.weight',
                       'conv38.weight', 'conv39.weight', 'conv40.weight', 'conv41.weight', 'conv42.weight', 'conv43.weight', 'conv44.weight', 'conv45.weight', 'conv46.weight',
                       'conv47.weight', 'conv48.weight', 'conv49.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight', 'shortcut4.weight',
                       'fc.weight']
        optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=0)  # 创建优化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

        if not os.path.exists('model_' + model_name + '_original_parameters.pth'):
            train(model_original, model_name, weight_decay_1, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch[0])  # 训练模型
        model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
        original_accuracy, _ = test(model_original, device, test_loader)  # 获得原始模型的准确率
        print(original_accuracy)

        pattern_value_number = [OU_size, 1, OU_size, 1, 1, OU_size, 1, 1, OU_size, 1,
                                1, OU_size, 1, 1, OU_size, 1, 1, OU_size, 1,
                                1, OU_size, 1, 1, OU_size, 1, 1, OU_size, 1,
                                1, OU_size, 1, 1, OU_size, 1, 1, OU_size, 1,
                                1, OU_size, 1, 1, OU_size, 1, 1, OU_size, 1,
                                1, OU_size, 1,
                                1, 1, 1, 1,
                                1]
        channel_number = [1, OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                          OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                          OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                          OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                          OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                          OU_size, 1, OU_size,
                          OU_size, OU_size, OU_size, OU_size,
                          OU_size]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0,
                           1.0]

        if 'shape' in translate_name:
            pattern_value_number = [8, 1, 8, 1, 1, 8, 1, 1, 8, 1,
                                    1, 4, 1, 1, 4, 1, 1, 4, 1,
                                    1, 4, 1, 1, 4, 1, 1, 4, 1,
                                    1, 4, 1, 1, 4, 1, 1, 4, 1,
                                    1, 4, 1, 1, 2, 1, 1, 2, 1,
                                    1, 2, 1,
                                    1, 1, 1, 1,
                                    1]
            channel_number = [1, OU_size, 1, OU_size, OU_size, 1, OU_size, OU_size, 1, OU_size,
                              OU_size, 2, OU_size, 2 * OU_size, 2, OU_size, 2 * OU_size, 2, OU_size,
                              2 * OU_size, 2, OU_size, 2 * OU_size, 2, OU_size, 2 * OU_size, 2, OU_size,
                              2 * OU_size, 2, OU_size, 2 * OU_size, 2, OU_size, 2 * OU_size, 2, OU_size,
                              2 * OU_size, 2, OU_size, 2 * OU_size, 4, 2 * OU_size, 2 * OU_size, 4, 2 * OU_size,
                              2 * OU_size, 4, 2 * OU_size,
                              OU_size, OU_size, OU_size, OU_size,
                              OU_size]

        if 'structure' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                               0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                               0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                               0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                               0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                               0.75, 0.65, 0.75,
                               0.75, 0.75, 0.75, 0.75,
                               1.0]

        if 'ORC' in translate_name:
            best_keep_ratio = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2,
                               0.2, 0.2, 0.2, 0.2,
                               1.0]

        if 'value_identical' in translate_name:
            best_keep_ratio = [1.0, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6,
                               0.6, 0.6, 0.6, 0.6,
                               1.0]

        if 'value_similar' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.75, 0.25, 0.75, 0.75, 0.25, 0.75, 0.75, 0.25,
                               0.5, 0.5, 0.125, 0.5, 0.5, 0.125, 0.5, 0.5, 0.125,
                               0.5, 0.5, 0.125, 0.25, 0.25, 0.0625, 0.25, 0.25, 0.0625,
                               0.25, 0.25, 0.0625, 0.25, 0.25, 0.0625, 0.25, 0.25, 0.0625,
                               0.25, 0.25, 0.0625, 0.125, 0.125, 0.03125, 0.125, 0.125, 0.03125,
                               0.125, 0.125, 0.03125,
                               0.25, 0.125, 0.125, 0.03125,
                               1.0]

        # 创建剪枝矩阵
        value_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        mask = dict(zip(weight_name, value_list))
        for i in range(0, len(weight_name)):
            if 'fc' in weight_name[i]:
                mask[weight_name[i]] = torch.ones(layer_out_channel[i], layer_in_channel[i])

        # 创建weight-pattern重用映射矩阵
        layer_map_list = [torch.ones((layer_in_channel[i], layer_out_channel[i], 2)) for i in range(0, len(weight_name))]
        map_information = dict(zip(weight_name, layer_map_list))

        # 创建weight-pattern倍数关系矩阵
        layer_multiple_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))

        # 记录每一层weight-pattern的重用率
        layer_reuse_ratio_list = [torch.zeros(1) for i in range(0, len(weight_name))]
        reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))
        ft_layer_enabled = [True] * len(weight_name)
        ft_group_target_ratio = [0.75] * len(weight_name)
        ft_target_group_size = [ft_target_group_size_default] * len(weight_name)
        ft_similarity_threshold = [ft_similarity_threshold_default] * len(weight_name)
        group_information = {layer: None for layer in weight_name}

        if is_ft_grouping_translate(translate_name):
            mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                model_original=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                layer_in_channel=layer_in_channel,
                layer_out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                pattern_value_number=pattern_value_number,
                mask=mask,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_layer_enabled=ft_layer_enabled,
                ft_group_target_ratio=ft_group_target_ratio,
                ft_target_group_size=ft_target_group_size,
                ft_similarity_threshold=ft_similarity_threshold,
                checkpoint_epoch=base_checkpoint_epoch,
                force_rebuild=force_rebuild,
                artifact_dir=artifact_dir,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
            )
            if build_only:
                save_ft_build_only_projection(
                    model=model_original,
                    model_name=model_name,
                    translate_name=translate_name,
                    weight_name=weight_name,
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=base_checkpoint_epoch,
                    artifact_dir=artifact_dir,
                    projection_strength=ft_budget_config['projection_strength'],
                    evaluate_projected=ft_budget_config['evaluate_projected'],
                    device=device,
                    test_loader=test_loader,
                    original_accuracy=original_accuracy,
                )
                raise SystemExit(0)

        if 'structure_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_structure_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_structure_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'ORC_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_ORC_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_ORC_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'weight_pattern_shape' in translate_name:
            if not os.path.exists('model_' + model_name + '_pattern_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    mask[weight_name[i]] = get_shape_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], channel_number[i], pattern_value_number[i], pattern_shape_number, OU_size)  # 计算剪枝矩阵
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'value_identical' in translate_name:
            if not os.path.exists('model_' + model_name + '_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if 'value_similar' in translate_name:
            if not os.path.exists('model_' + model_name + '_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'structure_pruning_and_weight_pattern_value_identical_translate':
            if not os.path.exists('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = structure_and_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'weight_pattern_shape_and_value_similar_translate':
            if not os.path.exists('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_shape_and_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if is_ft_grouping_translate(translate_name):
            ft_group_translate_train(
                model=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                in_channel=layer_in_channel,
                out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                ft_layer_enabled=ft_layer_enabled,
                mask=mask,
                group_information=group_information,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_mask_lambda=ft_mask_lambda,
                ft_proto_lambda=ft_proto_lambda,
                ft_balance_lambda=ft_balance_lambda,
                ft_sep_lambda=ft_sep_lambda,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                test_loader=test_loader,
                max_epoches=ft_end_epoch,
                translate_epoch=ft_translate_epoch,
                checkpoint_epoch=base_checkpoint_epoch,
                group_refresh_epoch=ft_group_refresh_epoch,
                min_group_size=ft_min_group_size,
                target_group_size=ft_target_group_size,
                sim_threshold=ft_similarity_threshold,
                exact_threshold=ft_exact_similarity_threshold,
                scale_candidates=ft_scale_candidates,
                pattern_value_number=pattern_value_number,
                pattern_shape_number=pattern_shape_number,
                OU_size=OU_size,
                ft_reg_interval=ft_reg_interval,
                ft_reg_min_coverage=ft_reg_min_coverage,
                ft_reg_min_groups=ft_reg_min_groups,
                ft_reg_boost_after_refresh=ft_reg_boost_after_refresh,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
                ft_projection_ramp_start=ft_codebook_adapt_config['projection_ramp_start'],
                ft_projection_ramp_end=ft_codebook_adapt_config['projection_ramp_end'],
                ft_projection_ramp_epochs=ft_codebook_adapt_config['projection_ramp_epochs'],
                ft_projection_loss_lambda=ft_codebook_adapt_config['projection_loss_lambda'],
                ft_projection_reg_max_links=ft_codebook_adapt_config['projection_reg_max_links'],
                ft_codebook_freeze_grouping=ft_codebook_adapt_config['codebook_freeze_grouping'],
                ft_codebook_use_legacy_regularization=ft_codebook_adapt_config['codebook_use_legacy_regularization'],
                output_dir=artifact_dir,
            )
        else:
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)


    # 创建并训练模型
    if model_name == 'WRN':
        model_original = WRN(num_classes)
        model_original = model_original.to(device)
        kernel_size = [3, 3, 3, 3, 3, 3, 3,
                       3, 3, 3, 3, 3, 3,
                       1, 1, 1,
                       1]
        layer_in_channel = [3, 16, 128, 128, 128, 128, 256,
                            256, 256, 256, 512, 512, 512,
                            16, 128, 256,
                            512]
        layer_out_channel = [16, 128, 128, 128, 128, 256, 256,
                             256, 256, 512, 512, 512, 512,
                             128, 256, 512,
                             num_classes]
        weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
                       'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
                       'shortcut1.weight', 'shortcut2.weight', 'shortcut3.weight',
                       'fc.weight']
        optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=0)  # 创建优化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

        if not os.path.exists('model_' + model_name + '_original_parameters.pth'):
            train(model_original, model_name, weight_decay_1, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch[0])  # 训练模型
        model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
        original_accuracy, _ = test(model_original, device, test_loader)  # 获得原始模型的准确率
        print(original_accuracy)

        pattern_value_number = [OU_size, OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                1, 1, 1,
                                1]
        channel_number = [1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1,
                          OU_size, OU_size, OU_size,
                          OU_size]
        best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                           1.0, 1.0, 1.0,
                           1.0]

        if 'shape' in translate_name:
            pattern_value_number = [8, 4, 4, 4, 4, 4, 4,
                                    4, 4, 4, 2, 2, 2,
                                    1, 1, 1,
                                    1]
            channel_number = [1, 2, 2, 2, 2, 2, 2,
                              2, 2, 2, 4, 4, 4,
                              OU_size, OU_size, OU_size,
                              OU_size]

        if 'structure' in translate_name:
            best_keep_ratio = [1.0, 1.0, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                               0.65, 0.65, 0.65,
                               1.0]

        if 'ORC' in translate_name:
            best_keep_ratio = [1.0, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                               0.15, 0.15, 0.15, 0.15, 0.15, 0.15,
                               0.15, 0.15, 0.15,
                               1.0]

        if 'value_identical' in translate_name:
            best_keep_ratio = [1.0, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                               0.7, 0.7, 0.7, 0.7, 0.7, 0.7,
                               0.7, 0.7, 0.7,
                               1.0]

        if 'value_similar' in translate_name:
            best_keep_ratio = [1.0, 0.75, 0.75, 0.75, 0.75, 0.375, 0.375,
                               0.375, 0.375, 0.1875, 0.1875, 0.1875, 0.1875,
                               0.75, 0.375, 0.1875,
                               1.0]

        # 创建剪枝矩阵
        value_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        mask = dict(zip(weight_name, value_list))
        for i in range(0, len(weight_name)):
            if 'fc' in weight_name[i]:
                mask[weight_name[i]] = torch.ones(layer_out_channel[i], layer_in_channel[i])

        # 创建weight-pattern重用映射矩阵
        layer_map_list = [torch.ones((layer_in_channel[i], layer_out_channel[i], 2)) for i in range(0, len(weight_name))]
        map_information = dict(zip(weight_name, layer_map_list))

        # 创建weight-pattern倍数关系矩阵
        layer_multiple_list = [torch.ones((layer_out_channel[i], layer_in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
        multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))

        # 记录每一层weight-pattern的重用率
        layer_reuse_ratio_list = [torch.zeros(1) for i in range(0, len(weight_name))]
        reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))
        ft_layer_enabled = [True] * len(weight_name)
        ft_group_target_ratio = [0.75] * len(weight_name)
        ft_target_group_size = [ft_target_group_size_default] * len(weight_name)
        ft_similarity_threshold = [ft_similarity_threshold_default] * len(weight_name)
        group_information = {layer: None for layer in weight_name}

        if is_ft_grouping_translate(translate_name):
            mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information = prepare_ft_artifacts(
                model_original=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                layer_in_channel=layer_in_channel,
                layer_out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                pattern_value_number=pattern_value_number,
                mask=mask,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_layer_enabled=ft_layer_enabled,
                ft_group_target_ratio=ft_group_target_ratio,
                ft_target_group_size=ft_target_group_size,
                ft_similarity_threshold=ft_similarity_threshold,
                checkpoint_epoch=base_checkpoint_epoch,
                force_rebuild=force_rebuild,
                artifact_dir=artifact_dir,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
            )
            if build_only:
                save_ft_build_only_projection(
                    model=model_original,
                    model_name=model_name,
                    translate_name=translate_name,
                    weight_name=weight_name,
                    mask=mask,
                    group_information=group_information,
                    checkpoint_epoch=base_checkpoint_epoch,
                    artifact_dir=artifact_dir,
                    projection_strength=ft_budget_config['projection_strength'],
                    evaluate_projected=ft_budget_config['evaluate_projected'],
                    device=device,
                    test_loader=test_loader,
                    original_accuracy=original_accuracy,
                )
                raise SystemExit(0)

        if 'structure_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_structure_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_structure_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'ORC_pruning' in translate_name:
            if not os.path.exists('model_' + model_name + '_ORC_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    if best_keep_ratio[i] != 1.0:
                        print(weight_name[i])
                        mask[weight_name[i]] = get_ORC_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], best_keep_ratio[i])  # 计算剪枝矩阵
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_ORC_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'weight_pattern_shape' in translate_name:
            if not os.path.exists('model_' + model_name + '_pattern_mask' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    mask[weight_name[i]] = get_shape_mask(model_original, weight_name[i], layer_in_channel[i], layer_out_channel[i], kernel_size[i], channel_number[i], pattern_value_number[i], pattern_shape_number, OU_size)  # 计算剪枝矩阵
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

        if 'value_identical' in translate_name:
            if not os.path.exists('model_' + model_name + '_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_identical_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if 'value_similar' in translate_name:
            if not os.path.exists('model_' + model_name + '_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'structure_pruning_and_weight_pattern_value_identical_translate':
            if not os.path.exists('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = structure_and_value_identical_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_structure_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_structure_and_value_identical_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if translate_name == 'weight_pattern_shape_and_value_similar_translate':
            if not os.path.exists('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if best_keep_ratio[i] != 1.0:
                        map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]] = pattern_shape_and_value_similar_translate(model_original, layer_in_channel[i], layer_out_channel[i], weight_name[i], best_keep_ratio[i], kernel_size[i], channel_number[i], mask[weight_name[i]])  # 计算剪枝矩阵
                        map_information[weight_name[i]] = map_information[weight_name[i]].type(torch.long)
                        print(reuse_ratio_information[weight_name[i]])
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_similar_map_information' + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_multiple_relationship_information' + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_shape_and_value_reuse_ratio_information' + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for i in range(0, len(weight_name)):
                    best_keep_ratio[i] = 1.0 - reuse_ratio_information[weight_name[i]]

        if is_ft_grouping_translate(translate_name):
            ft_group_translate_train(
                model=model_original,
                model_name=model_name,
                translate_name=translate_name,
                weight_name=weight_name,
                in_channel=layer_in_channel,
                out_channel=layer_out_channel,
                kernel_size=kernel_size,
                channel_number=channel_number,
                ft_layer_enabled=ft_layer_enabled,
                mask=mask,
                group_information=group_information,
                map_information=map_information,
                multiple_relationship_information=multiple_relationship_information,
                reuse_ratio_information=reuse_ratio_information,
                ft_mask_lambda=ft_mask_lambda,
                ft_proto_lambda=ft_proto_lambda,
                ft_balance_lambda=ft_balance_lambda,
                ft_sep_lambda=ft_sep_lambda,
                device=device,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                test_loader=test_loader,
                max_epoches=ft_end_epoch,
                translate_epoch=ft_translate_epoch,
                checkpoint_epoch=base_checkpoint_epoch,
                group_refresh_epoch=ft_group_refresh_epoch,
                min_group_size=ft_min_group_size,
                target_group_size=ft_target_group_size,
                sim_threshold=ft_similarity_threshold,
                exact_threshold=ft_exact_similarity_threshold,
                scale_candidates=ft_scale_candidates,
                pattern_value_number=pattern_value_number,
                pattern_shape_number=pattern_shape_number,
                OU_size=OU_size,
                ft_reg_interval=ft_reg_interval,
                ft_reg_min_coverage=ft_reg_min_coverage,
                ft_reg_min_groups=ft_reg_min_groups,
                ft_reg_boost_after_refresh=ft_reg_boost_after_refresh,
                ft_grouping_mode=ft_budget_config['grouping_mode'],
                ft_budget_config=ft_budget_config,
                ft_mask_density_sweep=args.ft_mask_density_sweep,
                ft_mask_keep_ratios=ft_mask_keep_ratios,
                ft_target_coverage=args.ft_target_coverage,
                ft_prefer_sparser_mask=args.ft_prefer_sparser_mask,
                ft_score_singleton_penalty=args.ft_score_singleton_penalty,
                ft_score_zero_scale_penalty=args.ft_score_zero_scale_penalty,
                ft_projection_ramp_start=ft_codebook_adapt_config['projection_ramp_start'],
                ft_projection_ramp_end=ft_codebook_adapt_config['projection_ramp_end'],
                ft_projection_ramp_epochs=ft_codebook_adapt_config['projection_ramp_epochs'],
                ft_projection_loss_lambda=ft_codebook_adapt_config['projection_loss_lambda'],
                ft_projection_reg_max_links=ft_codebook_adapt_config['projection_reg_max_links'],
                ft_codebook_freeze_grouping=ft_codebook_adapt_config['codebook_freeze_grouping'],
                ft_codebook_use_legacy_regularization=ft_codebook_adapt_config['codebook_use_legacy_regularization'],
                output_dir=artifact_dir,
            )
        else:
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)
