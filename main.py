import os
import sys
import argparse
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
    ft_group_translate_train,
)


model_name = 'Vgg16'  # select one from[Vgg16, Res18, Res50, WRN]
translate_name = 'ft_group_cluster_translate'  # 新默认方法：面向容错的OU分组剪枝/映射

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
    parser.add_argument('--build-only', action='store_true', help='only build FT grouping artifacts and projected parameters; skip FT fine-tuning')
    parser.add_argument('--force-rebuild', action='store_true', help='ignore cached FT artifacts and rebuild them from the base checkpoint')
    parser.add_argument('--ft-cost-preset', type=str, default='none', choices=['none', 'fast', 'balanced', 'full'], help='cost preset for FT training schedule and regularization')
    parser.add_argument('--ft-low-cost', action='store_true', help='use a lower-cost FT training preset with sparse FT regularization and a shorter FT fine-tuning window')
    parser.add_argument('--ft-end-epoch', type=int, default=None, help='final epoch for FT fine-tuning; default keeps 200, ft-low-cost defaults to 160')
    parser.add_argument('--ft-reg-interval', type=int, default=None, help='apply FT regularization every N batches; default 1, ft-low-cost defaults to 10')
    parser.add_argument('--ft-reg-min-coverage', type=float, default=None, help='minimum layer coverage ratio required for FT regularization; default 0.0, ft-low-cost defaults to 0.1')
    parser.add_argument('--ft-reg-min-groups', type=int, default=None, help='minimum repairable group count required for FT regularization on a layer; default 1, ft-low-cost defaults to 64')
    parser.add_argument('--ft-reg-boost-after-refresh', action='store_true', help='halve the effective FT regularization interval during refresh epochs and the following epoch')
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


def normalize_schedule_epochs(epoch_values, checkpoint_epoch, end_epoch, ensure_final=False):
    filtered_epochs = sorted({epoch for epoch in epoch_values if checkpoint_epoch < epoch <= end_epoch})
    if ensure_final and end_epoch > checkpoint_epoch and end_epoch not in filtered_epochs:
        filtered_epochs.append(end_epoch)
    return filtered_epochs


def _flag_present(argv, flag_name):
    return any(token == flag_name or token.startswith(flag_name + '=') for token in argv)


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


def prepare_ft_artifacts(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel,
                         kernel_size, channel_number, pattern_value_number, mask, map_information,
                         multiple_relationship_information, reuse_ratio_information, ft_layer_enabled,
                         ft_group_target_ratio, ft_target_group_size, ft_similarity_threshold,
                         checkpoint_epoch, force_rebuild=False):
    group_information = {layer: None for layer in weight_name}

    mask_file = f'model_{model_name}_{translate_name}_mask.pkl'
    map_file = f'model_{model_name}_{translate_name}_map_information.pkl'
    mult_file = f'model_{model_name}_{translate_name}_multiple_relationship_information.pkl'
    coverage_file = f'model_{model_name}_{translate_name}_coverage_ratio_information.pkl'
    reuse_file = f'model_{model_name}_{translate_name}_reuse_ratio_information.pkl'
    group_file = f'model_{model_name}_{translate_name}_group_information.pkl'

    cache_files = [mask_file, map_file, mult_file, group_file]
    need_regenerate = force_rebuild or any(not os.path.exists(file_path) for file_path in cache_files)
    if not need_regenerate and not (os.path.exists(coverage_file) or os.path.exists(reuse_file)):
        need_regenerate = True

    if need_regenerate:
        if force_rebuild:
            print('[FT artifacts] force rebuild enabled; ignoring cached artifacts')
        checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth')
        model_original.load_state_dict(checkpoint['model'])

        for i in range(0, len(weight_name)):
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
                continue

            target_group_size = ft_target_group_size[i]
            similarity_threshold = ft_similarity_threshold[i]
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
            )
            print('[FTScore seed] {} strategy={} coverage={:.4f} avg_group_size={:.2f} fallback={}'.format(
                weight_name[i],
                group_seed_info.get('selected_strategy'),
                group_seed_info.get('estimated_coverage', 0.0),
                group_seed_info.get('estimated_avg_group_size', 0.0),
                group_seed_info.get('fallback_used', False),
            ))
            map_information[weight_name[i]], multiple_relationship_information[weight_name[i]], reuse_ratio_information[weight_name[i]], group_information[weight_name[i]] = ft_group_cluster_translate(
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
            )
            group_information[weight_name[i]]['target_ratio'] = ft_group_target_ratio[i]
            group_information[weight_name[i]]['seed_info'] = group_seed_info

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

    return mask, map_information, multiple_relationship_information, reuse_ratio_information, group_information


def save_ft_build_only_projection(model, model_name, translate_name, weight_name, mask, group_information, checkpoint_epoch):
    checkpoint_path = 'model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'])
    apply_ft_group_projection(model, weight_name, mask, group_information)
    torch.save(model.state_dict(), 'model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth')
    parameters_to_txt(model, model_name, translate_name)
    print('[FT build-only] saved projected parameters from epoch {}'.format(checkpoint_epoch))


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
    print('[FT schedule] preset={} low_cost={} end_epoch={} translate_epochs={} refresh_epochs={} reg_interval={} reg_min_coverage={:.3f} reg_min_groups={} reg_boost_after_refresh={}'.format(
        ft_config['selected_preset'],
        ft_low_cost,
        ft_end_epoch,
        ft_translate_epoch,
        ft_group_refresh_epoch,
        ft_reg_interval,
        ft_reg_min_coverage,
        ft_reg_min_groups,
        ft_reg_boost_after_refresh,
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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

        if translate_name == 'ft_group_cluster_translate':
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
            )
        else:
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)
