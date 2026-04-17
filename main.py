import os
import torch
import torch.utils.data
import pickle as pkl
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from model import Vgg16, Res18, Res50, WRN
from train_model import train, test
from cut import pattern_translate, get_structure_mask, get_ORC_mask, get_shape_mask, pattern_value_identical_translate, pattern_value_similar_translate, structure_and_value_identical_translate, pattern_shape_and_value_similar_translate


model_name = 'Vgg16'  # select one from[Vgg16, Res18, Res50, WRN]
translate_name = 'weight_pattern_shape_and_value_similar_translate'  # select one from['structure_pruning', 'ORC_pruning', 'weight_pattern_shape_translate', 'weight_pattern_value_identical_translate', 'weight_pattern_value_similar_translate', 'structure_pruning_and_weight_pattern_value_identical_translate', 'weight_pattern_shape_and_value_similar_translate']

lr = 0.1
epoches = 200
batch_size = 128
weight_decay_1 = 0.001
weight_decay_2 = 0.001

OU_size = 8
pattern_shape_number = 8
translate_epoch = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]


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


if __name__ == '__main__':
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

        pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)
