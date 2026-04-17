import os
import torch
import torch.utils.data
import pickle as pkl
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from model import Vgg16
from train_model import test
from cut import get_shape_mask, pattern_value_similar_translate, pattern_translate


# 超参数
lr = 0.1
weight_decay_1 = 0.001
weight_decay_2 = 0.001
epoches = 200
batch_size = 128

# 剪枝参数
OU_size = 8
pattern_value_number_kernel = [8, 4, 2, 1]
kernel_keep_ratios = [0.5, 0.25, 0.125, 0.0625]
translate_epoch = [150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200]
translate_name = 'weight_pattern_shape_translate'
# translate_name = 'weight_pattern_value_similar_translate'

# 模型参数
model_name = 'Vgg16'
kernel_size = [3, 3, 3, 3, 3, 3, 3,
               3, 3, 3, 3, 3, 3,
               1, 1, 1]
layer_in_channel = [3, 64, 64, 128, 128, 256, 256,
                    256, 512, 512, 512, 512, 512,
                    512, 4096, 4096]
layer_out_channel = [64, 64, 128, 128, 256, 256, 256,
                     512, 512, 512, 512, 512, 512,
                     4096, 4096, 10]
weight_name = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'conv4.weight', 'conv5.weight', 'conv6.weight', 'conv7.weight',
               'conv8.weight', 'conv9.weight', 'conv10.weight', 'conv11.weight', 'conv12.weight', 'conv13.weight',
               'fc1.weight', 'fc2.weight', 'fc3.weight']

# 打印数组时不再显示省略号改为全部显示
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


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
        cifar10_train_loader = torch.utils.data.DataLoader(cifar10_train_dataset, batch_size=batch_size, shuffle=False)
        cifar10_test_dataset = datasets.CIFAR10('./cifar10_data', train=False, transform=transform_test, download=False)
        cifar10_test_loader = torch.utils.data.DataLoader(cifar10_test_dataset, batch_size=batch_size, shuffle=False)
        return cifar10_train_loader, cifar10_test_loader, cifar10_train_size, cifar10_test_size, cifar10_classes, cifar10_input_size


def kernel_pattern_value_number_and_model_accuracy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 在gpu上训练
    if device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = True  # 不改变给定的神经网络结构的情况下，大大提升其训练和预测的速度
    train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集
    model_original = Vgg16(num_classes).to(device)  # 创建原始模型
    model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
    optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 创建优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

    accuracy_record_kernel = np.zeros((4, 13))  # 记录经过模式转换后的模型准确率
    accuracy_loss_kernel = np.zeros((4, 13))  # 记录模式转换造成的准确率损失
    accuracy_original = 0.9375
    result_all_kernel = pd.DataFrame()

    for i in range(0, 4):
        for j in range(1, 13):
            # 创建剪枝矩阵
            value_list = [torch.ones((layer_out_channel[k], layer_in_channel[k], kernel_size[k], kernel_size[k])) for k in range(0, len(weight_name))]
            mask = dict(zip(weight_name, value_list))
            for k in range(0, len(weight_name)):
                if 'fc' in weight_name[k]:
                    mask[weight_name[k]] = torch.ones(layer_out_channel[k], layer_in_channel[k])

            # 创建weight-pattern重用映射矩阵
            layer_map_list = [torch.ones((layer_in_channel[k], layer_out_channel[k], 2)) for k in range(0, len(weight_name))]
            map_information = dict(zip(weight_name, layer_map_list))

            # 创建weight-pattern倍数关系矩阵
            layer_multiple_list = [torch.ones((layer_out_channel[k], layer_in_channel[k], kernel_size[k], kernel_size[k])) for k in range(0, len(weight_name))]
            multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))
            for k in range(0, len(weight_name)):
                if 'fc' in weight_name[k]:
                    multiple_relationship_information[weight_name[k]] = torch.ones(layer_out_channel[k], layer_in_channel[k])

            channel_number = [1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              OU_size, OU_size, OU_size]

            pattern_value_number = [OU_size, OU_size, OU_size, OU_size, OU_size, OU_size, OU_size, OU_size,
                                    OU_size, OU_size, OU_size, OU_size, OU_size,
                                    1, 1, 1]

            best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0]

            pattern_value_number[j] = pattern_value_number_kernel[i]
            channel_number[j] = int(OU_size / pattern_value_number[j])
            if not os.path.exists('model_' + model_name + '_pattern_mask_' + str(j + 1) + '_pattern_value_number_' + str(pattern_value_number_kernel[i]) + '.pkl'):
                checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点
                model_original.load_state_dict(checkpoint['model'])  # 加载断点模型参数
                mask[weight_name[j]] = get_shape_mask(model_original, weight_name[j], layer_in_channel[j], layer_out_channel[j], kernel_size[j], channel_number[j], pattern_value_number[j], 8, OU_size)  # 计算剪枝矩阵
                with open('model_' + model_name + '_pattern_mask' + str(j + 1) + '_pattern_value_number_' + str(pattern_value_number_kernel[i]) + '.pkl', 'wb') as f:
                    pkl.dump(mask, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_pattern_mask' + str(j + 1) + '_pattern_value_number_' + str(pattern_value_number_kernel[i]) + '.pkl', 'rb') as f:
                    mask = pkl.load(f)
                    f.close()

            # 探索kernel稀疏率
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)
            model_translate = Vgg16(num_classes).to(device)  # 创建原始模型
            model_translate.load_state_dict(torch.load('model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth'))  # 加载训练好的原始模型
            accuracy_record_kernel[i][j], _ = test(model_translate, device, test_loader)
            accuracy_loss_kernel[i][j] = accuracy_original - accuracy_record_kernel[i][j]

        result_all_kernel['keep_ratio_' + str(pattern_value_number_kernel[i])] = accuracy_record_kernel[i].tolist()
        result_all_kernel['accuracy_loss_' + str(pattern_value_number_kernel[i])] = accuracy_loss_kernel[i].tolist()

    # 将结果存储到csv文件
    result_all_kernel.to_csv('kernel_pattern_value_number_and_model_accuracy.csv')


def layer_reuse_ratio_and_model_accuracy():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # 在gpu上训练
    if device == 'cuda':
        cudnn.deterministic = True
        cudnn.benchmark = True  # 不改变给定的神经网络结构的情况下，大大提升其训练和预测的速度
    train_loader, test_loader, train_size, test_size, num_classes, input_size = get_dataloader('cifar10')  # 构建训练集、测试集
    model_original = Vgg16(num_classes).to(device)  # 创建原始模型
    model_original.load_state_dict(torch.load('model_' + model_name + '_original_parameters.pth'))  # 加载训练好的原始模型
    optimizer = optim.SGD(model_original.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)  # 创建优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches)  # 动态学习率

    accuracy_record_kernel = np.zeros((4, 13))  # 记录经过模式转换后的模型准确率
    accuracy_loss_kernel = np.zeros((4, 13))  # 记录模式转换造成的准确率损失
    accuracy_original = 0.9375
    result_all_kernel = pd.DataFrame()

    for i in range(0, 4):
        for j in range(0, 1):
            # 创建剪枝矩阵
            value_list = [torch.ones((layer_out_channel[k], layer_in_channel[k], kernel_size[k], kernel_size[k])) for k in range(0, len(weight_name))]
            mask = dict(zip(weight_name, value_list))
            for k in range(0, len(weight_name)):
                if 'fc' in weight_name[k]:
                    mask[weight_name[k]] = torch.ones(layer_out_channel[k], layer_in_channel[k])

            # 创建weight-pattern重用映射矩阵
            layer_map_list = [torch.ones((layer_in_channel[k], layer_out_channel[k], 2)) for k in range(0, len(weight_name))]
            map_information = dict(zip(weight_name, layer_map_list))

            # 创建weight-pattern倍数关系矩阵
            layer_multiple_list = [torch.ones((layer_out_channel[k], layer_in_channel[k], kernel_size[k], kernel_size[k])) for k in range(0, len(weight_name))]
            multiple_relationship_information = dict(zip(weight_name, layer_multiple_list))
            for k in range(0, len(weight_name)):
                if 'fc' in weight_name[k]:
                    multiple_relationship_information[weight_name[k]] = torch.ones(layer_out_channel[k], layer_in_channel[k])

            # 记录每一层weight-pattern的重用率
            layer_reuse_ratio_list = [torch.zeros(1) for k in range(0, len(weight_name))]
            reuse_ratio_information = dict(zip(weight_name, layer_reuse_ratio_list))

            channel_number = [1, 1, 1, 1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              OU_size, OU_size, OU_size]

            best_keep_ratio = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                               1.0, 1.0, 1.0]

            best_keep_ratio[j] = kernel_keep_ratios[i]
            if not os.path.exists('model_' + model_name + '_value_normalized_map_information_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl'):
                map_information[weight_name[j]], multiple_relationship_information[weight_name[j]], reuse_ratio_information[weight_name[j]] = pattern_value_similar_translate(model_original, layer_in_channel[j], layer_out_channel[j], weight_name[j], best_keep_ratio[j], kernel_size[j], channel_number[j])  # 计算剪枝矩阵
                map_information[weight_name[j]] = map_information[weight_name[j]].type(torch.long)
                print(reuse_ratio_information[weight_name[j]])
                with open('model_' + model_name + '_value_normalized_map_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'wb') as f:
                    pkl.dump(map_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'wb') as f:
                    pkl.dump(multiple_relationship_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'wb') as f:
                    pkl.dump(reuse_ratio_information, f, pkl.HIGHEST_PROTOCOL)
                    f.close()
            else:
                with open('model_' + model_name + '_value_normalized_map_information_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'rb') as f:
                    map_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_multiple_relationship_information_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'rb') as f:
                    multiple_relationship_information = pkl.load(f)
                    f.close()
                with open('model_' + model_name + '_value_reuse_ratio_information_information_layer_' + str(j + 1) + '_reuse_ratio_' + str(kernel_keep_ratios[i]) + '.pkl', 'rb') as f:
                    reuse_ratio_information = pkl.load(f)
                    f.close()
                for k in range(0, len(weight_name)):
                    best_keep_ratio[k] = 1.0 - reuse_ratio_information[weight_name[k]]

            # 探索kernel稀疏率
            pattern_translate(model_original, model_name, translate_name, weight_name, layer_in_channel, layer_out_channel, kernel_size, best_keep_ratio, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, epoches, translate_epoch)
            model_translate = Vgg16(num_classes).to(device)  # 创建原始模型
            model_translate.load_state_dict(torch.load('model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth'))  # 加载训练好的原始模型
            accuracy_record_kernel[i][j], _ = test(model_translate, device, test_loader)
            accuracy_loss_kernel[i][j] = accuracy_original - accuracy_record_kernel[i][j]

        result_all_kernel['keep_ratio_' + str(kernel_keep_ratios[i])] = accuracy_record_kernel[i].tolist()
        result_all_kernel['accuracy_loss_' + str(kernel_keep_ratios[i])] = accuracy_loss_kernel[i].tolist()

    # 将结果存储到csv文件
    result_all_kernel.to_csv('kernel_reuse_ratio_and_model_accuracy.csv')


if __name__ == '__main__':
    kernel_pattern_value_number_and_model_accuracy()
    # layer_reuse_ratio_and_model_accuracy()
