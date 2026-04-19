import copy
import math
import time
import pickle as pkl
from typing import Any, Dict, List, Optional, Tuple

import torch
import pandas as pd
import torch.nn.functional as F


# 将模型参数以txt形式存储
def parameters_to_txt(model, model_name, translate_name):
    str_parameters = ''
    for parameters in model.parameters():
        str_parameters = str_parameters + str(parameters) + str('\n')
    f_parameters = open('parameters_' + model_name + '_' + translate_name + '.txt', 'w', encoding='utf-8')
    f_parameters.write(str_parameters)
    f_parameters.close()


# 测试模型
def test(model, device, test_loader):
    model.eval()  # 不启用Batch Normalization 和 Dropout
    total = 0
    correct = 0
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)

            test_loss = test_loss + loss.item()
            _, predicted = outputs.max(1)
            total = total + targets.size(0)
            correct = correct + predicted.eq(targets).sum().item()

    test_accuracy = correct / total
    return test_accuracy, test_loss


# 创建用于排序的数据结构
class Node:
    def __init__(self, index=0, value=0.0):
        self.index = index
        self.value = value


def get_structure_mask(model, weight_name, in_channel, out_channel, kernel_size, keep_ratio):
    keep_number = int(in_channel * keep_ratio)
    if 'conv' in weight_name or 'shortcut' in weight_name:
        pattern_mask = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
        weight_matrix = torch.ones(out_channel, in_channel, kernel_size, kernel_size)
        channel_importance = [Node(i, 0.0) for i in range(0, in_channel)]  # 统计通道的绝对值大小
        for c_out in range(0, out_channel):
            weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
        for c_in in range(0, in_channel):
            for c_out in range(0, out_channel):
                channel_importance[c_in].value = channel_importance[c_in].value + weight_matrix[c_out][c_in].abs().sum().item()
        channel_importance.sort(key=lambda f: f.value, reverse=True)
        for i in range(0, keep_number):
            for c_out in range(0, out_channel):
                pattern_mask[c_out][channel_importance[i].index] = torch.ones(kernel_size, kernel_size)  # 构造剪枝矩阵

        return pattern_mask

    if 'fc' in weight_name:
        pattern_mask = torch.zeros(out_channel, in_channel)
        weight_matrix = torch.ones(out_channel, in_channel)
        channel_importance = [Node(i, 0.0) for i in range(0, in_channel)]  # 统计通道的绝对值大小
        for c_out in range(0, out_channel):
            weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
        for c_in in range(0, in_channel):
            for c_out in range(0, out_channel):
                channel_importance[c_in].value = channel_importance[c_in].value + weight_matrix[c_out][c_in].abs().item()
        channel_importance.sort(key=lambda f: f.value, reverse=True)
        for i in range(0, keep_number):
            for c_out in range(0, out_channel):
                pattern_mask[c_out][channel_importance[i].index] = 1.0  # 构造剪枝矩阵

        return pattern_mask


def get_ORC_mask(model, weight_name, in_channel, out_channel, kernel_size, keep_ratio):
    if 'conv' in weight_name or 'shortcut' in weight_name:
        const1 = kernel_size
        const2 = kernel_size * kernel_size
        keep_number = int(in_channel * kernel_size * kernel_size * keep_ratio)
        pattern_mask = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
        weight_matrix = torch.ones(out_channel, in_channel, kernel_size, kernel_size)
        for c_out in range(0, out_channel):
            weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
        for c_out in range(0, out_channel):
            weight_importance = [Node(i, 0.0) for i in range(0, in_channel * kernel_size * kernel_size)]  # 统计通道的绝对值大小
            for c_in in range(0, in_channel):
                for h in range(0, kernel_size):
                    for w in range(0, kernel_size):
                        weight_importance[c_in * const2 + h * const1 + w].value = weight_matrix[c_out][c_in][h][w].abs().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            for i in range(0, keep_number):
                c_in = int(weight_importance[i].index / const2)
                weight_importance[i].index = weight_importance[i].index % const2
                h = int(weight_importance[i].index / const1)
                w = weight_importance[i].index % const1
                pattern_mask[c_out][c_in][h][w] = 1.0  # 构造剪枝矩阵

        return pattern_mask

    if 'fc' in weight_name:
        keep_number = int(in_channel * keep_ratio)
        pattern_mask = torch.zeros(out_channel, in_channel)
        weight_matrix = torch.ones(out_channel, in_channel)
        weight_importance = [Node(i, 0.0) for i in range(0, in_channel)]  # 统计通道的绝对值大小
        for c_out in range(0, out_channel):
            weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
        for c_out in range(0, out_channel):
            for c_in in range(0, in_channel):
                weight_importance[c_in].value = weight_matrix[c_out][c_in].abs().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            for i in range(0, keep_number):
                c_in = weight_importance[i].index
                pattern_mask[c_out][c_in] = 1.0  # 构造剪枝矩阵

        return pattern_mask


# 模式形状转换
def get_shape_mask(model, weight_name, in_channel, out_channel, kernel_size, channel_number, pattern_value_number, pattern_shape_number, OU_size):
    if 'conv' in weight_name and kernel_size != 1:
        pattern_mask = torch.ones(out_channel, in_channel, kernel_size, kernel_size)
        weight_matrix = torch.ones(out_channel, in_channel, kernel_size, kernel_size)
        for c_out in range(0, out_channel):
            weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
        # 遍历所有kernel统计出现次数最多的形状id
        for c_in in range(0, in_channel, channel_number):
            pattern_importance = [Node(i, 0.0) for i in range(0, 9877)]
            for c_out in range(0, out_channel):
                # 获取模式形状编号
                if pattern_value_number == 8:
                    weight_importance = [Node(i + 1, weight_matrix[c_out][c_in][int(i / kernel_size)][int(i % kernel_size)].abs()) for i in range(0, kernel_size * kernel_size)]
                    weight_importance.sort(key=lambda f: f.value, reverse=True)
                    pattern_id = 1000 * weight_importance[8].index
                    pattern_importance[pattern_id].value = pattern_importance[pattern_id].value + 1  # 统计每个模式出现的次数

                if pattern_value_number == 4:
                    weight_importance = [Node(i + 1, weight_matrix[c_out][c_in][int(i / kernel_size)][int(i % kernel_size)].abs() + weight_matrix[c_out][c_in + 1][int(i / kernel_size)][int(i % kernel_size)].abs()) for i in range(0, kernel_size * kernel_size)]
                    weight_importance.sort(key=lambda f: f.value, reverse=True)
                    weight_importance = weight_importance[0:pattern_value_number]
                    weight_importance.sort(key=lambda f: f.index, reverse=True)
                    pattern_id = 1000 * weight_importance[0].index + 100 * weight_importance[1].index + 10 * weight_importance[2].index + weight_importance[3].index  # 获得每个kernel的形状id
                    pattern_importance[pattern_id].value = pattern_importance[pattern_id].value + 1  # 统计每个模式出现的次数

                if pattern_value_number == 2:
                    weight_importance = [Node(i + 1, weight_matrix[c_out][c_in][int(i / kernel_size)][int(i % kernel_size)].abs() + weight_matrix[c_out][c_in + 1][int(i / kernel_size)][int(i % kernel_size)].abs() + weight_matrix[c_out][c_in + 2][int(i / kernel_size)][int(i % kernel_size)].abs() + weight_matrix[c_out][c_in + 3][int(i / kernel_size)][int(i % kernel_size)].abs()) for i in range(0, kernel_size * kernel_size)]
                    weight_importance.sort(key=lambda f: f.value, reverse=True)
                    weight_importance = weight_importance[0:pattern_value_number]
                    weight_importance.sort(key=lambda f: f.index, reverse=True)
                    pattern_id = 1000 * weight_importance[0].index + 100 * weight_importance[1].index
                    pattern_importance[pattern_id].value = pattern_importance[pattern_id].value + 1  # 统计每个模式出现的次数

            # 构造出现次数最多模式的形状
            pattern_importance.sort(key=lambda f: f.value, reverse=True)  # 对各个模式按出现的次数排序
            important_pattern_shape = torch.zeros(pattern_shape_number, channel_number, kernel_size, kernel_size)
            for i in range(0, pattern_shape_number):
                pattern_id = pattern_importance[i].index
                flag = 1000
                if pattern_value_number <= 4:
                    for j in range(0, pattern_value_number):
                        location = pattern_id / flag - 1
                        pattern_id = pattern_id % flag
                        flag = flag / 10
                        for k in range(0, channel_number):
                            important_pattern_shape[i][k][int(location / kernel_size)][int(location % kernel_size)] = 1.0
                else:
                    important_pattern_shape[i] = torch.ones(channel_number, kernel_size, kernel_size)
                    for j in range(0, kernel_size * kernel_size - pattern_value_number):
                        location = pattern_id / flag - 1
                        pattern_id = pattern_id % flag
                        flag = flag / 10
                        for k in range(0, channel_number):
                            important_pattern_shape[i][k][int(location / kernel_size)][int(location % kernel_size)] = 0.0

            # 遍历所有kernel，对每个kernel进行模式剪枝
            for c_out in range(0, out_channel):
                # 获得最相似的模式
                weight_value = torch.zeros(channel_number, kernel_size, kernel_size)
                for i in range(0, channel_number):
                    weight_value[i] = weight_matrix[c_out][c_in + i]
                select_number = (important_pattern_shape * weight_value).abs().sum(axis=3).sum(axis=2).sum(axis=1).argmax().item()
                for i in range(0, channel_number):
                    pattern_mask[c_out][c_in + i] = important_pattern_shape[select_number][i]  # 构造剪枝矩阵

        return pattern_mask

    if ('conv' in weight_name or 'shortcut' in weight_name) and kernel_size == 1:
        if channel_number == OU_size:
            pattern_mask = torch.ones(out_channel, in_channel, 1, 1)

            return pattern_mask

        else:
            pattern_mask = torch.zeros(out_channel, in_channel, 1, 1)
            weight_matrix = torch.ones(out_channel, in_channel, 1, 1)
            for c_out in range(0, out_channel):
                weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
            for c_in in range(0, in_channel, channel_number):
                # 获取模式形状编号
                channel_importance = [Node(i, 0.0) for i in range(0, channel_number)]  # 统计通道的绝对值大小
                for c_out in range(0, out_channel):
                    for i in range(0, channel_number):
                        channel_importance[i].value = channel_importance[i].value + weight_matrix[c_out][c_in + i].abs().sum().item()
                channel_importance.sort(key=lambda f: f.value, reverse=True)
                for c_out in range(0, out_channel):
                    for i in range(0, OU_size):
                        pattern_mask[c_out][c_in + channel_importance[i].index][0][0] = 1.0  # 构造剪枝矩阵

            return pattern_mask

    if 'fc' in weight_name:
        if channel_number == OU_size:
            pattern_mask = torch.ones(out_channel, in_channel)

            return pattern_mask

        else:
            pattern_mask = torch.zeros(out_channel, in_channel)
            weight_matrix = torch.ones(out_channel, in_channel)
            for c_out in range(0, out_channel):
                weight_matrix[c_out] = model.state_dict()[weight_name][c_out]
            for c_in in range(0, in_channel, channel_number):
                # 获取模式形状编号
                channel_importance = [Node(i, 0.0) for i in range(0, channel_number)]  # 统计通道的绝对值大小
                for c_out in range(0, out_channel):
                    for i in range(0, channel_number):
                        channel_importance[i].value = channel_importance[i].value + weight_matrix[c_out][c_in + i].abs().item()
                channel_importance.sort(key=lambda f: f.value, reverse=True)
                for c_out in range(0, out_channel):
                    for i in range(0, OU_size):
                        pattern_mask[c_out][c_in + channel_importance[i].index] = 1.0  # 构造剪枝矩阵

            return pattern_mask


def pattern_value_identical_translate(model, in_channel, out_channel, weight_name, threshold, kernel_size, channel_number):
    total_translate_weight_pattern = 0  # 统计该层的weight-pattern重用率

    # 构造矩阵加速代码执行速度
    weight_matrix = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    if 'fc' in weight_name:
        weight_matrix = torch.zeros(out_channel, in_channel)

    # 先对权重进行量化
    max_value = torch.max(model.state_dict()[weight_name]).item()
    min_value = torch.min(model.state_dict()[weight_name]).item()
    if max_value < -min_value:
        max_value = -min_value
    scale = (max_value - 0) / 127
    for c_out in range(0, out_channel):
        weight_matrix[c_out] = torch.round(model.state_dict()[weight_name][c_out] / scale)

    if 'fc' in weight_name:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        for c_in in range(0, in_channel, channel_number):
            # 模式大小排序
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            threshold_value = weight_importance[int(out_channel * threshold)].value
            # 创建相关变量
            translate_value = 0  # 记录每个in_channel转换多少weight-pattern
            keep_number = [0] * out_channel  # 记录保留weight-pattern的索引
            keep_value = 1  # 记录保留多少weight-pattern
            keep_number[0] = weight_importance[0].index  # 将最大的weight-pattern加入保留集合
            keep_weight_matrix = torch.zeros(out_channel, channel_number)  # 记录保留的weight-pattern
            for j in range(0, channel_number):
                keep_weight_matrix[0][j] = weight_matrix[keep_number[0]][c_in + j]
            # 依次统计每个输入通道的映射表
            for i in range(1, out_channel):
                weightx_value = weight_matrix[weight_importance[i].index][c_in: c_in + channel_number]  # 记录转换模式中的权重数值
                weighty_value = keep_weight_matrix[0:keep_value + 1]  # 记录保留模式中的权重数值
                select_number = (weighty_value - weightx_value).abs().sum(axis=1).argmin().item()  # 找到最相近的weight-pattern
                weight_pattern_difference = (weighty_value[select_number] - weightx_value).abs().sum().item()  # 计算两个weight-pattern之间的差异
                # 判断每个weight-pattern是否可以重用
                if weight_pattern_difference <= threshold_value:
                    for j in range(0, channel_number):
                        # 修改映射表
                        if select_number != keep_value:
                            map_table[c_in + j][translate_value][0] = weight_importance[i].index
                            map_table[c_in + j][translate_value][1] = weight_importance[keep_number[select_number]].index
                        else:
                            map_table[c_in + j][translate_value][0] = weight_importance[i].index
                            map_table[c_in + j][translate_value][1] = -1
                    # 统计重用weight-pattern的个数
                    translate_value = translate_value + 1
                else:
                    # 记录保留weight-pattern的索引
                    keep_number[keep_value] = weight_importance[i].index
                    # 将该weight-pattern加入保留weight-pattern矩阵
                    for j in range(0, channel_number):
                        keep_weight_matrix[keep_value][j] = weight_matrix[weight_importance[i].index][c_in + j]
                    # 统计保留weight-pattern的个数
                    keep_value = keep_value + 1
            # 统计该层重用weight-pattern数量
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            for j in range(0, channel_number):
                map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, weight_pattern_reuse_ratio

    else:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        for c_in in range(0, in_channel, channel_number):
            # 模式大小排序
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            threshold_value = weight_importance[int(out_channel * threshold)].value
            # 创建相关变量
            translate_value = 0  # 记录每个in_channel转换多少weight-pattern
            keep_number = [0] * out_channel  # 记录保留weight-pattern的索引
            keep_value = 1  # 记录保留多少weight-pattern
            keep_number[0] = weight_importance[0].index  # 将最大的weight-pattern加入保留集合
            keep_weight_matrix = torch.zeros(out_channel, channel_number, kernel_size, kernel_size)  # 记录保留的weight-pattern
            for j in range(0, channel_number):
                keep_weight_matrix[0][j] = weight_matrix[keep_number[0]][c_in + j]
            # 依次统计每个输入通道的映射表
            for i in range(1, out_channel):
                weightx_value = weight_matrix[weight_importance[i].index][c_in: c_in + channel_number]  # 记录转换模式中的权重数值
                weighty_value = keep_weight_matrix[0:keep_value + 1]  # 记录保留模式中的权重数值
                select_number = (weighty_value - weightx_value).abs().sum(axis=3).sum(axis=2).sum(axis=1).argmin().item()  # 找到最相近的weight-pattern
                weight_pattern_difference = (weighty_value[select_number] - weightx_value).abs().sum().item()  # 计算两个weight-pattern之间的差异
                # 判断每个weight-pattern是否可以重用
                if weight_pattern_difference <= threshold_value:
                    # 修改映射表
                    if select_number != keep_value:
                        map_table[c_in][translate_value][0] = weight_importance[i].index
                        map_table[c_in][translate_value][1] = weight_importance[keep_number[select_number]].index
                    else:
                        map_table[c_in][translate_value][0] = weight_importance[i].index
                        map_table[c_in][translate_value][1] = -1
                    # 统计重用weight-pattern的个数
                    translate_value = translate_value + 1
                else:
                    # 记录保留weight-pattern的索引
                    keep_number[keep_value] = weight_importance[i].index
                    # 将该weight-pattern加入保留weight-pattern矩阵
                    for j in range(0, channel_number):
                        keep_weight_matrix[keep_value][j] = weight_matrix[weight_importance[i].index][c_in + j]
                    # 统计保留weight-pattern的个数
                    keep_value = keep_value + 1
            # 统计该层重用weight-pattern数量
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            for j in range(0, channel_number):
                map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, weight_pattern_reuse_ratio


def pattern_value_similar_translate(model, in_channel, out_channel, weight_name, threshold, kernel_size, channel_number):
    total_translate_weight_pattern = 0  # 统计重用weight-pattern的总数

    # 构造矩阵加速代码执行速度
    weight_matrix = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    if 'fc' in weight_name:
        weight_matrix = torch.zeros(out_channel, in_channel)

    # 先对权重进行量化
    max_value = torch.max(model.state_dict()[weight_name]).item()
    min_value = torch.min(model.state_dict()[weight_name]).item()
    if max_value < -min_value:
        max_value = -min_value
    scale = (max_value - 0) / (127 * 256)
    for c_out in range(0, out_channel):
        weight_matrix[c_out] = torch.round(model.state_dict()[weight_name][c_out] / scale)

    if 'fc' in weight_name:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        multiple_relationship_table = torch.ones(out_channel, in_channel)  # 存储weight-pattern之间的倍数关系
        # 进行kernel级模式匹配
        for c_in in range(0, in_channel, channel_number):
            threshold_value = int(out_channel * threshold)  # 最终保留的模式数
            translate_value = out_channel - threshold_value  # 最终转换的模式数
            # 找到参数绝对值最小的模式
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            # 统计保留模式的索引
            keep_number = [0] * threshold_value  # 保留模式的索引
            y_importance = torch.zeros(threshold_value + 1)  # 记录保留模式中的权重标准值
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weight_matrix[weight_importance[i].index][c_in + j] = torch.round(weight_matrix[weight_importance[i].index][c_in + j] / math.pow(2, 8)) * math.pow(2, 8)
                    y_importance[i] = y_importance[i] + weight_matrix[weight_importance[i].index][c_in + j].abs().sum().item()
                if y_importance[i] != 0:
                    keep_number[i] = weight_importance[i].index
                else:
                    threshold_value = i
                    translate_value = out_channel - threshold_value
                    break
            # 统计转换模式的索引
            translate_number = [0] * translate_value  # 转换模式的索引
            x_importance = torch.zeros(translate_value)  # 记录转换模式中的权重标准值
            similarity = torch.zeros(threshold_value + 1)  # 记录相似系数
            weightx_value = torch.zeros(translate_value, channel_number)  # 记录转换模式中的权重数值
            weighty_value = torch.zeros(threshold_value + 1, channel_number)  # 记录保留模式中的权重数值
            similar_value = torch.zeros(threshold_value + 1, channel_number)  # 记录乘以相似系数后保留模式中的权重数值
            # 构造保留weight-pattern矩阵
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weighty_value[i][j] = weight_matrix[keep_number[i]][c_in + j]
            # 将绝对值最小的模式加入转换模式集合
            for i in range(0, translate_value):
                translate_number[i] = weight_importance[i + threshold_value].index
                for j in range(0, channel_number):
                    weightx_value[i][j] = weight_matrix[translate_number[i]][c_in + j]
                x_importance[i] = weightx_value[i].abs().sum().item()
            # 为每个要剪枝的模式匹配剩余最相似的模式
            for i in range(0, translate_value):
                if x_importance[i] != 0:
                    for j in range(0, threshold_value):
                        similarity[j] = math.ceil(math.pow(2, 8) * math.pow(2, round(math.log2(x_importance[i] / y_importance[j])))) / math.pow(2, 8)
                        similar_value[j] = weighty_value[j] * similarity[j]
                    select_number = (similar_value - weightx_value[i]).abs().sum(axis=1).argmin().item()
                    for j in range(0, channel_number):
                        # 构建映射表
                        map_table[c_in + j][i][0] = translate_number[i]
                        if select_number != threshold_value:
                            map_table[c_in + j][i][1] = keep_number[select_number]
                            multiple_relationship_table[translate_number[i]][c_in + j] = similarity[select_number]
                        else:
                            map_table[c_in + j][i][1] = -1
                            multiple_relationship_table[translate_number[i]][c_in + j] = 0
                else:
                    for j in range(0, channel_number):
                        # 构建映射表
                        map_table[c_in + j][i][0] = translate_number[i]
                        map_table[c_in + j][i][1] = -1
                        multiple_relationship_table[translate_number[i]][c_in + j] = 0
            # 统计重用weight-pattern的总数
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            if translate_value < out_channel:
                for j in range(0, channel_number):
                    map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, multiple_relationship_table, weight_pattern_reuse_ratio

    else:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        multiple_relationship_table = torch.ones(out_channel, in_channel, kernel_size, kernel_size)  # 存储weight-pattern之间的倍数关系
        # 进行kernel级模式匹配
        for c_in in range(0, in_channel, channel_number):
            threshold_value = int(out_channel * threshold)  # 最终保留的模式数
            translate_value = out_channel - threshold_value  # 最终转换的模式数
            # 找到参数绝对值最小的模式
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            # 统计保留模式的索引
            keep_number = [0] * threshold_value  # 保留模式的索引
            y_importance = torch.zeros(threshold_value + 1)  # 记录保留模式中的权重标准值
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weight_matrix[weight_importance[i].index][c_in + j] = torch.round(weight_matrix[weight_importance[i].index][c_in + j] / math.pow(2, 8)) * math.pow(2, 8)
                    y_importance[i] = y_importance[i] + weight_matrix[weight_importance[i].index][c_in + j].abs().sum().item()
                if y_importance[i] != 0:
                    keep_number[i] = weight_importance[i].index
                else:
                    threshold_value = i
                    translate_value = out_channel - threshold_value
                    break
            # 统计转换模式的索引
            translate_number = [0] * translate_value  # 转换模式的索引
            x_importance = torch.zeros(translate_value)  # 记录转换模式中的权重标准值
            similarity = torch.zeros(threshold_value + 1)  # 记录相似系数
            weightx_value = torch.zeros(translate_value, channel_number, kernel_size, kernel_size)  # 记录转换模式中的权重数值
            weighty_value = torch.zeros(threshold_value + 1, channel_number, kernel_size, kernel_size)  # 记录保留模式中的权重数值
            similar_value = torch.zeros(threshold_value + 1, channel_number, kernel_size, kernel_size)  # 记录乘以相似系数后保留模式中的权重数值
            # 构造保留weight-pattern矩阵
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weighty_value[i][j] = weight_matrix[keep_number[i]][c_in + j]
            # 将绝对值最小的模式加入转换模式集合
            for i in range(0, translate_value):
                translate_number[i] = weight_importance[i + threshold_value].index
                for j in range(0, channel_number):
                    weightx_value[i][j] = weight_matrix[translate_number[i]][c_in + j]
                x_importance[i] = weightx_value[i].abs().sum().item()
            # 为每个要剪枝的模式匹配剩余最相似的模式
            for i in range(0, translate_value):
                if x_importance[i] != 0:
                    for j in range(0, threshold_value):
                        similarity[j] = math.ceil(math.pow(2, 8) * math.pow(2, round(math.log2(x_importance[i] / y_importance[j])))) / math.pow(2, 8)
                        similar_value[j] = weighty_value[j] * similarity[j]
                    select_number = (similar_value - weightx_value[i]).abs().sum(axis=3).sum(axis=2).sum(axis=1).argmin().item()
                    for j in range(0, channel_number):
                        # 构建映射表
                        map_table[c_in + j][i][0] = translate_number[i]
                        if select_number != threshold_value:
                            map_table[c_in + j][i][1] = keep_number[select_number]
                            multiple_relationship_table[translate_number[i]][c_in + j] = similarity[select_number]
                        else:
                            map_table[c_in + j][i][1] = -1
                            multiple_relationship_table[translate_number[i]][c_in + j] = 0
                else:
                    for j in range(0, channel_number):
                        # 构建映射表
                        map_table[c_in + j][i][0] = translate_number[i]
                        map_table[c_in + j][i][1] = -1
                        multiple_relationship_table[translate_number[i]][c_in + j] = 0

            # 统计重用weight-pattern的总数
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            if translate_value < out_channel:
                for j in range(0, channel_number):
                    map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, multiple_relationship_table, weight_pattern_reuse_ratio


def structure_and_value_identical_translate(model, in_channel, out_channel, weight_name, threshold, kernel_size, mask):
    total_translate_weight_pattern = 0  # 统计该层的weight-pattern重用率

    # 构造矩阵加速代码执行速度
    weight_matrix = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    weight_matrix_pruning = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    if 'fc' in weight_name:
        weight_matrix = torch.zeros(out_channel, in_channel)
        weight_matrix_pruning = torch.zeros(out_channel, in_channel)

    # 先对权重进行量化
    max_value = torch.max(model.state_dict()[weight_name]).item()
    min_value = torch.min(model.state_dict()[weight_name]).item()
    if max_value < -min_value:
        max_value = -min_value
    scale = (max_value - 0) / 127
    for c_out in range(0, out_channel):
        weight_matrix[c_out] = torch.round(model.state_dict()[weight_name][c_out] / scale)
        weight_matrix_pruning[c_out] = weight_matrix[c_out] * mask[c_out]

    total_in_channel = in_channel

    if 'fc' in weight_name:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        for c_in in range(0, in_channel, 8):
            # 模式大小排序
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, 8):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix_pruning[i][c_in + j].abs().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            threshold_value = weight_importance[int(out_channel * threshold)].value
            # 创建相关变量
            translate_value = 0  # 记录每个in_channel转换多少weight-pattern
            keep_number = [0] * out_channel  # 记录保留weight-pattern的索引
            keep_value = 1  # 记录保留多少weight-pattern
            keep_number[0] = weight_importance[0].index  # 将最大的weight-pattern加入保留集合
            keep_weight_matrix = torch.zeros(out_channel, 8)  # 记录保留的weight-pattern
            for j in range(0, 8):
                keep_weight_matrix[0][j] = weight_matrix_pruning[keep_number[0]][c_in + j]
            # 依次统计每个输入通道的映射表
            for i in range(1, out_channel):
                weightx_value = weight_matrix_pruning[weight_importance[i].index][c_in: c_in + 8]  # 记录转换模式中的权重数值
                weighty_value = keep_weight_matrix[0:keep_value + 1]  # 记录保留模式中的权重数值
                select_number = (weighty_value - weightx_value).abs().sum(axis=1).argmin().item()  # 找到最相近的weight-pattern
                weight_pattern_difference = (weighty_value[select_number] - weightx_value).abs().sum().item()  # 计算两个weight-pattern之间的差异
                # 判断每个weight-pattern是否可以重用
                if weight_pattern_difference <= threshold_value:
                    for j in range(0, 8):
                        # 修改映射表
                        if select_number != keep_value:
                            map_table[c_in + j][translate_value][0] = weight_importance[i].index
                            map_table[c_in + j][translate_value][1] = weight_importance[keep_number[select_number]].index
                        else:
                            map_table[c_in + j][translate_value][0] = weight_importance[i].index
                            map_table[c_in + j][translate_value][1] = -1
                    # 统计重用weight-pattern的个数
                    translate_value = translate_value + 1
                else:
                    # 记录保留weight-pattern的索引
                    keep_number[keep_value] = weight_importance[i].index
                    # 将该weight-pattern加入保留weight-pattern矩阵
                    for j in range(0, 8):
                        keep_weight_matrix[keep_value][j] = weight_matrix_pruning[weight_importance[i].index][c_in + j]
                    # 统计保留weight-pattern的个数
                    keep_value = keep_value + 1
            # 统计该层重用weight-pattern数量
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            for j in range(0, 8):
                map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / 8 * out_channel)
        return map_table, weight_pattern_reuse_ratio

    else:
        if kernel_size != 1:
            map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
            for c_in in range(0, in_channel):
                # 模式大小排序
                weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
                for i in range(0, out_channel):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix_pruning[i][c_in].abs().sum().item()
                weight_importance.sort(key=lambda f: f.value, reverse=True)
                threshold_value = weight_importance[int(out_channel * threshold)].value
                if weight_importance[0].value == 0:
                    # 给map_table设置结束标志
                    map_table[c_in][0][0] = -1
                    total_in_channel = total_in_channel - 1
                    continue
                # 创建相关变量
                translate_value = 0  # 记录每个in_channel转换多少weight-pattern
                keep_number = [0] * out_channel  # 记录保留weight-pattern的索引
                keep_value = 1  # 记录保留多少weight-pattern
                keep_number[0] = weight_importance[0].index  # 将最大的weight-pattern加入保留集合
                keep_weight_matrix = torch.zeros(out_channel, kernel_size, kernel_size)  # 记录保留的weight-pattern
                keep_weight_matrix[0] = weight_matrix_pruning[keep_number[0]][c_in]
                # 依次统计每个输入通道的映射表
                for i in range(1, out_channel):
                    weightx_value = weight_matrix_pruning[weight_importance[i].index][c_in]  # 记录转换模式中的权重数值
                    weighty_value = keep_weight_matrix[0:keep_value + 1]  # 记录保留模式中的权重数值
                    select_number = (weighty_value - weightx_value).abs().sum(axis=2).sum(axis=1).argmin().item()  # 找到最相近的weight-pattern
                    weight_pattern_difference = (weighty_value[select_number] - weightx_value).abs().sum().item()  # 计算两个weight-pattern之间的差异
                    # 判断每个weight-pattern是否可以重用
                    if weight_pattern_difference <= threshold_value:
                        # 修改映射表
                        if select_number != keep_value:
                            map_table[c_in][translate_value][0] = weight_importance[i].index
                            map_table[c_in][translate_value][1] = weight_importance[keep_number[select_number]].index
                        else:
                            map_table[c_in][translate_value][0] = weight_importance[i].index
                            map_table[c_in][translate_value][1] = -1
                        # 统计重用weight-pattern的个数
                        translate_value = translate_value + 1
                    else:
                        # 记录保留weight-pattern的索引
                        keep_number[keep_value] = weight_importance[i].index
                        # 将该weight-pattern加入保留weight-pattern矩阵
                        keep_weight_matrix[keep_value] = weight_matrix_pruning[weight_importance[i].index][c_in]
                        # 统计保留weight-pattern的个数
                        keep_value = keep_value + 1
                # 统计该层重用weight-pattern数量
                total_translate_weight_pattern = total_translate_weight_pattern + translate_value
                # 给map_table设置结束标志
                map_table[c_in][translate_value][0] = -1

            # 计算该层weight-pattern重用率
            weight_pattern_reuse_ratio = total_translate_weight_pattern / (total_in_channel * out_channel)
            return map_table, weight_pattern_reuse_ratio

        else:
            map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
            for c_in in range(0, in_channel, 8):
                # 模式大小排序
                weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
                for i in range(0, out_channel):
                    for j in range(0, 8):
                        weight_importance[i].value = weight_importance[i].value + weight_matrix_pruning[i][c_in + j].abs().sum().item()
                weight_importance.sort(key=lambda f: f.value, reverse=True)
                threshold_value = weight_importance[int(out_channel * threshold)].value
                # 创建相关变量
                translate_value = 0  # 记录每个in_channel转换多少weight-pattern
                keep_number = [0] * out_channel  # 记录保留weight-pattern的索引
                keep_value = 1  # 记录保留多少weight-pattern
                keep_number[0] = weight_importance[0].index  # 将最大的weight-pattern加入保留集合
                keep_weight_matrix = torch.zeros(out_channel, 8, kernel_size, kernel_size)  # 记录保留的weight-pattern
                for j in range(0, 8):
                    keep_weight_matrix[0][j] = weight_matrix_pruning[keep_number[0]][c_in + j]
                # 依次统计每个输入通道的映射表
                for i in range(1, out_channel):
                    weightx_value = weight_matrix_pruning[weight_importance[i].index][c_in: c_in + 8]  # 记录转换模式中的权重数值
                    weighty_value = keep_weight_matrix[0:keep_value + 1]  # 记录保留模式中的权重数值
                    select_number = (weighty_value - weightx_value).abs().sum(axis=1).argmin().item()  # 找到最相近的weight-pattern
                    weight_pattern_difference = (weighty_value[select_number] - weightx_value).abs().sum().item()  # 计算两个weight-pattern之间的差异
                    # 判断每个weight-pattern是否可以重用
                    if weight_pattern_difference <= threshold_value:
                        for j in range(0, 8):
                            # 修改映射表
                            if select_number != keep_value:
                                map_table[c_in + j][translate_value][0] = weight_importance[i].index
                                map_table[c_in + j][translate_value][1] = weight_importance[keep_number[select_number]].index
                            else:
                                map_table[c_in + j][translate_value][0] = weight_importance[i].index
                                map_table[c_in + j][translate_value][1] = -1
                        # 统计重用weight-pattern的个数
                        translate_value = translate_value + 1
                    else:
                        # 记录保留weight-pattern的索引
                        keep_number[keep_value] = weight_importance[i].index
                        # 将该weight-pattern加入保留weight-pattern矩阵
                        for j in range(0, 8):
                            keep_weight_matrix[keep_value][j] = weight_matrix_pruning[weight_importance[i].index][c_in + j]
                        # 统计保留weight-pattern的个数
                        keep_value = keep_value + 1
                # 统计该层重用weight-pattern数量
                total_translate_weight_pattern = total_translate_weight_pattern + translate_value
                # 给map_table设置结束标志
                for j in range(0, 8):
                    map_table[c_in + j][translate_value][0] = -1

            # 计算该层weight-pattern重用率
            weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / 8 * out_channel)
            return map_table, weight_pattern_reuse_ratio


def pattern_shape_and_value_similar_translate(model, in_channel, out_channel, weight_name, threshold, kernel_size, channel_number, mask):
    total_translate_weight_pattern = 0  # 统计重用weight-pattern的总数

    # 构造矩阵加速代码执行速度
    weight_matrix = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    weight_matrix_pruning = torch.zeros(out_channel, in_channel, kernel_size, kernel_size)
    if 'fc' in weight_name:
        weight_matrix = torch.zeros(out_channel, in_channel)
        weight_matrix_pruning = torch.zeros(out_channel, in_channel)

    # 先对权重进行量化
    max_value = torch.max(model.state_dict()[weight_name]).item()
    min_value = torch.min(model.state_dict()[weight_name]).item()
    if max_value < -min_value:
        max_value = -min_value
    scale = (max_value - 0) / (127 * 256)
    for c_out in range(0, out_channel):
        weight_matrix[c_out] = torch.round(model.state_dict()[weight_name][c_out] / scale)
        weight_matrix_pruning[c_out] = weight_matrix[c_out] * mask[c_out]

    if 'fc' in weight_name:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        multiple_relationship_table = torch.ones(out_channel, in_channel)  # 存储weight-pattern之间的倍数关系
        # 进行kernel级模式匹配
        for c_in in range(0, in_channel, channel_number):
            threshold_value = int(out_channel * threshold)  # 最终保留的模式数
            translate_value = out_channel - threshold_value  # 最终转换的模式数
            # 找到参数绝对值最小的模式
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix_pruning[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            # 统计保留模式的索引
            keep_number = [0] * threshold_value  # 保留模式的索引
            y_importance = torch.zeros(threshold_value + 1)  # 记录保留模式中的权重标准值
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weight_matrix_pruning[weight_importance[i].index][c_in + j] = torch.round(weight_matrix_pruning[weight_importance[i].index][c_in + j] / math.pow(2, 8)) * math.pow(2, 8)
                    y_importance[i] = y_importance[i] + weight_matrix_pruning[weight_importance[i].index][c_in + j].abs().sum().item()
                if y_importance[i] != 0:
                    keep_number[i] = weight_importance[i].index
                else:
                    threshold_value = i
                    translate_value = out_channel - threshold_value
                    break
            # 统计转换模式的索引
            translate_number = [0] * translate_value  # 转换模式的索引
            x_importance = torch.zeros(threshold_value)  # 记录转换模式中的权重标准值
            similarity = torch.zeros(threshold_value + 1)  # 记录相似系数
            y_mask = torch.zeros(threshold_value, channel_number)  # 构造保留模式的形状矩阵
            weightx_value = torch.zeros(translate_value, channel_number)  # 记录转换模式中的权重数值
            weightx_value_pruning = torch.zeros(threshold_value + 1, channel_number)  # 记录剪枝后转换模式中的权重数值
            weighty_value = torch.zeros(threshold_value, channel_number)  # 记录保留模式中的权重数值
            similar_value = torch.zeros(threshold_value + 1, channel_number)  # 记录乘以相似系数后保留模式中的权重数值
            # 构造保留weight-pattern矩阵
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    y_mask[i][j] = mask[keep_number[i]][c_in + j]
                    weighty_value[i][j] = weight_matrix_pruning[keep_number[i]][c_in + j]
            # 将绝对值最小的模式加入转换模式集合
            for i in range(0, translate_value):
                translate_number[i] = weight_importance[i + threshold_value].index
                for j in range(0, channel_number):
                    weightx_value[i][j] = weight_matrix[translate_number[i]][c_in + j]
            # 为每个要剪枝的模式匹配剩余最相似的模式
            for i in range(0, translate_value):
                for j in range(0, threshold_value):
                    x_importance[j] = 0
                    weightx_value_pruning[j] = weightx_value[i] * y_mask[j]
                    x_importance[j] = weightx_value_pruning[j].abs().sum().item()
                    if x_importance[j] != 0:
                        similarity[j] = math.ceil(math.pow(2, 8) * math.pow(2, round(math.log2(x_importance[j] / y_importance[j])))) / math.pow(2, 8)
                        similar_value[j] = weighty_value[j] * similarity[j]
                    else:
                        similarity[j] = 0
                        similar_value[j] = 0
                for j in range(0, channel_number):
                    weightx_value_pruning[threshold_value][j] = weightx_value[i][j] * mask[translate_number[i]][c_in + j]
                select_number = (similar_value - weightx_value_pruning).abs().sum(axis=1).argmin().item()
                for j in range(0, channel_number):
                    # 构建映射表
                    map_table[c_in + j][i][0] = translate_number[i]
                    if select_number != threshold_value and similarity[select_number] != 0:
                        map_table[c_in + j][i][1] = keep_number[select_number]
                        multiple_relationship_table[translate_number[i]][c_in + j] = similarity[select_number]
                    else:
                        map_table[c_in + j][i][1] = -1
                        multiple_relationship_table[translate_number[i]][c_in + j] = 0

            # 统计重用weight-pattern的总数
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            if translate_value < out_channel:
                for j in range(0, channel_number):
                    map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, multiple_relationship_table, weight_pattern_reuse_ratio

    else:
        map_table = torch.zeros(in_channel, out_channel, 2)  # 映射表，最后一维第一个数存被转换weight-pattern的索引，第二个数存目标weight-pattern的索引
        multiple_relationship_table = torch.ones(out_channel, in_channel, kernel_size, kernel_size)  # 存储weight-pattern之间的倍数关系
        # 进行kernel级模式匹配
        for c_in in range(0, in_channel, channel_number):
            threshold_value = int(out_channel * threshold)  # 最终保留的模式数
            translate_value = out_channel - threshold_value  # 最终转换的模式数
            # 找到参数绝对值最小的模式
            weight_importance = [Node(i, 0.0) for i in range(0, out_channel)]  # 统计模式的绝对值大小
            for i in range(0, out_channel):
                for j in range(0, channel_number):
                    weight_importance[i].value = weight_importance[i].value + weight_matrix_pruning[i][c_in + j].abs().sum().item()
            weight_importance.sort(key=lambda f: f.value, reverse=True)
            # 统计保留模式的索引
            keep_number = [0] * threshold_value  # 保留模式的索引
            y_importance = torch.zeros(threshold_value + 1)  # 记录保留模式中的权重标准值
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    weight_matrix_pruning[weight_importance[i].index][c_in + j] = torch.round(weight_matrix_pruning[weight_importance[i].index][c_in + j] / math.pow(2, 8)) * math.pow(2, 8)
                    y_importance[i] = y_importance[i] + weight_matrix_pruning[weight_importance[i].index][c_in + j].abs().sum().item()
                if y_importance[i] != 0:
                    keep_number[i] = weight_importance[i].index
                else:
                    threshold_value = i
                    translate_value = out_channel - threshold_value
                    break
            # 统计转换模式的索引
            translate_number = [0] * translate_value  # 转换模式的索引
            x_importance = torch.zeros(threshold_value)  # 记录转换模式中的权重标准值
            similarity = torch.zeros(threshold_value + 1)  # 记录相似系数
            y_mask = torch.zeros(threshold_value, channel_number, kernel_size, kernel_size)  # 构造保留模式的形状矩阵
            weightx_value = torch.zeros(translate_value, channel_number, kernel_size, kernel_size)  # 记录转换模式中的权重数值
            weightx_value_pruning = torch.zeros(threshold_value + 1, channel_number, kernel_size, kernel_size)  # 记录剪枝后转换模式中的权重数值
            weighty_value = torch.zeros(threshold_value, channel_number, kernel_size, kernel_size)  # 记录保留模式中的权重数值
            similar_value = torch.zeros(threshold_value + 1, channel_number, kernel_size, kernel_size)  # 记录乘以相似系数后保留模式中的权重数值
            # 构造保留weight-pattern矩阵
            for i in range(0, threshold_value):
                for j in range(0, channel_number):
                    y_mask[i][j] = mask[keep_number[i]][c_in + j]
                    weighty_value[i][j] = weight_matrix_pruning[keep_number[i]][c_in + j]
            # 将绝对值最小的模式加入转换模式集合
            for i in range(0, translate_value):
                translate_number[i] = weight_importance[i + threshold_value].index
                for j in range(0, channel_number):
                    weightx_value[i][j] = weight_matrix[translate_number[i]][c_in + j]
            # 为每个要剪枝的模式匹配剩余最相似的模式
            for i in range(0, translate_value):
                for j in range(0, threshold_value):
                    x_importance[j] = 0
                    weightx_value_pruning[j] = weightx_value[i] * y_mask[j]
                    x_importance[j] = weightx_value_pruning[j].abs().sum().item()
                    if x_importance[j] != 0:
                        similarity[j] = math.ceil(math.pow(2, 8) * math.pow(2, round(math.log2(x_importance[j] / y_importance[j])))) / math.pow(2, 8)
                        similar_value[j] = weighty_value[j] * similarity[j]
                    else:
                        similarity[j] = 0
                        similar_value[j] = torch.zeros(channel_number, kernel_size, kernel_size)
                for j in range(0, channel_number):
                    weightx_value_pruning[threshold_value][j] = weightx_value[i][j] * mask[translate_number[i]][c_in + j]
                select_number = (similar_value - weightx_value_pruning).abs().sum(axis=3).sum(axis=2).sum(axis=1).argmin().item()
                for j in range(0, channel_number):
                    # 构建映射表
                    map_table[c_in + j][i][0] = translate_number[i]
                    if select_number != threshold_value and similarity[select_number] != 0:
                        map_table[c_in + j][i][1] = keep_number[select_number]
                        multiple_relationship_table[translate_number[i]][c_in + j] = similarity[select_number]
                    else:
                        map_table[c_in + j][i][1] = -1
                        multiple_relationship_table[translate_number[i]][c_in + j] = 0

            # 统计重用weight-pattern的总数
            total_translate_weight_pattern = total_translate_weight_pattern + translate_value
            # 给map_table设置结束标志
            if translate_value < out_channel:
                for j in range(0, channel_number):
                    map_table[c_in + j][translate_value][0] = -1

        # 计算该层weight-pattern重用率
        weight_pattern_reuse_ratio = total_translate_weight_pattern / (in_channel / channel_number * out_channel)
        return map_table, multiple_relationship_table, weight_pattern_reuse_ratio


def _get_block_step(channel_number: int) -> int:
    return max(int(channel_number) if channel_number is not None else 1, 1)


def _get_channel_span(in_channel: int, in_ch_start: int, channel_number: int) -> int:
    return max(1, min(_get_block_step(channel_number), in_channel - in_ch_start))


def _extract_block_tensor(weight_tensor: torch.Tensor, out_ch: int, in_ch_start: int, channel_span: int) -> torch.Tensor:
    if weight_tensor.dim() == 4:
        return weight_tensor[out_ch, in_ch_start:in_ch_start + channel_span, :, :].clone()
    return weight_tensor[out_ch, in_ch_start:in_ch_start + channel_span].clone()


def _assign_block_tensor(weight_tensor: torch.Tensor, out_ch: int, in_ch_start: int, channel_span: int, block_value: torch.Tensor):
    if weight_tensor.dim() == 4:
        weight_tensor[out_ch, in_ch_start:in_ch_start + channel_span, :, :] = block_value
    else:
        weight_tensor[out_ch, in_ch_start:in_ch_start + channel_span] = block_value


def _tensor_mask_signature(mask_tensor: torch.Tensor) -> bytes:
    return mask_tensor.flatten().to(torch.uint8).cpu().numpy().tobytes()


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _build_shared_topk_mask(model, weight_name, in_channel, out_channel, channel_number, keep_ratio):
    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    pattern_mask = torch.zeros_like(weight_tensor)
    block_step = _get_block_step(channel_number)

    for in_ch_start in range(0, in_channel, block_step):
        channel_span = _get_channel_span(in_channel, in_ch_start, channel_number)
        block_weight = weight_tensor[:, in_ch_start:in_ch_start + channel_span].abs().sum(dim=0)
        flat_block = block_weight.reshape(-1)
        keep_count = max(1, min(flat_block.numel(), int(round(flat_block.numel() * keep_ratio))))
        keep_indices = torch.topk(flat_block, keep_count).indices
        block_mask = torch.zeros_like(flat_block)
        block_mask[keep_indices] = 1.0
        block_mask = block_mask.reshape(block_weight.shape)
        for out_ch in range(0, out_channel):
            _assign_block_tensor(pattern_mask, out_ch, in_ch_start, channel_span, block_mask)

    return pattern_mask


def _build_per_out_topk_mask(model, weight_name, in_channel, out_channel, channel_number, keep_ratio):
    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    pattern_mask = torch.zeros_like(weight_tensor)
    block_step = _get_block_step(channel_number)

    for out_ch in range(0, out_channel):
        for in_ch_start in range(0, in_channel, block_step):
            channel_span = _get_channel_span(in_channel, in_ch_start, channel_number)
            block_weight = _extract_block_tensor(weight_tensor, out_ch, in_ch_start, channel_span).abs()
            flat_block = block_weight.reshape(-1)
            keep_count = max(1, min(flat_block.numel(), int(round(flat_block.numel() * keep_ratio))))
            keep_indices = torch.topk(flat_block, keep_count).indices
            block_mask = torch.zeros_like(flat_block)
            block_mask[keep_indices] = 1.0
            block_mask = block_mask.reshape(block_weight.shape)
            _assign_block_tensor(pattern_mask, out_ch, in_ch_start, channel_span, block_mask)

    return pattern_mask


def extract_ou_patterns(model, weight_name, in_channel, out_channel, kernel_size, channel_number, mask=None):
    """按真实OU/block边界提取模式，保留成员的block元数据。"""
    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    if mask is None:
        mask_tensor = torch.ones_like(weight_tensor)
    else:
        mask_tensor = mask.detach().cpu()
        if mask_tensor.shape != weight_tensor.shape:
            mask_tensor = torch.ones_like(weight_tensor)

    block_step = _get_block_step(channel_number)
    pattern_list = []
    for out_ch in range(0, out_channel):
        for in_ch_start in range(0, in_channel, block_step):
            channel_span = _get_channel_span(in_channel, in_ch_start, channel_number)
            raw = _extract_block_tensor(weight_tensor, out_ch, in_ch_start, channel_span)
            current_mask = _extract_block_tensor(mask_tensor, out_ch, in_ch_start, channel_span)
            masked = raw * current_mask
            pattern_list.append({
                'out_ch': out_ch,
                'in_ch_start': in_ch_start,
                'channel_span': channel_span,
                'raw': raw,
                'masked': masked,
                'norm1': masked.abs().sum().item(),
                'norm2': torch.norm(masked.reshape(-1).float(), p=2).item(),
                'mask_signature': _tensor_mask_signature(current_mask),
                'multiplier': 1.0,
                'block_kind': 'fc' if 'fc' in weight_name else ('conv1x1' if kernel_size == 1 else 'conv3x3'),
                'block_index': in_ch_start // block_step,
            })
    return pattern_list


def search_best_power_of_two_scale(source_tensor, target_tensor, scale_candidates=None):
    """在离散power-of-two候选中搜索最佳缩放关系。"""
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]

    source = source_tensor.reshape(-1).float()
    target = target_tensor.reshape(-1).float()
    source_norm = source.abs().sum().item()
    target_norm = target.abs().sum().item()

    if target_norm <= 1e-8:
        best_error = source.abs().sum().item()
        best_similarity = 1.0 if source_norm <= 1e-8 else max(0.0, 1.0 - best_error / (source_norm + 1e-8))
        return 0.0, best_error, best_similarity

    best_scale = 1.0
    best_error = None
    best_similarity = -1.0

    for scale in scale_candidates:
        error = (source - scale * target).abs().sum().item()
        similarity = max(0.0, 1.0 - error / (source_norm + 1e-8))
        if best_error is None or error < best_error or (abs(error - best_error) <= 1e-8 and similarity > best_similarity):
            best_scale = float(scale)
            best_error = error
            best_similarity = similarity

    return best_scale, best_error, best_similarity


def select_group_prototype(pattern_list, scale_candidates=None):
    """选择组原型，优先选与其他成员总误差最小的medoid。"""
    if not pattern_list:
        return 0
    if len(pattern_list) == 1:
        return 0

    best_index = 0
    best_score = None
    for i, candidate in enumerate(pattern_list):
        total_error = 0.0
        total_similarity = 0.0
        for j, other in enumerate(pattern_list):
            if i == j:
                continue
            _, error, similarity = search_best_power_of_two_scale(
                other['masked'], candidate['masked'], scale_candidates
            )
            total_error += error
            total_similarity += similarity
        candidate_score = (total_error, -total_similarity, -candidate['norm1'])
        if best_score is None or candidate_score < best_score:
            best_score = candidate_score
            best_index = i

    return best_index


def build_ft_groups_for_block(pattern_list, min_group_size, target_group_size, sim_threshold, exact_threshold, scale_candidates):
    """在同一个输入通道block内按mask形状和缩放相似度构建FT groups。"""
    if not pattern_list:
        return []

    buckets: Dict[bytes, List[Dict[str, Any]]] = {}
    for pattern in pattern_list:
        buckets.setdefault(pattern['mask_signature'], []).append(pattern)

    block_groups: List[Dict[str, Any]] = []
    for mask_signature, bucket_patterns in buckets.items():
        unassigned = sorted(bucket_patterns, key=lambda item: item['norm1'], reverse=True)

        while unassigned:
            prototype_seed = unassigned.pop(0)
            current_patterns = [prototype_seed]
            candidates_kept = []

            for candidate in unassigned:
                _, _, similarity = search_best_power_of_two_scale(
                    candidate['masked'], prototype_seed['masked'], scale_candidates
                )
                if similarity >= sim_threshold and len(current_patterns) < target_group_size:
                    current_patterns.append(candidate)
                else:
                    candidates_kept.append(candidate)
            unassigned = candidates_kept

            prototype_index = select_group_prototype(current_patterns, scale_candidates)
            prototype_pattern = current_patterns[prototype_index]

            member_entries = []
            exact_group = True
            for member in current_patterns:
                if member is prototype_pattern:
                    scale = 1.0
                    similarity = 1.0
                    role = 'prototype'
                else:
                    scale, _, similarity = search_best_power_of_two_scale(
                        member['masked'], prototype_pattern['masked'], scale_candidates
                    )
                    role = 'member'
                    if abs(scale - 1.0) > 1e-6:
                        exact_group = False
                    if similarity < exact_threshold:
                        exact_group = False

                member_entries.append({
                    'pattern': member,
                    'multiplier': float(scale),
                    'similarity': float(similarity),
                    'role': role,
                })

            block_groups.append({
                'mask_signature': mask_signature,
                'prototype': prototype_pattern,
                'members': member_entries,
                'repair_mode': 'exact' if exact_group else 'scaled',
            })

    non_singleton_groups = [group for group in block_groups if len(group['members']) >= min_group_size]
    singleton_groups = [group for group in block_groups if len(group['members']) < min_group_size]

    for singleton in singleton_groups:
        member = singleton['members'][0]
        best_target = None
        best_similarity = -1.0
        for target_group in non_singleton_groups:
            if len(target_group['members']) >= target_group_size:
                continue
            if target_group['mask_signature'] != singleton['mask_signature']:
                continue

            scale, _, similarity = search_best_power_of_two_scale(
                member['pattern']['masked'],
                target_group['prototype']['masked'],
                scale_candidates,
            )
            if similarity >= sim_threshold and similarity > best_similarity:
                best_target = (target_group, scale, similarity)
                best_similarity = similarity

        if best_target is not None:
            target_group, scale, similarity = best_target
            member['multiplier'] = float(scale)
            member['similarity'] = float(similarity)
            member['role'] = 'member'
            target_group['members'].append(member)
            if abs(scale - 1.0) > 1e-6 or similarity < exact_threshold:
                target_group['repair_mode'] = 'scaled'

    merged_singletons = {id(group) for group in singleton_groups if any(group['members'][0] is member['pattern'] for target in non_singleton_groups for member in target['members'])}
    final_groups = non_singleton_groups + [group for group in singleton_groups if id(group) not in merged_singletons]
    final_groups.sort(key=lambda item: (item['prototype']['in_ch_start'], item['prototype']['out_ch']))
    return final_groups


def _build_layer_ft_groups(pattern_list, min_group_size, target_group_size, sim_threshold, exact_threshold, scale_candidates):
    block_map: Dict[int, List[Dict[str, Any]]] = {}
    for pattern in pattern_list:
        block_map.setdefault(pattern['in_ch_start'], []).append(pattern)

    layer_groups = []
    group_id = 0
    repairable_block_count = 0
    repairable_ou_count = 0
    singleton_group_count = 0
    exact_group_count = 0
    scaled_group_count = 0

    for block_start in sorted(block_map.keys()):
        block_patterns = block_map[block_start]
        for group in build_ft_groups_for_block(
            block_patterns,
            min_group_size=min_group_size,
            target_group_size=target_group_size,
            sim_threshold=sim_threshold,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
        ):
            normalized_members = []
            for member in group['members']:
                pattern = member['pattern']
                normalized_members.append({
                    'out_ch': pattern['out_ch'],
                    'in_ch_start': pattern['in_ch_start'],
                    'channel_span': pattern['channel_span'],
                    'mask_signature': pattern['mask_signature'].hex(),
                    'multiplier': float(member['multiplier']),
                    'similarity': float(member['similarity']),
                    'role': member['role'],
                })

            member_count = len(normalized_members)
            covered_ou_count = int(sum(member['channel_span'] for member in normalized_members))
            layer_groups.append({
                'group_id': group_id,
                'prototype': {
                    'out_ch': group['prototype']['out_ch'],
                    'in_ch_start': group['prototype']['in_ch_start'],
                    'channel_span': group['prototype']['channel_span'],
                    'mask_signature': group['prototype']['mask_signature'].hex(),
                    'multiplier': 1.0,
                },
                'members': normalized_members,
                'mask_signature': group['mask_signature'].hex(),
                'group_size': member_count,
                'member_count': member_count,
                'covered_ou_count': covered_ou_count,
                'repair_mode': group['repair_mode'],
            })
            if member_count >= min_group_size:
                repairable_block_count += member_count
                repairable_ou_count += covered_ou_count
                if group['repair_mode'] == 'exact':
                    exact_group_count += 1
                else:
                    scaled_group_count += 1
            else:
                singleton_group_count += 1
            group_id += 1

    total_block_count = len(pattern_list)
    total_ou_count = int(sum(pattern['channel_span'] for pattern in pattern_list))
    layer_summary = {
        'group_count': len(layer_groups),
        'block_count': total_block_count,
        'ou_count': total_ou_count,
        'repairable_block_count': repairable_block_count,
        'repairable_ou_count': repairable_ou_count,
        'coverage_ratio': repairable_ou_count / max(total_ou_count, 1),
        'block_coverage_ratio': repairable_block_count / max(total_block_count, 1),
        'singleton_group_count': singleton_group_count,
        'singleton_ratio': singleton_group_count / max(len(layer_groups), 1),
        'avg_group_size': _mean_or_zero([group['group_size'] for group in layer_groups]),
        'exact_group_count': exact_group_count,
        'scaled_group_count': scaled_group_count,
        'exact_group_ratio': exact_group_count / max(len(layer_groups), 1),
        'scaled_group_ratio': scaled_group_count / max(len(layer_groups), 1),
    }
    return layer_groups, layer_summary


def summarize_group_information(group_information, layer_names=None, ft_layer_enabled=None):
    if not group_information:
        return {
            'coverage_ratio': 0.0,
            'block_coverage_ratio': 0.0,
            'group_count': 0,
            'singleton_ratio': 0.0,
            'avg_group_size': 0.0,
            'exact_group_proportion': 0.0,
            'scaled_group_proportion': 0.0,
            'repairable_ou_count': 0,
            'total_ou_count': 0,
            'repairable_block_count': 0,
            'total_block_count': 0,
        }

    total_ou_count = 0
    total_block_count = 0
    repairable_ou_count = 0
    repairable_block_count = 0
    total_groups = 0
    singleton_groups = 0
    exact_groups = 0
    scaled_groups = 0
    group_sizes = []

    target_layers = layer_names if layer_names is not None else list(group_information.keys())
    for layer_idx, layer_name in enumerate(target_layers):
        if ft_layer_enabled is not None and layer_idx < len(ft_layer_enabled) and not ft_layer_enabled[layer_idx]:
            continue
        layer_group_info = group_information.get(layer_name)
        if not layer_group_info:
            continue

        total_ou_count += int(layer_group_info.get('ou_count', 0))
        total_block_count += int(layer_group_info.get('block_count', layer_group_info.get('ou_count', 0)))
        repairable_ou_count += int(layer_group_info.get('repairable_ou_count', 0))
        repairable_block_count += int(layer_group_info.get('repairable_block_count', 0))
        layer_groups = layer_group_info.get('groups', [])
        total_groups += len(layer_groups)

        for group in layer_groups:
            group_size = int(group.get('group_size', len(group.get('members', []))))
            group_sizes.append(group_size)
            if group_size < 2:
                singleton_groups += 1
            elif group.get('repair_mode', 'scaled') == 'exact':
                exact_groups += 1
            else:
                scaled_groups += 1

    return {
        'coverage_ratio': repairable_ou_count / max(total_ou_count, 1),
        'block_coverage_ratio': repairable_block_count / max(total_block_count, 1),
        'group_count': total_groups,
        'singleton_ratio': singleton_groups / max(total_groups, 1),
        'avg_group_size': _mean_or_zero(group_sizes),
        'exact_group_proportion': exact_groups / max(total_groups, 1),
        'scaled_group_proportion': scaled_groups / max(total_groups, 1),
        'repairable_ou_count': repairable_ou_count,
        'total_ou_count': total_ou_count,
        'repairable_block_count': repairable_block_count,
        'total_block_count': total_block_count,
    }


def ft_group_score_mask(model, weight_name, in_channel, out_channel, kernel_size,
                        channel_number, pattern_value_number, pattern_shape_number,
                        OU_size, target_group_size=4, sim_threshold=0.85):
    """用FTScore在多个mask候选间搜索最适合冗余组构建的shape/mask。"""
    try:
        shape_seed_mask = get_shape_mask(
            model=model,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            channel_number=channel_number,
            pattern_value_number=pattern_value_number,
            pattern_shape_number=pattern_shape_number,
            OU_size=OU_size,
        )
    except Exception:
        shape_seed_mask = torch.ones_like(model.state_dict()[weight_name].detach().cpu())

    raw_weight = model.state_dict()[weight_name].detach().cpu()
    seed_density = float(shape_seed_mask.sum().item() / max(shape_seed_mask.numel(), 1))
    keep_ratios = sorted({
        max(0.15, min(1.0, round(seed_density * 0.75, 4))),
        max(0.15, min(1.0, round(seed_density, 4))),
        max(0.15, min(1.0, round(seed_density * 1.2 + 0.02, 4))),
    })

    candidate_masks = [('shape_seed', shape_seed_mask)]
    for keep_ratio in keep_ratios:
        candidate_masks.append((f'shared_topk_{keep_ratio:.4f}', _build_shared_topk_mask(
            model=model,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            channel_number=channel_number,
            keep_ratio=keep_ratio,
        )))
        candidate_masks.append((f'per_out_topk_{keep_ratio:.4f}', _build_per_out_topk_mask(
            model=model,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            channel_number=channel_number,
            keep_ratio=keep_ratio,
        )))
    candidate_masks.append(('dense_mask', torch.ones_like(raw_weight)))

    valid_candidates = []
    candidate_summaries = []
    for strategy_name, candidate_mask in candidate_masks:
        pattern_list = extract_ou_patterns(
            model=model,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            channel_number=channel_number,
            mask=candidate_mask,
        )
        estimated_groups, estimated_summary = _build_layer_ft_groups(
            pattern_list=pattern_list,
            min_group_size=2,
            target_group_size=target_group_size,
            sim_threshold=sim_threshold,
            exact_threshold=0.98,
            scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
        )

        member_similarities = []
        for group in estimated_groups:
            for member in group.get('members', []):
                if member.get('role') != 'prototype':
                    member_similarities.append(float(member.get('similarity', 1.0)))

        pruning_distortion = (raw_weight - raw_weight * candidate_mask).abs().sum().item() / (raw_weight.abs().sum().item() + 1e-8)
        avg_similarity = _mean_or_zero(member_similarities) if member_similarities else (1.0 if estimated_groups else 0.0)
        avg_group_size = float(estimated_summary.get('avg_group_size', 0.0))
        normalized_group_size = min(avg_group_size / max(float(target_group_size), 1.0), 1.0)
        ft_score = (
            0.42 * estimated_summary['coverage_ratio']
            + 0.22 * avg_similarity
            + 0.18 * normalized_group_size
            + 0.08 * estimated_summary['exact_group_ratio']
            - 0.18 * pruning_distortion
        )

        candidate_summary = {
            'strategy': strategy_name,
            'score': float(ft_score),
            'estimated_coverage': float(estimated_summary['coverage_ratio']),
            'estimated_block_coverage': float(estimated_summary['block_coverage_ratio']),
            'estimated_avg_group_size': avg_group_size,
            'estimated_group_count': int(estimated_summary['group_count']),
            'estimated_singleton_ratio': float(estimated_summary['singleton_ratio']),
            'avg_intra_group_similarity': float(avg_similarity),
            'pruning_distortion': float(pruning_distortion),
            'selected': False,
        }
        candidate_summaries.append(candidate_summary)
        valid_candidates.append((candidate_summary, candidate_mask))

    fallback_used = False
    if valid_candidates:
        best_summary, best_mask = max(
            valid_candidates,
            key=lambda item: (
                item[0]['score'],
                item[0]['estimated_coverage'],
                item[0]['avg_intra_group_similarity'],
                item[0]['estimated_avg_group_size'],
                1 if item[0]['strategy'] != 'shape_seed' else 0,
            ),
        )
    else:
        best_summary = {
            'strategy': 'shape_seed',
            'score': 0.0,
            'estimated_coverage': 0.0,
            'estimated_block_coverage': 0.0,
            'estimated_avg_group_size': 0.0,
            'estimated_group_count': 0,
            'estimated_singleton_ratio': 1.0,
            'avg_intra_group_similarity': 0.0,
            'pruning_distortion': 0.0,
            'selected': True,
        }
        best_mask = shape_seed_mask
        fallback_used = True

    for summary in candidate_summaries:
        if summary['strategy'] == best_summary['strategy']:
            summary['selected'] = True

    group_seed_info = {
        'selected_strategy': best_summary['strategy'],
        'estimated_coverage': best_summary['estimated_coverage'],
        'estimated_block_coverage': best_summary['estimated_block_coverage'],
        'estimated_avg_group_size': best_summary['estimated_avg_group_size'],
        'estimated_group_count': best_summary['estimated_group_count'],
        'estimated_singleton_ratio': best_summary['estimated_singleton_ratio'],
        'avg_intra_group_similarity': best_summary['avg_intra_group_similarity'],
        'pruning_distortion': best_summary['pruning_distortion'],
        'candidate_count': len(candidate_summaries),
        'fallback_used': fallback_used,
        'candidate_summaries': candidate_summaries,
    }
    return best_mask, group_seed_info


def ft_group_cluster_translate(model, in_channel, out_channel, weight_name,
                               kernel_size, channel_number, mask,
                               min_group_size=2, target_group_size=4,
                               sim_threshold=0.85, exact_threshold=0.98,
                               scale_candidates=None):
    """构建FT-oriented group，并输出兼容旧接口的map/multiple文件。"""
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]

    pattern_list = extract_ou_patterns(
        model=model,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        channel_number=channel_number,
        mask=mask,
    )

    layer_groups, layer_summary = _build_layer_ft_groups(
        pattern_list=pattern_list,
        min_group_size=min_group_size,
        target_group_size=target_group_size,
        sim_threshold=sim_threshold,
        exact_threshold=exact_threshold,
        scale_candidates=scale_candidates,
    )

    map_table = torch.full((in_channel, out_channel, 2), -1, dtype=torch.long)
    if 'fc' in weight_name:
        multiple_relationship_table = torch.ones(out_channel, in_channel)
    else:
        multiple_relationship_table = torch.ones(out_channel, in_channel, kernel_size, kernel_size)

    write_cursors = [0] * in_channel
    for group in layer_groups:
        prototype = group['prototype']
        proto_out = int(prototype['out_ch'])
        proto_in = int(prototype['in_ch_start'])
        proto_span = int(prototype.get('channel_span', 1))

        for offset in range(0, proto_span):
            member_in = proto_in + offset
            if 'fc' in weight_name:
                multiple_relationship_table[proto_out][member_in] = 1.0
            else:
                multiple_relationship_table[proto_out][member_in] = torch.ones(kernel_size, kernel_size)

        for member in group['members']:
            member_out = int(member['out_ch'])
            member_in = int(member['in_ch_start'])
            member_span = int(member.get('channel_span', 1))
            multiplier = float(member.get('multiplier', 1.0))

            for offset in range(0, member_span):
                target_in = member_in + offset
                if 'fc' in weight_name:
                    multiple_relationship_table[member_out][target_in] = multiplier
                else:
                    multiple_relationship_table[member_out][target_in] = torch.ones(kernel_size, kernel_size) * multiplier

                if member.get('role') == 'prototype':
                    continue

                cursor = write_cursors[target_in]
                if cursor < out_channel:
                    map_table[target_in][cursor][0] = member_out
                    map_table[target_in][cursor][1] = proto_out
                    write_cursors[target_in] += 1

    layer_group_information = {
        'layer_name': weight_name,
        'group_count': layer_summary['group_count'],
        'block_count': layer_summary['block_count'],
        'ou_count': layer_summary['ou_count'],
        'repairable_block_count': layer_summary['repairable_block_count'],
        'repairable_ou_count': layer_summary['repairable_ou_count'],
        'coverage_ratio': layer_summary['coverage_ratio'],
        'block_coverage_ratio': layer_summary['block_coverage_ratio'],
        'singleton_group_count': layer_summary['singleton_group_count'],
        'singleton_ratio': layer_summary['singleton_ratio'],
        'avg_group_size': layer_summary['avg_group_size'],
        'exact_group_count': layer_summary['exact_group_count'],
        'scaled_group_count': layer_summary['scaled_group_count'],
        'exact_group_ratio': layer_summary['exact_group_ratio'],
        'scaled_group_ratio': layer_summary['scaled_group_ratio'],
        'groups': layer_groups,
    }
    return map_table, multiple_relationship_table, layer_summary['coverage_ratio'], layer_group_information


def _get_group_member_entries(layer_group_information):
    if not layer_group_information:
        return []
    return layer_group_information.get('groups', [])


def _apply_ft_group_projection(model, weight_name, mask, group_information):
    with torch.no_grad():
        state_dict = model.state_dict()
        for layer_name in weight_name:
            if layer_name not in state_dict:
                continue

            layer_weight = state_dict[layer_name]
            layer_mask = mask.get(layer_name)
            if layer_mask is not None and layer_mask.shape == layer_weight.shape:
                layer_weight.mul_(layer_mask.to(layer_weight.device))

            layer_group_information = group_information.get(layer_name)
            if not layer_group_information:
                continue

            for group in _get_group_member_entries(layer_group_information):
                prototype = group['prototype']
                proto_out = int(prototype['out_ch'])
                proto_in = int(prototype['in_ch_start'])
                proto_span = int(prototype.get('channel_span', 1))
                proto_multiplier = float(prototype.get('multiplier', 1.0))
                prototype_value = _extract_block_tensor(layer_weight, proto_out, proto_in, proto_span)

                for member in group['members']:
                    member_out = int(member['out_ch'])
                    member_in = int(member['in_ch_start'])
                    member_span = int(member.get('channel_span', 1))
                    member_multiplier = float(member.get('multiplier', 1.0))
                    if member_span != proto_span:
                        continue
                    if abs(proto_multiplier) > 1e-8:
                        scale = member_multiplier / proto_multiplier
                    else:
                        scale = 1.0
                    _assign_block_tensor(layer_weight, member_out, member_in, member_span, prototype_value * scale)


def _compute_ft_regularization(model, weight_name, ft_layer_enabled, mask, group_information, device):
    loss_mask = torch.zeros(1, device=device)
    loss_proto = torch.zeros(1, device=device)
    loss_balance = torch.zeros(1, device=device)
    loss_sep = torch.zeros(1, device=device)

    parameter_map = dict(model.named_parameters())
    for layer_index, layer_name in enumerate(weight_name):
        if not ft_layer_enabled[layer_index] or layer_name not in parameter_map:
            continue

        layer_param = parameter_map[layer_name]
        layer_mask = mask.get(layer_name)
        if layer_mask is not None and layer_mask.shape == layer_param.shape:
            layer_mask = layer_mask.to(device)
            projected = layer_param * layer_mask
            loss_mask = loss_mask + 0.5 * torch.sum(torch.pow(layer_param - projected, 2))
        else:
            projected = layer_param

        layer_group_information = group_information.get(layer_name)
        if not layer_group_information:
            continue

        for group in _get_group_member_entries(layer_group_information):
            group_size = int(group.get('group_size', len(group.get('members', []))))
            if group_size < 2:
                loss_balance = loss_balance + torch.tensor(float((2 - group_size) ** 2), device=device)
                continue

            prototype = group['prototype']
            proto_out = int(prototype['out_ch'])
            proto_in = int(prototype['in_ch_start'])
            proto_span = int(prototype.get('channel_span', 1))
            prototype_multiplier = float(prototype.get('multiplier', 1.0))
            prototype_weight = _extract_block_tensor(projected, proto_out, proto_in, proto_span)

            for member in group['members']:
                member_out = int(member['out_ch'])
                member_in = int(member['in_ch_start'])
                member_span = int(member.get('channel_span', 1))
                member_multiplier = float(member.get('multiplier', 1.0))
                if member_span != proto_span:
                    continue
                if abs(prototype_multiplier) > 1e-8:
                    scale = member_multiplier / prototype_multiplier
                else:
                    scale = 1.0
                member_weight = _extract_block_tensor(projected, member_out, member_in, member_span)
                loss_proto = loss_proto + torch.sum(torch.pow(member_weight - prototype_weight * scale, 2))

                proto_vec = prototype_weight.reshape(-1).float()
                member_vec = member_weight.reshape(-1).float()
                proto_norm = torch.norm(proto_vec, p=2)
                member_norm = torch.norm(member_vec, p=2)
                if proto_norm > 1e-8 and member_norm > 1e-8:
                    cosine = torch.dot(member_vec, proto_vec) / (member_norm * proto_norm)
                    loss_sep = loss_sep + (1.0 - cosine)

    return loss_mask, loss_proto, loss_balance, loss_sep


def _refresh_acceptance_score(summary: Dict[str, float]) -> float:
    normalized_group_size = min(float(summary.get('avg_group_size', 0.0)) / 4.0, 1.0)
    return (
        float(summary.get('coverage_ratio', 0.0))
        + 0.12 * normalized_group_size
        - 0.15 * float(summary.get('singleton_ratio', 0.0))
        + 0.05 * float(summary.get('exact_group_proportion', 0.0))
    )


def ft_group_translate_train(model, model_name, translate_name,
                             weight_name, in_channel, out_channel, kernel_size, channel_number,
                             ft_layer_enabled, mask, group_information,
                             map_information, multiple_relationship_information,
                             reuse_ratio_information,
                             ft_mask_lambda, ft_proto_lambda,
                             ft_balance_lambda, ft_sep_lambda,
                             device, optimizer, scheduler,
                             train_loader, test_loader,
                             max_epoches, translate_epoch,
                             group_refresh_epoch=None,
                             min_group_size=2, target_group_size=4,
                             sim_threshold=0.85, exact_threshold=0.98,
                             scale_candidates=None, pattern_value_number=None,
                             pattern_shape_number=8, OU_size=8):
    """最小可运行FT训练器。保持旧训练器不动，新方法单独落盘。"""
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    if group_refresh_epoch is None:
        group_refresh_epoch = []
    if pattern_value_number is None:
        pattern_value_number = [OU_size] * len(weight_name)
    if isinstance(target_group_size, list):
        target_group_size_list = target_group_size
    else:
        target_group_size_list = [target_group_size] * len(weight_name)
    if isinstance(sim_threshold, list):
        sim_threshold_list = sim_threshold
    else:
        sim_threshold_list = [sim_threshold] * len(weight_name)

    result_all = pd.DataFrame()
    before_translate_accuracy = [0.0] * len(translate_epoch)
    before_translate_loss = [0.0] * len(translate_epoch)
    after_translate_accuracy = [0.0] * len(translate_epoch)
    after_translate_loss = [0.0] * len(translate_epoch)
    model_accuracy_difference = [0.0] * len(translate_epoch)

    result = pd.DataFrame()
    train_accuracy_record = [0.0] * max_epoches
    train_loss_record = [0.0] * max_epoches
    test_accuracy_record = [0.0] * max_epoches
    test_loss_record = [0.0] * max_epoches

    current_iteration = 0
    start_time = time.time()
    checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')
    refresh_records = []
    refresh_log_path = 'model_' + model_name + '_' + translate_name + '_refresh_log.csv'

    for epoch in range(checkpoint['epoch'], max_epoches):
        if epoch == checkpoint['epoch']:
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_schedule'])
        else:
            model.train()
            total = 0
            correct = 0
            train_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_ce = F.cross_entropy(outputs, targets)
                loss_mask, loss_proto, loss_balance, loss_sep = _compute_ft_regularization(
                    model=model,
                    weight_name=weight_name,
                    ft_layer_enabled=ft_layer_enabled,
                    mask=mask,
                    group_information=group_information,
                    device=device,
                )
                loss = (
                    loss_ce
                    + ft_mask_lambda * loss_mask
                    + ft_proto_lambda * loss_proto
                    + ft_balance_lambda * loss_balance
                    + ft_sep_lambda * loss_sep
                )
                loss.backward()
                optimizer.step()

                train_loss = train_loss + loss.item()
                _, predicted = outputs.max(1)
                total = total + targets.size(0)
                correct = correct + predicted.eq(targets).sum().item()

            scheduler.step()
            train_accuracy_record[epoch] = correct / total
            train_loss_record[epoch] = train_loss
            print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
            test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
            print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')

        if (epoch + 1) in group_refresh_epoch:
            with torch.no_grad():
                before_summary = summarize_group_information(group_information, weight_name, ft_layer_enabled)
                candidate_mask = copy.deepcopy(mask)
                candidate_group_information = copy.deepcopy(group_information)
                candidate_map_information = copy.deepcopy(map_information)
                candidate_multiple_relationship_information = copy.deepcopy(multiple_relationship_information)
                candidate_reuse_ratio_information = copy.deepcopy(reuse_ratio_information)
                print('[FT refresh] epoch {} start rebuild'.format(epoch + 1))

                for i in range(0, len(weight_name)):
                    if not ft_layer_enabled[i]:
                        continue
                    print('[FT refresh] epoch {} layer {} seed search'.format(epoch + 1, weight_name[i]))
                    candidate_mask[weight_name[i]], group_seed_info = ft_group_score_mask(
                        model=model,
                        weight_name=weight_name[i],
                        in_channel=in_channel[i],
                        out_channel=out_channel[i],
                        kernel_size=kernel_size[i],
                        channel_number=channel_number[i],
                        pattern_value_number=pattern_value_number[i],
                        pattern_shape_number=pattern_shape_number,
                        OU_size=OU_size,
                        target_group_size=target_group_size_list[i],
                        sim_threshold=sim_threshold_list[i],
                    )
                    print('[FT refresh] epoch {} layer {} regroup'.format(epoch + 1, weight_name[i]))
                    candidate_map_information[weight_name[i]], candidate_multiple_relationship_information[weight_name[i]], candidate_reuse_ratio_information[weight_name[i]], candidate_group_information[weight_name[i]] = ft_group_cluster_translate(
                        model=model,
                        in_channel=in_channel[i],
                        out_channel=out_channel[i],
                        weight_name=weight_name[i],
                        kernel_size=kernel_size[i],
                        channel_number=channel_number[i],
                        mask=candidate_mask[weight_name[i]],
                        min_group_size=min_group_size,
                        target_group_size=target_group_size_list[i],
                        sim_threshold=sim_threshold_list[i],
                        exact_threshold=exact_threshold,
                        scale_candidates=scale_candidates,
                    )
                    candidate_group_information[weight_name[i]]['seed_info'] = group_seed_info

                after_summary = summarize_group_information(candidate_group_information, weight_name, ft_layer_enabled)
                accepted = _refresh_acceptance_score(after_summary) + 1e-8 >= _refresh_acceptance_score(before_summary)
                if accepted:
                    mask = candidate_mask
                    group_information = candidate_group_information
                    map_information = candidate_map_information
                    multiple_relationship_information = candidate_multiple_relationship_information
                    reuse_ratio_information = candidate_reuse_ratio_information
                else:
                    print('[FT refresh warning] epoch {} refresh rejected: coverage {:.4f}->{:.4f}, singleton {:.4f}->{:.4f}'.format(
                        epoch + 1,
                        before_summary['coverage_ratio'],
                        after_summary['coverage_ratio'],
                        before_summary['singleton_ratio'],
                        after_summary['singleton_ratio'],
                    ))

                refresh_record = {
                    'epoch': epoch + 1,
                    'accepted': int(accepted),
                    'before_coverage': before_summary['coverage_ratio'],
                    'after_coverage': after_summary['coverage_ratio'],
                    'before_block_coverage': before_summary['block_coverage_ratio'],
                    'after_block_coverage': after_summary['block_coverage_ratio'],
                    'before_group_count': before_summary['group_count'],
                    'after_group_count': after_summary['group_count'],
                    'before_singleton_ratio': before_summary['singleton_ratio'],
                    'after_singleton_ratio': after_summary['singleton_ratio'],
                    'before_avg_group_size': before_summary['avg_group_size'],
                    'after_avg_group_size': after_summary['avg_group_size'],
                    'before_exact_group_proportion': before_summary['exact_group_proportion'],
                    'after_exact_group_proportion': after_summary['exact_group_proportion'],
                }
                refresh_records.append(refresh_record)
                pd.DataFrame(refresh_records).to_csv(refresh_log_path, index=False)
                print('[FT refresh] epoch {} accepted={} coverage {:.4f}->{:.4f} group_count {}->{} singleton {:.4f}->{:.4f}'.format(
                    epoch + 1,
                    accepted,
                    before_summary['coverage_ratio'],
                    after_summary['coverage_ratio'],
                    before_summary['group_count'],
                    after_summary['group_count'],
                    before_summary['singleton_ratio'],
                    after_summary['singleton_ratio'],
                ))

        if epoch + 1 in translate_epoch:
            before_translate_accuracy[current_iteration], before_translate_loss[current_iteration] = test(model, device, test_loader)
            print('Before_translate_accuracy: ' + str(before_translate_accuracy[current_iteration]) + ' Before_translate_loss: ' + str(before_translate_loss[current_iteration]))
            _apply_ft_group_projection(model, weight_name, mask, group_information)
            after_translate_accuracy[current_iteration], after_translate_loss[current_iteration] = test(model, device, test_loader)
            print('After_translate_accuracy: ' + str(after_translate_accuracy[current_iteration]) + ' After_translate_loss: ' + str(after_translate_loss[current_iteration]))
            model_accuracy_difference[current_iteration] = before_translate_accuracy[current_iteration] - after_translate_accuracy[current_iteration]
            print('Model_accuracy_difference: ' + str(model_accuracy_difference[current_iteration]))
            current_iteration = current_iteration + 1

    _apply_ft_group_projection(model, weight_name, mask, group_information)

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))
    final_summary = summarize_group_information(group_information, weight_name, ft_layer_enabled)
    print('Final coverage: {:.4f}'.format(final_summary['coverage_ratio']))
    print('Final group_count: {}'.format(final_summary['group_count']))
    print('Final singleton_ratio: {:.4f}'.format(final_summary['singleton_ratio']))
    print('Final exact_group_proportion: {:.4f}'.format(final_summary['exact_group_proportion']))
    print('Final scaled_group_proportion: {:.4f}'.format(final_summary['scaled_group_proportion']))

    torch.save(model.state_dict(), 'model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth')

    with open('model_' + model_name + '_' + translate_name + '_mask.pkl', 'wb') as f_mask:
        pkl.dump(mask, f_mask, pkl.HIGHEST_PROTOCOL)
    with open('model_' + model_name + '_' + translate_name + '_map_information.pkl', 'wb') as f_map:
        pkl.dump(map_information, f_map, pkl.HIGHEST_PROTOCOL)
    with open('model_' + model_name + '_' + translate_name + '_multiple_relationship_information.pkl', 'wb') as f_mult:
        pkl.dump(multiple_relationship_information, f_mult, pkl.HIGHEST_PROTOCOL)
    with open('model_' + model_name + '_' + translate_name + '_coverage_ratio_information.pkl', 'wb') as f_coverage:
        pkl.dump(reuse_ratio_information, f_coverage, pkl.HIGHEST_PROTOCOL)
    with open('model_' + model_name + '_' + translate_name + '_reuse_ratio_information.pkl', 'wb') as f_reuse:
        pkl.dump(reuse_ratio_information, f_reuse, pkl.HIGHEST_PROTOCOL)
    with open('model_' + model_name + '_' + translate_name + '_group_information.pkl', 'wb') as f_group:
        pkl.dump(group_information, f_group, pkl.HIGHEST_PROTOCOL)

    result_all['Before_Translate_Accuracy'] = before_translate_accuracy
    result_all['Before_Translate_Loss'] = before_translate_loss
    result_all['After_Translate_Accuracy'] = after_translate_accuracy
    result_all['After_Translate_Loss'] = after_translate_loss
    result_all['Model_Difference'] = model_accuracy_difference
    result_all.to_csv('model_' + model_name + '_' + translate_name + '.csv')

    result['Train_Accuracy'] = train_accuracy_record
    result['Train_Loss'] = train_loss_record
    result['Test_Accuracy'] = test_accuracy_record
    result['Test_Loss'] = test_loss_record
    result.to_csv('model_' + model_name + '_' + translate_name + '_train_info.csv')

    parameters_to_txt(model, model_name, translate_name)


# 多种模式转化迭代训练
def pattern_translate(model, model_name, translate_name, weight_name, in_channel, out_channel, kernel_size, threshold, mask, map_information, multiple_relationship_information, weight_decay_1, weight_decay_2, device, optimizer, scheduler, train_loader, test_loader, max_epoches, translate_epoch):
    result_all = pd.DataFrame()  # 记录全部结果存储到csv文件
    before_translate_accuracy = [0.0] * len(translate_epoch)  # 记录转换前模型准确率
    before_translate_loss = [0.0] * len(translate_epoch)  # 记录转换前模型损失值
    after_translate_accuracy = [0.0] * len(translate_epoch)  # 记录转换后模型准确率
    after_translate_loss = [0.0] * len(translate_epoch)  # 记录转换后模型损失值
    model_accuracy_difference = [0.0] * len(translate_epoch)  # 记录模型转换前后的误差

    current_iteration = 0  # 记录当前训练epoch数
    start_time = time.time()  # 统计训练时间
    checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(translate_epoch[0]) + '_ckpt.pth')  # 加载断点

    result = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    train_accuracy_record = [0.0] * max_epoches  # 记录每个epoch训练集的准确率
    train_loss_record = [0.0] * max_epoches  # 记录每个epoch训练集的损失值
    test_accuracy_record = [0.0] * max_epoches  # 记录每个epoch测试集的准确率
    test_loss_record = [0.0] * max_epoches  # 记录每个epoch测试集的损失值

    reusing_layer_name = []  # 构造重用weight-pattern层列表
    value_list = [torch.ones((out_channel[i], in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
    similar_weight_pattern = dict(zip(weight_name, value_list))  # 构造相似weight-pattern矩阵
    multiple_list = [torch.ones((out_channel[i], in_channel[i], kernel_size[i], kernel_size[i])) for i in range(0, len(weight_name))]
    multiple_matrix = dict(zip(weight_name, multiple_list))  # 构造倍数关系倒数矩阵

    if 'value' in translate_name:
        # 构造关系倍数矩阵
        if 'similar' in translate_name:
            print('get multiple relationship matrix')
            for i in range(0, len(weight_name)):
                print(weight_name[i])
                if threshold[i] != 1.0:
                    reusing_layer_name.append(weight_name[i])
                if 'fc' in weight_name[i]:
                    multiple_matrix[weight_name[i]] = torch.ones(out_channel[i], in_channel[i])
                for c_out in range(0, out_channel[i]):
                    for c_in in range(0, in_channel[i]):
                        if multiple_relationship_information[weight_name[i]][c_out][c_in].abs().sum() != 0:
                            multiple_matrix[weight_name[i]][c_out][c_in] = 1.0 / multiple_relationship_information[weight_name[i]][c_out][c_in]
                        else:
                            multiple_matrix[weight_name[i]][c_out][c_in] = 1.0 / (1 / math.pow(2, -8))
        # 修改mask
        if 'shape' in translate_name:
            print('modify weight pattern mask')
            for i in range(0, len(weight_name)):
                print(weight_name[i])
                if threshold[i] != 1.0:
                    for c_in in range(0, in_channel[i]):
                        for j in range(0, map_information[weight_name[i]].shape[1]):
                            if map_information[weight_name[i]][c_in][j][0] == -1:
                                break
                            if map_information[weight_name[i]][c_in][j][1] != -1:
                                mask[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = mask[weight_name[i]][map_information[weight_name[i]][c_in][j][1]][c_in]
                            else:
                                if 'fc' in weight_name[i]:
                                    mask[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = 0.0
                                else:
                                    mask[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = torch.zeros(kernel_size[i], kernel_size[i])

    for epoch in range(checkpoint['epoch'], max_epoches):
        if epoch == checkpoint['epoch']:
            model.load_state_dict(checkpoint['model'])  # 加载模型参数
            optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
            scheduler.load_state_dict(checkpoint['lr_schedule'])  # 加载学习率优化器

        else:
            model.train()  # 启用batch normalization和drop out
            total = 0  # 记录样本数
            correct = 0  # 记录总正确数
            train_loss = 0.0  # 记录总损失值

            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(device), targets.to(device)  # 将数据加载到gpu
                optimizer.zero_grad()  # 计算梯度
                outputs = model(inputs)  # 前向传播
                # 计算损失
                loss_ce = F.cross_entropy(outputs, targets)  # 交叉熵损失
                loss_re_1 = loss_re_2 = 0  # 正则化损失，loss_re_1为剪枝正则化损失，loss_re_2为重用正则化损失
                for name, par in model.named_parameters():
                    if ('shape' in translate_name or 'pruning' in translate_name) and name in weight_name:  # 对模型进行模式形状转换
                        mask[name] = mask[name].to(device)
                        Z = mask[name] * par
                        loss_re_1 = loss_re_1 + weight_decay_1 * 0.5 * torch.sum(torch.pow(par - Z, 2))
                    if 'value_similar' in translate_name and name in reusing_layer_name:
                        multiple_matrix[name] = multiple_matrix[name].to(device)
                        similar_weight_pattern[name] = similar_weight_pattern[name].to(device)
                        loss_re_2 = loss_re_2 + weight_decay_2 * torch.sum(torch.pow(multiple_matrix[name] * (par - similar_weight_pattern[name]), 2))
                loss = loss_ce + loss_re_1 + loss_re_2
                loss.backward()  # 后向传播
                optimizer.step()  # 更新优化器

                # 可视化训练过程
                train_loss = train_loss + loss.item()  # 计算当前损失值
                _, predicted = outputs.max(1)
                total = total + targets.size(0)
                correct = correct + predicted.eq(targets).sum().item()  # 计算当前准确率

            scheduler.step()  # 余弦退火调整学习率

            # 记录最优模型
            train_accuracy_record[epoch] = correct / total
            train_loss_record[epoch] = train_loss
            print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
            test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
            print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')

        # 构造相似weight-pattern矩阵
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
        if 'value' in translate_name:
            if 'similar' in translate_name:
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if 'fc' in weight_name[i]:
                        similar_weight_pattern[weight_name[i]] = torch.ones(out_channel[i], in_channel[i])
                    # 先对权重矩阵和相似矩阵进行量化
                    max_value = torch.max(model.state_dict()[weight_name[i]]).item()
                    min_value = torch.min(model.state_dict()[weight_name[i]]).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / (127 * 256)
                    for c_out in range(0, out_channel[i]):
                        similar_weight_pattern[weight_name[i]][c_out] = torch.round(model.state_dict()[weight_name[i]][c_out] / scale)
                        similar_weight_pattern[weight_name[i]][c_out] = torch.round(similar_weight_pattern[weight_name[i]][c_out] / math.pow(2, 8)) * math.pow(2, 8)
                    # 获得相似权重矩阵
                    if threshold[i] != 1.0:
                        similar_weight_pattern[weight_name[i]] = similar_weight_pattern[weight_name[i]].to(device='cpu')
                        for c_in in range(0, in_channel[i]):
                            for j in range(0, map_information[weight_name[i]].shape[1]):
                                if map_information[weight_name[i]][c_in][j][0] == -1:
                                    break
                                if map_information[weight_name[i]][c_in][j][1] != -1:
                                    similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][1]][c_in]
                                else:
                                    if 'fc' in weight_name[i]:
                                        similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = 0.0
                                    else:
                                        similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = torch.zeros(kernel_size[i], kernel_size[i])
                        for c_out in range(0, out_channel[i]):
                            similar_weight_pattern[weight_name[i]][c_out] = similar_weight_pattern[weight_name[i]][c_out] * multiple_relationship_information[weight_name[i]][c_out] * scale

            if 'identical' in translate_name and (epoch + 1) in translate_epoch:
                for i in range(0, len(weight_name)):
                    print(weight_name[i])
                    if 'fc' in weight_name[i]:
                        similar_weight_pattern[weight_name[i]] = torch.ones(out_channel[i], in_channel[i])
                    # 先对权重矩阵和相似矩阵进行量化
                    max_value = torch.max(model.state_dict()[weight_name[i]]).item()
                    min_value = torch.min(model.state_dict()[weight_name[i]]).item()
                    if max_value < -min_value:
                        max_value = -min_value
                    scale = (max_value - 0) / 127
                    for c_out in range(0, out_channel[i]):
                        similar_weight_pattern[weight_name[i]][c_out] = torch.round(model.state_dict()[weight_name[i]][c_out] / scale)
                    if threshold[i] != 1.0:
                        similar_weight_pattern[weight_name[i]] = similar_weight_pattern[weight_name[i]].to(device='cpu')
                        for c_in in range(0, in_channel[i]):
                            for j in range(0, map_information[weight_name[i]].shape[1]):
                                if map_information[weight_name[i]][c_in][j][0] == -1:
                                    break
                                if map_information[weight_name[i]][c_in][j][1] != -1:
                                    similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][1]][c_in]
                                else:
                                    if 'fc' in weight_name[i]:
                                        similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = 0.0
                                    else:
                                        similar_weight_pattern[weight_name[i]][map_information[weight_name[i]][c_in][j][0]][c_in] = torch.zeros(kernel_size[i], kernel_size[i])
                        for c_out in range(0, out_channel[i]):
                            similar_weight_pattern[weight_name[i]][c_out] = similar_weight_pattern[weight_name[i]][c_out] * scale
        print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

        if epoch + 1 in translate_epoch:
            before_translate_accuracy[current_iteration], before_translate_loss[current_iteration] = test(model, device, test_loader)  # 测试转换前模型准确率
            print('Before_translate_accuracy: ' + str(before_translate_accuracy[current_iteration]) + ' Before_translate_loss: ' + str(before_translate_loss[current_iteration]))
            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))

            if 'value' in translate_name:  # 对模型进行模式数值转换
                for i in range(0, len(weight_name)):
                    if threshold[i] != 1.0:
                        for c_out in range(0, out_channel[i]):
                            model.state_dict()[weight_name[i]][c_out] = similar_weight_pattern[weight_name[i]][c_out]

            if 'shape' in translate_name or 'pruning' in translate_name:  # 对模型进行模式形状转换
                for i in range(0, len(weight_name)):
                    mask[weight_name[i]] = mask[weight_name[i]].to(device)
                    weight_pattern = model.state_dict()[weight_name[i]] * mask[weight_name[i]]
                    for c_out in range(0, out_channel[i]):
                        model.state_dict()[weight_name[i]][c_out] = weight_pattern[c_out]

            print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
            after_translate_accuracy[current_iteration], after_translate_loss[current_iteration] = test(model, device, test_loader)  # 测试转换后模型准确率
            print('After_translate_accuracy: ' + str(after_translate_accuracy[current_iteration]) + ' After_translate_loss: ' + str(after_translate_loss[current_iteration]))
            model_accuracy_difference[current_iteration] = before_translate_accuracy[current_iteration] - after_translate_accuracy[current_iteration]  # 计算模型转换损失
            print('Model_accuracy_difference: ' + str(model_accuracy_difference[current_iteration]))
            current_iteration = current_iteration + 1

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))
    torch.save(model.state_dict(), 'model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth')  # 保存转换后最优模型参数

    # 将剪枝过程数据保存到csv文件
    result_all['Before_Translate_Accuracy'] = before_translate_accuracy
    result_all['Before_Translate_Loss'] = before_translate_loss
    result_all['After_Translate_Accuracy'] = after_translate_accuracy
    result_all['After_Translate_Loss'] = after_translate_loss
    result_all['Model_Difference'] = model_accuracy_difference
    result_all.to_csv('model_' + model_name + '_' + translate_name + '.csv')

    # 将训练过程数据保存到csv文件
    result['Train_Accuracy'] = train_accuracy_record
    result['Train_Loss'] = train_loss_record
    result['Test_Accuracy'] = test_accuracy_record
    result['Test_Loss'] = test_loss_record
    result.to_csv('model_' + model_name + '_' + translate_name + '_train_info' + '.csv')

    # 检查训练模型参数是否正确
    parameters_to_txt(model, model_name, translate_name)  # 以文本形式保存模型参数
