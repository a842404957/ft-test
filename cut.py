import copy
import json
import math
import time
import pickle as pkl
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F


# 将模型参数以txt形式存储
def _artifact_output_path(output_dir, filename):
    output_root = Path(output_dir or '.')
    output_root.mkdir(parents=True, exist_ok=True)
    return output_root / filename


def parameters_to_txt(model, model_name, translate_name, output_dir='.'):
    str_parameters = ''
    for parameters in model.parameters():
        str_parameters = str_parameters + str(parameters) + str('\n')
    f_parameters = open(_artifact_output_path(output_dir, 'parameters_' + model_name + '_' + translate_name + '.txt'), 'w', encoding='utf-8')
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


def _tensor_mask_channel_counts(mask_tensor: torch.Tensor) -> Tuple[int, Tuple[int, ...]]:
    if mask_tensor.dim() == 0:
        reshaped = mask_tensor.reshape(1, 1).float()
    elif mask_tensor.dim() == 1:
        reshaped = mask_tensor.reshape(mask_tensor.shape[0], 1).float()
    else:
        reshaped = mask_tensor.reshape(mask_tensor.shape[0], -1).float()
    channel_counts = tuple(int(value) for value in reshaped.sum(dim=1).tolist())
    return int(sum(channel_counts)), channel_counts


def _normalize_budget_bucket_key(bucket_key: Any):
    if isinstance(bucket_key, tuple):
        return tuple(_normalize_budget_bucket_key(item) for item in bucket_key)
    if isinstance(bucket_key, bytes):
        return bucket_key.hex()
    return bucket_key


def _budget_bucket_key_from_mask(mask_tensor: torch.Tensor, bucket_mode: str):
    if bucket_mode == 'exact_mask':
        return ('exact_mask', _tensor_mask_signature(mask_tensor))
    nonzero_count, channel_counts = _tensor_mask_channel_counts(mask_tensor)
    if bucket_mode == 'shape_family':
        return ('shape_family', nonzero_count, tuple(sorted(channel_counts, reverse=True)))
    return ('nonzero_count', nonzero_count)


def _budget_bucket_key_from_pattern(pattern: Dict[str, Any], bucket_mode: str):
    if bucket_mode == 'exact_mask':
        return ('exact_mask', pattern['mask_signature'])
    nonzero_count = int(pattern.get('mask_nonzero_count', 0))
    channel_counts = tuple(int(value) for value in pattern.get('mask_channel_counts', (nonzero_count,)))
    if bucket_mode == 'shape_family':
        return ('shape_family', nonzero_count, tuple(sorted(channel_counts, reverse=True)))
    return ('nonzero_count', nonzero_count)


def _mean_or_zero(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _format_multiplier_key(multiplier: float) -> str:
    if not np.isfinite(multiplier):
        return 'nan_or_inf'
    rounded = round(float(multiplier), 6)
    if abs(rounded) <= 1e-8:
        rounded = 0.0
    return f'{rounded:.6f}'


def _compute_group_multiplier_stats(layer_groups: List[Dict[str, Any]]) -> Dict[str, Any]:
    scale_distribution: Dict[str, int] = {}
    total_members = 0
    zero_multiplier_members = 0
    nonzero_multiplier_members = 0

    for group in layer_groups:
        for member in group.get('members', []):
            multiplier = float(member.get('multiplier', 1.0))
            scale_key = _format_multiplier_key(multiplier)
            scale_distribution[scale_key] = scale_distribution.get(scale_key, 0) + 1
            total_members += 1
            if abs(multiplier) <= 1e-8 or not np.isfinite(multiplier):
                zero_multiplier_members += 1
            else:
                nonzero_multiplier_members += 1

    return {
        'zero_multiplier_count': int(zero_multiplier_members),
        'nonzero_multiplier_count': int(nonzero_multiplier_members),
        'zero_multiplier_ratio': zero_multiplier_members / max(total_members, 1),
        'nonzero_multiplier_ratio': nonzero_multiplier_members / max(total_members, 1),
        'scale_distribution': scale_distribution,
    }


def _build_layer_group_summary(layer_groups: List[Dict[str, Any]],
                               total_block_count: int,
                               total_ou_count: int,
                               repairable_block_count: int,
                               repairable_ou_count: int,
                               singleton_group_count: int,
                               exact_group_count: int,
                               scaled_group_count: int) -> Dict[str, Any]:
    multiplier_stats = _compute_group_multiplier_stats(layer_groups)
    group_sizes = [group['group_size'] for group in layer_groups]
    return {
        'group_count': len(layer_groups),
        'block_count': total_block_count,
        'ou_count': total_ou_count,
        'repairable_block_count': repairable_block_count,
        'repairable_ou_count': repairable_ou_count,
        'coverage_ratio': repairable_ou_count / max(total_ou_count, 1),
        'block_coverage_ratio': repairable_block_count / max(total_block_count, 1),
        'singleton_group_count': singleton_group_count,
        'singleton_ratio': singleton_group_count / max(len(layer_groups), 1),
        'avg_group_size': _mean_or_zero(group_sizes),
        'max_group_size': max(group_sizes) if group_sizes else 0,
        'exact_group_count': exact_group_count,
        'scaled_group_count': scaled_group_count,
        'exact_group_ratio': exact_group_count / max(len(layer_groups), 1),
        'scaled_group_ratio': scaled_group_count / max(len(layer_groups), 1),
        'zero_multiplier_count': multiplier_stats['zero_multiplier_count'],
        'nonzero_multiplier_count': multiplier_stats['nonzero_multiplier_count'],
        'zero_multiplier_ratio': multiplier_stats['zero_multiplier_ratio'],
        'nonzero_multiplier_ratio': multiplier_stats['nonzero_multiplier_ratio'],
        'scale_distribution': multiplier_stats['scale_distribution'],
    }


FT_GROUP_COMPARE_WINDOW = 512
FT_GROUP_COMPARE_WINDOW_FC = 256
FT_GROUP_SCALE_CHUNK = 1024


def _get_block_kind(weight_name: str, kernel_size: int) -> str:
    if 'fc' in weight_name:
        return 'fc'
    return 'conv1x1' if kernel_size == 1 else 'conv3x3'


def _pack_even_layer_blocks(weight_tensor: torch.Tensor, mask_tensor: torch.Tensor, weight_name: str,
                            in_channel: int, out_channel: int, kernel_size: int, channel_number: int):
    block_step = _get_block_step(channel_number)
    if in_channel <= 0 or out_channel <= 0 or in_channel % block_step != 0:
        return None

    num_blocks = in_channel // block_step
    weight_tensor = weight_tensor.detach().cpu().contiguous()
    mask_tensor = mask_tensor.detach().cpu().contiguous()

    if weight_tensor.dim() == 4:
        raw_blocks = weight_tensor.reshape(out_channel, num_blocks, block_step, kernel_size, kernel_size)
        mask_blocks = mask_tensor.reshape(out_channel, num_blocks, block_step, kernel_size, kernel_size)
    else:
        raw_blocks = weight_tensor.reshape(out_channel, num_blocks, block_step)
        mask_blocks = mask_tensor.reshape(out_channel, num_blocks, block_step)

    raw_flat = raw_blocks.reshape(out_channel, num_blocks, -1)
    mask_flat = mask_blocks.reshape(out_channel, num_blocks, -1)
    masked_flat = raw_flat * mask_flat

    return {
        'raw_blocks': raw_blocks,
        'mask_blocks': mask_blocks,
        'raw_flat': raw_flat,
        'mask_flat': mask_flat,
        'masked_flat': masked_flat,
        'norm1': masked_flat.abs().sum(dim=2),
        'norm2': torch.norm(masked_flat.float(), p=2, dim=2),
        'out_channel': out_channel,
        'num_blocks': num_blocks,
        'channel_span': block_step,
        'in_starts': torch.arange(0, in_channel, block_step, dtype=torch.long),
        'block_kind': _get_block_kind(weight_name, kernel_size),
        'block_shape': tuple(raw_blocks.shape[2:]),
        'kernel_size': kernel_size,
    }


def _pattern_from_packed_block(packed: Dict[str, Any], out_index: int, block_index: int) -> Dict[str, Any]:
    current_mask = packed['mask_blocks'][out_index, block_index].clone()
    raw = packed['raw_blocks'][out_index, block_index].clone()
    masked = raw * current_mask
    mask_nonzero_count, mask_channel_counts = _tensor_mask_channel_counts(current_mask)
    return {
        'out_ch': int(out_index),
        'in_ch_start': int(packed['in_starts'][block_index].item()),
        'channel_span': int(packed['channel_span']),
        'raw': raw,
        'masked': masked,
        'norm1': float(packed['norm1'][out_index, block_index].item()),
        'norm2': float(packed['norm2'][out_index, block_index].item()),
        'mask_signature': _tensor_mask_signature(current_mask),
        'mask_nonzero_count': int(mask_nonzero_count),
        'mask_channel_counts': mask_channel_counts,
        'multiplier': 1.0,
        'block_kind': packed['block_kind'],
        'block_index': int(block_index),
    }


def _batched_best_power_of_two_scale(source_matrix: torch.Tensor, target_vector: torch.Tensor, scale_candidates=None,
                                     chunk_size: int = FT_GROUP_SCALE_CHUNK):
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    if source_matrix.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty

    source_matrix = source_matrix.float()
    target_vector = target_vector.reshape(1, -1).float()
    scale_tensor = torch.tensor(scale_candidates, dtype=torch.float32).reshape(1, -1, 1)
    source_norm = source_matrix.abs().sum(dim=1)
    target_norm = target_vector.abs().sum().item()

    if target_norm <= 1e-8:
        best_error = source_norm.clone()
        best_similarity = torch.where(
            source_norm <= 1e-8,
            torch.ones_like(best_error),
            torch.clamp(1.0 - best_error / (source_norm + 1e-8), min=0.0),
        )
        best_scale = torch.zeros_like(best_error)
        return best_scale, best_error, best_similarity

    best_scale_chunks = []
    best_error_chunks = []
    best_similarity_chunks = []

    for start in range(0, source_matrix.shape[0], chunk_size):
        end = min(source_matrix.shape[0], start + chunk_size)
        source_chunk = source_matrix[start:end]
        error_matrix = (source_chunk.unsqueeze(1) - scale_tensor * target_vector.unsqueeze(0)).abs().sum(dim=2)
        similarity_matrix = torch.clamp(
            1.0 - error_matrix / (source_norm[start:end].unsqueeze(1) + 1e-8),
            min=0.0,
        )

        best_indices = torch.zeros(source_chunk.shape[0], dtype=torch.long)
        best_errors = error_matrix[:, 0].clone()
        best_similarities = similarity_matrix[:, 0].clone()

        for candidate_idx in range(1, error_matrix.shape[1]):
            candidate_errors = error_matrix[:, candidate_idx]
            candidate_similarities = similarity_matrix[:, candidate_idx]
            better = candidate_errors < (best_errors - 1e-8)
            tie = (candidate_errors - best_errors).abs() <= 1e-8
            tie_better = tie & (candidate_similarities > best_similarities)
            update_mask = better | tie_better
            best_indices[update_mask] = candidate_idx
            best_errors = torch.where(update_mask, candidate_errors, best_errors)
            best_similarities = torch.where(update_mask, candidate_similarities, best_similarities)

        best_scale_chunks.append(scale_tensor.reshape(-1)[best_indices])
        best_error_chunks.append(best_errors)
        best_similarity_chunks.append(best_similarities)

    return (
        torch.cat(best_scale_chunks, dim=0),
        torch.cat(best_error_chunks, dim=0),
        torch.cat(best_similarity_chunks, dim=0),
    )


def _get_bucket_compare_window(bucket_size: int, block_kind: str) -> int:
    if bucket_size <= 0:
        return 0
    if block_kind == 'fc' and bucket_size > FT_GROUP_COMPARE_WINDOW_FC:
        return FT_GROUP_COMPARE_WINDOW_FC
    if bucket_size > FT_GROUP_COMPARE_WINDOW:
        return FT_GROUP_COMPARE_WINDOW
    return bucket_size


def _build_shared_topk_mask(model, weight_name, in_channel, out_channel, channel_number, keep_ratio):
    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    packed = _pack_even_layer_blocks(
        weight_tensor=weight_tensor,
        mask_tensor=torch.ones_like(weight_tensor),
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=1 if weight_tensor.dim() == 2 else weight_tensor.shape[-1],
        channel_number=channel_number,
    )
    if packed is not None:
        flat_block = packed['raw_flat'].abs().sum(dim=0)
        keep_count = max(1, min(flat_block.shape[1], int(round(flat_block.shape[1] * keep_ratio))))
        keep_indices = torch.topk(flat_block, keep_count, dim=1).indices
        block_mask = torch.zeros_like(flat_block)
        block_mask.scatter_(1, keep_indices, 1.0)
        expanded_mask = block_mask.unsqueeze(0).expand(out_channel, -1, -1)
        return expanded_mask.reshape_as(weight_tensor)

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
    packed = _pack_even_layer_blocks(
        weight_tensor=weight_tensor,
        mask_tensor=torch.ones_like(weight_tensor),
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=1 if weight_tensor.dim() == 2 else weight_tensor.shape[-1],
        channel_number=channel_number,
    )
    if packed is not None:
        flat_block = packed['raw_flat'].abs()
        keep_count = max(1, min(flat_block.shape[2], int(round(flat_block.shape[2] * keep_ratio))))
        keep_indices = torch.topk(flat_block, keep_count, dim=2).indices
        block_mask = torch.zeros_like(flat_block)
        block_mask.scatter_(2, keep_indices, 1.0)
        return block_mask.reshape_as(weight_tensor)

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

    packed = _pack_even_layer_blocks(
        weight_tensor=weight_tensor,
        mask_tensor=mask_tensor,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        channel_number=channel_number,
    )
    if packed is not None:
        pattern_list = []
        for out_ch in range(0, out_channel):
            for block_index in range(0, packed['num_blocks']):
                pattern_list.append(_pattern_from_packed_block(packed, out_ch, block_index))
        return pattern_list

    block_step = _get_block_step(channel_number)
    pattern_list = []
    for out_ch in range(0, out_channel):
        for in_ch_start in range(0, in_channel, block_step):
            channel_span = _get_channel_span(in_channel, in_ch_start, channel_number)
            raw = _extract_block_tensor(weight_tensor, out_ch, in_ch_start, channel_span)
            current_mask = _extract_block_tensor(mask_tensor, out_ch, in_ch_start, channel_span)
            masked = raw * current_mask
            mask_nonzero_count, mask_channel_counts = _tensor_mask_channel_counts(current_mask)
            pattern_list.append({
                'out_ch': out_ch,
                'in_ch_start': in_ch_start,
                'channel_span': channel_span,
                'raw': raw,
                'masked': masked,
                'norm1': masked.abs().sum().item(),
                'norm2': torch.norm(masked.reshape(-1).float(), p=2).item(),
                'mask_signature': _tensor_mask_signature(current_mask),
                'mask_nonzero_count': int(mask_nonzero_count),
                'mask_channel_counts': mask_channel_counts,
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


def _resolve_budgeted_layer_config(budget_config: Optional[Dict[str, Any]], layer_name: str) -> Dict[str, Any]:
    budget_config = budget_config or {}
    resolved = {
        'target_coverage': float(budget_config.get('target_coverage', 0.6)),
        'max_singleton': float(budget_config.get('max_singleton', 0.5)),
        'min_avg_group_size': float(budget_config.get('min_avg_group_size', 2.0)),
        'prototype_budget_ratio': float(budget_config.get('prototype_budget_ratio', 0.25)),
        'prototype_budget_min': int(budget_config.get('prototype_budget_min', 4)),
        'prototype_budget_max': int(budget_config.get('prototype_budget_max', 256)),
        'relax_threshold': float(budget_config.get('relax_threshold', 0.85)),
        'max_scale_error': float(budget_config.get('max_scale_error', 0.25)),
        'bucket_mode': str(budget_config.get('bucket_mode', 'nonzero_count')),
        'mask_family': list(budget_config.get('mask_family', ['shape_seed', 'shared_topk', 'per_out_topk'])),
        'mask_keep_ratios': [float(value) for value in budget_config.get('mask_keep_ratios', [0.6667, 0.4444])],
    }
    layer_overrides = budget_config.get('layer_overrides', {}) or {}
    layer_override = layer_overrides.get(layer_name, {})
    if isinstance(layer_override, dict):
        for key in resolved:
            if key in layer_override:
                if key.endswith('_min') or key.endswith('_max'):
                    resolved[key] = int(layer_override[key])
                elif key == 'bucket_mode':
                    resolved[key] = str(layer_override[key])
                elif key in {'mask_family'}:
                    resolved[key] = [str(value) for value in layer_override[key]]
                elif key in {'mask_keep_ratios'}:
                    resolved[key] = [float(value) for value in layer_override[key]]
                else:
                    resolved[key] = float(layer_override[key])
    return resolved


def _resolve_prototype_budget(num_patterns: int, config: Dict[str, Any]) -> int:
    if num_patterns <= 1:
        return num_patterns
    desired = int(round(num_patterns * float(config.get('prototype_budget_ratio', 0.25))))
    desired = max(int(config.get('prototype_budget_min', 4)), desired)
    desired = min(int(config.get('prototype_budget_max', 256)), desired)
    desired = min(desired, max(1, num_patterns // 2))
    return max(1, min(num_patterns, desired))


def budgeted_select_prototypes(pattern_list: List[Dict[str, Any]], prototype_budget: int) -> List[int]:
    """Deterministic farthest-first prototype selection on normalized masked vectors."""
    if not pattern_list:
        return []
    if len(pattern_list) == 1:
        return [0]

    feature_matrix = torch.stack([pattern['masked'].reshape(-1).float() for pattern in pattern_list], dim=0)
    norms = torch.norm(feature_matrix, p=2, dim=1)
    normalized = feature_matrix / torch.clamp(norms.unsqueeze(1), min=1e-8)
    importance = torch.tensor([float(pattern.get('norm1', feature_matrix[i].abs().sum().item())) for i, pattern in enumerate(pattern_list)])

    target_budget = max(1, min(len(pattern_list), int(prototype_budget)))
    selected = [max(range(len(pattern_list)), key=lambda index: (float(importance[index]), -index))]
    min_distance = torch.full((len(pattern_list),), float('inf'))
    min_distance[selected[0]] = -1.0

    while len(selected) < target_budget:
        newest = selected[-1]
        distances = 1.0 - torch.mv(normalized, normalized[newest])
        min_distance = torch.minimum(min_distance, distances)
        for existing_index in selected:
            min_distance[existing_index] = -1.0

        candidate_index = max(
            (index for index in range(len(pattern_list)) if index not in selected),
            key=lambda index: (float(min_distance[index]), float(importance[index]), -index),
        )
        selected.append(candidate_index)

    return selected


def budgeted_assign_members(pattern_list: List[Dict[str, Any]],
                            prototype_indices: List[int],
                            scale_candidates,
                            max_scale_error: float,
                            exact_threshold: float,
                            bucket_key: Any = None):
    groups = []
    if not pattern_list:
        return groups

    prototype_set = set(prototype_indices)
    groups_by_index = {}
    for prototype_index in prototype_indices:
        prototype_pattern = pattern_list[prototype_index]
        groups_by_index[prototype_index] = {
            'prototype': prototype_pattern,
            'members': [{
                'pattern': prototype_pattern,
                'multiplier': 1.0,
                'similarity': 1.0,
                'normalized_error': 0.0,
                'role': 'prototype',
            }],
            'repair_mode': 'exact',
            'bucket_key': bucket_key,
        }

    for pattern_index, pattern in enumerate(pattern_list):
        if pattern_index in prototype_set:
            continue

        best_assignment = None
        member_norm = float(pattern.get('norm1', pattern['masked'].abs().sum().item()))
        for prototype_index in prototype_indices:
            prototype_pattern = pattern_list[prototype_index]
            scale, error, similarity = search_best_power_of_two_scale(
                pattern['masked'],
                prototype_pattern['masked'],
                scale_candidates=scale_candidates,
            )
            normalized_error = float(error / (member_norm + 1e-8))
            if (
                not np.isfinite(scale)
                or abs(scale) <= 1e-8
                or not np.isfinite(normalized_error)
            ):
                continue

            candidate_assignment = {
                'prototype_index': prototype_index,
                'multiplier': float(scale),
                'similarity': float(similarity),
                'normalized_error': normalized_error,
            }
            if best_assignment is None or (
                candidate_assignment['normalized_error'],
                -candidate_assignment['similarity'],
                candidate_assignment['prototype_index'],
            ) < (
                best_assignment['normalized_error'],
                -best_assignment['similarity'],
                best_assignment['prototype_index'],
            ):
                best_assignment = candidate_assignment

        if best_assignment is not None and best_assignment['normalized_error'] <= float(max_scale_error):
            target_group = groups_by_index[best_assignment['prototype_index']]
            target_group['members'].append({
                'pattern': pattern,
                'multiplier': best_assignment['multiplier'],
                'similarity': best_assignment['similarity'],
                'normalized_error': best_assignment['normalized_error'],
                'role': 'member',
            })
            if abs(best_assignment['multiplier'] - 1.0) > 1e-6 or best_assignment['similarity'] < exact_threshold:
                target_group['repair_mode'] = 'scaled'
        else:
            groups_by_index[pattern_index] = {
                'prototype': pattern,
                'members': [{
                    'pattern': pattern,
                    'multiplier': 1.0,
                    'similarity': 1.0,
                    'normalized_error': 0.0,
                    'role': 'prototype',
                }],
                'repair_mode': 'exact',
                'bucket_key': bucket_key,
            }

    groups.extend(groups_by_index.values())
    groups.sort(key=lambda item: (item['prototype']['in_ch_start'], item['prototype']['out_ch']))
    return groups


def budgeted_merge_singletons(groups: List[Dict[str, Any]],
                              scale_candidates,
                              max_scale_error: float,
                              exact_threshold: float,
                              min_group_size: int = 2):
    if not groups:
        return groups

    final_groups = list(groups)
    singleton_groups = [group for group in final_groups if len(group['members']) < min_group_size]
    non_singleton_groups = [group for group in final_groups if len(group['members']) >= min_group_size]
    merged_singleton_ids = set()

    for singleton_group in singleton_groups:
        singleton_pattern = singleton_group['prototype']
        singleton_norm = float(singleton_pattern.get('norm1', singleton_pattern['masked'].abs().sum().item()))
        best_target = None

        for target_group in non_singleton_groups:
            if singleton_group.get('bucket_key') != target_group.get('bucket_key'):
                continue
            scale, error, similarity = search_best_power_of_two_scale(
                singleton_pattern['masked'],
                target_group['prototype']['masked'],
                scale_candidates=scale_candidates,
            )
            normalized_error = float(error / (singleton_norm + 1e-8))
            if (
                not np.isfinite(scale)
                or abs(scale) <= 1e-8
                or not np.isfinite(normalized_error)
                or normalized_error > float(max_scale_error)
            ):
                continue

            candidate = {
                'target_group': target_group,
                'multiplier': float(scale),
                'similarity': float(similarity),
                'normalized_error': normalized_error,
            }
            if best_target is None or (
                candidate['normalized_error'],
                -candidate['similarity'],
                candidate['target_group']['prototype']['out_ch'],
            ) < (
                best_target['normalized_error'],
                -best_target['similarity'],
                best_target['target_group']['prototype']['out_ch'],
            ):
                best_target = candidate

        if best_target is None:
            continue

        best_target['target_group']['members'].append({
            'pattern': singleton_pattern,
            'multiplier': best_target['multiplier'],
            'similarity': best_target['similarity'],
            'normalized_error': best_target['normalized_error'],
            'role': 'member',
        })
        if abs(best_target['multiplier'] - 1.0) > 1e-6 or best_target['similarity'] < exact_threshold:
            best_target['target_group']['repair_mode'] = 'scaled'
        merged_singleton_ids.add(id(singleton_group))

    return [group for group in final_groups if id(group) not in merged_singleton_ids]


def summarize_budgeted_grouping(layer_summary: Dict[str, Any],
                                assignment_errors: List[float],
                                config: Dict[str, Any],
                                prototype_budget: int,
                                prototype_count: int,
                                relaxed: bool,
                                relax_steps: int) -> Dict[str, Any]:
    if assignment_errors:
        assignment_error_mean = float(np.mean(assignment_errors))
        assignment_error_p95 = float(np.percentile(assignment_errors, 95))
    else:
        assignment_error_mean = 0.0
        assignment_error_p95 = 0.0

    achieved_coverage = float(layer_summary.get('coverage_ratio', 0.0))
    singleton_ratio = float(layer_summary.get('singleton_ratio', 1.0))
    avg_group_size = float(layer_summary.get('avg_group_size', 0.0))
    return {
        'grouping_mode': 'budgeted',
        'bucket_mode': str(config.get('bucket_mode', 'nonzero_count')),
        'prototype_budget': int(prototype_budget),
        'prototype_budget_ratio': float(config.get('prototype_budget_ratio', 0.25)),
        'prototype_count': int(prototype_count),
        'target_coverage': float(config.get('target_coverage', 0.6)),
        'achieved_coverage': achieved_coverage,
        'coverage_gap': max(float(config.get('target_coverage', 0.6)) - achieved_coverage, 0.0),
        'singleton_ratio': singleton_ratio,
        'avg_group_size': avg_group_size,
        'assignment_error_mean': assignment_error_mean,
        'assignment_error_p95': assignment_error_p95,
        'max_scale_error': float(config.get('max_scale_error', 0.25)),
        'budget_max_singleton': float(config.get('max_singleton', 0.5)),
        'budget_min_avg_group_size': float(config.get('min_avg_group_size', 2.0)),
        'prototype_budget_min': int(config.get('prototype_budget_min', 4)),
        'prototype_budget_max': int(config.get('prototype_budget_max', 256)),
        'relax_threshold': float(config.get('relax_threshold', 0.85)),
        'mask_family': list(config.get('mask_family', [])),
        'mask_keep_ratios': [float(value) for value in config.get('mask_keep_ratios', [])],
        'relaxed': int(bool(relaxed)),
        'relax_steps': int(relax_steps),
    }


def _budgeted_candidate_key(layer_summary: Dict[str, Any], budget_stats: Dict[str, Any], config: Dict[str, Any]) -> Tuple[float, ...]:
    coverage_gap = max(float(config.get('target_coverage', 0.6)) - float(layer_summary.get('coverage_ratio', 0.0)), 0.0)
    singleton_excess = max(float(layer_summary.get('singleton_ratio', 1.0)) - float(config.get('max_singleton', 0.5)), 0.0)
    avg_group_gap = max(float(config.get('min_avg_group_size', 2.0)) - float(layer_summary.get('avg_group_size', 0.0)), 0.0)
    return (
        coverage_gap,
        singleton_excess,
        avg_group_gap,
        float(budget_stats.get('assignment_error_p95', 0.0)),
        float(budget_stats.get('assignment_error_mean', 0.0)),
        float(layer_summary.get('singleton_ratio', 1.0)),
        -float(layer_summary.get('coverage_ratio', 0.0)),
    )


def _build_budgeted_groups_for_patterns(pattern_list: List[Dict[str, Any]],
                                        min_group_size: int,
                                        exact_threshold: float,
                                        scale_candidates,
                                        config: Dict[str, Any],
                                        bucket_key: Any = None):
    if not pattern_list:
        empty_summary = _build_layer_group_summary([], 0, 0, 0, 0, 0, 0, 0)
        empty_summary.update(summarize_budgeted_grouping(empty_summary, [], config, 0, 0, False, 0))
        return [], empty_summary

    num_patterns = len(pattern_list)
    current_config = dict(config)
    best_result = None
    relax_steps = 0
    max_relax_steps = 4

    for current_step in range(max_relax_steps + 1):
        prototype_budget = _resolve_prototype_budget(num_patterns, current_config)
        prototype_indices = budgeted_select_prototypes(pattern_list, prototype_budget)
        current_groups = budgeted_assign_members(
            pattern_list=pattern_list,
            prototype_indices=prototype_indices,
            scale_candidates=scale_candidates,
            max_scale_error=float(current_config.get('max_scale_error', 0.25)),
            exact_threshold=exact_threshold,
            bucket_key=bucket_key,
        )
        current_groups = budgeted_merge_singletons(
            current_groups,
            scale_candidates=scale_candidates,
            max_scale_error=float(current_config.get('max_scale_error', 0.25)),
            exact_threshold=exact_threshold,
            min_group_size=min_group_size,
        )

        repairable_block_count = 0
        repairable_ou_count = 0
        singleton_group_count = 0
        exact_group_count = 0
        scaled_group_count = 0
        assignment_errors = []

        for group in current_groups:
            group_size = len(group['members'])
            covered_ou_count = int(sum(int(member['pattern'].get('channel_span', 1)) for member in group['members']))
            if group_size >= min_group_size:
                repairable_block_count += group_size
                repairable_ou_count += covered_ou_count
                if group['repair_mode'] == 'exact':
                    exact_group_count += 1
                else:
                    scaled_group_count += 1
                assignment_errors.extend(
                    float(member.get('normalized_error', 0.0))
                    for member in group['members']
                    if member.get('role') != 'prototype'
                )
            else:
                singleton_group_count += 1

        total_ou_count = int(sum(int(pattern.get('channel_span', 1)) for pattern in pattern_list))
        layer_summary = _build_layer_group_summary(
            layer_groups=[{'group_size': len(group['members']), 'members': group['members'], 'repair_mode': group['repair_mode']} for group in current_groups],
            total_block_count=num_patterns,
            total_ou_count=total_ou_count,
            repairable_block_count=repairable_block_count,
            repairable_ou_count=repairable_ou_count,
            singleton_group_count=singleton_group_count,
            exact_group_count=exact_group_count,
            scaled_group_count=scaled_group_count,
        )
        budget_stats = summarize_budgeted_grouping(
            layer_summary=layer_summary,
            assignment_errors=assignment_errors,
            config=current_config,
            prototype_budget=prototype_budget,
            prototype_count=len(current_groups),
            relaxed=current_step > 0,
            relax_steps=current_step,
        )
        layer_summary.update(budget_stats)

        candidate_result = {
            'groups': current_groups,
            'layer_summary': layer_summary,
            'config': dict(current_config),
        }
        if best_result is None or _budgeted_candidate_key(candidate_result['layer_summary'], budget_stats, current_config) < _budgeted_candidate_key(best_result['layer_summary'], best_result['layer_summary'], best_result['config']):
            best_result = candidate_result

        if (
            float(layer_summary['coverage_ratio']) >= float(current_config.get('target_coverage', 0.6))
            and float(layer_summary['singleton_ratio']) <= float(current_config.get('max_singleton', 0.5))
            and float(layer_summary['avg_group_size']) >= float(current_config.get('min_avg_group_size', 2.0))
        ):
            break

        if current_step == max_relax_steps:
            break
        relax_steps += 1
        current_config['prototype_budget_ratio'] = max(0.05, float(current_config['prototype_budget_ratio']) * float(current_config.get('relax_threshold', 0.85)))
        current_config['max_scale_error'] = min(1.0, float(current_config['max_scale_error']) / max(float(current_config.get('relax_threshold', 0.85)), 1e-6))

    return best_result['groups'], best_result['layer_summary']


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


def _build_ft_groups_for_packed_bucket(packed: Dict[str, Any], block_index: int, bucket_out_indices: torch.Tensor,
                                       min_group_size: int, target_group_size: int,
                                       sim_threshold: float, exact_threshold: float,
                                       scale_candidates) -> List[Dict[str, Any]]:
    if bucket_out_indices.numel() == 0:
        return []

    bucket_norm = packed['norm1'][bucket_out_indices, block_index]
    ordered_out_indices = bucket_out_indices[torch.argsort(bucket_norm, descending=True)].tolist()
    block_groups: List[Dict[str, Any]] = []

    while ordered_out_indices:
        prototype_seed_out = ordered_out_indices[0]
        remaining_out_indices = ordered_out_indices[1:]
        compare_window = _get_bucket_compare_window(len(remaining_out_indices), packed['block_kind'])
        compare_out_indices = remaining_out_indices[:compare_window]

        selected_member_out_indices: List[int] = []
        if compare_out_indices:
            candidate_tensor = torch.tensor(compare_out_indices, dtype=torch.long)
            _, _, similarities = _batched_best_power_of_two_scale(
                packed['masked_flat'][candidate_tensor, block_index, :],
                packed['masked_flat'][prototype_seed_out, block_index, :],
                scale_candidates=scale_candidates,
            )
            for candidate_out, similarity in zip(compare_out_indices, similarities.tolist()):
                if similarity >= sim_threshold and len(selected_member_out_indices) < target_group_size - 1:
                    selected_member_out_indices.append(candidate_out)

        selected_set = set(selected_member_out_indices)
        ordered_out_indices = [out_idx for out_idx in compare_out_indices if out_idx not in selected_set] + remaining_out_indices[compare_window:]

        current_pattern_indices = [prototype_seed_out] + selected_member_out_indices
        current_patterns = [
            _pattern_from_packed_block(packed, out_idx, block_index)
            for out_idx in current_pattern_indices
        ]
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
                if abs(scale - 1.0) > 1e-6 or similarity < exact_threshold:
                    exact_group = False

            member_entries.append({
                'pattern': member,
                'multiplier': float(scale),
                'similarity': float(similarity),
                'role': role,
            })

        block_groups.append({
            'mask_signature': prototype_pattern['mask_signature'],
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


def _build_layer_ft_groups_from_packed(packed: Dict[str, Any], min_group_size: int, target_group_size: int,
                                       sim_threshold: float, exact_threshold: float,
                                       scale_candidates):
    layer_groups = []
    group_id = 0
    repairable_block_count = 0
    repairable_ou_count = 0
    singleton_group_count = 0
    exact_group_count = 0
    scaled_group_count = 0

    for block_index in range(0, packed['num_blocks']):
        block_mask_flat = packed['mask_flat'][:, block_index, :]
        _, inverse = torch.unique(block_mask_flat, dim=0, return_inverse=True)

        for mask_group_id in range(int(inverse.max().item()) + 1):
            bucket_out_indices = torch.nonzero(inverse == mask_group_id, as_tuple=False).flatten()
            if bucket_out_indices.numel() == 0:
                continue

            block_groups = _build_ft_groups_for_packed_bucket(
                packed=packed,
                block_index=block_index,
                bucket_out_indices=bucket_out_indices,
                min_group_size=min_group_size,
                target_group_size=target_group_size,
                sim_threshold=sim_threshold,
                exact_threshold=exact_threshold,
                scale_candidates=scale_candidates,
            )

            for group in block_groups:
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

    total_block_count = packed['out_channel'] * packed['num_blocks']
    total_ou_count = total_block_count * packed['channel_span']
    layer_summary = _build_layer_group_summary(
        layer_groups=layer_groups,
        total_block_count=total_block_count,
        total_ou_count=total_ou_count,
        repairable_block_count=repairable_block_count,
        repairable_ou_count=repairable_ou_count,
        singleton_group_count=singleton_group_count,
        exact_group_count=exact_group_count,
        scaled_group_count=scaled_group_count,
    )
    return layer_groups, layer_summary


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
    layer_summary = _build_layer_group_summary(
        layer_groups=layer_groups,
        total_block_count=total_block_count,
        total_ou_count=total_ou_count,
        repairable_block_count=repairable_block_count,
        repairable_ou_count=repairable_ou_count,
        singleton_group_count=singleton_group_count,
        exact_group_count=exact_group_count,
        scaled_group_count=scaled_group_count,
    )
    return layer_groups, layer_summary


def _build_layer_budgeted_groups_from_packed(packed: Dict[str, Any],
                                             weight_name: str,
                                             min_group_size: int,
                                             exact_threshold: float,
                                             scale_candidates,
                                             budget_config: Dict[str, Any]):
    layer_groups = []
    group_id = 0
    budgeted_layer_summaries = []
    layer_config = _resolve_budgeted_layer_config(budget_config, weight_name)
    bucket_mode = str(layer_config.get('bucket_mode', 'nonzero_count'))

    for block_index in range(0, packed['num_blocks']):
        bucketed_patterns: Dict[Any, List[Dict[str, Any]]] = {}
        for out_idx in range(0, packed['out_channel']):
            pattern = _pattern_from_packed_block(packed, out_idx, block_index)
            bucket_key = _budget_bucket_key_from_pattern(pattern, bucket_mode)
            bucketed_patterns.setdefault(bucket_key, []).append(pattern)

        for bucket_key, bucket_patterns in sorted(
            bucketed_patterns.items(),
            key=lambda item: str(_normalize_budget_bucket_key(item[0])),
        ):
            bucket_groups, bucket_summary = _build_budgeted_groups_for_patterns(
                pattern_list=bucket_patterns,
                min_group_size=min_group_size,
                exact_threshold=exact_threshold,
                scale_candidates=scale_candidates,
                config=layer_config,
                bucket_key=bucket_key,
            )
            budgeted_layer_summaries.append(bucket_summary)

            for group in bucket_groups:
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
                        'normalized_error': float(member.get('normalized_error', 0.0)),
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
                    'mask_signature': group['prototype']['mask_signature'].hex(),
                    'group_size': member_count,
                    'member_count': member_count,
                    'covered_ou_count': covered_ou_count,
                    'repair_mode': group['repair_mode'],
                    'bucket_key': _normalize_budget_bucket_key(group.get('bucket_key')),
                })
                group_id += 1

    total_block_count = packed['out_channel'] * packed['num_blocks']
    total_ou_count = total_block_count * packed['channel_span']
    repairable_block_count = 0
    repairable_ou_count = 0
    singleton_group_count = 0
    exact_group_count = 0
    scaled_group_count = 0
    assignment_errors = []

    for group in layer_groups:
        if group['group_size'] >= min_group_size:
            repairable_block_count += group['group_size']
            repairable_ou_count += int(group['covered_ou_count'])
            if group['repair_mode'] == 'exact':
                exact_group_count += 1
            else:
                scaled_group_count += 1
            assignment_errors.extend(
                float(member.get('normalized_error', 0.0))
                for member in group['members']
                if member.get('role') != 'prototype'
            )
        else:
            singleton_group_count += 1

    layer_summary = _build_layer_group_summary(
        layer_groups=layer_groups,
        total_block_count=total_block_count,
        total_ou_count=total_ou_count,
        repairable_block_count=repairable_block_count,
        repairable_ou_count=repairable_ou_count,
        singleton_group_count=singleton_group_count,
        exact_group_count=exact_group_count,
        scaled_group_count=scaled_group_count,
    )
    aggregate_budget_config = layer_config
    prototype_budget = int(sum(int(summary.get('prototype_budget', 0)) for summary in budgeted_layer_summaries))
    prototype_count = len(layer_groups)
    relaxed = any(int(summary.get('relaxed', 0)) for summary in budgeted_layer_summaries)
    relax_steps = max((int(summary.get('relax_steps', 0)) for summary in budgeted_layer_summaries), default=0)
    layer_summary.update(summarize_budgeted_grouping(
        layer_summary=layer_summary,
        assignment_errors=assignment_errors,
        config=aggregate_budget_config,
        prototype_budget=prototype_budget,
        prototype_count=prototype_count,
        relaxed=relaxed,
        relax_steps=relax_steps,
    ))
    return layer_groups, layer_summary


def _build_layer_budgeted_groups(pattern_list, weight_name, min_group_size, exact_threshold, scale_candidates, budget_config):
    layer_config = _resolve_budgeted_layer_config(budget_config, weight_name)
    bucket_mode = str(layer_config.get('bucket_mode', 'nonzero_count'))
    block_map: Dict[Tuple[int, Any], List[Dict[str, Any]]] = {}
    for pattern in pattern_list:
        bucket_key = _budget_bucket_key_from_pattern(pattern, bucket_mode)
        block_map.setdefault((pattern['in_ch_start'], bucket_key), []).append(pattern)

    layer_groups = []
    group_id = 0
    budgeted_layer_summaries = []

    for (block_start, bucket_key), block_patterns in sorted(block_map.items(), key=lambda item: (item[0][0], str(_normalize_budget_bucket_key(item[0][1])))):
        block_groups, block_summary = _build_budgeted_groups_for_patterns(
            pattern_list=block_patterns,
            min_group_size=min_group_size,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
            config=layer_config,
            bucket_key=bucket_key,
        )
        budgeted_layer_summaries.append(block_summary)

        for group in block_groups:
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
                    'normalized_error': float(member.get('normalized_error', 0.0)),
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
                'mask_signature': group['prototype']['mask_signature'].hex(),
                'group_size': member_count,
                'member_count': member_count,
                'covered_ou_count': covered_ou_count,
                'repair_mode': group['repair_mode'],
                'bucket_key': _normalize_budget_bucket_key(group.get('bucket_key')),
            })
            group_id += 1

    repairable_block_count = 0
    repairable_ou_count = 0
    singleton_group_count = 0
    exact_group_count = 0
    scaled_group_count = 0
    assignment_errors = []
    for group in layer_groups:
        if group['group_size'] >= min_group_size:
            repairable_block_count += group['group_size']
            repairable_ou_count += int(group['covered_ou_count'])
            if group['repair_mode'] == 'exact':
                exact_group_count += 1
            else:
                scaled_group_count += 1
            assignment_errors.extend(
                float(member.get('normalized_error', 0.0))
                for member in group['members']
                if member.get('role') != 'prototype'
            )
        else:
            singleton_group_count += 1

    total_block_count = len(pattern_list)
    total_ou_count = int(sum(pattern['channel_span'] for pattern in pattern_list))
    layer_summary = _build_layer_group_summary(
        layer_groups=layer_groups,
        total_block_count=total_block_count,
        total_ou_count=total_ou_count,
        repairable_block_count=repairable_block_count,
        repairable_ou_count=repairable_ou_count,
        singleton_group_count=singleton_group_count,
        exact_group_count=exact_group_count,
        scaled_group_count=scaled_group_count,
    )
    aggregate_budget_config = layer_config
    prototype_budget = int(sum(int(summary.get('prototype_budget', 0)) for summary in budgeted_layer_summaries))
    prototype_count = len(layer_groups)
    relaxed = any(int(summary.get('relaxed', 0)) for summary in budgeted_layer_summaries)
    relax_steps = max((int(summary.get('relax_steps', 0)) for summary in budgeted_layer_summaries), default=0)
    layer_summary.update(summarize_budgeted_grouping(
        layer_summary=layer_summary,
        assignment_errors=assignment_errors,
        config=aggregate_budget_config,
        prototype_budget=prototype_budget,
        prototype_count=prototype_count,
        relaxed=relaxed,
        relax_steps=relax_steps,
    ))
    return layer_groups, layer_summary


def _materialize_layer_group_outputs(layer_groups, layer_summary, weight_name, in_channel, out_channel, kernel_size, grouping_mode='ftscore', extra_fields=None):
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
        proto_multiplier = float(prototype.get('multiplier', 1.0))

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
            member_multiplier = float(member.get('multiplier', 1.0))

            for offset in range(0, member_span):
                target_in = member_in + offset
                scale_value = member_multiplier / proto_multiplier if abs(proto_multiplier) > 1e-8 else member_multiplier
                if 'fc' in weight_name:
                    multiple_relationship_table[member_out][target_in] = scale_value
                else:
                    multiple_relationship_table[member_out][target_in] = torch.ones(kernel_size, kernel_size) * scale_value

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
        'max_group_size': layer_summary.get('max_group_size', 0),
        'exact_group_count': layer_summary['exact_group_count'],
        'scaled_group_count': layer_summary['scaled_group_count'],
        'exact_group_ratio': layer_summary['exact_group_ratio'],
        'scaled_group_ratio': layer_summary['scaled_group_ratio'],
        'zero_multiplier_count': layer_summary.get('zero_multiplier_count', 0),
        'nonzero_multiplier_count': layer_summary.get('nonzero_multiplier_count', 0),
        'zero_multiplier_ratio': layer_summary.get('zero_multiplier_ratio', 0.0),
        'nonzero_multiplier_ratio': layer_summary.get('nonzero_multiplier_ratio', 0.0),
        'scale_distribution': layer_summary.get('scale_distribution', {}),
        'grouping_mode': grouping_mode,
        'groups': layer_groups,
    }
    if extra_fields:
        layer_group_information.update(extra_fields)
    return map_table, multiple_relationship_table, layer_summary['coverage_ratio'], layer_group_information


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


def _compile_ft_regularization_state(weight_name, ft_layer_enabled, group_information,
                                     min_coverage=0.0, min_repairable_groups=1):
    compiled_layers = {}
    summary = {
        'layer_count': 0,
        'repairable_group_count': 0,
        'member_link_count': 0,
        'singleton_group_count': 0,
        'skipped_low_coverage_layers': 0,
        'skipped_small_group_layers': 0,
        'skipped_no_repairable_group_layers': 0,
        'disabled_layer_count': 0,
    }
    layer_rows = []

    for layer_index, layer_name in enumerate(weight_name):
        layer_enabled = bool(ft_layer_enabled[layer_index])
        row = {
            'layer': layer_name,
            'ft_layer_enabled': int(layer_enabled),
            'coverage_ratio': 0.0,
            'repairable_groups': 0,
            'member_links': 0,
            'mask_reg_enabled': 0,
            'group_reg_enabled': 0,
            'reg_enabled': 0,
            'skip_reason': 'disabled_layer',
        }

        if not layer_enabled:
            summary['disabled_layer_count'] += 1
            layer_rows.append(row)
            continue

        row['mask_reg_enabled'] = 1

        layer_group_information = group_information.get(layer_name)
        if not layer_group_information:
            summary['skipped_no_repairable_group_layers'] += 1
            row['skip_reason'] = 'no_repairable_group'
            layer_rows.append(row)
            continue
        layer_coverage = float(layer_group_information.get('coverage_ratio', 0.0))
        row['coverage_ratio'] = layer_coverage
        if layer_coverage < float(min_coverage):
            summary['skipped_low_coverage_layers'] += 1
            row['skip_reason'] = 'low_coverage'
            layer_rows.append(row)
            continue

        compiled_groups = []
        singleton_group_count = 0
        for group in _get_group_member_entries(layer_group_information):
            group_size = int(group.get('group_size', len(group.get('members', []))))
            if group_size < 2:
                singleton_group_count += 1
                continue

            prototype = group['prototype']
            proto_span = int(prototype.get('channel_span', 1))
            compiled_members = []
            for member in group.get('members', []):
                if member.get('role') == 'prototype':
                    continue
                member_span = int(member.get('channel_span', 1))
                if member_span != proto_span:
                    continue
                compiled_members.append({
                    'out_ch': int(member['out_ch']),
                    'in_ch_start': int(member['in_ch_start']),
                    'channel_span': member_span,
                    'multiplier': float(member.get('multiplier', 1.0)),
                })

            if not compiled_members:
                continue

            compiled_groups.append({
                'prototype': {
                    'out_ch': int(prototype['out_ch']),
                    'in_ch_start': int(prototype['in_ch_start']),
                    'channel_span': proto_span,
                    'multiplier': float(prototype.get('multiplier', 1.0)),
                },
                'members': compiled_members,
            })

        row['repairable_groups'] = len(compiled_groups)
        row['member_links'] = sum(len(group['members']) for group in compiled_groups)

        if len(compiled_groups) == 0:
            summary['skipped_no_repairable_group_layers'] += 1
            summary['singleton_group_count'] += singleton_group_count
            row['skip_reason'] = 'no_repairable_group'
            layer_rows.append(row)
            continue

        if len(compiled_groups) < int(min_repairable_groups):
            summary['skipped_small_group_layers'] += 1
            summary['singleton_group_count'] += singleton_group_count
            row['skip_reason'] = 'small_group_count'
            layer_rows.append(row)
            continue

        compiled_layers[layer_name] = compiled_groups
        summary['layer_count'] += 1
        summary['repairable_group_count'] += len(compiled_groups)
        summary['member_link_count'] += sum(len(group['members']) for group in compiled_groups)
        summary['singleton_group_count'] += singleton_group_count
        row['group_reg_enabled'] = 1
        row['reg_enabled'] = 1
        row['skip_reason'] = 'enabled'
        layer_rows.append(row)

    return {
        'layers': compiled_layers,
        'summary': summary,
        'layer_rows': layer_rows,
    }


def _write_regularization_layers_report(model_name, translate_name, regularization_state, output_dir='.'):
    report_path = _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_regularization_layers.csv')
    pd.DataFrame(regularization_state.get('layer_rows', [])).to_csv(report_path, index=False)
    return str(report_path)


def write_mask_sweep_report(model_name, translate_name, group_information, output_dir='.'):
    rows = []
    for layer_name, layer_group_info in (group_information or {}).items():
        seed_info = {}
        if isinstance(layer_group_info, dict):
            seed_info = layer_group_info.get('seed_info', {}) or {}
        candidate_summaries = seed_info.get('candidate_summaries', []) or []
        selected_candidate = next((item for item in candidate_summaries if item.get('selected')), None)
        if selected_candidate is None and candidate_summaries:
            selected_strategy = seed_info.get('selected_strategy', '')
            selected_candidate = next(
                (item for item in candidate_summaries if item.get('strategy') == selected_strategy),
                None,
            )
        mask_density = seed_info.get('mask_density', None)
        if mask_density is None and selected_candidate is not None:
            mask_density = selected_candidate.get('mask_density', 0.0)
        if mask_density is None:
            mask_density = 0.0
        rows.append({
            'layer': layer_name,
            'selected_mask_strategy': seed_info.get('selected_strategy', ''),
            'mask_density': mask_density,
            'selected_candidate_coverage': seed_info.get('estimated_coverage', 0.0),
            'selected_candidate_block_coverage': seed_info.get('estimated_block_coverage', 0.0),
            'selected_candidate_avg_group_size': seed_info.get('estimated_avg_group_size', 0.0),
            'selected_candidate_singleton_ratio': seed_info.get('estimated_singleton_ratio', 0.0),
            'selected_candidate_zero_multiplier_ratio': seed_info.get('estimated_zero_multiplier_ratio', 0.0),
            'selected_candidate_exact_group_ratio': seed_info.get('estimated_exact_group_ratio', 0.0),
            'selected_candidate_scaled_group_ratio': seed_info.get('estimated_scaled_group_ratio', 0.0),
            'selected_candidate_distortion': seed_info.get('pruning_distortion', 0.0),
            'target_coverage': seed_info.get('target_coverage', None),
            'prefer_sparser_mask': seed_info.get('prefer_sparser_mask', False),
            'singleton_penalty': seed_info.get('singleton_penalty', 0.0),
            'zero_scale_penalty': seed_info.get('zero_scale_penalty', 0.0),
            'candidate_count': seed_info.get('candidate_count', 0),
            'fallback_used': seed_info.get('fallback_used', False),
            'candidate_summaries': json.dumps(candidate_summaries, ensure_ascii=False),
        })

    csv_path = _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_mask_sweep_report.csv')
    json_path = _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_mask_sweep_report.json')
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding='utf-8')
    return str(csv_path), str(json_path)


def _write_training_profile(profile_records, profile_path):
    pd.DataFrame(profile_records).to_csv(profile_path, index=False)


def _effective_ft_reg_interval(epoch_number, base_interval, refresh_epochs, boost_after_refresh):
    if base_interval <= 1 or not boost_after_refresh:
        return max(int(base_interval), 1)

    refresh_set = set(int(epoch) for epoch in refresh_epochs)
    if epoch_number in refresh_set or (epoch_number - 1) in refresh_set:
        return max(1, int(base_interval) // 2)
    return max(int(base_interval), 1)


def ft_group_score_mask(model, weight_name, in_channel, out_channel, kernel_size,
                        channel_number, pattern_value_number, pattern_shape_number,
                        OU_size, target_group_size=4, sim_threshold=0.85,
                        mask_density_sweep=False, mask_keep_ratios=None,
                        target_coverage=None, prefer_sparser_mask=False,
                        singleton_penalty=0.18, zero_scale_penalty=0.12):
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
    keep_ratios = set()
    if mask_keep_ratios:
        keep_ratios.update(max(0.05, min(1.0, round(float(ratio), 4))) for ratio in mask_keep_ratios)
    if mask_density_sweep or not keep_ratios:
        keep_ratios.update({
            max(0.05, min(1.0, round(seed_density * 0.5, 4))),
            max(0.05, min(1.0, round(seed_density * 0.75, 4))),
            max(0.05, min(1.0, round(seed_density, 4))),
            max(0.05, min(1.0, round(seed_density * 1.2 + 0.02, 4))),
        })
    keep_ratios = sorted(keep_ratios)

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

    deduplicated_candidate_masks = []
    for strategy_name, candidate_mask in candidate_masks:
        if any(torch.equal(candidate_mask, existing_mask) for _, existing_mask in deduplicated_candidate_masks):
            continue
        deduplicated_candidate_masks.append((strategy_name, candidate_mask))

    valid_candidates = []
    candidate_summaries = []
    for strategy_name, candidate_mask in deduplicated_candidate_masks:
        packed = _pack_even_layer_blocks(
            weight_tensor=raw_weight,
            mask_tensor=candidate_mask,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            channel_number=channel_number,
        )
        if packed is not None:
            estimated_groups, estimated_summary = _build_layer_ft_groups_from_packed(
                packed=packed,
                min_group_size=2,
                target_group_size=target_group_size,
                sim_threshold=sim_threshold,
                exact_threshold=0.98,
                scale_candidates=[0.25, 0.5, 1.0, 2.0, 4.0],
            )
        else:
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
        mask_density = float(candidate_mask.sum().item() / max(candidate_mask.numel(), 1))
        avg_similarity = _mean_or_zero(member_similarities) if member_similarities else (1.0 if estimated_groups else 0.0)
        avg_group_size = float(estimated_summary.get('avg_group_size', 0.0))
        normalized_group_size = min(avg_group_size / max(float(target_group_size), 1.0), 1.0)
        singleton_ratio = float(estimated_summary.get('singleton_ratio', 1.0))
        zero_multiplier_ratio = float(estimated_summary.get('zero_multiplier_ratio', 0.0))
        repairable_ou_ratio = float(estimated_summary.get('coverage_ratio', 0.0))
        exact_group_ratio = float(estimated_summary.get('exact_group_ratio', 0.0))
        ft_score = (
            0.48 * repairable_ou_ratio
            + 0.18 * normalized_group_size
            + 0.08 * avg_similarity
            + 0.06 * exact_group_ratio
            - singleton_penalty * singleton_ratio
            - 0.18 * pruning_distortion
            - zero_scale_penalty * zero_multiplier_ratio
        )
        if prefer_sparser_mask:
            ft_score -= 0.05 * mask_density

        candidate_summary = {
            'strategy': strategy_name,
            'score': float(ft_score),
            'mask_density': mask_density,
            'estimated_coverage': float(estimated_summary['coverage_ratio']),
            'estimated_block_coverage': float(estimated_summary['block_coverage_ratio']),
            'estimated_repairable_ou_ratio': repairable_ou_ratio,
            'estimated_avg_group_size': avg_group_size,
            'estimated_group_count': int(estimated_summary['group_count']),
            'estimated_singleton_ratio': float(estimated_summary['singleton_ratio']),
            'estimated_zero_multiplier_ratio': zero_multiplier_ratio,
            'estimated_exact_group_ratio': exact_group_ratio,
            'estimated_scaled_group_ratio': float(estimated_summary.get('scaled_group_ratio', 0.0)),
            'estimated_max_group_size': int(estimated_summary.get('max_group_size', 0)),
            'estimated_scale_distribution': estimated_summary.get('scale_distribution', {}),
            'avg_intra_group_similarity': float(avg_similarity),
            'pruning_distortion': float(pruning_distortion),
            'target_coverage_satisfied': bool(target_coverage is not None and repairable_ou_ratio >= float(target_coverage)),
            'selected': False,
        }
        candidate_summaries.append(candidate_summary)
        valid_candidates.append((candidate_summary, candidate_mask))

    fallback_used = False
    if valid_candidates:
        target_candidates = []
        if target_coverage is not None:
            target_candidates = [
                item for item in valid_candidates
                if item[0]['estimated_repairable_ou_ratio'] >= float(target_coverage)
            ]
        selection_pool = target_candidates if target_candidates else valid_candidates
        if target_candidates:
            best_summary, best_mask = min(
                selection_pool,
                key=lambda item: (
                    item[0]['pruning_distortion'],
                    item[0]['mask_density'] if prefer_sparser_mask else 0.0,
                    -item[0]['estimated_repairable_ou_ratio'],
                    -item[0]['score'],
                ),
            )
        else:
            best_summary, best_mask = max(
                selection_pool,
                key=lambda item: (
                    item[0]['score'],
                    item[0]['estimated_repairable_ou_ratio'],
                    -item[0]['estimated_singleton_ratio'],
                    item[0]['avg_intra_group_similarity'],
                    item[0]['estimated_avg_group_size'],
                    -item[0]['mask_density'] if prefer_sparser_mask else 0.0,
                    1 if item[0]['strategy'] != 'shape_seed' else 0,
                ),
            )
    else:
        best_summary = {
            'strategy': 'shape_seed',
            'score': 0.0,
            'mask_density': seed_density,
            'estimated_coverage': 0.0,
            'estimated_block_coverage': 0.0,
            'estimated_repairable_ou_ratio': 0.0,
            'estimated_avg_group_size': 0.0,
            'estimated_group_count': 0,
            'estimated_singleton_ratio': 1.0,
            'estimated_zero_multiplier_ratio': 0.0,
            'estimated_exact_group_ratio': 0.0,
            'estimated_scaled_group_ratio': 0.0,
            'estimated_max_group_size': 0,
            'estimated_scale_distribution': {},
            'avg_intra_group_similarity': 0.0,
            'pruning_distortion': 0.0,
            'target_coverage_satisfied': False,
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
        'estimated_zero_multiplier_ratio': best_summary['estimated_zero_multiplier_ratio'],
        'estimated_exact_group_ratio': best_summary['estimated_exact_group_ratio'],
        'estimated_scaled_group_ratio': best_summary['estimated_scaled_group_ratio'],
        'estimated_max_group_size': best_summary['estimated_max_group_size'],
        'estimated_scale_distribution': best_summary['estimated_scale_distribution'],
        'mask_density': best_summary['mask_density'],
        'avg_intra_group_similarity': best_summary['avg_intra_group_similarity'],
        'pruning_distortion': best_summary['pruning_distortion'],
        'target_coverage': float(target_coverage) if target_coverage is not None else None,
        'prefer_sparser_mask': bool(prefer_sparser_mask),
        'singleton_penalty': float(singleton_penalty),
        'zero_scale_penalty': float(zero_scale_penalty),
        'candidate_count': len(candidate_summaries),
        'fallback_used': fallback_used,
        'candidate_summaries': candidate_summaries,
    }
    return best_mask, group_seed_info


def _build_budgeted_mask_family_candidates(model, weight_name, in_channel, out_channel, kernel_size,
                                           channel_number, pattern_value_number, pattern_shape_number,
                                           OU_size, budget_config):
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
    family = [item for item in budget_config.get('mask_family', ['shape_seed', 'shared_topk', 'per_out_topk']) if item]
    keep_ratios = [max(0.05, min(1.0, float(ratio))) for ratio in budget_config.get('mask_keep_ratios', [0.6667, 0.4444])]
    candidate_masks = []
    if 'shape_seed' in family:
        candidate_masks.append(('shape_seed_budgeted', shape_seed_mask))
    for keep_ratio in keep_ratios:
        if 'shared_topk' in family:
            candidate_masks.append((f'shared_topk_{keep_ratio:.4f}_budgeted', _build_shared_topk_mask(
                model=model,
                weight_name=weight_name,
                in_channel=in_channel,
                out_channel=out_channel,
                channel_number=channel_number,
                keep_ratio=keep_ratio,
            )))
        if 'per_out_topk' in family:
            candidate_masks.append((f'per_out_topk_{keep_ratio:.4f}_budgeted', _build_per_out_topk_mask(
                model=model,
                weight_name=weight_name,
                in_channel=in_channel,
                out_channel=out_channel,
                channel_number=channel_number,
                keep_ratio=keep_ratio,
            )))
    if not candidate_masks:
        candidate_masks.append(('dense_mask_budgeted', torch.ones_like(raw_weight)))

    deduplicated = []
    for strategy_name, candidate_mask in candidate_masks:
        if any(torch.equal(candidate_mask, existing_mask) for _, existing_mask in deduplicated):
            continue
        deduplicated.append((strategy_name, candidate_mask))
    return deduplicated


def _budgeted_candidate_selection_key(candidate_summary, prefer_sparser=True):
    return (
        max(float(candidate_summary.get('target_coverage', 0.0)) - float(candidate_summary.get('estimated_coverage', 0.0)), 0.0),
        max(float(candidate_summary.get('estimated_singleton_ratio', 1.0)) - float(candidate_summary.get('budget_max_singleton', 1.0)), 0.0),
        max(float(candidate_summary.get('budget_min_avg_group_size', 0.0)) - float(candidate_summary.get('estimated_avg_group_size', 0.0)), 0.0),
        float(candidate_summary.get('estimated_assignment_error_p95', 0.0)),
        float(candidate_summary.get('estimated_assignment_error_mean', 0.0)),
        float(candidate_summary.get('mask_density', 0.0)) if prefer_sparser else 0.0,
        -float(candidate_summary.get('estimated_coverage', 0.0)),
        float(candidate_summary.get('estimated_singleton_ratio', 1.0)),
        -float(candidate_summary.get('estimated_avg_group_size', 0.0)),
    )


def ft_budgeted_select_mask_candidate(model, weight_name, in_channel, out_channel, kernel_size,
                                      channel_number, pattern_value_number, pattern_shape_number,
                                      OU_size, min_group_size=2, exact_threshold=0.98,
                                      scale_candidates=None, budget_config=None):
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]
    budget_config = dict(budget_config or {})
    raw_weight = model.state_dict()[weight_name].detach().cpu()
    candidate_masks = _build_budgeted_mask_family_candidates(
        model=model,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        channel_number=channel_number,
        pattern_value_number=pattern_value_number,
        pattern_shape_number=pattern_shape_number,
        OU_size=OU_size,
        budget_config=budget_config,
    )

    valid_candidates = []
    candidate_summaries = []
    for strategy_name, candidate_mask in candidate_masks:
        map_info, multiple_info, coverage_ratio, group_info = ft_budgeted_group_translate(
            model=model,
            in_channel=in_channel,
            out_channel=out_channel,
            weight_name=weight_name,
            kernel_size=kernel_size,
            channel_number=channel_number,
            mask=candidate_mask,
            min_group_size=min_group_size,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
            budget_config=budget_config,
        )
        pruning_distortion = float((raw_weight - raw_weight * candidate_mask).abs().sum().item() / (raw_weight.abs().sum().item() + 1e-8))
        mask_density = float(candidate_mask.sum().item() / max(candidate_mask.numel(), 1))
        candidate_summary = {
            'strategy': strategy_name,
            'mask_density': mask_density,
            'estimated_coverage': float(group_info.get('coverage_ratio', coverage_ratio)),
            'estimated_block_coverage': float(group_info.get('block_coverage_ratio', 0.0)),
            'estimated_repairable_ou_ratio': float(group_info.get('coverage_ratio', coverage_ratio)),
            'estimated_avg_group_size': float(group_info.get('avg_group_size', 0.0)),
            'estimated_group_count': int(group_info.get('group_count', 0)),
            'estimated_singleton_ratio': float(group_info.get('singleton_ratio', 1.0)),
            'estimated_zero_multiplier_ratio': float(group_info.get('zero_multiplier_ratio', 0.0)),
            'estimated_exact_group_ratio': float(group_info.get('exact_group_ratio', 0.0)),
            'estimated_scaled_group_ratio': float(group_info.get('scaled_group_ratio', 0.0)),
            'estimated_max_group_size': int(group_info.get('max_group_size', 0)),
            'estimated_scale_distribution': group_info.get('scale_distribution', {}),
            'estimated_assignment_error_mean': float(group_info.get('assignment_error_mean', 0.0)),
            'estimated_assignment_error_p95': float(group_info.get('assignment_error_p95', 0.0)),
            'pruning_distortion': pruning_distortion,
            'target_coverage': float(group_info.get('target_coverage', budget_config.get('target_coverage', 0.0))),
            'budget_max_singleton': float(group_info.get('budget_max_singleton', budget_config.get('max_singleton', 0.5))),
            'budget_min_avg_group_size': float(group_info.get('budget_min_avg_group_size', budget_config.get('min_avg_group_size', 2.0))),
            'target_coverage_satisfied': bool(float(group_info.get('coverage_ratio', coverage_ratio)) >= float(budget_config.get('target_coverage', 0.0))),
            'target_singleton_satisfied': bool(float(group_info.get('singleton_ratio', 1.0)) <= float(budget_config.get('max_singleton', 1.0))),
            'target_avg_group_size_satisfied': bool(float(group_info.get('avg_group_size', 0.0)) >= float(budget_config.get('min_avg_group_size', 0.0))),
            'selected': False,
        }
        candidate_summaries.append(candidate_summary)
        valid_candidates.append((candidate_summary, candidate_mask, (map_info, multiple_info, coverage_ratio, group_info)))

    best_summary, best_mask, best_outputs = min(
        valid_candidates,
        key=lambda item: _budgeted_candidate_selection_key(item[0]),
    )
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
        'estimated_zero_multiplier_ratio': best_summary['estimated_zero_multiplier_ratio'],
        'estimated_exact_group_ratio': best_summary['estimated_exact_group_ratio'],
        'estimated_scaled_group_ratio': best_summary['estimated_scaled_group_ratio'],
        'estimated_max_group_size': best_summary['estimated_max_group_size'],
        'estimated_scale_distribution': best_summary['estimated_scale_distribution'],
        'mask_density': best_summary['mask_density'],
        'avg_intra_group_similarity': 0.0,
        'pruning_distortion': best_summary['pruning_distortion'],
        'candidate_count': len(candidate_summaries),
        'fallback_used': False,
        'candidate_summaries': candidate_summaries,
        'budgeted_mask_family': list(budget_config.get('mask_family', [])),
        'budgeted_mask_keep_ratios': [float(value) for value in budget_config.get('mask_keep_ratios', [])],
        'budgeted_bucket_mode': str(budget_config.get('bucket_mode', 'nonzero_count')),
    }
    return best_mask, group_seed_info, best_outputs


def ft_group_cluster_translate(model, in_channel, out_channel, weight_name,
                               kernel_size, channel_number, mask,
                               min_group_size=2, target_group_size=4,
                               sim_threshold=0.85, exact_threshold=0.98,
                               scale_candidates=None, grouping_mode='ftscore', budget_config=None):
    """构建FT-oriented group，并输出兼容旧接口的map/multiple文件。"""
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]

    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    mask_tensor = mask.detach().cpu() if mask is not None else torch.ones_like(weight_tensor)
    packed = _pack_even_layer_blocks(
        weight_tensor=weight_tensor,
        mask_tensor=mask_tensor,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        channel_number=channel_number,
    )
    if packed is not None:
        layer_groups, layer_summary = _build_layer_ft_groups_from_packed(
            packed=packed,
            min_group_size=min_group_size,
            target_group_size=target_group_size,
            sim_threshold=sim_threshold,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
        )
    else:
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

    return _materialize_layer_group_outputs(
        layer_groups=layer_groups,
        layer_summary=layer_summary,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        grouping_mode='ftscore',
    )


def ft_budgeted_group_translate(model, in_channel, out_channel, weight_name,
                                kernel_size, channel_number, mask,
                                min_group_size=2, target_group_size=4,
                                sim_threshold=0.85, exact_threshold=0.98,
                                scale_candidates=None, grouping_mode='budgeted',
                                budget_config=None):
    """Active redundancy grouping with prototype budgets and bounded assignment error."""
    if scale_candidates is None:
        scale_candidates = [0.25, 0.5, 1.0, 2.0, 4.0]

    weight_tensor = model.state_dict()[weight_name].detach().cpu()
    mask_tensor = mask.detach().cpu() if mask is not None else torch.ones_like(weight_tensor)
    packed = _pack_even_layer_blocks(
        weight_tensor=weight_tensor,
        mask_tensor=mask_tensor,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        channel_number=channel_number,
    )
    if packed is not None:
        layer_groups, layer_summary = _build_layer_budgeted_groups_from_packed(
            packed=packed,
            weight_name=weight_name,
            min_group_size=min_group_size,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
            budget_config=budget_config or {},
        )
    else:
        pattern_list = extract_ou_patterns(
            model=model,
            weight_name=weight_name,
            in_channel=in_channel,
            out_channel=out_channel,
            kernel_size=kernel_size,
            channel_number=channel_number,
            mask=mask,
        )
        layer_groups, layer_summary = _build_layer_budgeted_groups(
            pattern_list=pattern_list,
            weight_name=weight_name,
            min_group_size=min_group_size,
            exact_threshold=exact_threshold,
            scale_candidates=scale_candidates,
            budget_config=budget_config or {},
        )

    return _materialize_layer_group_outputs(
        layer_groups=layer_groups,
        layer_summary=layer_summary,
        weight_name=weight_name,
        in_channel=in_channel,
        out_channel=out_channel,
        kernel_size=kernel_size,
        grouping_mode='budgeted',
        extra_fields={
            'bucket_mode': layer_summary.get('bucket_mode', 'nonzero_count'),
            'prototype_budget': int(layer_summary.get('prototype_budget', 0)),
            'prototype_budget_ratio': float(layer_summary.get('prototype_budget_ratio', 0.0)),
            'prototype_count': int(layer_summary.get('prototype_count', len(layer_groups))),
            'target_coverage': float(layer_summary.get('target_coverage', 0.0)),
            'achieved_coverage': float(layer_summary.get('achieved_coverage', layer_summary.get('coverage_ratio', 0.0))),
            'coverage_gap': float(layer_summary.get('coverage_gap', 0.0)),
            'assignment_error_mean': float(layer_summary.get('assignment_error_mean', 0.0)),
            'assignment_error_p95': float(layer_summary.get('assignment_error_p95', 0.0)),
            'max_scale_error': float(layer_summary.get('max_scale_error', 0.0)),
            'budget_max_singleton': float(layer_summary.get('budget_max_singleton', 0.0)),
            'budget_min_avg_group_size': float(layer_summary.get('budget_min_avg_group_size', 0.0)),
            'prototype_budget_min': int(layer_summary.get('prototype_budget_min', 0)),
            'prototype_budget_max': int(layer_summary.get('prototype_budget_max', 0)),
            'relax_threshold': float(layer_summary.get('relax_threshold', 0.0)),
            'mask_family': list(layer_summary.get('mask_family', [])),
            'mask_keep_ratios': [float(value) for value in layer_summary.get('mask_keep_ratios', [])],
            'relaxed': int(layer_summary.get('relaxed', 0)),
            'relax_steps': int(layer_summary.get('relax_steps', 0)),
        },
    )


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


def apply_ft_group_projection(model, weight_name, mask, group_information):
    """Public wrapper for projecting a model to the current FT grouping state."""
    _apply_ft_group_projection(model, weight_name, mask, group_information)


def _compute_ft_regularization(model, weight_name, ft_layer_enabled, mask, group_information, device,
                               regularization_state=None):
    loss_mask = torch.zeros(1, device=device)
    loss_proto = torch.zeros(1, device=device)
    loss_balance = torch.zeros(1, device=device)
    loss_sep = torch.zeros(1, device=device)

    parameter_map = dict(model.named_parameters())
    compiled_layers = regularization_state.get('layers', {}) if regularization_state is not None else None
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

        if compiled_layers is not None:
            layer_groups = compiled_layers.get(layer_name, [])
        else:
            layer_group_information = group_information.get(layer_name)
            if not layer_group_information:
                continue
            layer_groups = _get_group_member_entries(layer_group_information)

        for group in layer_groups:
            if compiled_layers is None:
                group_size = int(group.get('group_size', len(group.get('members', []))))
                if group_size < 2:
                    continue

            prototype = group['prototype']
            proto_out = int(prototype['out_ch'])
            proto_in = int(prototype['in_ch_start'])
            proto_span = int(prototype.get('channel_span', 1))
            prototype_multiplier = float(prototype.get('multiplier', 1.0))
            prototype_weight = _extract_block_tensor(projected, proto_out, proto_in, proto_span)

            for member in group['members']:
                if compiled_layers is None and member.get('role') == 'prototype':
                    continue
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
                             checkpoint_epoch=150,
                             group_refresh_epoch=None,
                             min_group_size=2, target_group_size=4,
                             sim_threshold=0.85, exact_threshold=0.98,
                             scale_candidates=None, pattern_value_number=None,
                             pattern_shape_number=8, OU_size=8,
                             ft_reg_interval=1,
                             ft_reg_min_coverage=0.0,
                             ft_reg_min_groups=1,
                             ft_reg_boost_after_refresh=False,
                             ft_grouping_mode='ftscore',
                             ft_budget_config=None,
                             ft_mask_density_sweep=False,
                             ft_mask_keep_ratios=None,
                             ft_target_coverage=None,
                             ft_prefer_sparser_mask=False,
                             ft_score_singleton_penalty=0.18,
                             ft_score_zero_scale_penalty=0.12,
                             output_dir='.'):
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
    checkpoint = torch.load('model_' + model_name + '_original_parameter_epoch' + str(checkpoint_epoch) + '_ckpt.pth')
    refresh_records = []
    refresh_log_path = _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_refresh_log.csv')
    training_profile_records = []
    training_profile_path = _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_training_profile.csv')
    regularization_state = _compile_ft_regularization_state(
        weight_name,
        ft_layer_enabled,
        group_information,
        min_coverage=ft_reg_min_coverage,
        min_repairable_groups=ft_reg_min_groups,
    )
    regularization_report_path = _write_regularization_layers_report(model_name, translate_name, regularization_state, output_dir=output_dir)
    write_mask_sweep_report(model_name, translate_name, group_information, output_dir=output_dir)
    print('[FT regularization] layers={} repairable_groups={} member_links={} singleton_groups_skipped={} low_coverage_layers_skipped={} small_group_layers_skipped={} reg_interval={} report={}'.format(
        regularization_state['summary']['layer_count'],
        regularization_state['summary']['repairable_group_count'],
        regularization_state['summary']['member_link_count'],
        regularization_state['summary']['singleton_group_count'],
        regularization_state['summary']['skipped_low_coverage_layers'],
        regularization_state['summary']['skipped_small_group_layers'],
        ft_reg_interval,
        regularization_report_path,
    ))

    for epoch in range(checkpoint['epoch'], max_epoches):
        epoch_start_time = time.time()
        batch_count = 0
        reg_batch_count = 0
        ce_loss_sum = 0.0
        ft_reg_loss_sum = 0.0
        mask_loss_sum = 0.0
        proto_loss_sum = 0.0
        balance_loss_sum = 0.0
        sep_loss_sum = 0.0
        reg_batch_mask_loss_sum = 0.0
        reg_batch_proto_loss_sum = 0.0
        reg_batch_balance_loss_sum = 0.0
        reg_batch_sep_loss_sum = 0.0
        reg_batch_ft_reg_loss_sum = 0.0
        reg_time_sec = 0.0
        refresh_time_sec = 0.0
        projection_time_sec = 0.0
        effective_reg_interval = _effective_ft_reg_interval(
            epoch_number=epoch + 1,
            base_interval=ft_reg_interval,
            refresh_epochs=group_refresh_epoch,
            boost_after_refresh=ft_reg_boost_after_refresh,
        )

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
                batch_count += 1
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss_ce = F.cross_entropy(outputs, targets)
                ce_loss_sum += float(loss_ce.item())
                apply_ft_reg = ((batch_idx % effective_reg_interval) == 0) or ((batch_idx + 1) == len(train_loader))
                if apply_ft_reg:
                    reg_start_time = time.time()
                    loss_mask, loss_proto, loss_balance, loss_sep = _compute_ft_regularization(
                        model=model,
                        weight_name=weight_name,
                        ft_layer_enabled=ft_layer_enabled,
                        mask=mask,
                        group_information=group_information,
                        device=device,
                        regularization_state=regularization_state,
                    )
                    reg_time_sec += time.time() - reg_start_time
                    reg_batch_count += 1
                else:
                    loss_mask = torch.zeros(1, device=device)
                    loss_proto = torch.zeros(1, device=device)
                    loss_balance = torch.zeros(1, device=device)
                    loss_sep = torch.zeros(1, device=device)
                mask_loss_sum += float(loss_mask.item())
                proto_loss_sum += float(loss_proto.item())
                balance_loss_sum += float(loss_balance.item())
                sep_loss_sum += float(loss_sep.item())
                ft_reg_loss_value = float(
                    (ft_mask_lambda * loss_mask + ft_proto_lambda * loss_proto + ft_balance_lambda * loss_balance + ft_sep_lambda * loss_sep).item()
                )
                ft_reg_loss_sum += ft_reg_loss_value
                if apply_ft_reg:
                    reg_batch_mask_loss_sum += float(loss_mask.item())
                    reg_batch_proto_loss_sum += float(loss_proto.item())
                    reg_batch_balance_loss_sum += float(loss_balance.item())
                    reg_batch_sep_loss_sum += float(loss_sep.item())
                    reg_batch_ft_reg_loss_sum += ft_reg_loss_value
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

                if batch_idx == 0 or (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(train_loader):
                    print('[FT train] epoch {} batch {}/{} reg={} interval={} ce={:.4f} mask={:.4f} proto={:.4f} sep={:.4f}'.format(
                        epoch + 1,
                        batch_idx + 1,
                        len(train_loader),
                        int(apply_ft_reg),
                        effective_reg_interval,
                        float(loss_ce.item()),
                        float(loss_mask.item()),
                        float(loss_proto.item()),
                        float(loss_sep.item()),
                    ))

            scheduler.step()
            train_accuracy_record[epoch] = correct / total
            train_loss_record[epoch] = train_loss
            print('epoch: ' + str(epoch + 1) + '  train_loss: ' + str(train_loss_record[epoch]) + ';  train_accuracy: ' + str(train_accuracy_record[epoch] * 100) + '%')
            test_accuracy_record[epoch], test_loss_record[epoch] = test(model, device, test_loader)
            print('epoch: ' + str(epoch + 1) + '  test_loss: ' + str(test_loss_record[epoch]) + ';  test_accuracy: ' + str(test_accuracy_record[epoch] * 100) + '%')

        if (epoch + 1) in group_refresh_epoch:
            refresh_start_time = time.time()
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
                        mask_density_sweep=ft_mask_density_sweep,
                        mask_keep_ratios=ft_mask_keep_ratios,
                        target_coverage=ft_target_coverage,
                        prefer_sparser_mask=ft_prefer_sparser_mask,
                        singleton_penalty=ft_score_singleton_penalty,
                        zero_scale_penalty=ft_score_zero_scale_penalty,
                    )
                    print('[FT refresh] epoch {} layer {} regroup'.format(epoch + 1, weight_name[i]))
                    grouping_fn = ft_budgeted_group_translate if ft_grouping_mode == 'budgeted' else ft_group_cluster_translate
                    candidate_map_information[weight_name[i]], candidate_multiple_relationship_information[weight_name[i]], candidate_reuse_ratio_information[weight_name[i]], candidate_group_information[weight_name[i]] = grouping_fn(
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
                        grouping_mode=ft_grouping_mode,
                        budget_config=ft_budget_config,
                    )
                    candidate_group_information[weight_name[i]]['seed_info'] = group_seed_info
                    candidate_group_information[weight_name[i]]['grouping_mode'] = ft_grouping_mode

                after_summary = summarize_group_information(candidate_group_information, weight_name, ft_layer_enabled)
                accepted = _refresh_acceptance_score(after_summary) + 1e-8 >= _refresh_acceptance_score(before_summary)
                if accepted:
                    mask = candidate_mask
                    group_information = candidate_group_information
                    map_information = candidate_map_information
                    multiple_relationship_information = candidate_multiple_relationship_information
                    reuse_ratio_information = candidate_reuse_ratio_information
                    regularization_state = _compile_ft_regularization_state(
                        weight_name,
                        ft_layer_enabled,
                        group_information,
                        min_coverage=ft_reg_min_coverage,
                        min_repairable_groups=ft_reg_min_groups,
                    )
                    _write_regularization_layers_report(model_name, translate_name, regularization_state, output_dir=output_dir)
                    write_mask_sweep_report(model_name, translate_name, group_information, output_dir=output_dir)
                    print('[FT regularization] epoch {} cache layers={} repairable_groups={} member_links={} singleton_groups_skipped={} low_coverage_layers_skipped={} small_group_layers_skipped={}'.format(
                        epoch + 1,
                        regularization_state['summary']['layer_count'],
                        regularization_state['summary']['repairable_group_count'],
                        regularization_state['summary']['member_link_count'],
                        regularization_state['summary']['singleton_group_count'],
                        regularization_state['summary']['skipped_low_coverage_layers'],
                        regularization_state['summary']['skipped_small_group_layers'],
                    ))
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
            refresh_time_sec += time.time() - refresh_start_time

        if epoch + 1 in translate_epoch:
            before_translate_accuracy[current_iteration], before_translate_loss[current_iteration] = test(model, device, test_loader)
            print('Before_translate_accuracy: ' + str(before_translate_accuracy[current_iteration]) + ' Before_translate_loss: ' + str(before_translate_loss[current_iteration]))
            projection_start_time = time.time()
            _apply_ft_group_projection(model, weight_name, mask, group_information)
            projection_time_sec += time.time() - projection_start_time
            after_translate_accuracy[current_iteration], after_translate_loss[current_iteration] = test(model, device, test_loader)
            print('After_translate_accuracy: ' + str(after_translate_accuracy[current_iteration]) + ' After_translate_loss: ' + str(after_translate_loss[current_iteration]))
            model_accuracy_difference[current_iteration] = before_translate_accuracy[current_iteration] - after_translate_accuracy[current_iteration]
            print('Model_accuracy_difference: ' + str(model_accuracy_difference[current_iteration]))
            current_iteration = current_iteration + 1

        epoch_time_sec = time.time() - epoch_start_time
        average_denominator = max(batch_count, 1)
        reg_average_denominator = max(reg_batch_count, 1)
        training_profile_records.append({
            'epoch': epoch + 1,
            'batch_count': batch_count,
            'reg_batch_count': reg_batch_count,
            'effective_reg_interval': effective_reg_interval,
            'active_reg_layers': regularization_state['summary']['layer_count'],
            'skipped_low_coverage_layers': regularization_state['summary']['skipped_low_coverage_layers'],
            'skipped_small_group_layers': regularization_state['summary']['skipped_small_group_layers'],
            'ce_loss_avg': ce_loss_sum / average_denominator,
            'ft_reg_loss_avg': ft_reg_loss_sum / average_denominator,
            'ft_reg_loss_reg_batch_avg': reg_batch_ft_reg_loss_sum / reg_average_denominator,
            'mask_loss_avg': mask_loss_sum / average_denominator,
            'mask_loss_reg_batch_avg': reg_batch_mask_loss_sum / reg_average_denominator,
            'proto_loss_avg': proto_loss_sum / average_denominator,
            'proto_loss_reg_batch_avg': reg_batch_proto_loss_sum / reg_average_denominator,
            'balance_loss_avg': balance_loss_sum / average_denominator,
            'balance_loss_reg_batch_avg': reg_batch_balance_loss_sum / reg_average_denominator,
            'sep_loss_avg': sep_loss_sum / average_denominator,
            'sep_loss_reg_batch_avg': reg_batch_sep_loss_sum / reg_average_denominator,
            'epoch_time_sec': epoch_time_sec,
            'reg_time_sec': reg_time_sec,
            'refresh_time_sec': refresh_time_sec,
            'projection_time_sec': projection_time_sec,
        })
        _write_training_profile(training_profile_records, training_profile_path)

    final_projection_start_time = time.time()
    _apply_ft_group_projection(model, weight_name, mask, group_information)
    final_projection_time_sec = time.time() - final_projection_start_time
    if training_profile_records:
        training_profile_records[-1]['projection_time_sec'] = training_profile_records[-1].get('projection_time_sec', 0.0) + final_projection_time_sec
        _write_training_profile(training_profile_records, training_profile_path)

    time_now = time.time() - start_time
    print('Finished Training')
    print('Training complete in {:.0f}m {:.0f}s'.format(time_now // 60, time_now % 60))
    final_summary = summarize_group_information(group_information, weight_name, ft_layer_enabled)
    print('Final coverage: {:.4f}'.format(final_summary['coverage_ratio']))
    print('Final group_count: {}'.format(final_summary['group_count']))
    print('Final singleton_ratio: {:.4f}'.format(final_summary['singleton_ratio']))
    print('Final exact_group_proportion: {:.4f}'.format(final_summary['exact_group_proportion']))
    print('Final scaled_group_proportion: {:.4f}'.format(final_summary['scaled_group_proportion']))

    torch.save(model.state_dict(), _artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_after_translate_parameters.pth'))

    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_mask.pkl'), 'wb') as f_mask:
        pkl.dump(mask, f_mask, pkl.HIGHEST_PROTOCOL)
    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_map_information.pkl'), 'wb') as f_map:
        pkl.dump(map_information, f_map, pkl.HIGHEST_PROTOCOL)
    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_multiple_relationship_information.pkl'), 'wb') as f_mult:
        pkl.dump(multiple_relationship_information, f_mult, pkl.HIGHEST_PROTOCOL)
    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_coverage_ratio_information.pkl'), 'wb') as f_coverage:
        pkl.dump(reuse_ratio_information, f_coverage, pkl.HIGHEST_PROTOCOL)
    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_reuse_ratio_information.pkl'), 'wb') as f_reuse:
        pkl.dump(reuse_ratio_information, f_reuse, pkl.HIGHEST_PROTOCOL)
    with open(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_group_information.pkl'), 'wb') as f_group:
        pkl.dump(group_information, f_group, pkl.HIGHEST_PROTOCOL)
    write_mask_sweep_report(model_name, translate_name, group_information, output_dir=output_dir)

    result_all['Before_Translate_Accuracy'] = before_translate_accuracy
    result_all['Before_Translate_Loss'] = before_translate_loss
    result_all['After_Translate_Accuracy'] = after_translate_accuracy
    result_all['After_Translate_Loss'] = after_translate_loss
    result_all['Model_Difference'] = model_accuracy_difference
    result_all.to_csv(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '.csv'))

    result['Train_Accuracy'] = train_accuracy_record
    result['Train_Loss'] = train_loss_record
    result['Test_Accuracy'] = test_accuracy_record
    result['Test_Loss'] = test_loss_record
    result.to_csv(_artifact_output_path(output_dir, 'model_' + model_name + '_' + translate_name + '_train_info.csv'))

    parameters_to_txt(model, model_name, translate_name, output_dir=output_dir)


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
