import math
import time
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
