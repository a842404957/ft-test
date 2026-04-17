# -*-coding:utf-8-*-
import math
import os
import copy
import torch
import collections
import configparser
import numpy as np
from importlib import import_module


class TrainTestInterface(object):
    def __init__(self, network_module, dataset_module, SimConfig_path, mode):
        # load net, dataset, and weights
        self.network_module = network_module
        self.dataset_module = dataset_module
        self.mode = mode
        self.test_loader = None

        # load simconfig
        xbar_config = configparser.ConfigParser()
        xbar_config.read(SimConfig_path, encoding='UTF-8')
        self.hardware_config = collections.OrderedDict()
        xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.xbar_row = xbar_size[0]
        self.xbar_column = xbar_size[1]
        self.hardware_config['xbar_size'] = xbar_size[0]
        self.hardware_config['weight_bit'] = 1

        # group num
        self.PE_xbar_num = int(xbar_config.get('Process element level', 'Group_Num'))
        self.tile_size = list(map(int, xbar_config.get('Tile level', 'PE_Num').split(',')))
        self.tile_row = self.tile_size[0]
        self.tile_column = self.tile_size[1]

        self.device = torch.device('cpu')
        print(f'run on device {self.device}')

        if dataset_module.endswith('cifar10'):
            num_classes = 10
        elif dataset_module.endswith('cifar100'):
            num_classes = 100
        else:
            assert 0, f'unknown dataset'

        self.net = import_module('Interface.network').get_net(self.hardware_config, self.network_module, num_classes, mode)

    def get_structure(self):
        self.net.get_structure()
        self.net.set_reuse_and_prune_ratio()
        self.net.get_weights()
        net_structure_info = self.net.net_info
        net_bit_weights = self.net.net_bit_weights
        assert len(net_bit_weights) == len(net_structure_info)
        for i in range(0, len(net_structure_info)):
            print(net_structure_info[i])

        # set relative index to absolute index
        absolute_index = [0] * len(net_structure_info)
        absolute_count = 0
        for i in range(len(net_structure_info)):
            if not (len(net_structure_info[i]['Outputindex']) == 1 and net_structure_info[i]['Outputindex'][0] == 1):
                raise Exception('duplicate output')
            if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'fc']:
                absolute_index[i] = absolute_count
                absolute_count = absolute_count + 1
            else:
                if not len(net_structure_info[i]['Inputindex']) == 1:
                    raise Exception('duplicate input index')
                absolute_index[i] = absolute_index[i + net_structure_info[i]['Inputindex'][0]]

        graph = list()
        for i in range(len(net_structure_info)):
            if net_structure_info[i]['type'] in ['conv', 'pooling', 'element_sum', 'fc']:
                # layer num, layer type
                layer_num = absolute_index[i]
                layer_type = net_structure_info[i]['type']
                # layer input
                layer_input = list(map(lambda x: (absolute_index[i + x] if i + x != -1 else -1), net_structure_info[i]['Inputindex']))
                # layer output
                layer_output = list()
                for tmp_i in range(len(net_structure_info)):
                    if net_structure_info[tmp_i]['type'] in ['conv', 'pooling', 'element_sum', 'fc']:
                        tmp_layer_num = absolute_index[tmp_i]
                        tmp_layer_input = list(map(lambda x: (absolute_index[tmp_i + x] if tmp_i + x != -1 else -1), net_structure_info[tmp_i]['Inputindex']))
                        if layer_num in tmp_layer_input:
                            layer_output.append(tmp_layer_num)
                graph.append((layer_num, layer_type, layer_input, layer_output))
        for i in range(0, len(graph)):
            print(graph[i])

        # add to net array
        net_array = []
        for layer_num, (layer_bit_weights, layer_structure_info) in enumerate(zip(net_bit_weights, net_structure_info)):
            # change layer structure info
            layer_structure_info = copy.deepcopy(layer_structure_info)
            layer_count = absolute_index[layer_num]
            layer_structure_info['Layerindex'] = graph[layer_count][0]
            layer_structure_info['Inputindex'] = list(map(lambda x: x - graph[layer_count][0], graph[layer_count][2]))
            layer_structure_info['Outputindex'] = list(map(lambda x: x - graph[layer_count][0], graph[layer_count][3]))
            layer_structure_info['tile_number'] = 1
            layer_structure_info['mode'] = self.mode

            # add for element_sum and pooling
            if layer_structure_info['type'] in ['element_sum', 'pooling']:
                layer_structure_info['max_row'] = 0
                layer_structure_info['max_column'] = 0
                net_array.append([(layer_structure_info, None)])
                continue
            if layer_structure_info['type'] in ['bn', 'relu', 'view']:
                continue
            assert len(layer_bit_weights.keys()) == layer_structure_info['Crossbar_number']

            # generate pe array
            xbar_array = []
            tile_crossbar_number = self.PE_xbar_num * self.tile_row * self.tile_column
            if layer_structure_info['type'] == 'conv':
                complete_bar_row = layer_structure_info['OU_row_number'] / math.floor(self.hardware_config['xbar_size'] / self.net.OU_size[0])
                complete_bar_column = layer_structure_info['OU_column_number'] / math.floor(self.hardware_config['xbar_size'] / self.net.OU_size[1])
                tile_number_row = math.ceil(math.ceil(complete_bar_row) / (self.tile_row * self.tile_column))
                tile_number_column = math.ceil(math.ceil(complete_bar_column) / self.PE_xbar_num)
                layer_structure_info['tile_number'] = tile_number_row * tile_number_column
                layer_structure_info['max_row'] = math.ceil(min(layer_structure_info['Inputchannel'] * (layer_structure_info['Kernelsize'] ** 2), self.xbar_row) * (complete_bar_row / (tile_number_row * self.tile_row * self.tile_column)) / self.net.OU_size[0]) * int(self.net.OU_size[0])
                layer_structure_info['max_column'] = math.ceil(min(layer_structure_info['Outputchannel'] * layer_structure_info['Weightbit'], self.xbar_column) * (complete_bar_column / (tile_number_column * self.PE_xbar_num)) / self.net.OU_size[1]) * int(self.net.OU_size[1])
            if layer_structure_info['type'] == 'fc':
                layer_structure_info['tile_number'] = math.ceil(layer_structure_info['Crossbar_number'] / tile_crossbar_number)
                layer_structure_info['max_row'] = math.ceil(min(layer_structure_info['Inputchannel'], self.xbar_row) / self.net.OU_size[0]) * int(self.net.OU_size[0])
                layer_structure_info['max_column'] = math.ceil(min(layer_structure_info['Outputchannel'] * layer_structure_info['Weightbit'], self.xbar_column) / self.net.OU_size[1]) * int(self.net.OU_size[1])
            max_crossbar_number = layer_structure_info['tile_number'] * tile_crossbar_number
            layer_structure_info['Crossbar_number'] = max_crossbar_number
            assert layer_structure_info['max_row'] <= self.xbar_row
            assert layer_structure_info['max_column'] <= self.xbar_column
            for i in range(0, max_crossbar_number, self.PE_xbar_num):
                pe_array = []
                for j in range(0, self.PE_xbar_num):
                    pe_array.append(np.ones((layer_structure_info['max_row'], layer_structure_info['max_column'])))  # 暂且用1代表每个ceil已被使用，暂不考虑每个ceil具体的值
                xbar_array.append(pe_array)
            self.net.net_info[layer_num]['PE_number'] = len(xbar_array)
            layer_structure_info['PE_number'] = len(xbar_array)

            # store in xbar_array
            total_array = []
            for i in range(layer_structure_info['tile_number']):
                tile_array = []
                for h in range(self.tile_row):
                    for w in range(self.tile_column):
                        serial_number = i * self.tile_row * self.tile_column + h * self.tile_column + w
                        tile_array.append(xbar_array[serial_number])
                for j in range(0, int(math.pow(layer_structure_info['Multiple'], 2))):
                    total_array.append((layer_structure_info, tile_array))
            net_array.append(total_array)

        return net_array


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    __TestInterface = TrainTestInterface('Vgg16', 'cifar10', test_SimConfig_path, 'naive')
    structure_info = __TestInterface.get_structure()
