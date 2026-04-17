# -*-coding:utf-8-*-
import copy
import math
import collections
import numpy as np


class NetworkGraph:
    def __init__(self, hardware_config, layer_config_list, quantize_config_list, input_index_list, reuse_config_list, prune_config_list, input_params):
        super(NetworkGraph, self).__init__()
        # same length for layer_config_list , quantize_config_list and input_index_list
        assert len(layer_config_list) == len(quantize_config_list)
        assert len(layer_config_list) == len(input_index_list)

        self.hardware_config = copy.deepcopy(hardware_config)
        self.layer_config_list = copy.deepcopy(layer_config_list)
        self.quantize_config_list = copy.deepcopy(quantize_config_list)
        self.reuse_config_list = copy.deepcopy(reuse_config_list)
        self.prune_config_list = copy.deepcopy(prune_config_list)
        self.input_index_list = copy.deepcopy(input_index_list)
        self.input_params = copy.deepcopy(input_params)

        self.base_size = 8
        self.OU_size = [8, 8]

        self.net_info = []
        self.net_bit_weights = []

    def set_reuse_and_prune_ratio(self):
        layer_number = 0
        for i in range(0, len(self.layer_config_list)):
            if self.net_info[i]['type'] == 'conv' or self.net_info[i]['type'] == 'fc':
                self.net_info[i]['reuse_ratio'] = 1.0 - self.reuse_config_list[layer_number]
                self.net_info[i]['prune_ratio'] = 1.0 - self.prune_config_list[layer_number]
                layer_number = layer_number + 1
                print(str(layer_number) + ' reuse_ratio: ' + str(self.net_info[i]['reuse_ratio']) + ' prune_ratio: ' + str(self.net_info[i]['prune_ratio']))

    def get_weights(self):
        net_bit_weights = []
        for i in range(0, len(self.layer_config_list)):
            bit_weights = collections.OrderedDict()
            if self.net_info[i]['type'] == 'conv' or self.net_info[i]['type'] == 'fc':
                complete_bar_row = math.ceil(self.net_info[i]['OU_row_number'] / math.floor(self.hardware_config['xbar_size'] / self.OU_size[0]))
                complete_bar_column = math.ceil(self.net_info[i]['OU_column_number'] / math.floor(self.hardware_config['xbar_size'] / self.OU_size[1]))
                self.net_info[i]['Crossbar_number'] = complete_bar_row * complete_bar_column
                for j in range(0, self.net_info[i]['Crossbar_number']):
                    bit_weights[f'split{i}_weight{j}'] = np.ones((self.hardware_config['xbar_size'], self.hardware_config['xbar_size']))  # 暂且用1代表每个ceil已被使用，暂不考虑每个ceil具体的值
            net_bit_weights.append(bit_weights)

        self.net_bit_weights = net_bit_weights

    def get_structure(self):
        net_info = []
        input_size = [0] * len(self.layer_config_list)
        output_size = [0] * len(self.layer_config_list)
        for i in range(0, len(self.layer_config_list)):
            layer_info = collections.OrderedDict()
            layer_info['Multiple'] = 1
            layer_info['reuse_ratio'] = 0.0
            layer_info['prune_ratio'] = 0.0
            layer_info['OU_row_number'] = 0
            layer_info['OU_column_number'] = 0
            layer_info['Crossbar_number'] = 0
            layer_info['PE_number'] = 0

            if self.layer_config_list[i]['type'] == 'conv':
                layer_info['type'] = 'conv'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_channels']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_channels']
                layer_info['Kernelsize'] = self.layer_config_list[i]['kernel_size']
                layer_info['Stride'] = self.layer_config_list[i]['stride']
                layer_info['Padding'] = self.layer_config_list[i]['padding']

                layer_info['OU_row_number'] = math.ceil(self.layer_config_list[i]['in_channels'] * self.layer_config_list[i]['kernel_size'] * self.layer_config_list[i]['kernel_size'] / self.OU_size[0])
                layer_info['OU_column_number'] = math.ceil(self.layer_config_list[i]['out_channels'] * self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'] / self.OU_size[1])

                layer_info['reuse_ratio'] = self.layer_config_list[i]['reuse_ratio']
                layer_info['prune_ratio'] = self.layer_config_list[i]['prune_ratio']

                if i == 0:
                    input_size[i] = self.input_params['input_shape'][2]
                else:
                    input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = int((input_size[i] + 2 * self.layer_config_list[i]['padding'] - (self.layer_config_list[i]['kernel_size'] - 1)) / self.layer_config_list[i]['stride'])
                layer_info['Inputsize'] = [input_size[i], input_size[i]]
                layer_info['Outputsize'] = [output_size[i], output_size[i]]
                layer_info['Multiple'] = math.ceil(output_size[i] / self.base_size)

            elif self.layer_config_list[i]['type'] == 'fc':
                layer_info['type'] = 'fc'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_features']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_features']

                layer_info['OU_row_number'] = math.ceil(self.layer_config_list[i]['in_features'] / self.OU_size[0])
                layer_info['OU_column_number'] = math.ceil(self.layer_config_list[i]['out_features'] * self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'] / self.OU_size[1])

                layer_info['reuse_ratio'] = self.layer_config_list[i]['reuse_ratio']
                layer_info['prune_ratio'] = self.layer_config_list[i]['prune_ratio']

            elif self.layer_config_list[i]['type'] == 'pooling':
                layer_info['type'] = 'pooling'
                layer_info['Inputchannel'] = self.layer_config_list[i]['in_channels']
                layer_info['Outputchannel'] = self.layer_config_list[i]['out_channels']
                layer_info['Kernelsize'] = self.layer_config_list[i]['kernel_size']
                layer_info['Stride'] = self.layer_config_list[i]['stride']
                layer_info['Padding'] = self.layer_config_list[i]['padding']

                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = int(input_size[i] / self.layer_config_list[i]['stride'])
                layer_info['Inputsize'] = [input_size[i], input_size[i]]
                layer_info['Outputsize'] = [output_size[i], output_size[i]]
                layer_info['Multiple'] = math.ceil(output_size[i] / self.base_size)

            elif self.layer_config_list[i]['type'] == 'relu':
                layer_info['type'] = 'relu'
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            elif self.layer_config_list[i]['type'] == 'view':
                layer_info['type'] = 'view'

            elif self.layer_config_list[i]['type'] == 'bn':
                layer_info['type'] = 'bn'
                layer_info['features'] = self.layer_config_list[i]['features']
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            elif self.layer_config_list[i]['type'] == 'element_sum':
                layer_info['type'] = 'element_sum'
                input_size[i] = output_size[i + self.input_index_list[i][0]]
                output_size[i] = input_size[i]

            else:
                assert 0, f'not support {self.layer_config_list[i]["type"]}'

            layer_info['Inputbit'] = int(self.quantize_config_list[i]['activation_bit'])
            layer_info['Weightbit'] = int(self.quantize_config_list[i]['weight_bit'])
            if i != len(self.layer_config_list) - 1:
                layer_info['Outputbit'] = int(self.quantize_config_list[i+1]['activation_bit'])
            else:
                layer_info['Outputbit'] = int(self.quantize_config_list[i]['activation_bit'])
            layer_info['weight_cycle'] = math.ceil(self.quantize_config_list[i]['weight_bit'] / self.hardware_config['weight_bit'])
            if 'input_index' in self.layer_config_list[i]:
                layer_info['Inputindex'] = self.layer_config_list[i]['input_index']
            else:
                layer_info['Inputindex'] = [-1]
            layer_info['Outputindex'] = [1]

            net_info.append(layer_info)

        self.net_info = net_info


def get_net(hardware_config, cate, num_classes, mode):
    # initial config
    if hardware_config is None:
        hardware_config = {'xbar_size': 256, 'input_bit': 8, 'weight_bit': 8, 'quantize_bit': 8}
    layer_config_list = []
    quantize_config_list = []
    input_index_list = []
    reuse_config_list = []
    prune_config_list = []

    # layer by layer
    assert cate in ['Vgg16', 'Res18', 'Res50', 'WRN']
    if cate.startswith('Vgg16'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 2, 'stride': 2, 'padding': 0})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 2, 'stride': 2, 'padding': 0})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 2, 'stride': 2, 'padding': 0})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 2, 'padding': 0})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 2, 'stride': 2, 'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': 4096, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': 4096, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'fc', 'in_features': 4096, 'out_features': num_classes, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0]

        if 'structure' in mode:
            prune_config_list = [1.0, 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65, 0.65,
                                 0.25, 0.25, 1.0]

        if 'shape' in mode:
            prune_config_list = [8.0 / 9.0, 8.0 / 9.0, 8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0, 2.0 / 9.0,
                                 0.25, 0.25, 1.0]
        if 'ORC' in mode:
            prune_config_list = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.05, 0.05, 1.0]
        if 'value_identical' in mode:
            reuse_config_list = [1.0, 0.150146484375, 0.5103759765625, 0.447021484375, 0.390777587890625, 0.3853607177734375, 0.3890838623046875,
                                 0.2597808837890625, 0.3107032775878906, 0.24940109252929688, 0.14992523193359375, 0.16790390014648438, 0.20104217529296875,
                                 0.0591888427734375, 0.012379646301269531, 1.0]
        if 'value_similar' in mode:
            reuse_config_list = [1.0, 0.1875, 0.5, 0.48046875, 0.2421875, 0.2373046875, 0.2373046875,
                                 0.12255859375, 0.125, 0.12261962890625, 0.1234130859375, 0.12207412719726562, 0.1243133544921875,
                                 0.03125, 0.030131816864013672, 1.0]
        if 'shape_and_value_similar' in mode:
            reuse_config_list = [1.0, 0.1875, 0.5, 0.48046875, 0.2421875, 0.2373046875, 0.2373046875,
                                 0.12255859375, 0.125, 0.11902999877929688, 0.12331771850585938, 0.1225128173828125, 0.12428665161132812,
                                 0.02978515625, 0.029032230377197266, 1.0]

    elif cate.startswith('Res18'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 7
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 8
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})

        # output
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 4, 'stride': 4, 'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0]

        if 'structure' in mode:
            prune_config_list = [1.0, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65,
                                 1.0]

        if 'shape' in mode:
            prune_config_list = [8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 4.0 / 9.0, 1.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 4.0 / 9.0, 1.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 2.0 / 9.0, 1.0, 2.0 / 9.0, 2.0 / 9.0,
                                 1.0]
        if 'ORC' in mode:
            prune_config_list = [1.0, 0.15, 0.15, 0.15, 0.15,
                                 0.15, 0.15, 0.15, 0.15, 0.15,
                                 0.15, 0.15, 0.15, 0.15, 0.15,
                                 0.15, 0.15, 0.15, 0.15, 0.15,
                                 1.0]
        if 'value_identical' in mode:
            reuse_config_list = [1.0, 0.105224609375, 0.20849609375, 0.435302734375, 0.2978515625,
                                 0.2783203125, 0.35394287109375, 0.4306640625, 0.4722900390625, 0.25457763671875,
                                 0.23077392578125, 0.2772216796875, 0.30859375, 0.404876708984375, 0.2291259765625,
                                 0.23331451416015625, 0.31432342529296875, 0.292724609375, 0.24761199951171875, 0.042633056640625,
                                 1.0]
        if 'value_similar' in mode:
            reuse_config_list = [1.0, 0.18701171875, 0.38916015625, 0.5615234375, 0.6796875,
                                 0.375, 0.375, 0.375, 0.375, 0.357421875,
                                 0.1875, 0.1875, 0.1875, 0.1875, 0.16845703125,
                                 0.09375, 0.09375, 0.09375, 0.09375, 0.09375,
                                 1.0]
        if 'shape_and_value_similar' in mode:
            reuse_config_list = [1.0, 0.18701171875, 0.38916015625, 0.5615234375, 0.6796875,
                                 0.375, 0.375, 0.375, 0.375, 0.357421875,
                                 0.1875, 0.1875, 0.1875, 0.1875, 0.16845703125,
                                 0.09375, 0.09375, 0.09375, 0.09375, 0.09375,
                                 1.0]

    elif cate.startswith('Res50'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-6]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 64, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 64, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-6]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 7
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 8
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 1024, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-6]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 9
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 10
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 11
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 12
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 13
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 256, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 1024, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 14
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 2048, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 1024, 'out_channels': 2048, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-6]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 15
        layer_config_list.append({'type': 'conv', 'in_channels': 2048, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 2048, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})
        # block 16
        layer_config_list.append({'type': 'conv', 'in_channels': 2048, 'out_channels': 512, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 2048, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -6]})
        layer_config_list.append({'type': 'relu'})

        # output
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 4, 'stride': 4, 'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0,
                             1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0,
                             1.0]

        if 'structure' in mode:
            prune_config_list = [1.0, 0.75, 0.65, 0.75, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                                 0.75, 0.65, 0.75, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                                 0.75, 0.65, 0.75, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                                 0.75, 0.65, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                                 0.75, 0.65, 0.75, 0.75, 0.75, 0.65, 0.75, 0.75, 0.65, 0.75,
                                 0.75, 0.65, 0.75,
                                 1.0]

        if 'shape' in mode:
            prune_config_list = [8.0 / 9.0, 1.0, 8.0 / 9.0, 1.0, 1.0, 1.0, 8.0 / 9.0, 1.0, 1.0, 8.0 / 9.0, 1.0,
                                 1.0, 4.0 / 9.0, 1.0, 1.0, 0.5, 4.0 / 9.0, 1.0, 0.5, 4.0 / 9.0, 1.0,
                                 0.5, 4.0 / 9.0, 1.0, 1.0, 0.5, 4.0 / 9.0, 1.0, 0.5, 4.0 / 9.0, 1.0,
                                 0.5, 4.0 / 9.0, 1.0, 0.5, 4.0 / 9.0, 1.0, 0.5, 4.0 / 9.0, 1.0,
                                 0.5, 4.0 / 9.0, 1.0, 1.0, 0.5, 2.0 / 9.0, 0.5, 0.5, 2.0 / 9.0, 0.5,
                                 0.5, 2.0 / 9.0, 0.5,
                                 1.0]
        if 'ORC' in mode:
            prune_config_list = [1.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2,
                                 0.2, 0.2, 0.2,
                                 1.0]
        if 'value_identical' in mode:
            reuse_config_list = [1.0, 0.091796875, 0.06787109375, 0.48876953125, 0.326171875, 0.54931640625, 0.316650390625, 0.3701171875, 0.47021484375, 0.41064453125, 0.4609375,
                                 0.4189453125, 0.37225341796875, 0.3843994140625, 0.36981201171875, 0.2774658203125, 0.50848388671875, 0.4810791015625, 0.4544677734375, 0.3349609375, 0.32421875,
                                 0.40478515625, 0.4080810546875, 0.3065185546875, 0.282958984375, 0.2576904296875, 0.22119140625, 0.225677490234375, 0.28485107421875, 0.3317413330078125, 0.22735595703125,
                                 0.324127197265625, 0.353240966796875, 0.218597412109375, 0.336181640625, 0.375823974609375, 0.318206787109375, 0.348724365234375, 0.498382568359375, 0.460723876953125,
                                 0.361083984375, 0.4962158203125, 0.420654296875, 0.312225341796875, 0.2858467102050781, 0.38750457763671875, 0.40796661376953125, 0.1215057373046875, 0.2504692077636719, 0.356201171875,
                                 0.07648468017578125, 0.33281707763671875, 0.06191253662109375,
                                 1.0]
        if 'value_similar' in mode:
            reuse_config_list = [1.0, 0.091796875, 0.0615234375, 0.25, 0.25, 0.7001953125, 0.52099609375, 0.25, 0.72998046875, 0.53857421875, 0.25,
                                 0.5, 0.47265625, 0.125, 0.125, 0.5, 0.48785400390625, 0.125, 0.5, 0.3310546875, 0.125,
                                 0.5, 0.3839111328125, 0.125, 0.25, 0.25, 0.0625, 0.125, 0.25, 0.2490234375, 0.0625,
                                 0.25, 0.248046875, 0.0625, 0.25, 0.2431640625, 0.0625, 0.25, 0.2451171875, 0.0625,
                                 0.25, 0.244140625, 0.0625, 0.125, 0.12479782104492188, 0.03125, 0.03125, 0.12454986572265625, 0.12489700317382812, 0.03125,
                                 0.1241607666015625, 0.122528076171875, 0.03125,
                                 1.0]
        if 'shape_and_value_similar' in mode:
            reuse_config_list = [1.0, 0.091796875, 0.0615234375, 0.25, 0.25, 0.7001953125, 0.52099609375, 0.25, 0.72998046875, 0.53857421875, 0.25,
                                 0.5, 0.47265625, 0.125, 0.125, 0.5, 0.48785400390625, 0.125, 0.5, 0.3310546875, 0.125,
                                 0.5, 0.3839111328125, 0.125, 0.25, 0.25, 0.0625, 0.125, 0.25, 0.2490234375, 0.0625,
                                 0.25, 0.248046875, 0.0625, 0.25, 0.2431640625, 0.0625, 0.25, 0.2451171875, 0.0625,
                                 0.25, 0.244140625, 0.0625, 0.125, 0.12479782104492188, 0.03125, 0.03125, 0.12454986572265625, 0.12489700317382812, 0.03125,
                                 0.1241607666015625, 0.122528076171875, 0.03125,
                                 1.0]

    elif cate.startswith('WRN'):
        layer_config_list.append({'type': 'conv', 'in_channels': 3, 'out_channels': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        # block 1
        layer_config_list.append({'type': 'conv', 'in_channels': 16, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 16, 'out_channels': 128, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 2
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 3
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 128, 'out_channels': 256, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 4
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})
        # block 5
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'conv', 'in_channels': 256, 'out_channels': 512, 'kernel_size': 1, 'stride': 2, 'padding': 0, 'reuse_ratio': 0.0, 'prune_ratio': 0.0, 'input_index': [-4]})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -2]})
        layer_config_list.append({'type': 'relu'})
        # block 6
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'relu'})
        layer_config_list.append({'type': 'conv', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})
        layer_config_list.append({'type': 'element_sum', 'input_index': [-1, -4]})
        layer_config_list.append({'type': 'relu'})

        # output
        layer_config_list.append({'type': 'pooling', 'mode': 'MAX', 'in_channels': 512, 'out_channels': 512, 'kernel_size': 8, 'stride': 8, 'padding': 0})
        layer_config_list.append({'type': 'view'})
        layer_config_list.append({'type': 'fc', 'in_features': 512, 'out_features': num_classes, 'reuse_ratio': 0.0, 'prune_ratio': 0.0})

        prune_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0]
        reuse_config_list = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0, 1.0, 1.0, 1.0, 1.0,
                             1.0]

        if 'structure' in mode:
            prune_config_list = [1.0, 1.0, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65,
                                 0.65, 0.65, 0.65, 0.65, 0.65,
                                 1.0]

        if 'shape' in mode:
            prune_config_list = [8.0 / 9.0, 4.0 / 9.0, 4.0 / 9.0, 1.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 4.0 / 9.0, 1.0, 4.0 / 9.0, 4.0 / 9.0,
                                 4.0 / 9.0, 2.0 / 9.0, 1.0, 2.0 / 9.0, 2.0 / 9.0,
                                 1.0]
        if 'ORC' in mode:
            prune_config_list = [1.0, 0.15, 0.15, 0.15, 0.15, 0.15,
                                 0.15, 0.15, 0.15, 0.15, 0.15,
                                 0.15, 0.15, 0.15, 0.15, 0.15,
                                 1.0]
        if 'value_identical' in mode:
            reuse_config_list = [1.0, 0.095703125, 0.11102294921875, 0.6484375, 0.31524658203125, 0.29705810546875,
                                 0.377166748046875, 0.549285888671875, 0.569580078125, 0.3616180419921875, 0.1868743896484375,
                                 0.2427978515625, 0.26779937744140625, 0.271484375, 0.0947113037109375, 0.36454010009765625,
                                 1.0]
        if 'value_similar' in mode:
            reuse_config_list = [1.0, 0.09228515625, 0.10455322265625, 0.6484375, 0.31268310546875, 0.37164306640625,
                                 0.3251953125, 0.35302734375, 0.375, 0.28857421875, 0.17578125,
                                 0.1875, 0.1849365234375, 0.1875, 0.0509033203125, 0.16919326782226562,
                                 0.09375, 0.09375, 0.09375, 0.09375, 0.09375,
                                 1.0]
        if 'shape_and_value_similar' in mode:
            reuse_config_list = [1.0, 0.09228515625, 0.10455322265625, 0.6484375, 0.3111572265625, 0.37127685546875,
                                 0.3251953125, 0.35302734375, 0.375, 0.28857421875, 0.17578125,
                                 0.1875, 0.1849365234375, 0.1875, 0.0509033203125, 0.16904067993164062,
                                 0.09375, 0.09375, 0.09375, 0.09375, 0.09375,
                                 1.0]

    else:
        assert 0, f'not support {cate}'

    for i in range(len(layer_config_list)):
        quantize_config_list.append({'weight_bit': 8, 'activation_bit': 8})
        if 'input_index' in layer_config_list[i]:
            input_index_list.append(layer_config_list[i]['input_index'])
        else:
            input_index_list.append([-1])

    input_params = {'activation_bit': 8, 'input_shape': (1, 3, 32, 32)}

    # add bn for every conv
    L = len(layer_config_list)
    for i in range(L-1, -1, -1):
        if layer_config_list[i]['type'] == 'conv':
            layer_config_list.insert(i+1, {'type': 'bn', 'features': layer_config_list[i]['out_channels']})
            quantize_config_list.insert(i+1, {'weight_bit': 8, 'activation_bit': 8})
            input_index_list.insert(i+1, [-1])
            # conv层后面加了一层bn，所以这一层相对他后面的层的相对层数差要加一
            for j in range(i + 2, len(layer_config_list), 1):
                for relative_input_index in range(len(input_index_list[j])):
                    if j + input_index_list[j][relative_input_index] < i + 1:
                        input_index_list[j][relative_input_index] -= 1

    print(layer_config_list)
    print(quantize_config_list)
    print(input_index_list)

    # generate net
    net = NetworkGraph(hardware_config, layer_config_list, quantize_config_list, input_index_list, reuse_config_list, prune_config_list, input_params)

    return net


if __name__ == '__main__':
    hardware_config = {'xbar_size': 128, 'input_bit': 8, 'weight_bit': 8, 'quantize_bit': 8}
    net = get_net(hardware_config, 'Vgg16', 10, 'naive')
    net.get_structure()
    net.get_weights()
    print(net.net_info)
