#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
from Hardware_Model.Tile import tile
from Interface.interface import *


class TCG:
    def __init__(self, NetStruct, SimConfig_path, multiple=None):
        # NetStruct: layer structure, SimConfig_path: Hardware config path, multiple: allocate more resources for some layers
        TCG_config = cp.ConfigParser()
        TCG_config.read(SimConfig_path, encoding='UTF-8')
        self.OU_size = list(map(int, TCG_config.get('Crossbar level', 'OU_Size').split(',')))
        self.tile_num = list(map(int, TCG_config.get('Architecture level', 'Tile_Num').split(',')))
        if self.tile_num[0] == 0:
            self.tile_num[0] = 8
            self.tile_num[1] = 8
        assert self.tile_num[0] > 0, "Tile number < 0"
        assert self.tile_num[1] > 0, "Tile number < 0"
        self.tile_total_num = self.tile_num[0] * self.tile_num[1]

        self.tile = tile(SimConfig_path)
        self.net = NetStruct
        self.layer_num = len(self.net)
        self.layer_tileinfo = []
        self.mapping_order = -1 * np.ones(self.tile_num)  # 按一定顺序定义tilt映射矩阵每一个元素的顺序，如按蛇形、回型、Z型给每个元素标序号，每层权重按序号增加的顺序依次布置到每个tile上
        self.mapping_result = -1 * np.ones(self.tile_num)  # 记录最终每个tile存储的是第几层的权重，数值是层的序号

        start_tileid = 0  # the start tile id
        self.max_inbuf_size_pe = 0  # the maximum input buffer size of each PE, unit: KB
        self.max_outbuf_size_pe = 0  # the maximum output buffer size of each PE, unit: KB
        self.max_outbuf_size_tile = 0  # the maximum output buffer size of each tile, unit: KB
        self.max_outbuf_size_pe_index = 0  # the index information size of each PE, unit: KB

        self.multiple = multiple
        if multiple is None:
            self.multiple = [1] * len(NetStruct)
            for layer_id in range(self.layer_num):
                layer_dict = self.net[layer_id][0][0]
                self.multiple[layer_id] = layer_dict['Multiple']

        for layer_id in range(self.layer_num):
            layer_dict = self.net[layer_id][0][0]  # 第一维：层数   第二维：tile数   第三维：0为layer_info字典，1为权重矩阵
            layer_type = layer_dict['type']

            tmp_tileinfo = collections.OrderedDict()
            tmp_tileinfo['weight_precision'] = int(layer_dict['Weightbit'])
            tmp_tileinfo['startid'] = start_tileid

            data_inbuf_pe = 0
            data_outbuf_pe = 0
            data_outbuf_tile = 0

            if layer_type == 'conv':
                tmp_tileinfo['type'] = 'conv'
                """
                tmp_tileinfo['max_group'] = min(layer_dict['Crossbar_number'], self.tile.group_num)  # max_group: maximum used groups in one PE of this layer
                tmp_tileinfo['max_row'] = min(int(layer_dict['Inputchannel']) * (int(layer_dict['Kernelsize']) ** 2), self.tile.xbar_row)  # max_row: maximum used row in one crossbar of this layer
                tmp_tileinfo['max_column'] = min(int(layer_dict['Outputchannel'] * layer_dict['Weightbit']), self.tile.xbar_column)  # max_column: maximum used column in one crossbar of this layer
                """
                tmp_tileinfo['max_group'] = self.tile.PE_xbar_num
                tmp_tileinfo['max_row'] = layer_dict['max_row']
                tmp_tileinfo['max_column'] = layer_dict['max_column']
                tmp_tileinfo['PEnum'] = layer_dict['PE_number']

                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                # is_branchin: if this layer is the input layer of a branch
                tmp_tileinfo['is_branchout'] = 1
                # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    tmp_layer = self.net[i+layer_id][0][0]
                    if tmp_layer['type'] != 'element_sum':
                        tmp_tileinfo['is_branchout'] = -1

                input_size_list = list(map(int, layer_dict['Inputsize']))
                input_size = input_size_list[0] * input_size_list[1]
                inputchannel = int(layer_dict['Inputchannel'])
                output_size_list = list(map(int, layer_dict['Outputsize']))
                output_size = output_size_list[0] * output_size_list[1]
                outputchannel = int(layer_dict['Outputchannel'])

                # 计算PE层data_inbuf大小，Tile层data_outbuf大小 buffer_size: unit Byte
                data_inbuf_pe = input_size * inputchannel * int(layer_dict['Inputbit']) / 8
                data_outbuf_pe = math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num
                data_outbuf_tile = output_size * outputchannel * int(layer_dict['Outputbit']) / 8

            elif layer_type == 'fc':
                tmp_tileinfo['type'] = 'fc'
                """
                tmp_tileinfo['max_group'] = min(layer_dict['Crossbar_number'], self.tile.group_num)  # max_group: maximum used groups in one PE of this layer
                tmp_tileinfo['max_row'] = min(int(layer_dict['Infeature']), self.tile.xbar_row)  # max_row: maximum used row in one crossbar of this layer
                tmp_tileinfo['max_column'] = min(int(layer_dict['Outfeature']), self.tile.xbar_column)  # max_row: maximum used column in one crossbar of this layer
                """
                tmp_tileinfo['max_group'] = self.tile.PE_xbar_num
                tmp_tileinfo['max_row'] = layer_dict['max_row']
                tmp_tileinfo['max_column'] = layer_dict['max_column']
                tmp_tileinfo['PEnum'] = layer_dict['PE_number']

                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                tmp_tileinfo['is_branchout'] = 1
                # is_branchin: if this layer is the input layer of a branch
                # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    if (i+layer_id) < self.layer_num:
                        tmp_layer = self.net[i + layer_id][0][0]
                        if tmp_layer['type'] != 'element_sum':
                            tmp_tileinfo['is_branchout'] = -1

                # buffer_size: unit Byte
                data_inbuf_pe = int(layer_dict['Inputchannel']) * int(layer_dict['Inputbit']) / 8
                data_outbuf_pe = math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num
                data_outbuf_tile = int(layer_dict['Outputchannel']) * int(layer_dict['Outputbit']) / 8

            elif layer_type == 'pooling':
                tmp_tileinfo['type'] = 'pooling'
                tmp_tileinfo['max_group'] = 0
                tmp_tileinfo['max_row'] = 0
                tmp_tileinfo['max_column'] = 0
                tmp_tileinfo['PEnum'] = 0

                if 'Inputindex' not in layer_dict.keys():
                    tmp_tileinfo['Inputindex'] = [-1]
                else:
                    tmp_tileinfo['Inputindex'] = list(map(int, layer_dict['Inputindex']))
                # Inputindex: the relative index of the input layers of this layer
                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                # is_branchin: if this layer is the input layer of a branch
                tmp_tileinfo['is_branchout'] = 1
                # is_branchout: if this layer is the output layer of a branch (the next layer is element_sum)
                for i in tmp_tileinfo['Outputindex']:
                    tmp_layer = self.net[i + layer_id][0][0]
                    if tmp_layer['type'] != 'element_sum':
                        tmp_tileinfo['is_branchout'] = -1

                # assume the buffer size depends on the conv/fc layers
                data_inbuf_pe = 0
                data_outbuf_pe = 0
                data_outbuf_tile = 0

            elif layer_type == 'element_sum':
                tmp_tileinfo['type'] = 'element_sum'
                tmp_tileinfo['max_group'] = 0
                tmp_tileinfo['max_row'] = 0
                tmp_tileinfo['max_column'] = 0
                tmp_tileinfo['PEnum'] = 0

                if 'Outputindex' not in layer_dict.keys():
                    tmp_tileinfo['Outputindex'] = [1]
                else:
                    tmp_tileinfo['Outputindex'] = list(map(int, layer_dict['Outputindex']))
                # Outputindex: the relative index of the output layers of this layer
                if len(tmp_tileinfo['Outputindex']) == 1:
                    tmp_tileinfo['is_branchin'] = -1
                else:
                    tmp_tileinfo['is_branchin'] = 1
                # is_branchin: if this layer is the input layer of a branch
                tmp_tileinfo['is_branchout'] = -1
                Inputindex_list = list(map(int, layer_dict['Inputindex']))
                tmp_tileinfo['Inputindex'] = Inputindex_list
                assert len(Inputindex_list) > 1, "the number of element_sum's previous layers must > 1"

                previous_layer_dict = self.net[layer_id + Inputindex_list[0]][0][0]
                output_size_list = list(map(int, previous_layer_dict['Outputsize']))
                tmp_tileinfo['datanum_branchout'] = previous_layer_dict['Outputchannel'] * output_size_list[0] * output_size_list[1]  # the data number of each branch output
                tmp_tileinfo['bit_branchout'] = previous_layer_dict['Outputbit']  # the data precision of each branch output (bit)
                data_size = tmp_tileinfo['datanum_branchout'] * tmp_tileinfo['bit_branchout'] * len(Inputindex_list) / 8  # unit: Byte
                # buffer_size: unit Byte
                data_inbuf_pe = 0
                data_outbuf_pe = 0
                data_outbuf_tile = data_size

            tmp_tileinfo['tilenum'] = layer_dict['tile_number'] * (layer_dict['Multiple'] * layer_dict['Multiple'])
            tmp_tileinfo['max_PE'] = self.tile.tile_PE_total_num
            start_tileid += tmp_tileinfo['tilenum']
            self.layer_tileinfo.append(tmp_tileinfo)

            # unit: KB, restricted in 2^M KB
            if tmp_tileinfo['type'] == 'conv' or tmp_tileinfo['type'] == 'fc':
                data_inbuf_pe = data_inbuf_pe / (tmp_tileinfo['PEnum'] * layer_dict['Multiple'] * layer_dict['Multiple'])
                tmp_inbuf_size_pe = math.pow(2, math.ceil(math.log(data_inbuf_pe, 2))) / 1024

                data_outbuf_pe = data_outbuf_pe / (tmp_tileinfo['PEnum'] * layer_dict['Multiple'] * layer_dict['Multiple'])
                if layer_dict['reuse_ratio'] != 0:
                    if 'similar' in layer_dict['mode']:
                        data_outbuf_pe = data_outbuf_pe + 2 * math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num  # 暂存可以重用的部分和结果
                    else:
                        data_outbuf_pe = data_outbuf_pe + math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num  # 暂存可以重用的部分和结果
                tmp_outbuf_size_pe = math.pow(2, math.ceil(math.log(data_outbuf_pe, 2))) / 1024

                data_outbuf_tile = data_outbuf_tile / (layer_dict['Multiple'] * layer_dict['Multiple'])
                tmp_outbuf_size_tile = math.pow(2, math.ceil(math.log(data_outbuf_tile, 2))) / 1024

            else:
                tmp_inbuf_size_pe = 0
                tmp_outbuf_size_pe = 0
                tmp_outbuf_size_tile = 0

            if tmp_inbuf_size_pe > self.max_inbuf_size_pe:
                self.max_inbuf_size_pe = tmp_inbuf_size_pe
            if tmp_outbuf_size_pe > self.max_outbuf_size_pe:
                self.max_outbuf_size_pe = tmp_outbuf_size_pe
            if tmp_outbuf_size_tile > self.max_outbuf_size_tile:
                self.max_outbuf_size_tile = tmp_outbuf_size_tile

            data_outbuf_pe_index = 0
            if layer_dict['prune_ratio'] != 0:
                if 'shape' in layer_dict['mode']:
                    data_outbuf_pe_index = data_outbuf_pe_index + (math.ceil(tmp_tileinfo['max_row'] / self.OU_size[0])) * (4 + 1) * 8 * self.tile.PE_xbar_num  # 每个PE额外需要：OU行数 * (4个位置信息+1个数量信息) * weight_pattern种类个索引 * crossbar数
                if 'ORC' in layer_dict['mode']:
                    data_outbuf_pe_index = data_outbuf_pe_index + self.OU_size[0] * (math.ceil(tmp_tileinfo['max_row'] / self.OU_size[0])) * (math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1])) * self.tile.PE_xbar_num  # 每个OU额外需要8个索引
                if 'structure' in layer_dict['mode']:
                    data_outbuf_pe_index = data_outbuf_pe_index + self.OU_size[0] * (math.ceil(tmp_tileinfo['max_row'] / self.OU_size[0]))
            if layer_dict['reuse_ratio'] != 0:
                data_outbuf_pe_index = data_outbuf_pe_index + math.ceil(tmp_tileinfo['max_row'] / self.OU_size[0]) * math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num  # 每个部分和结果需要额外存储重用weight_pattern_id的索引
                if 'similar' in layer_dict['mode']:
                    data_outbuf_pe_index = data_outbuf_pe_index + math.ceil(tmp_tileinfo['max_row'] / self.OU_size[0]) * math.ceil(tmp_tileinfo['max_column'] / self.OU_size[1]) * self.tile.PE_xbar_num  # 每个部分和结果需要额外存储重用weight_pattern的放缩倍数
            if data_outbuf_pe_index != 0:
                tmp_outbuf_size_pe_index = math.pow(2, math.ceil(math.log(data_outbuf_pe_index, 2))) / 1024
            else:
                tmp_outbuf_size_pe_index = 0
            if tmp_outbuf_size_pe_index > self.max_outbuf_size_pe_index:
                self.max_outbuf_size_pe_index = tmp_outbuf_size_pe_index

        print('max_inbuf_size_pe', self.max_inbuf_size_pe)
        print('max_outbuf_size_pe', self.max_outbuf_size_pe)
        print('max_outbuf_size_tile', self.max_outbuf_size_tile)
        print('max_outbuf_size_pe_index', self.max_outbuf_size_pe_index)

        self.used_tile_num = start_tileid
        assert self.used_tile_num <= self.tile_total_num, "Tile number is not enough"


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    __TestInterface = TrainTestInterface('Vgg16', 'cifar10', test_SimConfig_path, 'naive')
    structure_file = __TestInterface.get_structure()
    test = TCG(structure_file, test_SimConfig_path)
