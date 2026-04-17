import pandas as pd
import configparser as cp
from Interface.interface import *
from Mapping_Model.Tile_connection_graph import TCG
from Latency_Model.Tile_latency import tile_latency_analysis
from Latency_Model.Pooling_latency import pooling_latency_analysis


def merge_interval(interval):
    if len(interval) == 0:
        return []
    result = []
    interval.sort()
    lower_bound = interval[0][0]
    upper_bound = interval[0][1]
    for index in range(1, len(interval)):
        if interval[index][0] > upper_bound:
            result.append([lower_bound, upper_bound])
            lower_bound = interval[index][0]
            upper_bound = interval[index][1]
        else:
            if interval[index][1] > upper_bound:
                upper_bound = interval[index][1]
    result.append([lower_bound, upper_bound])
    return result


class Model_latency:
    def __init__(self, NetStruct, SimConfig_path, multiple, TCG_mapping):
        self.SimConfig_path = SimConfig_path
        model_config = cp.ConfigParser()
        model_config.read(SimConfig_path, encoding='UTF-8')
        self.inter_tile_bandwidth = float(model_config.get('Tile level', 'Inter_Tile_Bandwidth'))  # 20Gps
        self.OU_size = list(map(int, model_config.get('Crossbar level', 'OU_Size').split(',')))

        self.NetStruct = NetStruct
        self.multiple = multiple
        self.graph = TCG_mapping

        self.begin_time = []
        self.finish_time = []
        self.layer_tile_latency = []
        self.occupancy = []

        self.buffer_r_latency = []
        self.buffer_w_latency = []
        self.inbuffer_latency = []
        self.outbuffer_latency = []
        self.buffer_latency = []
        self.DAC_latency = []
        self.xbar_latency = []
        self.ADC_latency = []
        self.iReg_latency = []
        self.oReg_latency = []
        self.input_demux_latency = []
        self.output_mux_latency = []
        self.shiftreg_latency = []
        self.adder_latency = []
        self.jointmodule_latency = []
        self.pooling_latency = []
        self.digital_latency = []

        self.computing_latency = []
        self.compute_interval = []
        self.intra_tile_latency = []
        self.inter_tile_latency = []
        self.tile_merge_latency = []
        self.tile_transfer_latency = []

        self.total_buffer_r_latency = []
        self.total_buffer_w_latency = []
        self.total_buffer_latency = []
        self.total_DAC_latency = []
        self.total_xbar_latency = []
        self.total_ADC_latency = []
        self.total_iReg_latency = []
        self.total_oReg_latency = []
        self.total_input_demux_latency = []
        self.total_output_mux_latency = []
        self.total_shiftreg_latency = []
        self.total_adder_latency = []
        self.total_jointmodule_latency = []
        self.total_pooling_latency = []
        self.total_digital_latency = []

        self.total_computing_latency = []
        self.total_intra_tile_latency = []
        self.total_inter_tile_latency = []
        self.total_tile_merge_latency = []
        self.total_tile_transfer_latency = []

        self.layer_type = []
        self.layer_split = []
        self.pre_max_time = 0
    
    def layer_latency_initial(self):
        self.begin_time.append([])
        self.finish_time.append([])

        self.buffer_r_latency.append([])
        self.buffer_w_latency.append([])
        self.inbuffer_latency.append([])
        self.outbuffer_latency.append([])
        self.buffer_latency.append([])
        self.DAC_latency.append([])
        self.xbar_latency.append([])
        self.ADC_latency.append([])
        self.iReg_latency.append([])
        self.oReg_latency.append([])
        self.input_demux_latency.append([])
        self.output_mux_latency.append([])
        self.shiftreg_latency.append([])
        self.adder_latency.append([])
        self.jointmodule_latency.append([])
        self.pooling_latency.append([])
        self.digital_latency.append([])

        self.compute_interval.append([])
        self.computing_latency.append([])
        self.intra_tile_latency.append([])
        self.inter_tile_latency.append([])
        self.tile_merge_latency.append([])
        self.tile_transfer_latency.append([])

    def pipe_result_update(self, layer_type='conv', begin_time=0, compute_time=0, layer_id=0, temp_tile_latency=None, temp_pooling_latency=None, merge_time=0, transfer_time=0, output_size=0):
        if layer_type == 'conv':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.tile_buf_rlatency + temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency + temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency + temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency + temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'fc':
            self.begin_time[layer_id] = output_size * [begin_time]
            self.finish_time[layer_id] = output_size * [compute_time]
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.tile_buf_wlatency + temp_tile_latency.PE_buf_rlatency + temp_tile_latency.PE_buf_wlatency)
            self.computing_latency[layer_id].append(temp_tile_latency.computing_latency)
            self.DAC_latency[layer_id].append(temp_tile_latency.DAC_latency)
            self.xbar_latency[layer_id].append(temp_tile_latency.xbar_latency)
            self.ADC_latency[layer_id].append(temp_tile_latency.ADC_latency)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency+temp_tile_latency.PE_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency+temp_tile_latency.PE_buf_wlatency)
            self.iReg_latency[layer_id].append(temp_tile_latency.iReg_latency)
            self.input_demux_latency[layer_id].append(temp_tile_latency.input_demux_latency)
            self.output_mux_latency[layer_id].append(temp_tile_latency.output_mux_latency)
            self.shiftreg_latency[layer_id].append(temp_tile_latency.shiftreg_latency)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(temp_tile_latency.oReg_latency)
            self.jointmodule_latency[layer_id].append(temp_tile_latency.jointmodule_latency)
            self.digital_latency[layer_id].append(temp_tile_latency.iReg_latency + temp_tile_latency.input_demux_latency + temp_tile_latency.output_mux_latency + temp_tile_latency.shiftreg_latency + temp_tile_latency.adder_latency + temp_tile_latency.oReg_latency + temp_tile_latency.jointmodule_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(temp_tile_latency.transfer_latency)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'pooling':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.inbuf_rlatency + temp_pooling_latency.outbuf_wlatency + temp_pooling_latency.outbuf_rlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(temp_pooling_latency.inbuf_rlatency + temp_pooling_latency.outbuf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_pooling_latency.inbuf_wlatency + temp_pooling_latency.outbuf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(0)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)
            self.digital_latency[layer_id].append(0)
            self.pooling_latency[layer_id].append(temp_pooling_latency.digital_latency)

            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

        elif layer_type == 'element_sum':
            self.begin_time[layer_id].append(begin_time)
            self.finish_time[layer_id].append(compute_time)
            self.compute_interval[layer_id].append([begin_time, compute_time])

            self.buffer_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency + temp_tile_latency.tile_buf_wlatency)
            self.computing_latency[layer_id].append(0)
            self.DAC_latency[layer_id].append(0)
            self.xbar_latency[layer_id].append(0)
            self.ADC_latency[layer_id].append(0)
            self.buffer_r_latency[layer_id].append(temp_tile_latency.tile_buf_rlatency)
            self.buffer_w_latency[layer_id].append(temp_tile_latency.tile_buf_wlatency)
            self.iReg_latency[layer_id].append(0)
            self.input_demux_latency[layer_id].append(0)
            self.output_mux_latency[layer_id].append(0)
            self.shiftreg_latency[layer_id].append(0)
            self.adder_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.oReg_latency[layer_id].append(0)
            self.jointmodule_latency[layer_id].append(0)
            self.digital_latency[layer_id].append(temp_tile_latency.adder_latency)
            self.pooling_latency[layer_id].append(0)

            self.intra_tile_latency[layer_id].append(0)
            self.inter_tile_latency[layer_id].append(merge_time + transfer_time)
            self.tile_merge_latency[layer_id].append(merge_time)
            self.tile_transfer_latency[layer_id].append(transfer_time)

    def model_latency_output(self, module_information=1, layer_information=1):
        print(' ')
        total_latency = [0] * len(self.begin_time)
        if layer_information:
            for i in range(len(self.begin_time)):
                print("Layer", i, " type:", self.NetStruct[i][0][0]['type'])
                total_latency[i] = self.total_buffer_latency[i] + self.total_computing_latency[i] + self.total_digital_latency[i] + self.total_intra_tile_latency[i] + self.total_inter_tile_latency[i] + self.total_pooling_latency[i]
                if module_information:
                    print("Total latency of layer", i, ":", total_latency[i])
                    print("Buffer latency of layer", i, ":", self.total_buffer_latency[i], '(', "%.2f" % (100 * self.total_buffer_latency[i] / total_latency[i]), '%)')
                    print("     read buffer latency of layer", i, ":", self.total_buffer_r_latency[i], '(', "%.2f" % (100 * self.total_buffer_r_latency[i] / total_latency[i]), '%)')
                    print("     write buffer latency of layer", i, ":", self.total_buffer_w_latency[i], '(', "%.2f" % (100 * self.total_buffer_w_latency[i] / total_latency[i]), '%)')
                    print("Computing latency of layer", i, ":", self.total_computing_latency[i], '(', "%.2f" % (100 * self.total_computing_latency[i] / total_latency[i]), '%)')
                    print("     DAC latency of layer", i, ":", self.total_DAC_latency[i], '(', "%.2f" % (100 * self.total_DAC_latency[i] / total_latency[i]), '%)')
                    print("     ADC latency of layer", i, ":", self.total_ADC_latency[i], '(', "%.2f" % (100 * self.total_ADC_latency[i] / total_latency[i]), '%)')
                    print("     xbar latency of layer", i, ":", self.total_xbar_latency[i], '(', "%.2f" % (100 * self.total_xbar_latency[i] / total_latency[i]), '%)')
                    print("Digital part latency of layer", i, ":", self.total_digital_latency[i], '(', "%.2f" % (100 * self.total_digital_latency[i] / total_latency[i]), '%)')
                    print("     iReg latency of layer", i, ":", self.total_iReg_latency[i], '(', "%.2f" % (100 * self.total_iReg_latency[i] / total_latency[i]), '%)')
                    print("     oReg latency of layer", i, ":", self.total_oReg_latency[i], '(', "%.2f" % (100 * self.total_oReg_latency[i] / total_latency[i]), '%)')
                    print("     input demux latency of layer", i, ":", self.total_input_demux_latency[i], '(', "%.2f" % (100 * self.total_input_demux_latency[i] / total_latency[i]), '%)')
                    print("     output mux latency of layer", i, ":", self.total_output_mux_latency[i], '(', "%.2f" % (100 * self.total_output_mux_latency[i] / total_latency[i]), '%)')
                    print("     shiftreg latency of layer", i, ":", self.total_shiftreg_latency[i], '(', "%.2f" % (100 * self.total_shiftreg_latency[i] / total_latency[i]), '%)')
                    print("     adder latency of layer", i, ":", self.total_adder_latency[i], '(', "%.2f" % (100 * self.total_adder_latency[i] / total_latency[i]), '%)')
                    print("     Jointmodule latency of layer", i, ":", self.total_jointmodule_latency[i], '(', "%.2f" % (100 * self.total_jointmodule_latency[i] / total_latency[i]), '%)')
                    print("Pooling module latency of layer", i, ":", self.total_pooling_latency[i], '(', "%.2f" % (100 * self.total_pooling_latency[i] / total_latency[i]), '%)')
                    print("Intra tile communication latency of layer", i, ":", self.total_intra_tile_latency[i], '(', "%.2f" % (100 * self.total_intra_tile_latency[i] / total_latency[i]), '%)')
                    print("Inter tile communication latency of layer", i, ":", self.total_inter_tile_latency[i], '(', "%.2f" % (100 * self.total_inter_tile_latency[i] / total_latency[i]), '%)')
                    print("     One layer merge latency of layer", i, ":", self.total_tile_merge_latency[i], '(', "%.2f" % (100 * self.total_tile_merge_latency[i] / total_latency[i]), '%)')
                    print("     Inter tile transfer latency of layer", i, ":", self.total_tile_transfer_latency[i], '(', "%.2f" % (100 * self.total_tile_transfer_latency[i] / total_latency[i]), '%)')
                print('----------------------------------------------')
        print("Entire latency:", str(sum(total_latency)), "ns")

        return sum(total_latency)


    def calculate_model_latency(self, mode):
        # fill in input data kernel size by kernel size (column direction)
        for layer_id in range(len(self.NetStruct)):
            layer_dict = self.NetStruct[layer_id][0][0]
            if layer_id == 0:
                self.layer_latency_initial()
                output_size = list(map(int, layer_dict['Outputsize']))
                kernelsize = int(layer_dict['Kernelsize'])
                stride = int(layer_dict['Stride'])
                outputchannel = int(layer_dict['Outputchannel'])
                padding = int(layer_dict['Padding'])
                inputbit = int(layer_dict['Inputbit'])
                outputbit = int(layer_dict['Outputbit'])

                read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path, max_row=self.graph.layer_tileinfo[layer_id]['max_row'], max_column=self.graph.layer_tileinfo[layer_id]['max_column'], inprecision=inputbit, PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'], default_outbuf_size_tile=self.graph.max_outbuf_size_tile, default_inbuf_size_pe=self.graph.max_inbuf_size_pe, default_outbuf_size_pe=self.graph.max_outbuf_size_pe)
                temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=(math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / 8))
                temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency
                merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                max_time = 0
                for i in range(int(output_size[0] / layer_dict['Multiple'])):
                    for j in range(int(output_size[1] / layer_dict['Multiple'])):
                        self.pre_max_time = max_time
                        if (i == 0) & (j == 0):  # the first output
                            indata = (max(kernelsize - padding, 0)**2) * inputbit / 8  # fill the line buffer
                            rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                            wdata = outputbit / 8
                            temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                        elif j == 0:
                            indata = stride * max(kernelsize-padding, 0) * inputbit / 8
                            rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                            wdata = outputbit / 8
                            temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                        else:
                            if i == 0:
                                indata = stride * kernelsize * inputbit / 8
                            else:
                                indata = stride ** 2 * inputbit / 8
                            rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                            wdata = outputbit / 8
                            temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                        begin_time = self.pre_max_time
                        compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                        self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                        max_time = compute_time

            else:
                if layer_dict['type'] == 'conv':
                    self.layer_latency_initial()
                    output_size = list(map(int, layer_dict['Outputsize']))
                    kernelsize = int(layer_dict['Kernelsize'])
                    stride = int(layer_dict['Stride'])
                    outputchannel = int(layer_dict['Outputchannel'])
                    padding = int(layer_dict['Padding'])
                    inputbit = int(layer_dict['Inputbit'])
                    outputbit = int(layer_dict['Outputbit'])

                    # the input channel number each PE processes
                    read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                    OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                    temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path, max_row=self.graph.layer_tileinfo[layer_id]['max_row'], max_column=self.graph.layer_tileinfo[layer_id]['max_column'], inprecision=inputbit, PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'], default_outbuf_size_tile=self.graph.max_outbuf_size_tile, default_inbuf_size_pe=self.graph.max_inbuf_size_pe, default_outbuf_size_pe=self.graph.max_outbuf_size_pe)
                    temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=(math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / 8))
                    temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency
                    merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                    transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                    max_time = 0
                    for i in range(int(output_size[0] / layer_dict['Multiple'])):
                        for j in range(int(output_size[1] / layer_dict['Multiple'])):
                            self.pre_max_time = max_time
                            if (i == 0) & (j == 0):  # the first output
                                indata = (max(kernelsize - padding, 0) ** 2) * inputbit / 8
                                rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                                wdata = outputbit / 8
                                temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                            elif j == 0:
                                indata = stride * max(kernelsize - padding, 0) * inputbit / 8
                                rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                                wdata = outputbit / 8
                                temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                            else:
                                if i == 0:
                                    indata = stride * kernelsize * inputbit / 8
                                else:
                                    indata = stride**2 * inputbit / 8
                                rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                                wdata = outputbit / 8
                                temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                            temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                            max_prelayer_time = 0  # the maximum time of the required input data (in all input layers)
                            for idx in temp_Inputindex:
                                tmp_time = self.finish_time[layer_id + idx][-1]
                                if tmp_time > max_prelayer_time:
                                    max_prelayer_time = tmp_time
                            begin_time = max(max_prelayer_time, self.pre_max_time)
                            OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / temp_tile_latency.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                            compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                            self.pipe_result_update(layer_type='conv', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)
                            max_time = compute_time

                else:
                    if layer_dict['type'] == 'fc':
                        self.layer_latency_initial()
                        output_size = int(layer_dict['Outputchannel'])
                        input_size = int(layer_dict['Inputchannel'])
                        self.layer_split.append([input_size])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['Outputbit'])

                        read_column = math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] * (1 - layer_dict['reuse_ratio']))
                        OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / self.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path, max_row=self.graph.layer_tileinfo[layer_id]['max_row'], max_column=self.graph.layer_tileinfo[layer_id]['max_column'], inprecision=inputbit, PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'], default_outbuf_size_tile=self.graph.max_outbuf_size_tile, default_inbuf_size_pe=self.graph.max_inbuf_size_pe, default_outbuf_size_pe=self.graph.max_outbuf_size_pe)
                        temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=(math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / 8))
                        temp_tile_latency.tile_buf_rlatency = temp_tile_latency.outbuf_tile.buf_rlatency
                        merge_time = temp_tile_latency.tile_buf_rlatency + temp_tile_latency.digital_period + math.ceil(self.graph.layer_tileinfo[layer_id]['max_column'] / temp_tile_latency.OU_size[1]) * self.graph.layer_tileinfo[layer_id]['max_PE'] * outputbit / self.inter_tile_bandwidth
                        transfer_time = int(output_size * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                        indata = self.graph.layer_tileinfo[layer_id]['max_row'] * inputbit / 8
                        rdata = temp_tile_latency.OU_size[0] * inputbit / 8  # OU-based Crossbar
                        wdata = outputbit / 8
                        temp_tile_latency.update_tile_latency(OU_row, read_column, indata, rdata, wdata, mode)

                        temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                        max_prelayer_time = 0
                        for idx in temp_Inputindex:
                            tmp_time = self.finish_time[layer_id+idx][-1]
                            if tmp_time > max_prelayer_time:
                                max_prelayer_time = tmp_time
                        begin_time = max_prelayer_time
                        OU_row = math.ceil(int(self.graph.layer_tileinfo[layer_id]['max_row'] / temp_tile_latency.OU_size[0]) * (1 - layer_dict['prune_ratio']))
                        compute_time = temp_tile_latency.tile_latency + merge_time * OU_row + transfer_time + begin_time
                        self.pipe_result_update(layer_type='fc', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time, output_size=output_size)

                    elif layer_dict['type'] == 'pooling':
                        self.layer_latency_initial()
                        output_size = list(map(int, layer_dict['Outputsize']))
                        input_size = list(map(int, layer_dict['Inputsize']))
                        self.layer_split.append([input_size[1]])
                        kernelsize = int(layer_dict['Kernelsize'])
                        stride = int(layer_dict['Stride'])
                        inputchannel = int(layer_dict['Inputchannel'])
                        outputchannel = int(layer_dict['Outputchannel'])
                        padding = int(layer_dict['Padding'])
                        inputbit = int(layer_dict['Inputbit'])
                        outputbit = int(layer_dict['Outputbit'])

                        temp_pooling_latency = pooling_latency_analysis(indata=0, rdata=0, outprecision=outputbit, default_inbuf_size=self.graph.max_inbuf_size_pe, default_outbuf_size=self.graph.max_outbuf_size_tile, default_inchannel=inputchannel)
                        temp_pooling_latency.outbuf.calculate_buf_read_latency(rdata=(outputchannel * outputbit / 8))
                        temp_pooling_latency.outbuf_rlatency = temp_pooling_latency.outbuf.buf_rlatency
                        merge_time = temp_pooling_latency.outbuf_rlatency
                        transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth / self.graph.layer_tileinfo[layer_id]['tilenum'])

                        self.pre_max_time = 0
                        for i in range(int(output_size[0] / layer_dict['Multiple'])):
                            for j in range(int(output_size[1] / layer_dict['Multiple'])):
                                if (i == 0) & (j == 0):
                                    indata = inputchannel * (max(kernelsize-padding, 0)**2) * inputbit / 8
                                    rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)

                                elif j == 0:
                                    indata = inputchannel * stride * max(kernelsize - padding, 0) * inputbit / 8
                                    rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)

                                else:
                                    indata = inputchannel * stride ** 2 * inputbit / 8
                                    rdata = inputchannel * kernelsize ** 2 * inputbit / 8
                                    temp_pooling_latency.update_pooling_latency(indata=indata, rdata=rdata)

                                temp_Inputindex = self.graph.layer_tileinfo[layer_id]['Inputindex']
                                max_prelayer_time = 0
                                for idx in temp_Inputindex:
                                    tmp_time = self.finish_time[layer_id + idx][-1]
                                    if tmp_time > max_prelayer_time:
                                        max_prelayer_time = tmp_time
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                compute_time = temp_pooling_latency.pooling_latency + merge_time + transfer_time + begin_time
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='pooling', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_pooling_latency=temp_pooling_latency, merge_time=merge_time, transfer_time=transfer_time)

                    elif layer_dict['type'] == 'element_sum':
                        self.layer_latency_initial()
                        Inputindex_list = list(map(int, layer_dict['Inputindex']))
                        assert len(Inputindex_list) > 1, "the number of element_sum's previous layers must > 1"
                        previous_layer_dict = self.NetStruct[layer_id + Inputindex_list[0]][0][0]
                        output_size = list(map(int, previous_layer_dict['Outputsize']))
                        inputchannel = int(previous_layer_dict['Outputchannel'])
                        outputchannel = int(previous_layer_dict['Outputchannel'])
                        inputbit = int(previous_layer_dict['Outputbit'])
                        outputbit = int(previous_layer_dict['Outputbit'])

                        merge_time = 0
                        transfer_time = int(outputchannel * outputbit / self.inter_tile_bandwidth)

                        temp_tile_latency = tile_latency_analysis(SimConfig_path=self.SimConfig_path, max_row=self.graph.layer_tileinfo[layer_id]['max_row'], max_column=self.graph.layer_tileinfo[layer_id]['max_column'], inprecision=inputbit, PE_num=self.graph.layer_tileinfo[layer_id]['max_PE'], default_outbuf_size_tile=self.graph.max_outbuf_size_tile, default_inbuf_size_pe=self.graph.max_inbuf_size_pe, default_outbuf_size_pe=self.graph.max_outbuf_size_pe)
                        temp_tile_latency.outbuf_tile.calculate_buf_read_latency(rdata=(len(Inputindex_list) * inputchannel * inputbit / 8))
                        temp_tile_latency.outbuf_tile.calculate_buf_write_latency(wdata=(inputchannel * inputbit / 8))
                        temp_tile_latency.adder_latency = temp_tile_latency.digital_period

                        self.pre_max_time = 0
                        for i in range(output_size[0]):
                            for j in range(output_size[1]):
                                max_prelayer_time = 0  # the maximum time of the required input data (in all input layers)
                                for idx in Inputindex_list:
                                    tmp_time = self.finish_time[layer_id + idx][-1]
                                    if tmp_time > max_prelayer_time:
                                        max_prelayer_time = tmp_time
                                begin_time = max(max_prelayer_time, self.pre_max_time)
                                compute_time = merge_time + transfer_time + temp_tile_latency.adder_latency + temp_tile_latency.outbuf_tile.buf_rlatency + temp_tile_latency.outbuf_tile.buf_wlatency + begin_time
                                self.pre_max_time = compute_time
                                self.pipe_result_update(layer_type='element_sum', begin_time=begin_time, compute_time=compute_time, layer_id=layer_id, temp_tile_latency=temp_tile_latency, merge_time=merge_time, transfer_time=transfer_time)

            self.compute_interval[layer_id] = merge_interval(self.compute_interval[layer_id])
            temp_runtime = 0
            for l in range(len(self.compute_interval[layer_id])):
                temp_runtime += (self.compute_interval[layer_id][l][1] - self.compute_interval[layer_id][l][0])

            self.occupancy.append(temp_runtime / (max(self.finish_time[layer_id]) - min(self.begin_time[layer_id])))
            self.total_buffer_latency.append(sum(self.buffer_latency[layer_id]))
            self.total_buffer_r_latency.append(sum(self.buffer_r_latency[layer_id]))
            self.total_buffer_w_latency.append(sum(self.buffer_w_latency[layer_id]))
            self.total_DAC_latency.append(sum(self.DAC_latency[layer_id]))
            self.total_xbar_latency.append(sum(self.xbar_latency[layer_id]))
            self.total_ADC_latency.append(sum(self.ADC_latency[layer_id]))
            self.total_iReg_latency.append(sum(self.iReg_latency[layer_id]))
            self.total_oReg_latency.append(sum(self.oReg_latency[layer_id]))
            self.total_input_demux_latency.append(sum(self.input_demux_latency[layer_id]))
            self.total_output_mux_latency.append(sum(self.output_mux_latency[layer_id]))
            self.total_shiftreg_latency.append(sum(self.shiftreg_latency[layer_id]))
            self.total_adder_latency.append(sum(self.adder_latency[layer_id]))
            self.total_jointmodule_latency.append(sum(self.jointmodule_latency[layer_id]))
            self.total_pooling_latency.append(sum(self.pooling_latency[layer_id]))
            self.total_digital_latency.append(sum(self.digital_latency[layer_id]))
            self.total_computing_latency.append(sum(self.computing_latency[layer_id]))
            self.total_inter_tile_latency.append(sum(self.inter_tile_latency[layer_id]))
            self.total_intra_tile_latency.append(sum(self.intra_tile_latency[layer_id]))
            self.total_tile_merge_latency.append(sum(self.tile_merge_latency[layer_id]))
            self.total_tile_transfer_latency.append(sum(self.tile_transfer_latency[layer_id]))


if __name__ == '__main__':
    model_name = ['Vgg16', 'Res18', 'Res50', 'WRN']
    mode = ['naive', 'structure', 'weight_pattern_shape_translate', 'ORC', 'weight_pattern_value_identical_translate', 'weight_pattern_value_similar_translate', 'structure_and weight_pattern_value_identical_translate', 'weight_pattern_shape_and_value_similar_translate']
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")

    result = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    performance_analyse = pd.DataFrame()  # 记录各部分执行时间占比
    model_latency = [0.0] * len(mode)  # 记录每次模拟的总延迟数值
    buffer_latency = [0.0] * len(mode)  # 记录每次模拟的buffer延迟数值
    dac_latency = [0.0] * len(mode)  # 记录每次模拟的DAC延迟数值
    crossbar_latency = [0.0] * len(mode)  # 记录每次模拟的Crossbar延迟数值
    adc_latency = [0.0] * len(mode)  # 记录每次模拟的ADC延迟数值
    shiftreg_latency = [0.0] * len(mode)  # 记录每次模拟的Shiftreg延迟数值
    adder_latency = [0.0] * len(mode)  # 记录每次模拟的Adder延迟数值
    latency_speedup = [0.0] * len(mode)  # 记录每次模拟的加速比
    model_baseline = 0

    for i in range(0, len(model_name)):
        for j in range(0, len(mode)):
            __TestInterface = TrainTestInterface(model_name[i], 'MNSIM.Interface.cifar10', test_SimConfig_path, mode[j])
            structure_file = __TestInterface.get_structure()
            __TCG_mapping = TCG(structure_file, test_SimConfig_path)
            __latency = Model_latency(structure_file, test_SimConfig_path, __TCG_mapping.multiple, __TCG_mapping)
            __latency.calculate_model_latency(mode[j])
            model_latency[j] = __latency.model_latency_output(1, 1)

            if mode[j] == 'naive':
                model_baseline = model_latency[j]
            latency_speedup[j] = model_baseline / model_latency[j]

        result[model_name[i] + '_model_latency'] = model_latency
        result[model_name[i] + '_latency_speedup'] = latency_speedup

    result.to_csv('latency_info' + '.csv')