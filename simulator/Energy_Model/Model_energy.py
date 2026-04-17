#!/usr/bin/python
# -*-coding:utf-8-*-
import pandas as pd
import configparser as cp
from Interface.interface import *
from Mapping_Model.Tile_connection_graph import TCG
from Power_Model.Model_inference_power import Model_inference_power
from Latency_Model.Model_latency import Model_latency


class Model_energy:
    def __init__(self, NetStruct, SimConfig_path, multiple, TCG_mapping, mode):
        self.NetStruct = NetStruct
        self.SimConfig_path = SimConfig_path
        modelL_config = cp.ConfigParser()
        modelL_config.read(self.SimConfig_path, encoding='UTF-8')
        self.graph = TCG_mapping
        self.total_layer_num = self.graph.layer_num

        self.model_latency = Model_latency(NetStruct, SimConfig_path, multiple, TCG_mapping)
        self.model_latency.calculate_model_latency(mode)
        self.model_power = Model_inference_power(NetStruct, SimConfig_path, multiple, TCG_mapping, mode)

        self.arch_energy = self.total_layer_num * [0]
        self.arch_xbar_energy = self.total_layer_num * [0]
        self.arch_ADC_energy = self.total_layer_num * [0]
        self.arch_DAC_energy = self.total_layer_num * [0]
        self.arch_digital_energy = self.total_layer_num * [0]
        self.arch_adder_energy = self.total_layer_num * [0]
        self.arch_shiftreg_energy = self.total_layer_num * [0]
        self.arch_iReg_energy = self.total_layer_num * [0]
        self.arch_oReg_energy = self.total_layer_num * [0]
        self.arch_input_demux_energy = self.total_layer_num * [0]
        self.arch_output_mux_energy = self.total_layer_num * [0]
        self.arch_jointmodule_energy = self.total_layer_num * [0]
        self.arch_buf_energy = self.total_layer_num * [0]
        self.arch_buf_r_energy = self.total_layer_num * [0]
        self.arch_buf_w_energy = self.total_layer_num * [0]
        self.arch_pooling_energy = self.total_layer_num * [0]

        self.arch_total_energy = 0
        self.arch_total_xbar_energy = 0
        self.arch_total_ADC_energy = 0
        self.arch_total_DAC_energy = 0
        self.arch_total_digital_energy = 0
        self.arch_total_adder_energy = 0
        self.arch_total_shiftreg_energy = 0
        self.arch_total_iReg_energy = 0
        self.arch_total_input_demux_energy = 0
        self.arch_total_jointmodule_energy = 0
        self.arch_total_buf_energy = 0
        self.arch_total_buf_r_energy = 0
        self.arch_total_buf_w_energy = 0
        self.arch_total_output_mux_energy = 0
        self.arch_total_pooling_energy = 0

        self.calculate_model_energy()

    def calculate_model_energy(self):
        for i in range(self.total_layer_num):
            self.arch_xbar_energy[i] = self.model_power.arch_xbar_power[i] * self.model_latency.total_xbar_latency[i]
            self.arch_ADC_energy[i] = self.model_power.arch_ADC_power[i] * self.model_latency.total_ADC_latency[i]
            self.arch_DAC_energy[i] = self.model_power.arch_DAC_power[i] * self.model_latency.total_DAC_latency[i]
            self.arch_adder_energy[i] = self.model_power.arch_adder_power[i] * self.model_latency.total_adder_latency[i]
            self.arch_shiftreg_energy[i] = self.model_power.arch_shiftreg_power[i] * self.model_latency.total_shiftreg_latency[i]
            self.arch_iReg_energy[i] = self.model_power.arch_iReg_power[i] * self.model_latency.total_iReg_latency[i]
            self.arch_oReg_energy[i] = self.model_power.arch_oReg_power[i] * self.model_latency.total_oReg_latency[i]
            self.arch_input_demux_energy[i] = self.model_power.arch_input_demux_power[i] * self.model_latency.total_input_demux_latency[i]
            self.arch_output_mux_energy[i] = self.model_power.arch_output_mux_power[i] * self.model_latency.total_output_mux_latency[i]
            self.arch_jointmodule_energy[i] = self.model_power.arch_jointmodule_power[i] * self.model_latency.total_jointmodule_latency[i]
            self.arch_buf_r_energy[i] = self.model_power.arch_buf_r_power[i] * self.model_latency.total_buffer_r_latency[i]
            self.arch_buf_w_energy[i] = self.model_power.arch_buf_w_power[i] * self.model_latency.total_buffer_w_latency[i]
            self.arch_buf_energy[i] = self.arch_buf_r_energy[i] + self.arch_buf_w_energy[i]
            self.arch_pooling_energy[i] = self.model_power.arch_pooling_power[i] * self.model_latency.total_pooling_latency[i]
            self.arch_digital_energy[i] = self.arch_shiftreg_energy[i] + self.arch_iReg_energy[i] + self.arch_oReg_energy[i] + self.arch_input_demux_energy[i] + self.arch_output_mux_energy[i] + self.arch_jointmodule_energy[i]
            self.arch_energy[i] = self.arch_xbar_energy[i] + self.arch_ADC_energy[i] + self.arch_DAC_energy[i] + self.arch_digital_energy[i] + self.arch_buf_energy[i] + self.arch_pooling_energy[i]

        self.arch_total_energy = sum(self.arch_energy)
        self.arch_total_xbar_energy = sum(self.arch_xbar_energy)
        self.arch_total_ADC_energy = sum(self.arch_ADC_energy)
        self.arch_total_DAC_energy = sum(self.arch_DAC_energy)
        self.arch_total_digital_energy = sum(self.arch_digital_energy)
        self.arch_total_adder_energy = sum(self.arch_adder_energy)
        self.arch_total_shiftreg_energy = sum(self.arch_shiftreg_energy)
        self.arch_total_iReg_energy = sum(self.arch_iReg_energy)
        self.arch_total_input_demux_energy = sum(self.arch_input_demux_energy)
        self.arch_total_output_mux_energy = sum(self.arch_output_mux_energy)
        self.arch_total_jointmodule_energy = sum(self.arch_jointmodule_energy)
        self.arch_total_buf_energy = sum(self.arch_buf_energy)
        self.arch_total_buf_r_energy = sum(self.arch_buf_r_energy)
        self.arch_total_buf_w_energy = sum(self.arch_buf_w_energy)
        self.arch_total_pooling_energy = sum(self.arch_pooling_energy)

    def model_energy_output(self, module_information=1, layer_information=1):
        print("Hardware energy:", self.arch_total_energy, "nJ")
        if module_information:
            print("		crossbar energy:", self.arch_total_xbar_energy, "nJ")
            print("		DAC energy:", self.arch_total_DAC_energy, "nJ")
            print("		ADC energy:", self.arch_total_ADC_energy, "nJ")
            print("		Buffer energy:", self.arch_total_buf_energy, "nJ")
            print("			|---read buffer energy:", self.arch_total_buf_r_energy, "nJ")
            print("			|---write buffer energy:", self.arch_total_buf_w_energy, "nJ")
            print("		Pooling energy:", self.arch_total_pooling_energy, "nJ")
            print("		Other digital part energy:", self.arch_total_digital_energy, "nJ")
            print("			|---adder energy:", self.arch_total_adder_energy, "nJ")
            print("			|---output-shift-reg energy:", self.arch_total_shiftreg_energy, "nJ")
            print("			|---input-reg energy:", self.arch_total_iReg_energy, "nJ")
            print("			|---input_demux energy:", self.arch_total_input_demux_energy, "nJ")
            print("			|---output_mux energy:", self.arch_total_output_mux_energy, "nJ")
            print("			|---joint_module energy:", self.arch_total_jointmodule_energy, "nJ")
        if layer_information:
            for i in range(self.total_layer_num):
                print("Layer", i, ":")
                print("     Hardware energy:", self.arch_energy[i], "nJ")

        return self.arch_total_energy


if __name__ == '__main__':
    model_name = ['Vgg16', 'Res18', 'Res50', 'WRN']
    mode = ['naive', 'structure', 'weight_pattern_shape_translate', 'ORC', 'weight_pattern_value_identical_translate', 'weight_pattern_value_similar_translate', 'structure_and weight_pattern_value_identical_translate', 'weight_pattern_shape_and_value_similar_translate']
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")

    result = pd.DataFrame()  # 记录训练过程数据并存储到csv文件
    model_energy = [0.0] * len(mode)  # 记录每次模拟的总能耗数值
    energy_efficiency = [0.0] * len(mode)  # 记录每次模拟的效率比
    energy_improvement = [0.0] * len(mode)  # 记录每次模拟的提高比
    model_baseline = 0

    for i in range(0, len(model_name)):
        for j in range(0, len(mode)):
            __TestInterface = TrainTestInterface(model_name[i], 'MNSIM.Interface.cifar10', test_SimConfig_path, mode[j])
            structure_file = __TestInterface.get_structure()
            __TCG_mapping = TCG(structure_file, test_SimConfig_path)
            __energy = Model_energy(structure_file, test_SimConfig_path, __TCG_mapping.multiple, __TCG_mapping, mode[j])
            model_energy[j] = __energy.model_energy_output(1, 1)

            if mode[j] == 'naive':
                model_baseline = model_energy[j]
            energy_efficiency[j] = model_energy[j] / model_baseline
            energy_improvement[j] = model_baseline / model_energy[j]

        result[model_name[i] + '_model_energy'] = model_energy
        result[model_name[i] + '_energy_efficiency'] = energy_efficiency
        result[model_name[i] + '_energy_improvement'] = energy_improvement

        result.to_csv('energy_info' + '.csv')
