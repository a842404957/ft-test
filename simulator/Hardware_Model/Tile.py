import os
import configparser as cp
from numpy import *
from Hardware_Model.PE import ProcessElement
from Hardware_Model.Adder import adder
from Hardware_Model.Buffer import buffer
from Hardware_Model.ShiftReg import shiftreg
from Hardware_Model.Reg import reg
from Hardware_Model.JointModule import JointModule
from Hardware_Model.Pooling import Pooling
test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")


class tile(ProcessElement):
	def __init__(self, SimConfig_path):
		# layer_num is a list with the size of 1xPE_num
		ProcessElement.__init__(self, SimConfig_path)
		tile_config = cp.ConfigParser()
		tile_config.read(SimConfig_path, encoding='UTF-8')
		self.tile_PE_num = list(map(int, tile_config.get('Tile level', 'PE_Num').split(',')))
		if self.tile_PE_num[0] == 0:
			self.tile_PE_num[0] = 4
			self.tile_PE_num[1] = 4
		assert self.tile_PE_num[0] > 0, "PE number in one PE < 0"
		assert self.tile_PE_num[1] > 0, "PE number in one PE < 0"
		self.tile_PE_total_num = self.tile_PE_num[0] * self.tile_PE_num[1]
		self.tile_PE_list = []
		self.tile_PE_enable = []
		for i in range(self.tile_PE_num[0]):
			self.tile_PE_list.append([])
			self.tile_PE_enable.append([])
			for j in range(self.tile_PE_num[1]):
				__PE = ProcessElement(SimConfig_path)
				self.tile_PE_list[i].append(__PE)
				self.tile_PE_enable[i].append(0)
		self.layer_type = 'conv'
		self.tile_layer_num = 0
		self.tile_activation_precision = 0
		self.tile_sliding_times = 0
		self.tile_adder_num = 0
		self.tile_jointmodule_num = 0
		self.tile_adder = adder()
		self.tile_shiftreg = shiftreg()
		self.tile_iReg = reg()
		self.tile_oReg = reg()
		self.tile_buffer = buffer(default_buf_size=64)
		self.tile_pooling = Pooling()
		self.tile_jointmodule = JointModule(SimConfig_path)

		self.tile_utilization = 0
		self.num_occupied_PE = 0

		self.tile_area = 0
		self.tile_xbar_area = 0
		self.tile_ADC_area = 0
		self.tile_DAC_area = 0
		self.tile_digital_area = 0
		self.tile_adder_area = 0
		self.tile_shiftreg_area = 0
		self.tile_iReg_area = 0
		self.tile_oReg_area = 0
		self.tile_input_demux_area = 0
		self.tile_output_mux_area = 0
		self.tile_jointmodule_area = 0
		self.tile_pooling_area = 0
		self.tile_buffer_area = 0

		self.tile_read_power = 0
		self.tile_xbar_read_power = 0
		self.tile_ADC_read_power = 0
		self.tile_DAC_read_power = 0
		self.tile_digital_read_power = 0
		self.tile_adder_read_power = 0
		self.tile_shiftreg_read_power = 0
		self.tile_iReg_read_power = 0
		self.tile_oReg_read_power = 0
		self.tile_input_demux_read_power = 0
		self.tile_output_mux_read_power = 0
		self.tile_jointmodule_read_power = 0
		self.tile_pooling_read_power = 0
		self.tile_buffer_power = 0
		self.tile_buffer_rpower = 0
		self.tile_buffer_wpower = 0

	def update_tile_buf_size(self, SimConfig_path, default_buf_size=64):
		self.tile_buffer = buffer(default_buf_size=default_buf_size)

	def calculate_tile_area(self, SimConfig_path, default_outbuf_size_tile, default_inbuf_size_pe, default_outbuf_size_pe, default_indexbuf_size_pe):
		# unit: um^2
		self.tile_area = 0
		self.tile_xbar_area = 0
		self.tile_ADC_area = 0
		self.tile_DAC_area = 0
		self.tile_input_demux_area = 0
		self.tile_output_mux_area = 0
		self.tile_shiftreg_area = 0
		self.tile_iReg_area = 0
		self.tile_oReg_area = 0
		self.tile_adder_area = 0
		self.tile_buffer_area = 0
		self.tile_digital_area = 0
		self.tile_buffer = buffer(default_buf_size=default_outbuf_size_tile)
		self.tile_buffer.calculate_buf_area()
		self.tile_jointmodule.calculate_jointmodule_area()

		for i in range(self.tile_PE_num[0]):
			for j in range(self.tile_PE_num[1]):
				self.tile_PE_list[i][j].calculate_PE_area(default_inbuf_size_pe, default_outbuf_size_pe, default_indexbuf_size_pe)
				self.tile_xbar_area += self.tile_PE_list[i][j].PE_xbar_area
				self.tile_ADC_area += self.tile_PE_list[i][j].PE_ADC_area
				self.tile_DAC_area += self.tile_PE_list[i][j].PE_DAC_area
				self.tile_input_demux_area += self.tile_PE_list[i][j].PE_input_demux_area
				self.tile_output_mux_area += self.tile_PE_list[i][j].PE_output_mux_area
				self.tile_shiftreg_area += self.tile_PE_list[i][j].PE_shiftreg_area
				self.tile_iReg_area += self.tile_PE_list[i][j].PE_iReg_area
				self.tile_oReg_area += self.tile_PE_list[i][j].PE_oReg_area
				self.tile_adder_area += self.tile_PE_list[i][j].PE_adder_area
				self.tile_buffer_area += self.tile_PE_list[i][j].PE_inbuf_area + self.tile_PE_list[i][j].PE_outbuf_area
		self.tile_jointmodule_area = self.tile_jointmodule_num * self.tile_jointmodule.jointmodule_area
		self.tile_digital_area = self.tile_input_demux_area + self.tile_output_mux_area + self.tile_adder_area + self.tile_shiftreg_area + self.tile_jointmodule_area + self.tile_iReg_area + self.tile_oReg_area
		self.tile_pooling_area = self.tile_pooling.Pooling_area
		self.tile_buffer_area += self.tile_buffer.buf_area
		self.tile_area = self.tile_xbar_area + self.tile_ADC_area + self.tile_DAC_area + self.tile_digital_area + self.tile_buffer_area + self.tile_pooling_area

	def calculate_tile_read_power_fast(self, max_PE=0, max_group=0, layer_type=None, SimConfig_path=None, default_outbuf_size_tile=64, default_inbuf_size_pe=16, default_outbuf_size_pe=16, default_indexbuf_size_pe=0):
		# unit: W
		# max_column: maximum used column in one crossbar in this tile
		# max_row: maximum used row in one crossbar in this tile
		# max_PE: maximum used PE in this tile
		# max_group: maximum used groups in one PE
		self.tile_read_power = 0
		self.tile_xbar_read_power = 0
		self.tile_ADC_read_power = 0
		self.tile_DAC_read_power = 0
		self.tile_digital_read_power = 0
		self.tile_adder_read_power = 0
		self.tile_shiftreg_read_power = 0
		self.tile_iReg_read_power = 0
		self.tile_oReg_read_power = 0
		self.tile_input_demux_read_power = 0
		self.tile_output_mux_read_power = 0
		self.tile_jointmodule_read_power = 0
		self.tile_pooling_read_power = 0
		self.tile_buffer_power = 0
		self.tile_buffer_rpower = 0
		self.tile_buffer_wpower = 0

		self.tile_buffer = buffer(default_buf_size=default_outbuf_size_tile)
		if layer_type == 'pooling':
			self.tile_pooling_read_power = self.tile_pooling.Pooling_power
		elif layer_type == 'conv' or layer_type == 'fc':
			self.calculate_PE_read_power_fast(max_group=max_group, SimConfig_path=SimConfig_path, default_inbuf_size_pe=default_inbuf_size_pe, default_outbuf_size_pe=default_outbuf_size_pe, default_indexbuf_size_pe=default_indexbuf_size_pe)
			self.tile_xbar_read_power = max_PE * self.PE_xbar_read_power
			self.tile_ADC_read_power = max_PE * self.PE_ADC_read_power
			self.tile_DAC_read_power = max_PE * self.PE_DAC_read_power
			self.tile_adder_read_power = max_PE * self.PE_adder_read_power
			self.tile_shiftreg_read_power = max_PE * self.PE_shiftreg_read_power
			self.tile_iReg_read_power = max_PE * self.PE_iReg_read_power
			self.tile_oReg_read_power = max_PE * self.PE_oReg_read_power
			self.tile_input_demux_read_power = max_PE * self.input_demux_read_power
			self.tile_output_mux_read_power = max_PE * self.output_mux_read_power
			self.tile_jointmodule_read_power = (max_PE - 1) * self.PE_ADC_num * self.tile_jointmodule.jointmodule_power
			self.tile_digital_read_power = self.tile_adder_read_power + self.tile_shiftreg_read_power + self.tile_input_demux_read_power + self.tile_output_mux_read_power + self.tile_jointmodule_read_power
			self.tile_buffer_rpower = max_PE * (self.PE_inbuf_rpower + self.PE_outbuf_rpower + self.PE_indexbuf_rpower)
			self.tile_buffer_wpower = max_PE * (self.PE_inbuf_wpower + self.PE_outbuf_wpower + self.PE_indexbuf_wpower)

		self.tile_buffer.calculate_buf_read_power()
		self.tile_buffer.calculate_buf_write_power()
		self.tile_buffer_rpower += self.tile_buffer.buf_rpower * 1e-3
		self.tile_buffer_wpower += self.tile_buffer.buf_wpower * 1e-3
		self.tile_buffer_power = self.tile_buffer_rpower + self.tile_buffer_wpower
		self.tile_digital_read_power = self.tile_adder_read_power + self.tile_shiftreg_read_power + self.tile_iReg_read_power + self.tile_oReg_read_power + self.tile_input_demux_read_power + self.tile_output_mux_read_power + self.tile_jointmodule_read_power
		self.tile_read_power = self.tile_xbar_read_power + self.tile_ADC_read_power + self.tile_DAC_read_power + self.tile_digital_read_power + self.tile_pooling_read_power + self.tile_buffer_power

	def tile_output(self):
		self.tile_PE_list[0][0].PE_output()
		print("-------------------------tile Configurations-------------------------")
		print("total PE number in one tile:", self.tile_PE_total_num, "(", self.tile_PE_num, ")")
		print("total adder number in one tile:", self.tile_adder_num)
		print("total jointmodule number in one tile:", self.tile_jointmodule_num)
		print("----------------------tile Area Simulation Results-------------------")
		print("tile area:", self.tile_area, "um^2")
		print("			crossbar area:", self.tile_xbar_area, "um^2")
		print("			DAC area:", self.tile_DAC_area, "um^2")
		print("			ADC area:", self.tile_ADC_area, "um^2")
		print("			digital part area:", self.tile_digital_area, "um^2")
		print("				|---adder area:", self.tile_adder_area, "um^2")
		print("				|---shift-reg area:", self.tile_shiftreg_area, "um^2")
		print("				|---input_demux area:", self.tile_input_demux_area, "um^2")
		print("				|---output_mux area:", self.tile_output_mux_area, "um^2")
		print("				|---JointModule area:", self.tile_jointmodule_area, "um^2")

		print("--------------------tile Power Simulation Results-------------------")
		print("tile read power:", self.tile_read_power, "W")
		print("			crossbar read power:", self.tile_xbar_read_power, "W")
		print("			DAC read power:", self.tile_DAC_read_power, "W")
		print("			ADC read power:", self.tile_ADC_read_power, "W")
		print("			digital part read power:", self.tile_digital_read_power, "W")
		print("				|---adder read power:", self.tile_adder_read_power, "W")
		print("				|---shift-reg read power:", self.tile_shiftreg_read_power, "W")
		print("				|---input demux read power:", self.tile_input_demux_read_power, "W")
		print("				|---output mux read power:", self.tile_output_mux_read_power, "W")
		print("				|---JointModule read power:", self.tile_jointmodule_read_power, "W")
		print("			buffer read power:", self.tile_buffer_power, "W")
		print("-----------------------------------------------------------------")


def tile_test():
	print("load file:", test_SimConfig_path)
	_tile = tile(test_SimConfig_path)
	_tile.calculate_tile_area(test_SimConfig_path, 64, 16, 2, 0)
	_tile.calculate_tile_read_power_fast(max_PE=9, max_group=8, layer_type='conv', SimConfig_path=test_SimConfig_path, default_outbuf_size_tile=64, default_inbuf_size_pe=16, default_outbuf_size_pe=2, default_indexbuf_size_pe=0)
	_tile.tile_output()


if __name__ == '__main__':
	tile_test()