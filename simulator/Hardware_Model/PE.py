import os
import configparser as cp
from numpy import *
from Hardware_Model.Crossbar import crossbar
from Hardware_Model.DAC import DAC
from Hardware_Model.ADC import ADC
from Hardware_Model.Adder import adder
from Hardware_Model.ShiftReg import shiftreg
from Hardware_Model.Reg import reg
from Hardware_Model.Buffer import buffer
test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")


class ProcessElement(crossbar, DAC, ADC):
	def __init__(self, SimConfig_path):
		crossbar.__init__(self, SimConfig_path)
		DAC.__init__(self)
		ADC.__init__(self)
		PE_config = cp.ConfigParser()
		PE_config.read(SimConfig_path, encoding='UTF-8')

		self.PE_adder = adder()
		self.PE_shiftreg = shiftreg()
		self.PE_iReg = reg()
		self.PE_oReg = reg()
		self.inbuf = buffer(default_buf_size=16)
		self.outbuf = buffer(default_buf_size=2)
		self.indexbuf = buffer(default_buf_size=0)

		self.PE_xbar_num = int(PE_config.get('Process element level', 'Group_Num'))
		self.OU_size = list(map(int, PE_config.get('Crossbar level', 'OU_Size').split(',')))
		self.Crossbar_DAC_num = 128
		self.Crossbar_ADC_num = 1
		self.PE_DAC_num = self.Crossbar_DAC_num * self.PE_xbar_num
		self.PE_ADC_num = self.Crossbar_ADC_num * self.PE_xbar_num
		self.PE_adder_num = 2 * self.PE_xbar_num
		self.PE_shiftreg_num = 2 * self.PE_xbar_num

		self.input_demux = 0
		self.input_demux_power = 0
		self.input_demux_area = 0
		self.output_mux = math.ceil(self.xbar_column / self.Crossbar_ADC_num)
		self.output_mux_power = 0
		self.output_mux_area = 0

		self.PE_area = 0
		self.PE_xbar_area = 0
		self.PE_ADC_area = 0
		self.PE_DAC_area = 0
		self.PE_adder_area = 0
		self.PE_shiftreg_area = 0
		self.PE_iReg_area = 0
		self.PE_oReg_area = 0
		self.PE_input_demux_area = 0
		self.PE_output_mux_area = 0
		self.PE_digital_area = 0
		self.PE_inbuf_area = 0
		self.PE_outbuf_area = 0
		self.PE_indexbuf_area = 0

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0
		self.PE_inbuf_power = 0
		self.PE_inbuf_rpower = 0
		self.PE_inbuf_wpower = 0
		self.PE_outbuf_power = 0
		self.PE_outbuf_rpower = 0
		self.PE_outbuf_wpower = 0
		self.PE_indexbuf_power = 0
		self.PE_indexbuf_rpower = 0
		self.PE_indexbuf_wpower = 0

	def calculate_demux_area(self):
		# unit: um^2
		transistor_area = 10 * 32 * 32 / 1000000
		demux_area_dict = {2: 8 * transistor_area,  # 2-1: 8 transistors
						   4: 24 * transistor_area,  # 4-1: 3 * 2-1
						   8: 72 * transistor_area,
						   16: 216 * transistor_area,
						   32: 648 * transistor_area,
						   64: 1944 * transistor_area
		}
		if self.input_demux <= 2:
			self.input_demux_area = demux_area_dict[2]
		elif self.input_demux <= 4:
			self.input_demux_area = demux_area_dict[4]
		elif self.input_demux <= 8:
			self.input_demux_area = demux_area_dict[8]
		elif self.input_demux <= 16:
			self.input_demux_area = demux_area_dict[16]
		elif self.input_demux <= 32:
			self.input_demux_area = demux_area_dict[32]
		else:
			self.input_demux_area = demux_area_dict[64]

	def calculate_mux_area(self):
		# unit: um^2
		transistor_area = 10 * 32 * 32 / 1000000
		mux_area_dict = {2: 8 * transistor_area,
						 4: 24 * transistor_area,
						 8: 72 * transistor_area,
						 16: 216 * transistor_area,
						 32: 648 * transistor_area,
						 64: 1944 * transistor_area
		}
		if self.output_mux <= 2:
			self.output_mux_area = mux_area_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_area = mux_area_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_area = mux_area_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_area = mux_area_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_area = mux_area_dict[32]
		else:
			self.output_mux_area = mux_area_dict[64]

	def calculate_PE_area(self, default_inbuf_size_pe, default_outbuf_size_pe, default_indexbuf_size_pe):
		# unit: um^2
		self.inbuf = buffer(default_buf_size=default_inbuf_size_pe)
		self.outbuf = buffer(default_buf_size=default_outbuf_size_pe)
		self.indexbuf = buffer(default_buf_size=default_indexbuf_size_pe)
		self.inbuf.calculate_buf_area()
		self.outbuf.calculate_buf_area()
		self.indexbuf.calculate_buf_area()
		self.calculate_xbar_area()
		self.calculate_demux_area()
		self.calculate_mux_area()
		self.PE_adder.calculate_adder_area()
		self.PE_shiftreg.calculate_shiftreg_area()
		self.PE_xbar_area = self.xbar_area * self.PE_xbar_num
		self.PE_DAC_area = self.DAC_area * self.PE_DAC_num
		self.PE_ADC_area = self.ADC_area * self.PE_ADC_num
		self.PE_adder_area = self.PE_adder.adder_area * self.PE_adder_num
		self.PE_shiftreg_area = self.PE_shiftreg.shiftreg_area * self.PE_shiftreg_num
		self.PE_iReg_area = self.PE_iReg.reg_area * self.PE_DAC_num
		self.PE_oReg_area = self.PE_oReg.reg_area * self.PE_ADC_num
		self.PE_input_demux_area = self.input_demux_area * self.PE_DAC_num
		self.PE_output_mux_area = self.output_mux_area * self.PE_ADC_num
		self.PE_digital_area = self.PE_adder_area + self.PE_shiftreg_area + self.PE_input_demux_area + self.PE_output_mux_area + self.PE_iReg_area + self.PE_oReg_area
		self.PE_inbuf_area = self.inbuf.buf_area
		self.PE_outbuf_area = self.outbuf.buf_area
		self.PE_indexbuf_area = self.indexbuf.buf_area
		self.PE_area = self.PE_xbar_area + self.PE_ADC_area + self.PE_DAC_area + self.PE_digital_area + self.PE_inbuf_area + self.PE_outbuf_area + self.indexbuf.buf_area

	def calculate_demux_power(self):
		# unit: W
		transistor_power = 10 * 1.2 / 1e9
		demux_power_dict = {2: 8 * transistor_power,
						 	4: 24 * transistor_power,
						 	8: 72 * transistor_power,
						 	16: 216 * transistor_power,
						 	32: 648 * transistor_power,
						 	64: 1944 * transistor_power
		}
		if self.input_demux <= 2:
			self.input_demux_power = demux_power_dict[2]
		elif self.input_demux <= 4:
			self.input_demux_power = demux_power_dict[4]
		elif self.input_demux <= 8:
			self.input_demux_power = demux_power_dict[8]
		elif self.input_demux <= 16:
			self.input_demux_power = demux_power_dict[16]
		elif self.input_demux <= 32:
			self.input_demux_power = demux_power_dict[32]
		else:
			self.input_demux_power = demux_power_dict[64]

	def calculate_mux_power(self):
		# unit: W
		transistor_power = 10 * 1.2 / 1e9
		mux_power_dict = {2: 8 * transistor_power,
						  4: 24 * transistor_power,
						  8: 72 * transistor_power,
						  16: 216 * transistor_power,
						  32: 648 * transistor_power,
						  64: 1944 * transistor_power
		}
		if self.output_mux <= 2:
			self.output_mux_power = mux_power_dict[2]
		elif self.output_mux <= 4:
			self.output_mux_power = mux_power_dict[4]
		elif self.output_mux <= 8:
			self.output_mux_power = mux_power_dict[8]
		elif self.output_mux <= 16:
			self.output_mux_power = mux_power_dict[16]
		elif self.output_mux <= 32:
			self.output_mux_power = mux_power_dict[32]
		else:
			self.output_mux_power = mux_power_dict[64]

	def calculate_PE_read_power_fast(self, max_group=0, SimConfig_path=None, default_inbuf_size_pe=16, default_outbuf_size_pe=16, default_indexbuf_size_pe=0):
		# unit: W
		# max_column: maximum used column in one crossbar in this tile
		# max_row: maximum used row in one crossbar in this tile
		# max_group: maximum used groups in one PE
		self.calculate_demux_power()
		self.calculate_mux_power()
		self.PE_shiftreg.calculate_shiftreg_power()
		self.PE_adder.calculate_adder_power()

		self.PE_read_power = 0
		self.PE_xbar_read_power = 0
		self.PE_ADC_read_power = 0
		self.PE_DAC_read_power = 0
		self.PE_adder_read_power = 0
		self.PE_shiftreg_read_power = 0
		self.PE_iReg_read_power = 0
		self.PE_oReg_read_power = 0
		self.input_demux_read_power = 0
		self.output_mux_read_power = 0
		self.PE_digital_read_power = 0

		self.calculate_xbar_read_power()
		self.PE_xbar_read_power = max_group * self.xbar_read_power / math.ceil(self.xbar_row / self.OU_size[0]) / math.ceil(self.xbar_column / self.Crossbar_ADC_num)
		self.PE_DAC_read_power = max_group * self.OU_size[0] * self.DAC_power
		self.PE_ADC_read_power = max_group * self.Crossbar_ADC_num * self.ADC_power
		self.input_demux_read_power = max_group * self.OU_size[0] * self.input_demux_power
		self.output_mux_read_power = max_group * self.Crossbar_ADC_num * self.output_mux_power
		self.PE_adder_read_power = max_group * 2 * self.PE_adder.adder_power
		self.PE_shiftreg_read_power = max_group * 2 * self.PE_shiftreg.shiftreg_power
		self.PE_iReg_read_power = max_group * self.OU_size[0] * self.PE_iReg.reg_power
		self.PE_oReg_read_power = max_group * self.Crossbar_ADC_num * self.PE_oReg.reg_power
		self.PE_digital_read_power = self.input_demux_read_power + self.output_mux_read_power + self.PE_adder_read_power + self.PE_shiftreg_read_power + self.PE_iReg_read_power + self.PE_oReg_read_power

		self.inbuf = buffer(default_buf_size=default_inbuf_size_pe)
		self.outbuf = buffer(default_buf_size=default_outbuf_size_pe)
		self.indexbuf = buffer(default_buf_size=default_indexbuf_size_pe)
		self.inbuf.calculate_buf_read_power()
		self.inbuf.calculate_buf_write_power()
		self.outbuf.calculate_buf_read_power()
		self.outbuf.calculate_buf_write_power()
		self.indexbuf.calculate_buf_read_power()
		self.indexbuf.calculate_buf_write_power()
		self.PE_inbuf_rpower = self.inbuf.buf_rpower * 1e-3
		self.PE_inbuf_wpower = self.inbuf.buf_wpower * 1e-3
		self.PE_outbuf_rpower = self.outbuf.buf_rpower * 1e-3
		self.PE_outbuf_wpower = self.outbuf.buf_wpower * 1e-3
		self.PE_indexbuf_rpower = self.indexbuf.buf_rpower * 1e-3
		self.PE_indexbuf_wpower = self.indexbuf.buf_wpower * 1e-3
		self.PE_inbuf_power = self.PE_inbuf_rpower + self.PE_inbuf_wpower
		self.PE_outbuf_power = self.PE_outbuf_rpower + self.PE_outbuf_wpower
		self.PE_indexbuf_power = self.PE_indexbuf_rpower + self.PE_indexbuf_wpower
		self.PE_read_power = self.PE_xbar_read_power + self.PE_DAC_read_power + self.PE_ADC_read_power + self.PE_digital_read_power + self.PE_inbuf_power + self.PE_outbuf_power + self.PE_indexbuf_power

	def PE_output(self):
		"""
		print("---------------------Crossbar Configurations-----------------------")
		crossbar.xbar_output(self)
		print("------------------------DAC Configurations-------------------------")
		DAC.DAC_output(self)
		print("------------------------ADC Configurations-------------------------")
		ADC.ADC_output(self)
		"""
		print("-------------------------PE Configurations-------------------------")
		print("total crossbar number in one PE:", self.PE_xbar_num)
		print("total DAC number in one PE:", self.PE_DAC_num)
		print("			the number of DAC in one set of interfaces:", self.Crossbar_DAC_num)
		print("total ADC number in one PE:", self.PE_ADC_num)
		print("			the number of ADC in one set of interfaces:", self.Crossbar_ADC_num)
		print("---------------------PE Area Simulation Results--------------------")
		print("PE area:", self.PE_area, "um^2")
		print("			crossbar area:", self.PE_xbar_area, "um^2")
		print("			DAC area:", self.PE_DAC_area, "um^2")
		print("			ADC area:", self.PE_ADC_area, "um^2")
		print("			digital part area:", self.PE_digital_area, "um^2")
		print("			|---adder area:", self.PE_adder_area, "um^2")
		print("			|---shift-reg area:", self.PE_shiftreg_area, "um^2")
		print("			|---input_demux area:", self.PE_input_demux_area, "um^2")
		print("			|---output_mux area:", self.PE_output_mux_area, "um^2")
		print("			|---input_register area:", self.PE_iReg_area, "um^2")
		print("			|---output_register area:", self.PE_oReg_area, "um^2")
		print("			buffer area:", self.PE_inbuf_area + self.PE_outbuf_area + self.indexbuf.buf_area, "um^2")
		print("--------------------PE Power Simulation Results-------------------")
		print("PE read power:", self.PE_read_power, "W")
		print("			crossbar read power:", self.PE_xbar_read_power, "W")
		print("			DAC read power:", self.PE_DAC_read_power, "W")
		print("			ADC read power:", self.PE_ADC_read_power, "W")
		print("			digital part read power:", self.PE_digital_read_power, "W")
		print("			|---adder power:", self.PE_adder_read_power, "W")
		print("			|---shift-reg power:", self.PE_shiftreg_read_power, "W")
		print("			|---input_demux power:", self.input_demux_read_power, "W")
		print("			|---output_mux power:", self.output_mux_read_power, "W")
		print("			buffer read power:", self.PE_inbuf_power + self.PE_outbuf_power + self.PE_indexbuf_power, "W")
		print("-----------------------------------------------------------------")


def PE_test():
	print("load file:", test_SimConfig_path)
	_PE = ProcessElement(test_SimConfig_path)
	_PE.calculate_PE_area(16, 2, 0)
	_PE.calculate_PE_read_power_fast(max_group=8, SimConfig_path=test_SimConfig_path, default_inbuf_size_pe=16, default_outbuf_size_pe=16, default_indexbuf_size_pe=0)
	_PE.PE_output()


if __name__ == '__main__':
	PE_test()