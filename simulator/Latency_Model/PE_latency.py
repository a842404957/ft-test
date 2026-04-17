#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
from Hardware_Model.PE import ProcessElement
from Hardware_Model.Buffer import buffer
from Interface.interface import *


class PE_latency_analysis:
    def __init__(self, SimConfig_path, max_row=0, max_column=0, inprecision=8, default_inbuf_size=16, default_outbuf_size=16, default_indexbuf_size=0):
        # read_row: activated WL number in crossbar
        # read_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # default_buf_size: default input buffer size (KB)
        PE_config = cp.ConfigParser()
        PE_config.read(SimConfig_path, encoding='UTF-8')
        self.inbuf_pe = buffer(default_buf_size=default_inbuf_size)
        self.outbuf_pe = buffer(default_buf_size=default_outbuf_size)
        self.indexbuf_pe = buffer(default_buf_size=default_indexbuf_size)
        self.PE = ProcessElement(SimConfig_path)
        self.digital_period = 1 / float(PE_config.get('Digital module', 'Digital_Frequency')) * 1e3  # unit: ns
        self.XBar_size = list(map(int, PE_config.get('Crossbar level', 'Xbar_Size').split(',')))
        self.OU_size = list(map(int, PE_config.get('Crossbar level', 'OU_Size').split(',')))
        self.max_row = max_row
        self.max_column = max_column
        self.inprecision = inprecision

        DAC_num = int(PE_config.get('Process element level', 'DAC_Num'))
        ADC_num = int(PE_config.get('Process element level', 'ADC_Num'))

        Row = self.XBar_size[0]
        Column = self.XBar_size[1]
        # ns  (using NVSim)
        decoderLatency_dict = {
            1: 0.27933  # 1:8, technology 65nm
        }
        decoder1_8 = decoderLatency_dict[1]
        Row_per_DAC = math.ceil(Row/DAC_num)
        m = 1
        while Row_per_DAC > 0:
            Row_per_DAC = Row_per_DAC // 8
            m += 1
        self.decoderLatency = m * decoder1_8

        # ns
        muxLatency_dict = {
            1: 32.744/1000
        }
        mux8_1 = muxLatency_dict[1]
        m = 1
        Column_per_ADC = math.ceil(Column / ADC_num)
        while Column_per_ADC > 0:
            Column_per_ADC = Column_per_ADC // 8
            m += 1
        self.muxLatency = m * mux8_1

        self.PE_buf_rlatency = 0
        self.PE_buf_wlatency = 0
        self.PE_inbuf_rlatency = 0
        self.PE_inbuf_wlatency = 0
        self.PE_outbuf_rlatency = 0
        self.PE_outbuf_wlatency = 0
        self.PE_indexbuf_rlatency = 0
        self.PE_indexbuf_wlatency = 0

        self.xbar_latency = 0
        self.DAC_latency = 0
        self.ADC_latency = 0
        self.iReg_latency = 0
        self.oReg_latency = 0
        self.input_demux_latency = 0
        self.output_mux_latency = 0
        self.shiftreg_latency = 0
        self.adder_latency = 0

        self.computing_latency = self.DAC_latency + self.xbar_latency + self.ADC_latency
        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + self.adder_latency + self.output_mux_latency + self.oReg_latency
        self.PE_buf_wlatency = self.PE_inbuf_wlatency + self.PE_outbuf_wlatency + self.PE_indexbuf_wlatency
        self.PE_buf_rlatency = self.PE_inbuf_rlatency + self.PE_outbuf_rlatency + self.PE_indexbuf_rlatency
        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency

    def update_PE_latency(self, OU_row, read_column, indata, rdata, wdata, mode):
        # update the latency computing when indata and rdata change
        multiple_time = math.ceil(self.inprecision / self.PE.DAC_precision) * OU_row * (read_column / self.OU_size[1]) * math.ceil(self.OU_size[1] / self.PE.Crossbar_ADC_num)

        self.PE.calculate_DAC_latency()
        self.PE.calculate_xbar_read_latency()
        self.PE.calculate_ADC_latency()

        self.iReg_latency = multiple_time * self.digital_period
        self.input_demux_latency = multiple_time * self.decoderLatency
        self.DAC_latency = multiple_time * self.PE.DAC_latency
        self.xbar_latency = multiple_time * self.PE.xbar_read_latency
        self.output_mux_latency = multiple_time * self.muxLatency
        self.ADC_latency = multiple_time * self.PE.ADC_latency
        self.shiftreg_latency = multiple_time * self.digital_period
        self.adder_latency = multiple_time * self.digital_period
        self.oReg_latency = OU_row * (read_column / self.OU_size[1]) * self.digital_period

        self.computing_latency = self.DAC_latency + self.xbar_latency + self.ADC_latency

        self.inbuf_pe.calculate_buf_read_latency(rdata)
        self.inbuf_pe.calculate_buf_write_latency(indata)
        if 'ORC' in mode:
            self.PE_inbuf_rlatency = self.inbuf_pe.buf_rlatency * OU_row * math.ceil(self.max_column / self.OU_size[1]) * self.PE.PE_xbar_num  # 读取feature_map数据
            self.PE_inbuf_wlatency = self.inbuf_pe.buf_wlatency * math.ceil(self.max_row / self.OU_size[0]) * math.ceil(self.max_column / self.OU_size[1])  # 写入feature_map数据
        else:
            self.PE_inbuf_rlatency = self.inbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num
            self.PE_inbuf_wlatency = self.inbuf_pe.buf_wlatency * math.ceil(self.max_row / self.OU_size[0])

        if 'value_similar' in mode:  # 采用更长的字节存储部分和结果确保不丢失移位部分的数据
            self.outbuf_pe.calculate_buf_read_latency(2 * wdata)
            self.outbuf_pe.calculate_buf_write_latency(2 * wdata)
        else:
            self.outbuf_pe.calculate_buf_read_latency(wdata)
            self.outbuf_pe.calculate_buf_write_latency(wdata)
        self.PE_outbuf_rlatency = self.outbuf_pe.buf_rlatency * math.ceil(self.max_column / self.OU_size[1]) * OU_row * self.PE.PE_xbar_num  # 读取每列OU的部分和结果
        self.PE_outbuf_wlatency = self.outbuf_pe.buf_wlatency * math.ceil(self.max_column / self.OU_size[1]) * OU_row * self.PE.PE_xbar_num  # 更新每列OU的部分和结果

        if 'value' in mode:
            self.outbuf_pe.calculate_buf_write_latency(wdata)
            self.PE_outbuf_wlatency = self.PE_outbuf_wlatency + self.outbuf_pe.buf_wlatency * (read_column / self.OU_size[1]) * OU_row * self.PE.PE_xbar_num  # 向buffer中写可以重用的部分和结果
            if 'similar' in mode:
                self.shiftreg_latency = self.shiftreg_latency + OU_row * (self.max_column / self.OU_size[1]) * self.digital_period  # 通过移位放缩部分和结果
        if 'value_similar' in mode:  # 采用更长的字节存储部分和结果确保不丢失移位部分的数据
            self.adder_latency = self.adder_latency + 2 * OU_row * (self.max_column / self.OU_size[1]) * self.digital_period  # 同列OU部分和结果累加
        else:
            self.adder_latency = self.adder_latency + OU_row * (self.max_column / self.OU_size[1]) * self.digital_period  # 同列OU部分和结果累加

        self.PE_digital_latency = self.iReg_latency + self.shiftreg_latency + self.input_demux_latency + self.adder_latency + self.output_mux_latency + self.oReg_latency

        self.PE_indexbuf_rlatency = 0
        if 'structure' in mode:
            self.indexbuf_pe.calculate_buf_read_latency(rdata=1)  # 为每个输入提供1个channel位置索引
            self.PE_indexbuf_rlatency = self.PE_indexbuf_rlatency + self.indexbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num  # 每行读8个索引
        if 'ORC' in mode:
            self.indexbuf_pe.calculate_buf_read_latency(rdata=8 + 8)  # 为每个输入提供(8个channel+8个kernel)位置索引
            self.PE_indexbuf_rlatency = self.PE_indexbuf_rlatency + self.indexbuf_pe.buf_rlatency * OU_row * (self.max_column / self.OU_size[1]) * self.PE.PE_xbar_num
        if 'shape' in mode:
            self.indexbuf_pe.calculate_buf_read_latency(rdata=5)  # 为每种输入提供4个位置索引和1个数量索引
            self.PE_indexbuf_rlatency = self.PE_indexbuf_rlatency + self.indexbuf_pe.buf_rlatency * OU_row * self.PE.PE_xbar_num
        if 'value' in mode:
            if 'similar' in mode:
                self.indexbuf_pe.calculate_buf_read_latency(rdata=1)  # 为每个OU提供一个索引
            else:
                self.indexbuf_pe.calculate_buf_read_latency(rdata=2)  # 为每个OU提供一个索引以及放缩倍数
            self.PE_indexbuf_rlatency = self.PE_indexbuf_rlatency + self.indexbuf_pe.buf_rlatency * OU_row * math.ceil(self.max_column / self.OU_size[1]) * self.PE.PE_xbar_num  # 读取每个OU索引信息

        self.PE_buf_wlatency = self.PE_inbuf_wlatency + self.PE_outbuf_wlatency + self.PE_indexbuf_wlatency
        self.PE_buf_rlatency = self.PE_inbuf_rlatency + self.PE_outbuf_rlatency + self.PE_indexbuf_rlatency

        self.PE_latency = self.PE_buf_wlatency + self.PE_buf_rlatency + self.computing_latency + self.PE_digital_latency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")
    _test = PE_latency_analysis(test_SimConfig_path, 256, 256, 8, 16, 2, 32)
    _test.update_PE_latency(OU_row=32, read_column=256, indata=1, rdata=8, wdata=1, mode='naive')
    print(_test.PE_latency)
