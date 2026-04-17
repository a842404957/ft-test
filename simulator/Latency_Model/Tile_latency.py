#!/usr/bin/python
# -*-coding:utf-8-*-
import configparser as cp
from Interface.interface import *
from Latency_Model.PE_latency import PE_latency_analysis
from Hardware_Model.Buffer import buffer


class tile_latency_analysis(PE_latency_analysis):
    def __init__(self, SimConfig_path, max_row=0, max_column=0, inprecision=8, PE_num=0, default_outbuf_size_tile=64, default_inbuf_size_pe=16, default_outbuf_size_pe=16, default_indexbuf_size_pe=0):
        # max_row: activated WL number in crossbar
        # max_column: activated BL number in crossbar
        # indata: volume of input data (for PE) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # outdata: volume of output data (for PE) (Byte)
        # inprecision: input data precision of each Xbar
        # PE_num: used PE_number in one tile
        # default_inbuf_size: the default PE-level input buffer size (unit: KB)
        # default_outbuf_size: the default Tile-level output buffer size (unit: KB)
        PE_latency_analysis.__init__(self, SimConfig_path, max_row=max_row, max_column=max_column, inprecision=inprecision, default_inbuf_size=default_inbuf_size_pe, default_outbuf_size=default_outbuf_size_pe, default_indexbuf_size=default_indexbuf_size_pe)
        tile_config = cp.ConfigParser()
        tile_config.read(SimConfig_path, encoding='UTF-8')
        self.outbuf_tile = buffer(default_buf_size=default_outbuf_size_tile)
        self.intra_tile_bandwidth = float(tile_config.get('Tile level', 'Intra_Tile_Bandwidth'))
        self.ADC_precision = int(tile_config.get('Interface level', 'ADC_Precision'))
        self.tile_PE_num = list(map(int, tile_config.get('Tile level', 'PE_Num').split(',')))

        if self.tile_PE_num[0] == 0:
            self.tile_PE_num[0] = 3
            self.tile_PE_num[1] = 3
        assert self.tile_PE_num[0] > 0, "PE number in one PE < 0"
        assert self.tile_PE_num[1] > 0, "PE number in one PE < 0"
        self.tile_PE_total_num = self.tile_PE_num[0] * self.tile_PE_num[1]
        assert PE_num <= self.tile_PE_total_num, "PE number exceeds the range"

        merge_time = math.ceil(math.log2(PE_num))
        self.jointmodule_latency = merge_time * self.digital_period
        self.transfer_latency = self.tile_PE_total_num * math.ceil(self.XBar_size[1] / self.OU_size[1]) * self.ADC_precision / self.intra_tile_bandwidth
        self.outbuf_tile.calculate_buf_write_latency(wdata=int(self.tile_PE_total_num * math.ceil(self.XBar_size[1] / self.OU_size[1]) * self.ADC_precision))
        self.tile_buf_rlatency = 0
        self.tile_buf_wlatency = self.outbuf_tile.buf_wlatency
        self.tile_latency = self.PE_latency + self.jointmodule_latency + self.transfer_latency + self.tile_buf_wlatency

    def update_tile_latency(self, OU_row, read_column, indata, rdata, wdata, mode):
        self.update_PE_latency(OU_row, read_column, indata, rdata, wdata, mode)
        self.tile_latency = self.PE_latency + self.jointmodule_latency + self.transfer_latency + self.tile_buf_rlatency + self.tile_buf_wlatency


if __name__ == '__main__':
    test_SimConfig_path = os.path.join('C:/Users/69562/Desktop/my_simulator', "SimConfig.ini")
    _test = tile_latency_analysis(test_SimConfig_path, 100, 100, 8, 8, 64, 16, 16)
    print(_test.tile_latency)