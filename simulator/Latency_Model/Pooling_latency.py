from Hardware_Model.Buffer import buffer
from Hardware_Model.Pooling import Pooling


class pooling_latency_analysis:
    def __init__(self, indata=0, rdata=0, outprecision=8, default_inbuf_size=16, default_outbuf_size=4, default_inchannel=64):
        # unit: ns
        # indata: volume of input data (for pooling) (Byte)
        # rdata: volume of data from buffer to iReg (Byte)
        # default_inbuf_size: the default PE-level input buffer size (unit: KB)
        # default_outbuf_size: the default Tile-level output buffer size (unit: KB)
        self.pooling = Pooling()
        self.inbuf = buffer(default_buf_size=default_inbuf_size)
        self.inbuf.calculate_buf_write_latency(indata)
        self.inbuf_wlatency = self.inbuf.buf_wlatency
        self.inbuf.calculate_buf_read_latency(rdata)
        self.inbuf_rlatency = self.inbuf.buf_rlatency
        self.digital_latency = self.pooling.Pooling_latency
        self.outbuf = buffer(default_buf_size=default_outbuf_size)
        self.outbuf.calculate_buf_write_latency(wdata=(default_inchannel * outprecision / 8))
        self.outbuf_rlatency = 0
        self.outbuf_wlatency = self.outbuf.buf_wlatency
        self.pooling_latency = self.inbuf_wlatency + self.inbuf_rlatency + self.digital_latency + self.outbuf_rlatency + self.outbuf_wlatency

    def update_pooling_latency(self, indata=0, rdata=0):
        # unit: ns
        # update the latency computing when indata and rdata change
        self.inbuf.calculate_buf_write_latency(indata)
        self.inbuf_wlatency = self.inbuf.buf_wlatency
        self.inbuf.calculate_buf_read_latency(rdata)
        self.inbuf_rlatency = self.inbuf.buf_rlatency
        self.pooling_latency = self.inbuf_wlatency + self.inbuf_rlatency + self.digital_latency + self.outbuf_rlatency + self.outbuf_wlatency


if __name__ == '__main__':
    _test = pooling_latency_analysis(8, 4)
    print(_test.pooling_latency)
