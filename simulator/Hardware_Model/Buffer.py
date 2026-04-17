import math


class buffer:
    def __init__(self, default_buf_size=64):
        self.buf_Tech = 32  # unit: nm
        self.buf_Size = default_buf_size  # unit: KB
        self.buf_bitwidth = 256  # unit: bit
        self.buf_cycle = 1.2  # unit: GSamples/s

        self.buf_area = 0  # unit: um^2
        self.buf_rlatency = 0  # unit: ns
        self.buf_wlatency = 0  # unit: ns
        self.buf_rpower = 0  # unit: mV
        self.buf_wpower = 0  # unit: mV
        self.buf_renergy = 0  # unit: nJ
        self.buf_wenergy = 0  # unit: nJ

        self.calculate_buf_area()
        self.calculate_buf_read_power()
        self.calculate_buf_write_power()

    def calculate_buf_area(self):
        self.buf_area = (self.buf_Size / 64) * 83000 * 1e6  # unit: um^2

    def calculate_buf_read_power(self):
        self.buf_rpower = (self.buf_Size / 64) * 20.7  # unit: mW

    def calculate_buf_write_power(self):
        self.buf_wpower = (self.buf_Size / 64) * 20.7  # unit: mW

    def calculate_buf_read_latency(self, rdata):
        self.buf_rlatency = math.ceil(rdata * 8 / self.buf_bitwidth) * self.buf_cycle  # unit: ns, Byte(data)

    def calculate_buf_write_latency(self, wdata):
        self.buf_wlatency = math.ceil(wdata * 8 / self.buf_bitwidth) * self.buf_cycle  # unit: ns, Byte(data)

    def calculate_buf_read_energy(self, rdata):
        self.buf_renergy = self.buf_cycle * (self.buf_rpower / 1e3) * math.ceil(rdata * 8 / self.buf_bitwidth)  # unit: nJ

    def calculate_buf_write_energy(self, wdata):
        self.buf_wenergy = self.buf_cycle * (self.buf_wpower / 1e3) * math.ceil(wdata * 8 / self.buf_bitwidth)  # unit: nJ

    def buf_output(self):
        print("buf_Size:", self.buf_Size, "KB")
        print("buf_Bitwidth:", self.buf_bitwidth, "bit")
        print("buf_Tech:", self.buf_Tech, "nm")
        print("buf_area:", self.buf_area, "um^2")
        print("buf_read_power:", self.buf_rpower, "mW")
        print("buf_read_energy:", self.buf_renergy, "nJ")
        print("buf_read_latency:", self.buf_rlatency, "ns")
        print("buf_write_power:", self.buf_wpower, "mW")
        print("buf_write_energy:", self.buf_wenergy, "nJ")
        print("buf_write_latency:", self.buf_wlatency, "ns")


def buf_test():
    _buf = buffer()
    _buf.calculate_buf_read_power()
    _buf.calculate_buf_read_latency(72 * 1024)
    _buf.calculate_buf_read_energy(72 * 1024)
    _buf.calculate_buf_write_power()
    _buf.calculate_buf_write_latency(72 * 1024)
    _buf.calculate_buf_write_energy(72 * 1024)
    _buf.buf_output()


if __name__ == '__main__':
    buf_test()

"""
buf_Size: 64 KB
buf_Bitwidth: 64 bit
buf_Tech: 32 nm
buf_area: 389828.369 um^2
buf_read_power: 57.26531668419402 mW
buf_dynamic_rpower: 25.15571668419402 mW
buf_read_energy: 572.099315023872 nJ
buf_read_latency: 9990.32832 ns
buf_write_power: 50.51230474714489 mW
buf_dynamic_wpower: 18.402704747144888 mW
buf_write_energy: 504.634508623872 nJ
buf_leakage_power: 32.1096 mW
buf_write_latency: 9990.32832 ns

buf_Size: 32 KB
buf_Bitwidth: 64 bit
buf_Tech: 32 nm
buf_area: 209317.30200000003 um^2
buf_read_power: 52.453328068780415 mW
buf_dynamic_rpower: 34.987668068780415 mW
buf_read_energy: 222.2631575296819 nJ
buf_read_latency: 4237.350912 ns
buf_write_power: 49.060850764318744 mW
buf_dynamic_wpower: 31.595190764318744 mW
buf_write_energy: 207.88804072968193 nJ
buf_leakage_power: 17.46566 mW
buf_write_latency: 4237.350912 ns
"""