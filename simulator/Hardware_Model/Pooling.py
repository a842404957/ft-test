class Pooling:
    def __init__(self):
        self.Pooling_unit_num = 64
        self.Pooling_Tech = 32
        self.Pooling_area = 240  # unit: um^2
        self.Pooling_power = 0.4 * 1e-3  # unit: W
        self.Pooling_frequency = 1.2  # unit: GSamples/s
        self.Pooling_latency = 1.0 / self.Pooling_frequency  # unit: ns
        self.Pooling_energy = 0  # unit: nJ

        self.calculate_Pooling_energy()

    def calculate_Pooling_energy(self):
        # unit nJ
        self.Pooling_energy = self.Pooling_power * self.Pooling_latency

    def Pooling_output(self):
        print("Pooling_area:", self.Pooling_area, "um^2")
        print("Pooling_power:", self.Pooling_power, "W")
        print("Pooling_latency:", self.Pooling_latency, "ns")
        print("Pooling_energy:", self.Pooling_energy, "nJ")


def Pooling_test():
    _Pooling = Pooling()
    _Pooling.calculate_Pooling_energy()
    _Pooling.Pooling_output()


if __name__ == '__main__':
    Pooling_test()
