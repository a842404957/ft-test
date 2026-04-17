import os
import configparser as cp
test_SimConfig_path = os.path.join(os.path.dirname(os.getcwd()), "SimConfig.ini")


class crossbar:
	def __init__(self, SimConfig_path):
		xbar_config = cp.ConfigParser()
		xbar_config.read(SimConfig_path, encoding='UTF-8')
		self.xbar_size = list(map(int, xbar_config.get('Crossbar level', 'Xbar_Size').split(',')))
		self.xbar_row = int(self.xbar_size[0])
		self.xbar_column = int(self.xbar_size[1])

		self.xbar_tech = 32  # unit: nm
		self.device_read_latency = 3.16  # unit: ns
		self.device_read_power = 0  # unit: W

		self.xbar_area = 0
		self.xbar_read_power = 0
		self.xbar_read_latency = 0
		self.xbar_read_energy = 0

		self.xbar_utilization = 1.0

		self.calculate_xbar_read_power()

	def calculate_xbar_area(self):
		# Area unit: um^2
		WL_ratio = 3  # WL_ratio is the technology parameter W/L of the transistor
		self.xbar_area = 3 * (WL_ratio + 1) * self.xbar_row * self.xbar_column * self.xbar_tech**2 * 1e-6

	def calculate_xbar_read_latency(self):
		# unit: ns
		size = self.xbar_row * self.xbar_column / 1024 / 8  # KB
		wire_latency = 0.001 * (0.0002 * size ** 2 + 5 * 10 ** -6 * size + 4 * 10 ** -14)  # ns
		self.xbar_read_latency = self.device_read_latency + wire_latency

	def calculate_xbar_read_power(self):
		# unit: W
		self.xbar_read_power = (self.xbar_row / 128) * (self.xbar_column / 128) * 0.3 * 1e-3

	def calculate_xbar_read_energy(self):
		# unit: nJ
		self.xbar_read_energy = self.xbar_read_power * self.xbar_read_latency

	def xbar_output(self):
		print("crossbar_size:", self.xbar_size)
		print("crossbar_area", self.xbar_area, "um^2")
		print("crossbar_utilization_rate", self.xbar_utilization)
		print("crossbar_read_power:", self.xbar_read_power, "W")
		print("crossbar_read_latency:", self.xbar_read_latency, "ns")
		print("crossbar_read_energy:", self.xbar_read_energy, "nJ")

	
def xbar_test():
	print("load file:", test_SimConfig_path)
	_xbar = crossbar(test_SimConfig_path)
	print('------------')
	_xbar.calculate_xbar_area()
	_xbar.calculate_xbar_read_latency()
	_xbar.calculate_xbar_read_power()
	_xbar.calculate_xbar_read_energy()
	_xbar.xbar_output()


if __name__ == '__main__':
	xbar_test()
