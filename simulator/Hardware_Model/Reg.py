class reg:
	def __init__(self):
		self.reg_tech = 32  # unit: nm
		self.size = 256  # unit: B
		self.reg_frequency = 1.2  # unit: GSamples/s
		self.reg_latency = 1.0 / self.reg_frequency  # unit: ns
		self.reg_power = 0.23 * 1e-3  # unit: W
		self.reg_area = 770  # unit: um^2
		self.reg_energy = 0  # unit: nJ

		self.calculate_reg_energy()

	def calculate_reg_energy(self):
		self.reg_energy = self.reg_latency * self.reg_power

	def reg_output(self):
		# unit: nJ
		print("reg_area:", self.reg_area, "um^2")
		print("reg_power:", self.reg_power, "W")
		print("reg_latency:", self.reg_latency, "ns")
		print("reg_energy:", self.reg_energy, "nJ")


def reg_test():
	_reg = reg()
	_reg.calculate_reg_energy()
	_reg.reg_output()


if __name__ == '__main__':
	reg_test()