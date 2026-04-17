class adder:
	def __init__(self, bitwidth=16):
		self.adder_tech = 32
		if bitwidth is None:
			self.adder_bitwidth = 16  # unit: bit
		else:
			self.adder_bitwidth = bitwidth
		self.adder_frequency = 1.2  # unit: GSamples/s
		self.adder_latency = 1.0 / self.adder_frequency  # unit: ns
		self.adder_area = 0  # unit: um^2
		self.adder_power = 0  # unit: W
		self.adder_energy = 0  # unit: nJ

		self.calculate_adder_area()
		self.calculate_adder_power()
		self.calculate_adder_energy()

	def calculate_adder_area(self):
		# unit: um^2
		self.adder_area = 10 * 14 * 32 * 32 / 1e6 * 16

	def calculate_adder_power(self):
		# unit: W
		self.adder_power = 0.05 * 1e-3 * (self.adder_bitwidth / 16)

	def calculate_adder_energy(self):
		# unit: nJ
		self.adder_energy = self.adder_latency * self.adder_power

	def adder_output(self):
		print("adder_area:", self.adder_area, "um^2")
		print("adder_bitwidth:", self.adder_bitwidth, "bit")
		print("adder_power:", self.adder_power, "W")
		print("adder_latency:", self.adder_latency, "ns")
		print("adder_energy:", self.adder_energy, "nJ")


def adder_test():
	_adder = adder()
	_adder.calculate_adder_area()
	_adder.calculate_adder_power()
	_adder.calculate_adder_energy()
	_adder.adder_output()


if __name__ == '__main__':
	adder_test()