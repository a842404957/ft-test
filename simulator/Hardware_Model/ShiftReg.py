class shiftreg:
	def __init__(self, max_shiftbase=16):
		self.shiftreg_tech = 32  # unit: nm
		if max_shiftbase is None:
			self.max_shiftbase = 16  # unit: bit
		else:
			self.max_shiftbase = max_shiftbase  # unit: bit
		self.shiftreg_frequency = 1.2  # unit: GSamples/s
		self.shiftreg_latency = 1.0 / self.shiftreg_frequency  # unit: ns

		self.shiftreg_area = 0  # unit: um^2
		self.shiftreg_power = 0  # unit: W
		self.shiftreg_energy = 0  # unit: nJ

		self.calculate_shiftreg_power()

	def calculate_shiftreg_area(self):
		# unit: um^2
		self.shiftreg_area = 1.42 * pow((self.shiftreg_tech / 65), 2)

	def calculate_shiftreg_power(self):
		# unit: W
		self.shiftreg_power = 0.05 * 1e-3 * (self.max_shiftbase / 16)

	def calculate_shiftreg_energy(self):
		self.shiftreg_energy = self.shiftreg_latency * self.shiftreg_power

	def shiftreg_output(self):
		print("shiftreg_area:", self.shiftreg_area, "um^2")
		print("shiftreg_bitwidth:", self.max_shiftbase, "bit")
		print("shiftreg_power:", self.shiftreg_power, "W")
		print("shiftreg_latency:", self.shiftreg_latency, "ns")
		print("shiftreg_energy:", self.shiftreg_energy, "nJ")


def shiftreg_test():
	_shiftreg = shiftreg()
	_shiftreg.calculate_shiftreg_area()
	_shiftreg.calculate_shiftreg_power()
	_shiftreg.calculate_shiftreg_energy()
	_shiftreg.shiftreg_output()


if __name__ == '__main__':
	shiftreg_test()