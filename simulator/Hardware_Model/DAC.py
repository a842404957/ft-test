class DAC:
	def __init__(self):
		# Data reference: ISAAC: A Convolutional Neural Network Accelerator with In-Situ Analog Arithmetic in Crossbars
		self.DAC_area = 0.166  # unit: um^2
		self.DAC_precision = 1  # bit
		self.DAC_power = 0.0039 * 1e-3  # unit: W
		self.DAC_sample_rate = 1.2  # unit: GSamples/s
		self.DAC_latency = 0  # unit: ns
		self.DAC_energy = 0  # unit: nJ

		self.calculate_DAC_latency()
		self.calculate_DAC_energy()

	def calculate_DAC_latency(self):
		# unit: ns
		self.DAC_latency = 1 / self.DAC_sample_rate * (self.DAC_precision + 2)

	def calculate_DAC_energy(self):
		# unit: nJ
		self.DAC_energy = self.DAC_latency * self.DAC_power

	def DAC_output(self):
		print("DAC_area:", self.DAC_area, "um^2")
		print("DAC_precision:", self.DAC_precision, "bit")
		print("DAC_power:", self.DAC_power, "W")
		print("DAC_sample_rate:", self.DAC_sample_rate, "Gbit/s")
		print("DAC_energy:", self.DAC_energy, "nJ")
		print("DAC_latency:", self.DAC_latency, "ns")


def DAC_test():
	_DAC = DAC()
	_DAC.calculate_DAC_energy()
	_DAC.DAC_output()


if __name__ == '__main__':
	DAC_test()