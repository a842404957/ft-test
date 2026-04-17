class ADC:
	def __init__(self):
		# reference: Sparse ReRAM Engine: Joint Exploration of Activation and Weight Sparsity in Compressed Neural Networks
		self.ADC_area = 1200  # unit: um^2
		self.ADC_precision = 6  # bit
		self.ADC_power = 0.6425 * 1e-3  # unit: W
		self.ADC_sample_rate = 1.2  # unit: GSamples/s
		self.ADC_latency = 0  # unit: ns
		self.ADC_energy = 0  # unit: nJ

		self.calculate_ADC_latency()
		self.calculate_ADC_energy()

	def calculate_ADC_latency(self):
		# unit: ns
		self.ADC_latency = 1 / self.ADC_sample_rate * (self.ADC_precision + 2)

	def calculate_ADC_energy(self):
		# unit: nJ
		self.ADC_energy = self.ADC_latency * self.ADC_power

	def ADC_output(self):
		print("ADC_area:", self.ADC_area, "um^2")
		print("ADC_precision:", self.ADC_precision, "bit")
		print("ADC_power:", self.ADC_power, "W")
		print("ADC_sample_rate:", self.ADC_sample_rate, "Gbit/s")
		print("ADC_latency:", self.ADC_latency, "ns")
		print("ADC_energy:", self.ADC_energy, "nJ")


def ADC_test():
	_ADC = ADC()
	_ADC.calculate_ADC_latency()
	_ADC.calculate_ADC_energy()
	_ADC.ADC_output()


if __name__ == '__main__':
	ADC_test()
