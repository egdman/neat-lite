import unittest
from neat import NumericParamSpec, NominalParamSpec, GeneSpec, NetworkSpec



class TestSpecs(unittest.TestCase):

	def setUp(self):

		neuron_sigma = 0.5
		connection_sigma = 0.5
		PS = NumericParamSpec

		self.correct_types = set(['Input', 'Simple', 'Sigmoid', 'Oscillator', 'default'])

		self.correct_sigmoid_spec =GeneSpec("Sigmoid",
			PS('bias', -1., 1., neuron_sigma),
			PS('gain', 0., 1., neuron_sigma)
		)

		neuron_specs = [
			GeneSpec("Input"),

			self.correct_sigmoid_spec,

			GeneSpec("Simple",
				PS('bias', -1., 1., neuron_sigma),
				PS('gain', 0., 1., neuron_sigma)
			),

			GeneSpec("Oscillator",
				PS("period", 0., 10., neuron_sigma),
	            PS("phase_offset", 0., 3.14, neuron_sigma),
	            PS("amplitude", 0., 10000., neuron_sigma)
	        ),
		]

		connection_specs = [
			GeneSpec("default",
				PS('weight', mutation_sigma=connection_sigma, mean_value=0.)
			),
		]

		self.netspec = NetworkSpec(neuron_specs, connection_specs)



	def test_numeric_spec(self):
		print("Testing NumericParamSpec")
		
		ps = NumericParamSpec('numpar', -87, 17, mutation_sigma = 0.5, mean_value = 5.)
		self.assertEquals(
			-87,
			ps.put_within_bounds(-999),
			msg = "NumericParamSpec.put_within_bounds() : Failed")


		self.assertEquals(
			17,
			ps.put_within_bounds(999),
			msg = "NumericParamSpec.put_within_bounds() : Failed")



	def test_network_spec(self):
		print("Testing NetworkSpec")

		correct_types = set(['Input', 'Simple', 'Sigmoid', 'Oscillator', 'default'])

		self.assertEquals(
			self.correct_types,
			set(self.netspec.gene_types()),
			msg="NetworkSpec.gene_types() method gave incorrect result"
			)


		self.assertEquals(
			self.correct_types,
			set(gt for gt in self.netspec),
			msg=("Iteration through NetworkSpec instance did not work"
				" ([gt for gt in self.netspec] gave incorrect result)")
		)


		self.assertEquals(
			self.netspec['Sigmoid'],
			self.correct_sigmoid_spec,
			msg = "NetworkSpec's [] operator (__getitem__) gave incorrect result"
			)

		
