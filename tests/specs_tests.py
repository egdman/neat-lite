import unittest
from neat import ParamSpec, NominalParamSpec, GeneSpec, NetworkSpec



class TestSpecs(unittest.TestCase):

	def setUp(self):

		neuron_sigma = 0.5
		connection_sigma = 0.5
		PS = ParamSpec

		self.correct_types = set(['Input', 'Simple', 'Sigmoid', 'Oscillator', 'default'])

		self.correct_sigmoid_spec =GeneSpec("Sigmoid",
			PS('bias', -1., 1., gen_uniform(), mut_gauss(neuron_sigma)),
			PS('gain', 0., 1., gen_uniform(), mut_gauss(neuron_sigma))
		)

		neuron_specs = [
			GeneSpec("Input"),

			self.correct_sigmoid_spec,

			GeneSpec("Simple",
				PS('bias', -1., 1., gen_uniform(), mut_gauss(neuron_sigma)),
				PS('gain', 0., 1., gen_uniform(), mut_gauss(neuron_sigma))
			),

			GeneSpec("Oscillator",
				PS("period", 0., 10., gen_uniform(), mut_gauss(neuron_sigma)),
	            PS("phase_offset", 0., 3.14, gen_uniform(), mut_gauss(neuron_sigma)),
	            PS("amplitude", 0., 10000., gen_uniform(), mut_gauss(neuron_sigma))
	        ),
		]

		connection_specs = [
			GeneSpec("default",
				PS('weight', None, None, gen_gauss(0, connection_sigma), mut_gauss(connection_sigma))
			),
		]

		self.netspec = NetworkSpec(neuron_specs, connection_specs)


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

		
