from os import path
import sys
import math

sys.path.append(path.dirname(path.abspath(__file__)) + '/../')

from neat import Mutator, crossover, NetworkSpec, GeneSpec, NumericParamSpec as PS
from itertools import izip


######## NEURAL NETWORK IMPLEMENTATION ########
def sigmoid(x, bias, gain):
	x = max(-60.0, min(60.0, x))
	return 1. / ( 1. + math.exp( -gain * (x - bias) ) )


class InputNode:
	def set(self, input_value)
		self.value = input_value
		self.ready = True

	def reset(self):
		self.ready = False


class ComputeNode:
	def __init__(self, act_func):
		self.value = 0
		self.ready = False
		self.inputs = []
		self.act_func

	def add_input(input_node):
		self.inputs.append(input_node)

	def compute(self):
		in_value = 0
		for inp, weight in self.inputs:
			if not inp.ready: return
			in_value += inp.value * weight

		self.ready = True
		self.value = self.act_func(in_value)

	def reset(self):
		self.ready = False



class NN:
	def __init__(self, genome):
		self.in_nodes = []
		self.comp_nodes = []
		self.out_nodes = []
		nodes = {}

		for ng in genome.neuron_genes:
			if ng.gene_type == 'sigmoid':
				node = ComputeNode(act_func = lambda x: sigmoid(x, ng.bias, ng.gain))
				self.comp_nodes.append(node)
			else:
				node = InputNode()
				self.in_nodes.append(node)

			nodes[ng.historical_mark] = node

		for cg in genome.connection_genes:
			nodes[cg.mark_to].add_input(nodes[cg.mark_from])

		#### create list of output nodes ####




	def compute(self, inputs):
		for in_node, in_value in izip(self.in_nodes, inputs):
			in_node.set(in_value)


		# compute
		done = False
		while not done:
			done = True
			for node in self.comp_nodes:
				node.compute()
				if not node.ready: done = False


		return list(out_node.value for out_node in self.out_nodes)





######## NEAT USAGE ########

def eval(genomes):
	xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
	xor_outputs = (0, 1, 1, 0)

	fitnesses = []
	for genome in genomes:
		fitn = 1.
		nn = NN(genome)
		for inp, true_outp in izip(xor_inputs, xor_outputs):
			# evaluate error
			fitn -= (nn.compute(inp) - true_outp) ** 2

		fitnesses.append(fitn)

	return zip(genomes, fitnesses)


	





netspec = NetworkSpec(
	[
		GeneSpec('input'),
		GeneSpec('sigmoid',
			PS('bias', -1., 1., neuron_sigma),
			PS('gain', 0, 1., neuron_sigma))
	],
	[
		GeneSpec('connection',
			PS('weight', mutation_sigma=conn_sigma, mean_value = 0.))
	]
)

mutator = Mutator(net_spec)