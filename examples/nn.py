import math
from itertools import izip
from copy import copy


def sigmoid(x, bias, gain):
    x = max(-60.0, min(60.0, x))
    return 1. / ( 1. + math.exp( -gain * (x - bias) ) )



class ComputeNode:
    def __init__(self, act_func):
        self.value = 0
        self._new_value = 0
        self.inputs = []
        self.act_func = act_func

    def add_input(self, input_node, weight):
        self.inputs.append((input_node, weight))

    def compute(self):
        in_value = 0
        for inp, weight in self.inputs:
            in_value += inp.value * weight

        self._new_value = self.act_func(in_value)

    def flip(self):
        self.value = self._new_value



class InputNode:
    def set(self, input_value):
        self.value = input_value

    def add_input(self, *params):
        pass

    def flip(self):
        pass



class NN:


    def __init__(self):
        self.in_nodes = []
        self.comp_nodes = []
        self.out_nodes = []


    def from_genome(self, genome):

        nodes = {}

        for ng in genome.neuron_genes:
            if ng.gene_type == 'sigmoid':

                # it is important to save these values in separate
                # variables that lambda can later access
                gain = ng.gain
                bias = ng.bias

                node = ComputeNode(act_func = lambda x: sigmoid(x, bias, gain))

                # output nodes go in both compute list and output list
                # hidden nodes only go in compute list
                if ng.layer in ['hidden', 'output']:
                    self.comp_nodes.append(node)
                if ng.layer == 'output':
                    self.out_nodes.append(node)

            elif ng.gene_type == 'input':
                node = InputNode()
                self.in_nodes.append(node)

            nodes[ng.historical_mark] = node

        for cg in genome.connection_genes:
            nodes[cg.mark_to].add_input(nodes[cg.mark_from], cg.weight)

        return self


    def compute(self, inputs):
        # set inputs
        for in_node, in_value in izip(self.in_nodes, inputs):
            in_node.set(in_value)

        # compute
        for _ in range(2):
            for node in self.comp_nodes: node.compute()
            for node in self.comp_nodes: node.flip()

        # get outputs
        return list(out_node.value for out_node in self.out_nodes)