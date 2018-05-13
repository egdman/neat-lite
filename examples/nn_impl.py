import math
from itertools import chain
from functools import partial
from copy import copy

try:
    from itertools import izip as zip
except ImportError:
    pass


def sigmoid(x, bias, gain):
    x = max(-60.0, min(60.0, x))
    return 1. / ( 1. + math.exp( -gain * (x - bias) ) )


class Node(object):
    def __init__(self):
        self.value = 0
        self._new_value = 0
        self.inputs = []

    def reset(self, value):
        self.value = value
        self._new_value = value

    def flip(self):
        self.value = self._new_value

    def add_input(self, input_node, weight):
        self.inputs.append((input_node, weight))


class InputNode(Node): pass


class ComputeNode(Node):
    def __init__(self, act_func):
        super(ComputeNode, self).__init__()
        self.act_func = act_func

    def compute(self):
        in_value = 0
        for inp, weight in self.inputs:
            in_value += inp.value * weight

        self._new_value = self.act_func(in_value)


class NN:


    def __init__(self):
        self.in_nodes = []
        self.comp_nodes = []
        self.out_nodes = []


    def from_genome(self, genome):

        nodes = {}

        for ng in genome.neuron_genes:
            if ng.gene_type == 'sigmoid':

                node = ComputeNode(
                    act_func = partial(sigmoid, bias = ng.bias, gain = ng.gain)
                )

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


    def reset(self):
        # reset node values
        for node in chain(self.in_nodes, self.comp_nodes, self.out_nodes):
            node.reset(0)


    def compute(self, inputs):
        # set inputs
        for in_node, in_value in zip(self.in_nodes, inputs):
            in_node.reset(in_value)

        # compute
        for _ in range(2):
            for node in self.comp_nodes: node.compute()
            for node in self.comp_nodes: node.flip()

        # get outputs
        return list(out_node.value for out_node in self.out_nodes)