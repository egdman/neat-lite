import math
from itertools import izip, chain
from copy import copy


def sigmoid(x, bias, gain):
    x = max(-60.0, min(60.0, x))
    return 1. / ( 1. + math.exp( -gain * (x - bias) ) )


class Node(object):
    def __init__(self):
        self.value = 0
        self._new_value = 0

    def reset(self, value):
        self.value = value
        self._new_value = value

    def flip(self):
        self.value = self._new_value


class InputNode(Node): pass


class ComputeNode(Node):
    def __init__(self, act_func):
        super(ComputeNode, self).__init__()
        self.inputs = []
        self.act_func = act_func

    def add_input(self, input_node, weight):
        self.inputs.append((input_node, weight))

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
                    act_func = lambda x, bias=ng.bias, gain=ng.gain: sigmoid(x, bias, gain)
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


    def compute(self, inputs):
        # reset node values
        for node in chain(self.in_nodes, self.comp_nodes, self.out_nodes):
            node.reset(0)

        # set inputs
        for in_node, in_value in izip(self.in_nodes, inputs):
            in_node.reset(in_value)

        # compute
        for _ in range(2):
            for node in self.comp_nodes: node.compute()
            for node in self.comp_nodes: node.flip()

        # get outputs
        return list(out_node.value for out_node in self.out_nodes)