import math
from functools import partial
from itertools import chain


def sigmoid(x, bias, gain):
    x = max(-60.0, min(60.0, x))
    return 1. / ( 1. + math.exp( -gain * (x - bias) ) )


class Node:
    def __init__(self, act_func):
        self.act_func = act_func
        self.upstream = []
        self.value = 0

    def add_upstream_node(self, input_node, weight):
        self.upstream.append((input_node, weight))

    def update_from_value(self, value):
        self.value = self.act_func(value)

    def update(self):
        in_value = sum((up.value * weight for up, weight in self.upstream))
        self.value = self.act_func(in_value)


class NN:
    def __init__(self, layers):
        self.layers = layers


    def compute(self, inputs):
        # set inputs
        for node, in_value in zip(self.layers[0], inputs):
            node.update_from_value(in_value)

        # calc all node output values
        for layer in self.layers[1:]:
            for node in layer:
                node.update()

        # collect outputs
        return tuple(node.value for node in self.layers[-1])


def _pop_layer(graph):
    l, g = [], []

    outputs = tuple(o for _, o in graph)
    for i, o in graph:
        if i in outputs:
            g.append((i, o))
        else:
            l.append((i, o))
    return tuple(l), g


class FeedForwardBuilder:
    def __init__(self, graph):
        layer, graph = _pop_layer(graph)

        layers = []
        while len(layer) > 0:
            layers.append(layer)
            layer, graph = _pop_layer(graph)

        if len(graph) > 0:
            raise RuntimeError("graph cannot be converted to a feed-forward network")

        self.compute_order = {}

        for i, _ in layers[0]:
            self.compute_order[i.type_name] = 0
        for idx, layer in enumerate(layers):
            for _, o in layer:
                self.compute_order[o.type_name] = idx + 1


    def __call__(self, genome):
        nodes_map = {}

        def _make_node(gene, gene_type):
            if gene_type.type_name.startswith('sigm'):
                bias, gain = gene.params

                node = Node(
                    act_func=partial(sigmoid, bias=bias, gain=gain),
                )
                nodes_map[gene.historical_mark] = node
                return node
            else:
                raise RuntimeError("unknown activation for type '{}'".format(gene_type))

        stack = [[]]

        for layer, neurons in genome.layers().items():
            if layer.type_name not in self.compute_order:
                raise RuntimeError(f"layer {layer.type_name} is not recognized by the neural network builder")

            idx = self.compute_order[layer.type_name]
            if idx >= len(stack):
                stack.extend(([] for _ in range(len(stack), idx + 1)))

            stack[idx].extend((_make_node(gene, layer) for gene in neurons.iter_non_empty()))


        for cg in genome.connection_genes():
            weight, = cg.params
            nodes_map[cg.mark_to].add_upstream_node(nodes_map[cg.mark_from], weight)

        return NN(stack)


    def from_yaml(self, y_genome):
        nodes_map = {}

        def _make_node(y_neuron, type_name):
            if type_name.startswith("sigm"):
                node = Node(
                    act_func=partial(sigmoid, bias=y_neuron["bias"], gain=y_neuron["gain"]),
                )
                nodes_map[y_neuron["historical_mark"]] = node
                return node
            else:
                raise RuntimeError("unknown activation for type '{}'".format(type_name))

        stack = [[]]

        for y_neuron in y_genome['neurons']:
            layer = y_neuron["gene_type"]

            if layer not in self.compute_order:
                raise RuntimeError(f"layer {layer} is not recognized by the neural network builder")

            idx = self.compute_order[layer]
            if idx >= len(stack):
                stack.extend(([] for _ in range(len(stack), idx + 1)))

            stack[idx].append(_make_node(y_neuron, layer))

        for y_connection in y_genome['connections']:
            nodes_map[y_connection["mark_to"]].add_upstream_node(
                nodes_map[y_connection["mark_from"]], y_connection["weight"])

        return NN(stack)
