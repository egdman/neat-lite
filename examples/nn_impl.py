import math
from functools import partial
from itertools import chain

try:
    import torch
    from torch.nn.functional import linear as torch_linear
except ImportError:
    torch = None


def sigmoid(x, bias):
    return 1. / (1. + math.exp(- x - bias))


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


class NodeLayer:
    def __init__(self, nodes):
        self._nodes = nodes

    def update_from_values(self, values):
        for value, node in zip(values, self._nodes):
            node.update_from_value(value)

    def update(self):
        for node in self._nodes:
            node.update()

    def get_values_list(self):
        return list(node.value for node in self._nodes)


class PytorchSigmoidLayer:
    def __init__(self, biases):
        self.biases = biases
        self.value = torch.zeros(self.biases.shape, dtype=torch.float64)
        self.upstream = []

    def add_upstream_layer(self, layer, weight_mtx):
        self.upstream.append((layer, weight_mtx))

    def update_from_values(self, values):
        self.value = torch.sigmoid(torch.DoubleTensor(values) + self.biases)

    def update(self):
        if len(self.upstream) == 0:
            self.value = torch.sigmoid(self.biases)

        else:
            layer, weight_mtx = self.upstream[0]
            in_values = torch_linear(layer.value, weight_mtx) + self.biases
            for layer, weight_mtx in self.upstream[1:]:
                in_values += torch_linear(layer.value, weight_mtx)

            self.value = torch.sigmoid(in_values)


    def get_values_list(self):
        return self.value.tolist()


class NN:
    def __init__(self, layers):
        self.layers = layers


    def compute(self, inputs):
        self.layers[0].update_from_values(inputs)

        for layer in self.layers[1:]:
            layer.update()

        return self.layers[-1].get_values_list()


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
            self.compute_order[i.type_id] = 0
        for idx, layer in enumerate(layers):
            for _, o in layer:
                self.compute_order[o.type_id] = idx + 1


    def build_pytorch(self, genome):
        stack = [[]]

        layers_map = {}
        hmarks_to_ids = {}

        for layer, neurons in genome.layers().items():
            if layer.type_id not in self.compute_order:
                raise RuntimeError(f"layer {layer.type_id} is not recognized by the neural network builder")

            stack_idx = self.compute_order[layer.type_id]

            if stack_idx >= len(stack):
                stack.extend(([] for _ in range(len(stack), stack_idx + 1)))

            layer_size = neurons.non_empty_count()
            layer_biases = torch.zeros((layer_size,), dtype=torch.float64)
            for idx, neuron in enumerate(neurons.iter_non_empty()):
                hmarks_to_ids[neuron.historical_mark] = idx
                bias, = neuron.params
                layer_biases[idx] = bias

            layer_impl = PytorchSigmoidLayer(layer_biases)
            layers_map[layer] = layer_impl

            stack[stack_idx].append(layer_impl)

        for channel, conn_genes in genome.channels().items():
            if conn_genes.non_empty_count() == 0:
                continue

            in_layer, out_layer = channel
            in_size = genome.num_neurons_in_layer(in_layer)
            out_size = genome.num_neurons_in_layer(out_layer)

            if in_size == 0 or out_size == 0:
                continue

            weight_mtx = torch.zeros((out_size, in_size), dtype=torch.float64)
            for conn in conn_genes.iter_non_empty():
                src_idx = hmarks_to_ids[conn.mark_from]
                dst_idx = hmarks_to_ids[conn.mark_to]
                weight, = conn.params
                weight_mtx[dst_idx, src_idx] = weight

            layers_map[out_layer].add_upstream_layer(layers_map[in_layer], weight_mtx)

        # flatten the stack into a 1d list:
        stack = tuple(chain(*stack))
        return NN(stack)


    def __call__(self, genome, use_pytorch=False):
        if use_pytorch:
            if torch is None:
                raise RuntimeError("called with use_pytorch=True, but PyTorch is not installed")
            return self.build_pytorch(genome)

        nodes_map = {}

        def _make_node(gene, gene_type):
            if gene_type.type_id.startswith('sigm'):
                bias, = gene.params

                node = Node(
                    act_func=partial(sigmoid, bias=bias)
                )
                nodes_map[gene.historical_mark] = node
                return node
            else:
                raise RuntimeError("unknown activation for type '{}'".format(gene_type))

        stack = [[]]

        for layer, neurons in genome.layers().items():
            if layer.type_id not in self.compute_order:
                raise RuntimeError(f"layer {layer.type_id} is not recognized by the neural network builder")

            idx = self.compute_order[layer.type_id]
            if idx >= len(stack):
                stack.extend(([] for _ in range(len(stack), idx + 1)))

            stack[idx].extend((_make_node(gene, layer) for gene in neurons.iter_non_empty()))


        for cg in genome.connection_genes():
            weight, = cg.params
            nodes_map[cg.mark_to].add_upstream_node(nodes_map[cg.mark_from], weight)

        stack = tuple(NodeLayer(nodes) for nodes in stack)
        return NN(stack)


    def from_yaml(self, y_genome):
        nodes_map = {}

        def _make_node(y_neuron, type_name):
            if type_name.startswith("sigm"):
                node = Node(
                    act_func=partial(sigmoid, bias=y_neuron["bias"])
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
