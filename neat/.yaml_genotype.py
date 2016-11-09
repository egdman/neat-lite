# Revolve
import yaml
from revolve.spec.msgs import Body, BodyPart, NeuralNetwork
from revolve.spec.exception import err

# ToL
from ..encoding import GeneticEncoding, Neuron, NeuronGene, ConnectionGene, Mutator



def parse_neurons(neurons, genotype, mutator):
    # map old marks to new marks:
    neuron_marks = {}
    for neuron_gene in neurons:
        id = neuron_gene['id']
        layer = neuron_gene['layer']
        neuron_type = neuron_gene['type']
        part_id = neuron_gene['part_id']
        params = neuron_gene['params']
        old_mark = neuron_gene['hist_mark']
        enabled = neuron_gene['enabled']

        if enabled:
            neuron = Neuron(neuron_id=id,
                            layer=layer,
                            neuron_type=neuron_type,
                            body_part_id=part_id,
                            neuron_params=params)

            new_mark = mutator.add_neuron(neuron, genotype)
            neuron_marks[old_mark] = new_mark

    return neuron_marks


def parse_connections(connections, genotype, mutator, neuron_marks):
    for conn in connections:
        enabled = conn['enabled']
        hist_mark = conn['hist_mark']
        from_mark = conn['from']
        to_mark = conn['to']
        weight = conn['weight']

        # this is optional field
        socket = conn.get('socket', None)

        if enabled:
            mutator.add_connection(mark_from=neuron_marks[from_mark],
                                   mark_to=neuron_marks[to_mark],
                                   weight=weight,
                                   genotype=genotype,
                                   socket=socket)

def get_neuron_genes(neurons):
    '''
    This method parses neurons keeping their original historical marks
    :param neurons:
    :return:
    '''

    neuron_genes = []
    for neuron_info in neurons:
        id = neuron_info['id']
        layer = neuron_info['layer']
        neuron_type = neuron_info['type']
        part_id = neuron_info['part_id']
        params = neuron_info['params']
        mark = neuron_info['hist_mark']
        enabled = neuron_info['enabled']

        neuron =  Neuron(
                neuron_id=id,
                layer=layer,
                neuron_type=neuron_type,
                body_part_id=part_id,
                neuron_params=params)

        neuron_gene = NeuronGene(neuron=neuron, innovation_number=mark, enabled=enabled)
        neuron_genes.append(neuron_gene)
    return neuron_genes


def get_connection_genes(connections):

    '''
    This method parses connections keeping their original historical marks
    :param connections:
    :return: connection_genes
    '''

    conn_genes = []
    for conn_info in connections:
        enabled = conn_info['enabled']
        hist_mark = conn_info['hist_mark']
        from_mark = conn_info['from']
        to_mark = conn_info['to']
        weight = conn_info['weight']

        # this is optional field
        socket = conn_info.get('socket', None)

        connection_gene = ConnectionGene(
            mark_from=from_mark,
            mark_to=to_mark,
            weight=weight,
            innovation_number=hist_mark,
            enabled=enabled,
            socket=socket
        )
        conn_genes.append(connection_gene)
    return conn_genes


def yaml_to_genotype(yaml_stream, brain_spec, keep_historical_marks=False):
    obj = yaml.load(yaml_stream)
    genotype = GeneticEncoding()
    neurons = obj['neurons']
    connections = obj['connections']

    if keep_historical_marks:
        neuron_genes = get_neuron_genes(neurons)
        connection_genes = get_connection_genes(connections)
        for neuron_gene in neuron_genes:
            genotype.add_neuron_gene(neuron_gene)
        for connection_gene in connection_genes:
            genotype.add_connection_gene(connection_gene)

    else:
        mutator = Mutator(brain_spec)
        neuron_marks = parse_neurons(neurons, genotype, mutator)
        parse_connections(connections, genotype, mutator, neuron_marks)
    return genotype




