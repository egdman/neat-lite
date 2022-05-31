from os import path
import sys

here_dir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(here_dir, '..'))

from neat import (GeneSpec,
                ParamSpec as PS, gen, mut, bounds,
                Mutator, neuron, connection, default_gene_factory)


neuron_sigma = 0.25 # gaussian distribution sigma for neuron params mutation
conn_sigma = 1.0    # gaussian distribution sigma for connection params mutation


input_neuron_spec = GeneSpec('input')
sigmoid_neuron_spec = GeneSpec('sigmoid',
    PS('bias', gen.uniform(), mut.gauss(neuron_sigma), bounds(-1., 1.)),
    PS('gain', gen.uniform(), mut.gauss(neuron_sigma), bounds(0., 1.)),
    PS('layer', gen.const('hidden')),
)
connection_spec = GeneSpec('default',
    PS('weight', gen.gauss(0, conn_sigma), mut.gauss(conn_sigma)),
)


mutator = Mutator(
    neuron_factory=default_gene_factory(sigmoid_neuron_spec),
    connection_factory=default_gene_factory(connection_spec),
)

sigmoid_params = sigmoid_neuron_spec.generate_parameter_values()
sigmoid_params['layer'] = 'output'

genome = mutator.produce_genome(
    in1=neuron(input_neuron_spec, non_removable=True, layer='input'),
    in2=neuron(input_neuron_spec, non_removable=True, layer='input'),
    out1=neuron(sigmoid_neuron_spec, non_removable=True, **sigmoid_params),
    connections=(
        connection(connection_spec, src='in1', dst='out1', weight=0.33433),
        connection(connection_spec, src='in2', dst='out1', weight=-0.77277),
    )
)
print(genome.to_yaml())
