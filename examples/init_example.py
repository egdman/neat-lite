from os import path
import sys

here_dir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(here_dir, '..'))

from neat import (GeneSpec,
                ParamSpec as PS, gen_uniform, gen_gauss, mut_gauss,
                Mutator, neuron, connection, default_gene_factory)


neuron_sigma = 0.25 # gaussian distribution sigma for neuron params mutation
conn_sigma = 1.0    # gaussian distribution sigma for connection params mutation


sigmoid_neuron_spec = GeneSpec('sigmoid',
    PS('bias', gen_uniform(), mut_gauss(neuron_sigma)).with_bounds(-1., 1.),
    PS('gain', gen_uniform(), mut_gauss(neuron_sigma)).with_bounds(0., 1.),
    PS('layer', lambda *a: 'hidden'),
)
connection_spec = GeneSpec('default',
    PS('weight', gen_gauss(0, conn_sigma), mut_gauss(conn_sigma)),
)


mutator = Mutator(
    neuron_factory=default_gene_factory(sigmoid_neuron_spec),
    connection_factory=default_gene_factory(connection_spec),
)

sigmoid_params = sigmoid_neuron_spec.get_random_parameters()
sigmoid_params['layer'] = 'output'

genome = mutator.produce_genome(
    in1=neuron('input', non_removable=True, layer='input'),
    in2=neuron('input', non_removable=True, layer='input'),
    out1=neuron('sigmoid', non_removable=True, **sigmoid_params),
    connections=(
        connection('default', src='in1', dst='out1', weight=0.33433),
        connection('default', src='in2', dst='out1', weight=-0.77277),
    )
)


with open('init_genome.yaml', 'w+') as outfile:
    outfile.write(genome.to_yaml())
