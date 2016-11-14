from os import path
import sys
import math
import random

sys.path.append(path.dirname(path.abspath(__file__)) + '/../')

from neat import (GeneticEncoding, Mutator,
                NetworkSpec, GeneSpec,
                NumericParamSpec as PS,
                NominalParamSpec as NPS,
                NEAT)

from itertools import izip
from operator import itemgetter

from nn import NN


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
neuron_sigma = 0.25                 # mutation sigma for neuron params
conn_sigma = 1.0                    # mutation sigma for connection params

conf = dict(
pop_size = 5,                       # population size
elite_size = 1,                     # size of the elite club
tournament_size = 4,                # size of the selection subsample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.5,       # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.5,   # probability to mutate each single connection in the genome
structural_augmentation_proba = 0.8,# probability to augment the topology of a newly created genome 
structural_removal_proba = 0.0,     # probability to diminish the topology of a newly created genome
speciation_threshold = 0.005        # genomes that are more similar than this value will be considered the same species
)
#### ###### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


## CREATE MUTATION SPEC ##
net_spec = NetworkSpec(
    [
        GeneSpec('input',
            NPS('layer', ['input'])
        ),
        GeneSpec('sigmoid',
            PS('bias', -1., 1., neuron_sigma),
            PS('gain', 0, 1., neuron_sigma),
            NPS('layer', ['hidden'])
        )
    ],
    [
        GeneSpec('connection',
            PS('weight', mutation_sigma=conn_sigma, mean_value = 0.))
    ]
)


## CREATE MUTATOR ##
mutator = Mutator(net_spec,
    allowed_neuron_types = ['sigmoid'],
    mutable_params = {'sigmoid': ['bias', 'gain'], 'input': []})


## CREATE MAIN NEAT OBJECT ##
neat_obj = NEAT(mutator = mutator, **conf)


## CREATE INITIAL GENOTYPE ##
init_ge = GeneticEncoding()

# add input neurons and protect them from deletion
mutator.protect_gene(mutator._add_neuron(init_ge, neuron_type='input', layer='input'))
mutator.protect_gene(mutator._add_neuron(init_ge, neuron_type='input', layer='input'))

# add output neuron and protect it from deletion
mutator.protect_gene(mutator._add_neuron(init_ge, neuron_type='sigmoid', layer='output'))


## CREATE INITIAL POPULATION ##
init_pop = []
for _ in range(pop_size):
    mutated_ge = init_ge.copy()
    
    mutator.mutate_connection_params(
        genotype=mutated_ge,
        probability=param_mut_proba)

    mutator.mutate_neuron_params(
        genotype=mutated_ge,
        probability=param_mut_proba)

    init_pop.append(mutated_ge)


## RUN EVOLUTION ##
def evaluate(genomes):
    xor_inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
    xor_outputs = (0, 1, 1, 0)

    fitnesses = []
    for genome in genomes:
        fitn = 1.
        nn = NN().from_genome(genome)
        for inp, true_outp in izip(xor_inputs, xor_outputs):
            # evaluate error
            nn_outp = nn.compute(inp)[0]
            fitn -= (nn_outp - true_outp) ** 2

        fitnesses.append(fitn)

    return zip(genomes, fitnesses)



current_gen = init_pop
num_generations = 1000

for num_gen in range(num_generations):
    evaluated_gen = evaluate(current_gen)
    current_gen = produce_new_generation(evaluated_gen)

    if num_gen % 10 == 0: 
        best_gen, best_fit = sorted(evaluated_gen, key = itemgetter(1))[-1]
        print("{}, size = {}N, {}C, fitness = {}"
            .format(
                num_gen,
                len(best_gen.neuron_genes),
                len(best_gen.connection_genes),
                best_fit))

        # # write genome as YAML file
        # if num_gen % 50 == 0:
        #     with open('genomes/gen_{}.yaml'.format(num_gen), 'w+') as genfile:
        #         genfile.write(best_gen.to_yaml())

        if 1. - best_fit < 0.000001: break 





