from os import path
import sys
import math
import random

from neat import (Mutator, NetworkSpec, GeneSpec,
                NumericParamSpec as PS,
                NominalParamSpec as NPS,
                NEAT, neuron)

from itertools import izip
from operator import itemgetter

from nn_impl import NN


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
neuron_sigma = 0.25                 # mutation sigma for neuron params
conn_sigma = 1.0                    # mutation sigma for connection params

conf = dict(
pop_size = 5,                       # population size
elite_size = 1,                     # size of the elite club
tournament_size = 4,                # size of the selection subsample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.5,       # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.5,   # probability to mutate each single connection in the genome
structural_augmentation_proba = 0.7,# probability to augment the topology of a newly created genome 
structural_removal_proba = 0.0,     # probability to diminish the topology of a newly created genome
speciation_threshold = 0.005        # genomes that are more similar than this value will be considered the same species
)
#### ###### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


## CREATE MUTATION SPEC ##
net_spec = NetworkSpec(
    [
        GeneSpec('input',
            NPS('layer', ['input'], mutable=False)
        ),
        GeneSpec('sigmoid',
            PS('bias', -1., 1., neuron_sigma, mutable=True),
            PS('gain', 0, 1., neuron_sigma, mutable=True),
            NPS('layer', ['hidden'], mutable=False)
        )
    ],
    [
        GeneSpec('connection',
            PS('weight', mutation_sigma=conn_sigma, mean_value = 0., mutable=True))
    ]
)


## CREATE MUTATOR ##
mutator = Mutator(net_spec, allowed_neuron_types = ['sigmoid'])


## CREATE MAIN NEAT OBJECT ##
neat_obj = NEAT(mutator = mutator, **conf)


## CREATE INITIAL GENOTYPE ##
# we specify initial input and output neurons and protect them from removal
init_genome = neat_obj.get_init_genome(
        in1=neuron('input', protected=True, layer='input'),
        in2=neuron('input', protected=True, layer='input'),
        out1=neuron('sigmoid', protected=True, layer='output'),
    )


## CREATE INITIAL GENERATION ##
init_gen = neat_obj.produce_init_generation(init_genome)


## RUN EVOLUTION ##
inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
true_outputs = (0, 1, 1, 0)


def rmse(X, Y):
    return math.sqrt( sum( (x - y)**2 for x, y in izip(X, Y) ) )


def evaluate(genomes):
    fitnesses = []
    for genome in genomes:
        nn = NN().from_genome(genome)

        nn_outputs = list(nn.compute(inp)[0] for inp in inputs)
        fitnesses.append(-rmse(true_outputs, nn_outputs))

    return zip(genomes, fitnesses)


print("\n//// AUGMENT PHASE ////")
current_gen = init_gen
num_generations = 10000

for num_gen in range(num_generations):
    evaluated_gen = evaluate(current_gen)
    current_gen = neat_obj.produce_new_generation(evaluated_gen)

    best_gen, best_fit = sorted(evaluated_gen, key = itemgetter(1))[-1]

    if num_gen % 10 == 0: 
        
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

    if abs(best_fit) < 0.00001: break


# Removal phase
print("\n//// REMOVAL PHASE ////")
num_generations = 1000
conf.update({'structural_removal_proba': 0.7, 'structural_augmentation_proba': 0.5})
neat_obj = NEAT(mutator = mutator, **conf)

for num_gen in range(num_generations):
    evaluated_gen = evaluate(current_gen)
    current_gen = neat_obj.produce_new_generation(evaluated_gen)

    best_gen, best_fit = sorted(evaluated_gen, key = itemgetter(1))[-1]

    if num_gen % 10 == 0: 
        
        print("{}, size = {}N, {}C, fitness = {}"
            .format(
                num_gen,
                len(best_gen.neuron_genes),
                len(best_gen.connection_genes),
                best_fit))




print("Final test:")

final_nn = NN().from_genome(best_gen)

for inp in inputs:
    print("{} -> {}".format(inp, final_nn.compute(inp)[0]))


# write final genome as YAML file
with open('xor_genome.yaml'.format(num_gen), 'w+') as genfile:
            genfile.write(best_gen.to_yaml())