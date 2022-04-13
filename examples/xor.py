from os import path
import sys
import math
import random
import heapq
from operator import itemgetter

try:
    from itertools import izip as zip
except ImportError:
    pass

try:
    AnyError = StandardError
except NameError:
    AnyError = Exception

try:
    range = xrange
except NameError:
    pass


here_dir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(here_dir, '..'))

from neat import (Mutator, NEAT, GeneSpec, ParamSpec as PS, validate_genome,
    gen_uniform, gen_gauss, mut_gauss,
    neuron, connection, default_gene_factory)

from nn_impl import NN


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
neuron_sigma = 0.25                 # mutation sigma for neuron params
conn_sigma = 10.                    # mutation sigma for connection params

popul = 10
elite_num = 1 # best performing genomes in a generation will be copied without change to the next generation

conf = dict(
selection_sample_size = int(.75 * popul), # size of the selection sample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.5,       # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.5,   # probability to mutate each single connection in the genome
topology_augmentation_proba = 0,    # probability to augment the topology of a newly created genome
topology_reduction_proba = 0,       # probability to reduce the topology of a newly created genome
speciation_threshold = 0.2          # genomes that are more similar than this value will be considered the same species
)
#### ###### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


## CREATE MUTATION SPEC ##

input_neuron_spec = GeneSpec(
    'input',
    PS('layer', lambda *a: 'input'),
)
sigmoid_neuron_spec = GeneSpec(
    'sigmoid',
    PS('layer', lambda *a: 'hidden'),
    PS('bias', gen_uniform(), mut_gauss(neuron_sigma)).with_bounds(-1., 1.),
    PS('gain', gen_uniform(), mut_gauss(neuron_sigma)).with_bounds(0., 1.),
)
connection_spec = GeneSpec(
    'connection',
    PS('weight', gen_gauss(0, conn_sigma), mut_gauss(conn_sigma)),
)

## CREATE MUTATOR ##
mutator = Mutator(
    neuron_factory=default_gene_factory(sigmoid_neuron_spec),
    connection_factory=default_gene_factory(connection_spec),
    pure_input_types=('input',),
)

## CREATE MAIN NEAT OBJECT ##
neat_obj = NEAT(
    topology_mutator=mutator,
    neuron_specs=(input_neuron_spec, sigmoid_neuron_spec),
    connection_specs=(connection_spec,),
    **conf)


def produce_new_generation(neat, genome_fitness_list):
    for _ in range(len(genome_fitness_list) - elite_num):
        g = neat.produce_new_genome(genome_fitness_list)
        # validate_genome(g, 'invalid genome')
        yield g

    # bringing the best parents into next generation:
    best_parents = heapq.nlargest(elite_num, genome_fitness_list, key=itemgetter(1))
    for genome, _ in best_parents:
        yield genome


## INPUTS AND CORRECT OUTPUTS FOR THE NETWORK ##
inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
true_outputs = (0, .75, .75, 0)

def rmse(X, Y):
    return math.sqrt( sum( (x - y)**2 for x, y in zip(X, Y) ) )


def evaluate(genome):
    nn = NN().from_genome(genome)
    nn_outputs = []
    for inp in inputs:
        nn.reset() # reset network otherwise it will remember previous state
        nn_outputs.append(nn.compute(inp)[0])

    outp0, outp1, outp2, outp3 = nn_outputs
    fitness = abs(outp0 - outp1) * abs(outp0 - outp2) * abs(outp3 - outp1) * abs(outp3 - outp2)
    # fitness = -rmse(true_outputs, nn_outputs)
    return fitness


def complexity(genomes):
    return (sum((len(ge.neuron_genes)) for ge in genomes),
    sum((len(ge.connection_genes)) for ge in genomes))

def get_stats(genomes, best_gen, best_fitness):
    n_neurons, n_conns = complexity(genomes)
    return ("evals: {}, best genome has: {}N, {}C, complexity: {}N, {}C, best fitness = {}"
        .format(evals_num,
            len(best_gen.neuron_genes), len(best_gen.connection_genes),
            n_neurons, n_conns, best_fitness))

def next_gen(current_gen):
    # evaluated_gen = list(evaluate(current_gen))
    evaluated_gen = list((genome, evaluate(genome)) for genome in current_gen)
    next_gen = list(produce_new_generation(neat_obj, evaluated_gen))
    best_genome, best_fitness = max(evaluated_gen, key=itemgetter(1))
    return next_gen, best_genome, best_fitness


num_epochs = 20 #1000000000
gens_per_epoch = 250
aug_proba = .9
red_proba = .9

evals_num = 0

augment_gens_per_epoch = int(.4 * gens_per_epoch)
reduct_gens_per_epoch = gens_per_epoch - augment_gens_per_epoch


## CREATE INITIAL GENOME ##
# we specify initial input and output neurons and protect them from removal
sigmoid_params = sigmoid_neuron_spec.get_random_parameters()
sigmoid_params['layer'] = 'output'

init_genome = mutator.produce_genome(
    in1=neuron('input', non_removable=True, layer='input'),
    in2=neuron('input', non_removable=True, layer='input'),
    out1=neuron('sigmoid', non_removable=True, **sigmoid_params),
    connections=(
        # connection('connection', src='in1', dst='out1'),
        # connection('connection', src='in2', dst='out1')
    )
)

# init_genome = mutator.produce_genome(
#     in1=neuron('input', non_removable=True, layer='input'),
#     in2=neuron('input', non_removable=True, layer='input'),
#     out1=neuron('sigmoid', non_removable=True, gain=1, bias=-0.2, layer='output'),
#     h11 = neuron('sigmoid', gain=1, bias=-0.6, layer='hidden'),
#     h11066=neuron('sigmoid', gain=1, bias=-1, layer='hidden'),
#     h11242=neuron('sigmoid', gain=1, bias=1, layer='hidden'),
#     connections=(
#         connection('connection', src='in1', dst='h11', weight=-125),
#         connection('connection', src='in1', dst='h11066', weight=-40),
#         connection('connection', src='in2', dst='h11', weight=-45),
#         connection('connection', src='in2', dst='out1', weight=35),
#         connection('connection', src='h11', dst='out1', weight=125),
#         connection('connection', src='h11066', dst='out1', weight=-50),
#         connection('connection', src='h11242', dst='out1', weight=-60),
#     )
# )


## CREATE INITIAL GENERATION ##
def copy_with_mutation(neat, source_genome):
    return neat.topology_augmentation_step(neat.parameters_mutation_step(source_genome.copy()))

current_gen = list(copy_with_mutation(neat_obj, init_genome) for _ in range(popul))


## RUN ALTERNATING AUGMENTATION AND REDUCTION STAGES ##
for epoch in range(num_epochs):
    try:
        print("Epoch #{}".format(epoch))

        conf.update({
            'topology_reduction_proba': 0,
            'topology_augmentation_proba': aug_proba})
        neat_obj = NEAT(
            topology_mutator=mutator,
            neuron_specs=(input_neuron_spec, sigmoid_neuron_spec),
            connection_specs=(connection_spec,),
            **conf)

        for _ in range(augment_gens_per_epoch):
            evals_num += len(current_gen)
            current_gen, best_genome, best_fitness = next_gen(current_gen)
        print("  AUG " + get_stats(current_gen, best_genome, best_fitness))


        conf.update({
            'topology_reduction_proba': red_proba,
            'topology_augmentation_proba': 0})
        neat_obj = NEAT(
            topology_mutator=mutator,
            neuron_specs=(input_neuron_spec, sigmoid_neuron_spec),
            connection_specs=(connection_spec,),
            **conf)

        for _ in range(reduct_gens_per_epoch):
            evals_num += len(current_gen)
            current_gen, best_genome, best_fitness = next_gen(current_gen)
        print("  RED " + get_stats(current_gen, best_genome, best_fitness))

        # if abs(best_fitness) < 1e-6:
        if abs(best_fitness - 1) < 1e-6:
            break

    except KeyboardInterrupt:
        break


print("Number of performed evaluations: {}, best fitness: {}".format(evals_num, best_fitness))
print("Final test:")

final_nn = NN().from_genome(best_genome)

test_inputs = list(inputs) * 4
random.shuffle(test_inputs)
for inp in test_inputs:
    final_nn.reset()
    print("{} -> {}".format(inp, final_nn.compute(inp)[0]))


# write final genome as YAML file
try:
    with open('xor_genome.yaml', 'w+') as genfile:
        genfile.write(best_genome.to_yaml())

except AnyError:
    pass
