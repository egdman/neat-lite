from os import path
import sys
import math
import random
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

from neat import (Mutator, NetworkSpec, GeneSpec, GeneticEncoding,
                NumericParamSpec as PS,
                NominalParamSpec as NPS,
                NEAT, neuron, connection)

from nn_impl import NN


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
neuron_sigma = 0.25                 # mutation sigma for neuron params
conn_sigma = 10.                    # mutation sigma for connection params

popul = 10
conf = dict(
pop_size = popul,                   # population size
elite_size = 1,                     # size of the elite club
tournament_size = int(.75 * popul), # size of the selection subsample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.5,       # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.5,   # probability to mutate each single connection in the genome
structural_augmentation_proba = 0,  # probability to augment the topology of a newly created genome 
structural_removal_proba = 0,       # probability to diminish the topology of a newly created genome
speciation_threshold = 0.2          # genomes that are more similar than this value will be considered the same species
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

## INPUTS AND CORRECT OUTPUTS FOR THE NETWORK ##
inputs = ((0, 0), (0, 1), (1, 0), (1, 1))
true_outputs = (0, 1, 1, 0)


## CREATE MUTATOR ##
mutator = Mutator(net_spec,
    allowed_neuron_types = ('sigmoid',),
    pure_input_types = ('input',)
)


## CREATE MAIN NEAT OBJECT ##
neat_obj = NEAT(mutator = mutator, **conf)


## CREATE INITIAL GENOTYPE ##
# we specify initial input and output neurons and protect them from removal
init_genome = neat_obj.get_init_genome(
        in1=neuron('input', protected=True, layer='input'),
        in2=neuron('input', protected=True, layer='input'),
        out1=neuron('sigmoid', protected=True, layer='output'),
        connections=[
        # connection('connection', protected=True, src='in1', dst='out1'),
        # connection('connection', protected=True, src='in2', dst='out1')
        ]
    )


## CREATE INITIAL GENERATION ##
init_gen = neat_obj.produce_init_generation(init_genome)


def rmse(X, Y):
    return math.sqrt( sum( (x - y)**2 for x, y in zip(X, Y) ) )


def evaluate(genomes):
    fitnesses = []
    for genome in genomes:
        nn = NN().from_genome(genome)
        nn_outputs = []
        for inp in inputs:
            nn.reset() # reset network otherwise it will remember previous state
            nn_outputs.append(nn.compute(inp)[0])

        fitnesses.append(-rmse(true_outputs, nn_outputs))
    return list(zip(genomes, fitnesses))

def complexity(genomes):
    return (sum((len(ge.neuron_genes)) for ge in genomes),
    sum((len(ge.connection_genes)) for ge in genomes))

def get_stats(genomes, best_gen, best_fitness):
    n_neurons, n_conns = complexity(genomes)
    return ("gen {}, best genome has: {}N, {}C, complexity: {}N, {}C, best fitness = {}"
        .format(gen_num,
            len(best_gen.neuron_genes), len(best_gen.connection_genes),
            n_neurons, n_conns, best_fitness))

def next_gen(current_gen):
    evaluated_gen = evaluate(current_gen)
    next_gen = list(neat_obj.produce_new_generation(evaluated_gen))
    best_genome, best_fitness = sorted(evaluated_gen, key = itemgetter(1))[-1]
    return next_gen, best_genome, best_fitness


num_epochs = 1000000000
gens_per_epoch = 100
aug_proba = .9
sim_proba = .9

gen_num = 0
current_gen = init_gen

## RUN ALTERNATING AUGMENTATION AND SIMPLIFICATION STAGES ##
for epoch in range(num_epochs):
    try:
        print("Epoch #{}".format(epoch))

        print("//// AUGMENT PHASE ////")
        conf.update({
            'structural_removal_proba': 0,
            'structural_augmentation_proba': aug_proba})
        neat_obj = NEAT(mutator = mutator, **conf)

        for _ in range(gens_per_epoch):
            current_gen, best_genome, best_fitness = next_gen(current_gen)
            gen_num += 1
            if gen_num % 1 == 0:
                print(get_stats(current_gen, best_genome, best_fitness))


        print("//// REMOVAL PHASE ////")
        conf.update({
            'structural_removal_proba': sim_proba,
            'structural_augmentation_proba': 0})
        neat_obj = NEAT(mutator = mutator, **conf)

        for _ in range(int(gens_per_epoch*1.5)):
            current_gen, best_genome, best_fitness = next_gen(current_gen)
            gen_num += 1
            if gen_num % 1 == 0:
                print(get_stats(current_gen, best_genome, best_fitness))

        if abs(best_fitness) < 1e-6:
            break

    except KeyboardInterrupt:
        break

print("Final test:")

# with open("xor_genome_reference.yaml", 'r') as gf:
#     best_genome = GeneticEncoding().from_yaml(yaml.load(gf.read()))

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
