from os import path
import sys
import math
import random
from random import seed
import heapq
from operator import itemgetter
from itertools import chain
import time
from datetime import datetime, timezone

here_dir = path.dirname(path.abspath(__file__))
sys.path.append(path.join(here_dir, '..'))

from neat import (Mutator, Pipeline, GeneSpec, ParamSpec as PS, validate_genome,
    gen, mut, bounds,
    neuron, connection, default_gene_factory)

from nn_impl import NN


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
generation_size = 20 # number of genomes in each generation across all species
num_species = 4      # number of species
elite_num = 1        # how many top performing members of a species will be
                     #   copied unchanged into the next generation

conf = dict(
selection_sample_size = int(.75 * generation_size) // num_species, # size of the selection sample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.8,     # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.8, # probability to mutate each single connection in the genome
topology_augmentation_proba = 0,  # probability to augment the topology of a newly created genome
topology_reduction_proba = 0,     # probability to reduce the topology of a newly created genome
)
#### ###### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

def make_pipeline(mutator, **args):
    conf_ = conf.copy()
    conf_.update(args)
    return Pipeline(
        topology_mutator=mutator,
        # # disable 2-way crossover
        # crossover_step=lambda genomes: genomes[0].copy(),
        **conf_)


def species_sizes(n, num_species):
    return (n // num_species,) * (num_species - 1) + (n // num_species + n % num_species,)


def count_members(species_list):
    return sum(len(species) for species in species_list)


## CREATE GENE SPECS ##
neuron_sigma = 0.25  # mutation sigma value for neuron params
conn_sigma = 10.     # mutation sigma value for connection params

input_neuron_spec = GeneSpec(
    'input',
    PS('layer', gen.const('input')),
)
sigmoid_neuron_spec = GeneSpec(
    'sigmoid',
    PS('layer', gen.const('hidden')),
    PS('bias', gen.uniform(), mut.gauss(neuron_sigma), bounds(-1., 1.)),
    PS('gain', gen.uniform(), mut.gauss(neuron_sigma), bounds(0., 1.)),
)
connection_spec = GeneSpec(
    'connection',
    PS('weight', gen.gauss(0, conn_sigma), mut.gauss(conn_sigma)),
)


def produce_new_generation(pipeline, genome_fitness_list):
    for _ in range(len(genome_fitness_list) - elite_num):
        g = pipeline.produce_new_genome(genome_fitness_list)
        # validate_genome(g, 'invalid genome')
        yield g

    # bringing the best genomes into next generation:
    best_genomes = heapq.nlargest(elite_num, genome_fitness_list, key=itemgetter(1))
    for genome, _ in best_genomes:
        yield genome


## INPUT VALUES FOR EVALUATING FITNESS OF THE SOLUTIONS ##
inputs = ((0, 0), (0, 1), (1, 0), (1, 1))

def evaluate(genome):
    nn = NN(genome)

    outp0 = nn.compute(inputs[0])[0]
    nn.reset() # reset network otherwise it will remember previous state
    outp1 = nn.compute(inputs[1])[0]
    nn.reset()
    outp2 = nn.compute(inputs[2])[0]
    nn.reset()
    outp3 = nn.compute(inputs[3])[0]

    fitness = abs(outp0 - outp1) * abs(outp0 - outp2) * abs(outp3 - outp1) * abs(outp3 - outp2)
    return fitness


def complexity(species_list):
    n = (ge.num_neuron_genes() for ge in chain(*species_list))
    c = (ge.num_connection_genes() for ge in chain(*species_list))
    return sum(n), sum(c)


total_eval_time = [0]
total_neat_time = [0]


class Attempt:
    def __init__(self):
        self.evals_num = 0
        self.best_fitness = None
        self.best_genome = None
        self.target_reached = False

    def make_next_generation(self, pipeline, current_gen):
        self.evals_num += count_members(current_gen)

        t0 = time.perf_counter()
        evaluated = [[(genome, evaluate(genome)) for genome in species] for species in current_gen]
        t1 = time.perf_counter()
        total_eval_time[0] += t1 - t0

        next_gen = [list(produce_new_generation(pipeline, species)) for species in evaluated]
        t2 = time.perf_counter()
        total_neat_time[0] += t2 - t1

        self.best_genome, self.best_fitness = max(chain(*evaluated), key=itemgetter(1))
        return next_gen


    def get_stats(self, genomes):
        n_neurons, n_conns = complexity(genomes)
        return ("evals: {}, best genome has: {}N, {}C, complexity: {}N, {}C, best fitness = {}"
            .format(self.evals_num,
                self.best_genome.num_neuron_genes(), self.best_genome.num_connection_genes(),
                n_neurons, n_conns, self.best_fitness))


# init_genome = mutator.produce_genome(
#     in1=neuron(input_neuron_spec, non_removable=True, layer='input'),
#     in2=neuron(input_neuron_spec, non_removable=True, layer='input'),
#     out1=neuron(sigmoid_neuron_spec, non_removable=True, gain=1, bias=-0.2, layer='output'),
#     h11 = neuron(sigmoid_neuron_spec, gain=1, bias=-0.6, layer='hidden'),
#     h11066=neuron(sigmoid_neuron_spec, gain=1, bias=-1, layer='hidden'),
#     h11242=neuron(sigmoid_neuron_spec, gain=1, bias=1, layer='hidden'),
#     connections=(
#         connection(connection_spec, src='in1', dst='h11', weight=-125),
#         connection(connection_spec, src='in1', dst='h11066', weight=-40),
#         connection(connection_spec, src='in2', dst='h11', weight=-45),
#         connection(connection_spec, src='in2', dst='out1', weight=35),
#         connection(connection_spec, src='h11', dst='out1', weight=125),
#         connection(connection_spec, src='h11066', dst='out1', weight=-50),
#         connection(connection_spec, src='h11242', dst='out1', weight=-60),
#     )
# )


## CREATE INITIAL GENERATION ##
def copy_with_mutation(pipeline, source_genome):
    return pipeline.topology_augmentation_step(pipeline.parameters_mutation_step(source_genome.copy()))

## CREATE MUTATOR ##
mutator = Mutator(
    neuron_factory=default_gene_factory(sigmoid_neuron_spec),
    connection_factory=default_gene_factory(connection_spec),
    pure_input_types=(input_neuron_spec,),
)

def make_attempt(num_epochs, gens_per_epoch):
    augmentation_proba = .9
    reduction_proba = .9

    augment_gens_per_epoch = int(.4 * gens_per_epoch)
    reduct_gens_per_epoch = gens_per_epoch - augment_gens_per_epoch

    ## CREATE THE REPRODUCTION PIPELINE ##
    pipeline = make_pipeline(mutator)

    ## CREATE INITIAL GENOME ##
    # we specify initial input and output neurons and protect them from removal

    init_genome = mutator.produce_genome(
        in1=neuron(input_neuron_spec, non_removable=True, layer='input'),
        in2=neuron(input_neuron_spec, non_removable=True, layer='input'),
        out1=neuron(sigmoid_neuron_spec, non_removable=True, layer='output'),
        connections=(
            # connection(connection_spec, src='in1', dst='out1'),
            # connection(connection_spec, src='in2', dst='out1')
        )
    )

    current_gen = list(
        list(copy_with_mutation(pipeline, init_genome) for _ in range(species_size)) \
        for species_size in species_sizes(generation_size, num_species))

    attempt = Attempt()

    ## RUN ALTERNATING AUGMENTATION AND REDUCTION STAGES ##
    for epoch in range(num_epochs):
        # print("Epoch #{}".format(epoch))

        pipeline = make_pipeline(mutator, topology_augmentation_proba=augmentation_proba, topology_reduction_proba=0)

        for _ in range(augment_gens_per_epoch):
            current_gen = attempt.make_next_generation(pipeline, current_gen)
        # print("  AUG " + attempt.get_stats(current_gen))

        pipeline = make_pipeline(mutator, topology_augmentation_proba=0, topology_reduction_proba=reduction_proba)

        for _ in range(reduct_gens_per_epoch):
            current_gen = attempt.make_next_generation(pipeline, current_gen)
        # print("  RED " + attempt.get_stats(current_gen))

        if abs(attempt.best_fitness - 1) < 1e-6:
            attempt.target_reached = True
            break
    return attempt


def timestamp():
    ts = datetime.fromtimestamp(time.time(), tz=timezone.utc)
    return ts.strftime("%b %d %y %H:%M")


## RUN MULTIPLE ATTEMPTS TO CREATE A XOR NETWORK ##
best_genome = None

num_attempts = 10
total_eval_num = 0
success_count = 0
start_timer = time.perf_counter()

for attempt_id in range(num_attempts):
    # attempt_id += 100#20400

    num_epochs = 20
    gens_per_epoch = 250

    seed(attempt_id)
    attempt = make_attempt(num_epochs=num_epochs, gens_per_epoch=gens_per_epoch)

    if attempt.target_reached:
        success_count += 1
        best_genome = attempt.best_genome
    total_eval_num += attempt.evals_num
    result = dict(
        id = attempt_id,
        t = timestamp(),
        species = tuple(species_sizes(generation_size, num_species)),
        gens_in_epoch = gens_per_epoch,
        n_epochs = num_epochs,
        n_evals = attempt.evals_num,
        fitness = attempt.best_fitness,
        target_reached = attempt.target_reached,
    )
    print(result)

total_time = time.perf_counter() - start_timer
print(f"took {total_time} ticks")
print(f"total eval time {total_eval_time[0]} ticks")
print(f"total neat time {total_neat_time[0]} ticks")
print(f"eval time per 10000 evals: {10000*total_eval_time[0] / total_eval_num} ticks")
print(f"neat time per 10000 evals: {10000*total_neat_time[0] / total_eval_num} ticks")
print(f"neat time percentage = {100*total_neat_time[0] / total_time}")
print(f"total number of innovations: {mutator.innovation_number}")
print(f"success rate: {success_count} / {num_attempts}")

time_per_success = total_time / success_count if success_count > 0 else "N/A"
print(f"time per success: {time_per_success}")


# print("Final test:")
# final_nn = NN(best_genome)

# test_inputs = list(inputs) * 4
# random.shuffle(test_inputs)
# for inp in test_inputs:
#     final_nn.reset()
#     print("{} -> {}".format(inp, final_nn.compute(inp)[0]))


# write final genome as YAML file
if best_genome is None:
    print("could not reach the target")
else:
    filename = path.join(here_dir, "xor_genome.yaml")
    try:
        with open(filename, 'w+') as genfile:
            genfile.write(best_genome.to_yaml())
        print(f"solution was written to file {filename}")

    except Exception:
        pass
