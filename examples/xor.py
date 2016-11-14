from os import path
import sys
import math
import random

sys.path.append(path.dirname(path.abspath(__file__)) + '/../')

from neat import (GeneticEncoding, Mutator, crossover,
                NetworkSpec, GeneSpec,
                NumericParamSpec as PS,
                NominalParamSpec as NPS)

from itertools import izip
from operator import itemgetter

from nn import NN


######## NEAT USAGE ########



#### CONFIG #### #### #### #### #### #### ####
pop_size = 5
elite_size = 1
tournament_size = 4
param_mut_proba = 0.5
structural_augmentation_probability = 0.8
structural_removal_probability = 0.0
speciation_threshold = 0.05

neuron_sigma = 0.25
conn_sigma = 1.0
#### ###### #### #### #### #### #### #### ####




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




## CREATE INITIAL GENOME ##
mutator = Mutator(net_spec,
    allowed_neuron_types = ['sigmoid'],
    mutable_params = {'sigmoid': ['bias', 'gain'], 'input': []})

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


# TODO:
# This should got into the NEAT module itself

def validate_genotype(genotype, error_msg):
    if not genotype.check_validity():
        raise RuntimeError(error_msg + '\n' + str(genotype))





def apply_structural_mutation(genotype):
    # apply augmentation mutation:
    if random.random() < structural_augmentation_probability:

        # if no connections, add connection
        if len(genotype.connection_genes) == 0:
            mutator.add_connection_mutation(genotype)
            validate_genotype(genotype, "inserting new CONNECTION created invalid genotype")

        # otherwise add connection or neuron with equal probability
        else:
            if random.random() < 0.5:
                mutator.add_connection_mutation(genotype)
                validate_genotype(genotype, "inserting new CONNECTION created invalid genotype")

            else:
                mutator.add_neuron_mutation(genotype)
                validate_genotype(genotype, "inserting new NEURON created invalid genotype")


    # apply removal mutation:
    if random.random() < structural_removal_probability:
        if random.random() < 0.5:
            mutator.remove_connection_mutation(genotype)
            validate_genotype(genotype, "removing a CONNECTION created invalid genotype")
        else:
            mutator.remove_neuron_mutation(genotype)
            validate_genotype(genotype, "removing a NEURON created invalid genotype")






def produce_child(parent1, parent2):
    # apply crossover:
    child_genotype = crossover(parent1, parent2)
    validate_genotype(child_genotype, "crossover created invalid genotype")

    # apply mutations:
    mutator.mutate_connection_params(
        genotype=child_genotype,
        probability=param_mut_proba)
    validate_genotype(child_genotype, "weight mutation created invalid genotype")

    mutator.mutate_neuron_params(
        genotype=child_genotype,
        probability=param_mut_proba)

    validate_genotype(child_genotype, "neuron parameters mutation created invalid genotype")

    # apply structural mutations:
    apply_structural_mutation(child_genotype)

    return child_genotype



def share_fitness(genomes_fitnesses):
    shared_fitness = []

    for cur_brain, cur_fitness in genomes_fitnesses:
        species_size = 1
        for other_brain, other_fitness in genomes_fitnesses:
            if not other_brain == cur_brain:
                distance = GeneticEncoding.get_dissimilarity(other_brain, cur_brain)
                if distance < speciation_threshold:
                    species_size += 1

        shared_fitness.append((cur_brain, cur_fitness / float(species_size)))
        
    return shared_fitness



def select_for_tournament(candidates, tournament_size):
    selected = sorted(
        random.sample(candidates, tournament_size),
        key = itemgetter(1),
        reverse=True)

    return list(genome for (genome, fitness) in selected)



def produce_new_generation(gen_fit):
    new_genomes = []

    gen_fit_shared = share_fitness(gen_fit)

    gen_fit = sorted(gen_fit, key=itemgetter(1), reverse=True)
    gen_fit_shared = sorted(gen_fit_shared, key=itemgetter(1), reverse=True)

    # create children:
    for _ in range(pop_size - elite_size):
        # we select genomes using their shared fitnesses:
        selected = select_for_tournament(gen_fit_shared, tournament_size)

        # select 2 best parents from the tournament and make child:
        child_genome = produce_child(selected[0], selected[1])
        new_genomes.append(child_genome)


    # bringing the best parents into next generation:
    for i in range(elite_size):
        new_genomes.append(gen_fit[i][0])

    return new_genomes




## RUN EVOLUTION ##

current_gen = init_pop
num_generations = 1000

for num_gen in range(num_generations):
    evaluated_gen = evaluate(current_gen)
    current_gen = produce_new_generation(evaluated_gen)

    best_fit = sorted(evaluated_gen, key = itemgetter(1))[-1][1]
    print("{}, fitness = {}".format(num_gen, best_fit))




