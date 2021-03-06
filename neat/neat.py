import random
import heapq
from operator import itemgetter

from .genes import GeneticEncoding
from .operators import crossover


class InvalidConfigError(RuntimeError): pass



class GeneInfo(object): pass


def neuron(gene_type, protected, **params):
    '''
    Helper function to use with NEAT.get_init_genome
    '''
    n = GeneInfo()
    n.protected = protected
    n.type = gene_type
    n.params = params
    return n


def connection(gene_type, protected, src, dst, **params):
    '''
    Helper function to use with NEAT.get_init_genome
    '''
    c = GeneInfo()
    c.protected = protected
    c.type = gene_type
    c.src = src
    c.dst = dst
    c.params = params
    return c



def validate_genotype(genotype, error_msg):
    if not genotype.check_validity():
        raise RuntimeError(error_msg + '\n' + str(genotype))



class NEAT(object):
    settings = {
        'pop_size': None,
        'tournament_size': None,
        'elite_size': 0,

        'structural_augmentation_proba': None,
        'structural_removal_proba': 0.,

        'neuron_param_mut_proba': None,
        'connection_param_mut_proba': None,

        'speciation_threshold': 0.,

        # coefficients for calculating genotype dissimilarity
        'excess_coef': 1.,
        'disjoint_coef': 1.,
        'neuron_diff_coef': 0.,
        'connection_diff_coef': 0.,
    }

    def __init__(self, mutator, **config):
        self.mutator = mutator

        for setting_name, default_value in NEAT.settings.items():
            provided_value = config.get(setting_name, None)

            if provided_value is not None:
                setattr(self, setting_name, provided_value)
            elif default_value is not None:
                setattr(self, setting_name, default_value)
            else:
                raise InvalidConfigError("NEAT instance: please provide value for {}".format(setting_name))

        # check validity of settings:
        if self.tournament_size > self.pop_size or self.tournament_size < 2:
            raise InvalidConfigError("NEAT instance: tournament_size must lie within [2, pop_size]")

        if self.elite_size > self.pop_size:
            raise InvalidConfigError("NEAT instance: elite_size must not be larger than pop_size")



    def get_init_genome(self, **kwargs):
        connections = kwargs.pop('connections', [])
        neurons = kwargs
        genome = GeneticEncoding()

        # if we want to protect a connection we also
        # want to protect its 2 adjacent neurons
        # because we cannot remove a neuron without removing
        # its adjacent connections
        for conn_info in connections:
            if conn_info.protected:
                neurons[conn_info.src].protected = True
                neurons[conn_info.dst].protected = True

        # add neuron genes to genome using mutator
        neuron_map = {}
        for neuron_id, neuron_info in neurons.items():
            hmark = self.mutator.add_neuron(
                genome,
                neuron_info.type,
                protected=neuron_info.protected,
                **neuron_info.params
            )
            neuron_map[neuron_id] = hmark

        # add connection genes to genome using mutator
        for conn_info in connections:
            self.mutator.add_connection(
                genome,
                conn_info.type,
                mark_from = neuron_map[conn_info.src],
                mark_to = neuron_map[conn_info.dst],
                protected=conn_info.protected,
                **conn_info.params
            )

        return genome




    def produce_init_generation(self, source_genome):
        init_pop = []
        for _ in range(self.pop_size):
            mutated_genome = source_genome.copy()
            
            self.mutator.mutate_connection_params(
                genotype=mutated_genome,
                probability=self.connection_param_mut_proba)

            self.mutator.mutate_neuron_params(
                genotype=mutated_genome,
                probability=self.neuron_param_mut_proba)

            self.apply_structural_mutation(mutated_genome)

            init_pop.append(mutated_genome)

        return init_pop



    def apply_structural_mutation(self, genotype):
        # apply augmentation mutation:
        if random.random() < self.structural_augmentation_proba:

            # if no connections, add connection
            if len(genotype.connection_genes) == 0:
                self.mutator.add_connection_mutation(genotype)
                # validate_genotype(genotype, "inserting new CONNECTION created invalid genotype")

            # otherwise add connection or neuron with equal probability
            else:
                if random.random() < 0.5:
                    self.mutator.add_connection_mutation(genotype)
                    # validate_genotype(genotype, "inserting new CONNECTION created invalid genotype")

                else:
                    self.mutator.add_neuron_mutation(genotype)
                    # validate_genotype(genotype, "inserting new NEURON created invalid genotype")


        # apply removal mutation:
        if random.random() < self.structural_removal_proba:
            if random.random() < 0.5:
                self.mutator.remove_connection_mutation(genotype)
                # validate_genotype(genotype, "removing a CONNECTION created invalid genotype")
            else:
                self.mutator.remove_neuron_mutation(genotype)
                # validate_genotype(genotype, "removing a NEURON created invalid genotype")



    def produce_child(self, parent1, parent2):
        # apply crossover:
        child_genotype = crossover(parent1, parent2)
        # validate_genotype(child_genotype, "crossover created invalid genotype")

        # apply mutations:
        self.mutator.mutate_connection_params(
            genotype=child_genotype,
            probability=self.connection_param_mut_proba)
        # validate_genotype(child_genotype, "weight mutation created invalid genotype")

        self.mutator.mutate_neuron_params(
            genotype=child_genotype,
            probability=self.neuron_param_mut_proba)
        # validate_genotype(child_genotype, "neuron parameters mutation created invalid genotype")

        # apply structural mutations:
        self.apply_structural_mutation(child_genotype)

        return child_genotype



    def share_fitness(self, genomes_fitnesses):

        def species_size(genome, fitness):
            size = 1
            for other_genome, other_fitness in genomes_fitnesses:
                if not other_genome == genome:

                    distance = GeneticEncoding.get_dissimilarity(
                        other_genome, genome,
                        excess_coef = self.excess_coef,
                        disjoint_coef = self.disjoint_coef,
                        neuron_diff_coef = self.neuron_diff_coef,
                        connection_diff_coef = self.connection_diff_coef
                    )
                    if distance < self.speciation_threshold: size += 1
            return size

        return list((g, f / species_size(g, f)) for g, f in genomes_fitnesses)



    def produce_new_generation(self, genome_fitness_list):
        gen_fit_shared = self.share_fitness(genome_fitness_list)

        # create children:
        for _ in range(self.pop_size - self.elite_size):
            # we select genomes using their shared fitnesses:
            subset = random.sample(gen_fit_shared, self.tournament_size)
            parent1, parent2 = heapq.nlargest(2, subset, key = itemgetter(1))

            yield self.produce_child(parent1[0], parent2[0])

        # bringing the best parents into next generation:
        best_parents = heapq.nlargest(self.elite_size, genome_fitness_list, key = itemgetter(1))
        for genome, _ in best_parents:
            yield genome
