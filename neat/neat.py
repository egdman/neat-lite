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


_defaults = {
    'selection_subsample_size': None,

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


def default_gene_factory(*gene_specs):
    def _generate():
        # select gene type at random
        gene_spec = random.choice(gene_specs)
        return gene_spec.type_name, gene_spec.get_random_parameters()
    return _generate


class NEAT(object):
    def __init__(self,
        mutator,
        neuron_factory=None,
        connection_factory=None,
        selection_step=None,
        crossover_step=None,
        parameters_mutation_step=None,
        topology_augmentation_step=None,
        topology_reduction_step=None,
        custom_reproduction_pipeline=None,
        **config):

        self.mutator = mutator

        self.neuron_factory = neuron_factory
        self.connection_factory = connection_factory

        self.selection_step = selection_step
        self.crossover_step = crossover_step
        self.parameters_mutation_step = parameters_mutation_step
        self.topology_augmentation_step = topology_augmentation_step
        self.topology_reduction_step = topology_reduction_step

        # use default order of operations for reproduction pipeline
        if custom_reproduction_pipeline is None:
            for setting_name, default_value in _defaults.items():
                provided_value = config.get(setting_name, None)

                if provided_value is not None:
                    setattr(self, setting_name, provided_value)
                elif default_value is not None:
                    setattr(self, setting_name, default_value)
                else:
                    raise InvalidConfigError("NEAT: please provide value for {}".format(setting_name))

            # check validity of settings:
            if self.selection_subsample_size < 2:
                raise InvalidConfigError("NEAT: selection_subsample_size must be greater than 1")


            if self.selection_step is None:
                self.selection_step = two_best_in_subsample(self.selection_subsample_size)
            if self.crossover_step is None:
                self.crossover_step = crossover_two_genomes()
            if self.parameters_mutation_step is None:
                self.parameters_mutation_step = parameters_mutation(self.network_spec, self.neuron_param_mut_proba, self.connection_param_mut_proba)
            if self.topology_augmentation_step is None:
                self.topology_augmentation_step = topology_augmentation(self.mutator, self.topology_augmentation_proba, neuron_factory, connection_factory)
            if self.topology_reduction_step is None:
                self.topology_reduction_step = topology_reduction(self.mutator, self.topology_reduction_proba)

        # use the provided custom reproduction pipeline
        else:
            self.custom_reproduction_pipeline = custom_reproduction_pipeline



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



    # def produce_new_generation(self, genome_fitness_list):
    #     gen_fit_shared = self.share_fitness(genome_fitness_list)

    #     # create children:
    #     for _ in range(self.pop_size - self.elite_size):
    #         # we select genomes using their shared fitnesses:
    #         subset = random.sample(gen_fit_shared, self.tournament_size)
    #         parent1, parent2 = heapq.nlargest(2, subset, key = itemgetter(1))

    #         yield self.produce_child(parent1[0], parent2[0])

    #     # bringing the best parents into next generation:
    #     best_parents = heapq.nlargest(self.elite_size, genome_fitness_list, key = itemgetter(1))
    #     for genome, _ in best_parents:
    #         yield genome


    def produce_new_genome(self, genome_and_fitness_list):
        if self.custom_reproduction_pipeline is None:
            pipeline = (
                self.selection_step,
                self.crossover_step,
                self.parameters_mutation_step,
                self.topology_augmentation_step,
                self.topology_reduction_step,
            )
        else:
            pipeline = self.custom_reproduction_pipeline

        value = genome_and_fitness_list
        for step in pipeline:
            value = step(value)
        return value


def two_best_in_subsample(subsample_size):
    def _impl(genome_and_fitness_list):
        subsample = random.sample(genome_and_fitness_list, subsample_size)
        parent1, parent2 = heapq.nlargest(2, subset, key=itemgetter(1))
        return parent1[0], parent2[0]
    return _impl


def parameters_mutation(network_spec, neuron_param_mut_proba, connection_param_mut_proba):
    def _mutate_gene_params(gene, probability):
        gene_spec = network_spec[gene.gene_type]

        for param_name, param_spec in gene_spec.param_specs.items():
            if random.random() < probability:
                current_value = gene[param_name]
                new_value = param_spec.mutate_value(current_value)
                gene[param_name] = new_value

    def _impl(genome):
        if neuron_param_mut_proba > 0:
            for neuron_gene in genome.neuron_genes:
                _mutate_gene_params(neuron_gene, neuron_param_mut_proba)

        if connection_param_mut_proba > 0:
            for connection_gene in genome.connection_genes:
                _mutate_gene_params(connection_gene, connection_param_mut_proba)

        return genome
    return _impl


def topology_augmentation(mutator, probability, neuron_factory, connection_factory):
    def _impl(genome):
        if random.random() < probability:
            # if no connections, add a connection
            if len(genome.connection_genes) == 0:
                mutator.add_random_connection(genome, connection_factory)

            # otherwise add connection or neuron with equal probability
            elif random.random() < 0.5:
                mutator.add_random_connection(genome, connection_factory)
            else:
                mutator.add_random_neuron(genome, neuron_factory, connection_factory)
        return genome

    return _impl


def topology_reduction(mutator, probability):
    def _impl(genome):
        if random.random() < probability:
            if random.random() < 0.5:
                mutator.remove_random_connection(genome)
            else:
                mutator.remove_random_neuron(genome)
        return genome
    return _impl


def crossover_two_genomes():
    def _impl(genomes):
        genome1, genome2 = genomes
        return crossover(genome1, genome2)
    return _impl
