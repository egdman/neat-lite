import random
import heapq
from operator import itemgetter

from .genes import Genome
from .operators import crossover


def validate_genome(genome, error_msg):
    if not genome.check_validity():
        raise RuntimeError(error_msg + '\n' + str(genome))


def default_gene_factory(*gene_specs):
    def _generate():
        # select gene type at random
        gene_spec = random.choice(gene_specs)
        return gene_spec.type_name, gene_spec.generate_parameter_values()
    return _generate


def two_best_in_sample(sample_size):
    def _select(genome_and_fitness_list):
        sample = random.sample(genome_and_fitness_list, sample_size)
        parent1, parent2 = heapq.nlargest(2, sample, key=itemgetter(1))
        return parent1[0], parent2[0]
    return _select


def parameters_mutation(neuron_specs, connection_specs, neuron_param_mut_proba, connection_param_mut_proba):
    neuron_specs = {spec.type_name: spec for spec in neuron_specs}
    connection_specs = {spec.type_name: spec for spec in connection_specs}

    def _mutate_gene_params(gene, spec, probability):
        if spec is None:
            return
        for param_name, param_spec in spec.param_specs.items():
            if random.random() < probability:
                current_value = gene[param_name]
                new_value = param_spec.mutate_value(current_value)
                gene[param_name] = new_value

    def _parameters_mutation(genome):
        if neuron_param_mut_proba > 0:
            for gene in genome.neuron_genes:
                spec = neuron_specs.get(gene.gene_type, None)
                _mutate_gene_params(gene, spec, neuron_param_mut_proba)

        if connection_param_mut_proba > 0:
            for gene in genome.connection_genes():
                spec = connection_specs.get(gene.gene_type, None)
                _mutate_gene_params(gene, spec, connection_param_mut_proba)

        return genome
    return _parameters_mutation


def topology_augmentation(mutator, probability):
    def _augment(genome):
        if random.random() < probability:
            # if no connections, add a connection
            if genome.num_connection_genes() == 0:
                mutator.add_random_connection(genome)

            # otherwise add connection or neuron with equal probability
            elif random.random() < 0.5:
                mutator.add_random_connection(genome)
            else:
                mutator.add_random_neuron(genome)
        return genome

    return _augment


def topology_reduction(mutator, probability):
    def _reduce(genome):
        if random.random() < probability:
            if random.random() < 0.5:
                mutator.remove_random_connection(genome)
            else:
                mutator.remove_random_neuron(genome)
        return genome
    return _reduce


def crossover_two_genomes():
    def _crossover(genomes):
        genome1, genome2 = genomes
        return crossover(genome1, genome2)
    return _crossover


_defaults = {
    'selection_sample_size': None,

    'topology_augmentation_proba': None,
    'topology_reduction_proba': 0.,

    'neuron_param_mut_proba': None,
    'connection_param_mut_proba': None,

    'speciation_threshold': 0.,

    # coefficients for calculating genome dissimilarity
    'excess_coef': 1.,
    'disjoint_coef': 1.,
    'neuron_diff_coef': 0.,
    'connection_diff_coef': 0.,
}

class InvalidConfigError(RuntimeError): pass

class NEAT(object):
    def __init__(self,
        topology_mutator,
        neuron_specs=(),
        connection_specs=(),
        selection_step=None,
        crossover_step=None,
        parameters_mutation_step=None,
        topology_augmentation_step=None,
        topology_reduction_step=None,
        custom_reproduction_pipeline=None,
        **config):

        self.selection_step = selection_step
        self.crossover_step = crossover_step
        self.parameters_mutation_step = parameters_mutation_step
        self.topology_augmentation_step = topology_augmentation_step
        self.topology_reduction_step = topology_reduction_step
        self.custom_reproduction_pipeline = custom_reproduction_pipeline

        # use default order of operations for reproduction pipeline if no override provided
        if self.custom_reproduction_pipeline is None:
            for setting_name, default_value in _defaults.items():
                provided_value = config.get(setting_name, None)

                if provided_value is not None:
                    setattr(self, setting_name, provided_value)
                elif default_value is not None:
                    setattr(self, setting_name, default_value)
                else:
                    raise InvalidConfigError("please provide value for {}".format(setting_name))

            # check validity of settings:
            if self.selection_sample_size < 2:
                raise InvalidConfigError("selection_sample_size must be greater than 1")


            if self.selection_step is None:
                self.selection_step = two_best_in_sample(self.selection_sample_size)
            if self.crossover_step is None:
                self.crossover_step = crossover_two_genomes()
            if self.parameters_mutation_step is None:
                self.parameters_mutation_step = parameters_mutation(neuron_specs, connection_specs, self.neuron_param_mut_proba, self.connection_param_mut_proba)
            if self.topology_augmentation_step is None:
                self.topology_augmentation_step = topology_augmentation(topology_mutator, self.topology_augmentation_proba)
            if self.topology_reduction_step is None:
                self.topology_reduction_step = topology_reduction(topology_mutator, self.topology_reduction_proba)


    def share_fitness(self, genomes_fitnesses):

        def species_size(genome, fitness):
            size = 1
            for other_genome, other_fitness in genomes_fitnesses:
                if not other_genome == genome:

                    distance = Genome.get_dissimilarity(
                        other_genome, genome,
                        excess_coef = self.excess_coef,
                        disjoint_coef = self.disjoint_coef,
                        neuron_diff_coef = self.neuron_diff_coef,
                        connection_diff_coef = self.connection_diff_coef
                    )
                    if distance < self.speciation_threshold: size += 1
            return size

        return list((g, f / species_size(g, f)) for g, f in genomes_fitnesses)


    def produce_new_genome(self, genome_and_fitness_list) -> Genome:
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
