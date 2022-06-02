import random
import heapq
from operator import itemgetter
from functools import partial

from .genes import Genome
from .operators import crossover


def validate_genome(genome, error_msg):
    if not genome.check_validity():
        raise RuntimeError(error_msg + '\n' + str(genome))


def default_gene_factory(*gene_specs):
    def _generate():
        # select gene type at random
        gene_spec = random.choice(gene_specs)
        return gene_spec, gene_spec.generate_parameter_values()
    return _generate


def two_best_in_sample(sample_size):
    def _select(genome_and_fitness_list):
        sample = random.sample(genome_and_fitness_list, sample_size)
        parent1, parent2 = heapq.nlargest(2, sample, key=itemgetter(1))
        return parent1[0], parent2[0]
    return _select


def parameters_mutation(neuron_param_mut_proba, connection_param_mut_proba):
    def _mutate_gene_params(genes_list, probability):
        for idx, gene in enumerate(genes_list):
            if gene is None:
                continue

            specs = enumerate(iter(gene.spec.mutable_param_specs))
            for param_idx, param_spec in specs:
                if random.random() < probability:
                    gene_copy = gene.copy()
                    params = gene_copy.params

                    current_value = params[param_idx]
                    params[param_idx] = param_spec.mutate_value(current_value)

                    for param_idx, param_spec in specs:
                        if random.random() < probability:
                            current_value = params[param_idx]
                            params[param_idx] = param_spec.mutate_value(current_value)

                    genes_list[idx] = gene_copy
                    # print(f"{tuple(gene.get_params_with_names())}\n{tuple(gene_copy.get_params_with_names())} ---")
                    break

    def _mutate_gene_params_always(genes_list):
        for idx, gene in enumerate(genes_list):
            if gene is None:
                continue

            specs = enumerate(iter(gene.spec.mutable_param_specs))
            for param_idx, param_spec in specs:
                gene_copy = gene.copy()
                params = gene_copy.params

                current_value = params[param_idx]
                params[param_idx] = param_spec.mutate_value(current_value)

                for param_idx, param_spec in specs:
                    current_value = params[param_idx]
                    params[param_idx] = param_spec.mutate_value(current_value)

                genes_list[idx] = gene_copy
                # print(f"{tuple(gene.get_params_with_names())}\n{tuple(gene_copy.get_params_with_names())} ---")
                break

    if neuron_param_mut_proba == 0:
        if connection_param_mut_proba == 0:
            def _parameters_mutation(genome):
                return genome

        elif connection_param_mut_proba == 1:
            def _parameters_mutation(genome):
                _mutate_gene_params_always(genome._conn_genes)
                return genome

        else:
            def _parameters_mutation(genome):
                _mutate_gene_params(genome._conn_genes, connection_param_mut_proba)
                return genome

    elif neuron_param_mut_proba == 1:
        if connection_param_mut_proba == 0:
            def _parameters_mutation(genome):
                _mutate_gene_params_always(genome._neuron_genes)
                return genome

        elif connection_param_mut_proba == 1:
            def _parameters_mutation(genome):
                _mutate_gene_params_always(genome._neuron_genes)
                _mutate_gene_params_always(genome._conn_genes)
                return genome

        else:
            def _parameters_mutation(genome):
                _mutate_gene_params_always(genome._neuron_genes)
                _mutate_gene_params(genome._conn_genes, connection_param_mut_proba)
                return genome
    else:
        if connection_param_mut_proba == 0:
            def _parameters_mutation(genome):
                _mutate_gene_params(genome._neuron_genes, neuron_param_mut_proba)
                return genome

        elif connection_param_mut_proba == 1:
            def _parameters_mutation(genome):
                _mutate_gene_params(genome._neuron_genes, neuron_param_mut_proba)
                _mutate_gene_params_always(genome._conn_genes)
                return genome

        else:
            def _parameters_mutation(genome):
                _mutate_gene_params(genome._neuron_genes, neuron_param_mut_proba)
                _mutate_gene_params(genome._conn_genes, connection_param_mut_proba)
                return genome
    return _parameters_mutation


def _to_tuple_and_trunc(tupl):
    # convert to tuple and truncate up to the first zero
    if not isinstance(tupl, tuple) and not isinstance(tupl, list):
        tupl = tupl,

    if 0 in tupl:
        return tupl[: tupl.index(0)]
    return tupl


def topology_augmentation(mutator, probas):
    def _augment_always(genome):
        # if no connections, add a connection
        if genome.num_connection_genes() == 0:
            mutator.add_random_connection(genome)

        # otherwise add connection or neuron with equal probability
        elif random.random() < 0.5:
            mutator.add_random_connection(genome)
        else:
            mutator.add_random_neuron(genome)
        return True

    def _augment(genome, probability):
        rv = random.random()
        if rv < probability:
            # if no connections, add a connection
            if genome.num_connection_genes() == 0:
                mutator.add_random_connection(genome)

            # otherwise add connection or neuron with equal probability
            elif rv < 0.5 * probability:
                mutator.add_random_connection(genome)
            else:
                mutator.add_random_neuron(genome)
            return True
        return False

    def _select_func(probability):
        if probability == 1:
            return _augment_always
        else:
            return partial(_augment, probability=probability)

    probas = _to_tuple_and_trunc(probas)
    _funcs = tuple(_select_func(p) for p in probas)

    def _augment_n(genome):
        all(f(genome) for f in _funcs)
        return genome
    return _augment_n


def topology_reduction(mutator, probas):
    def _reduce_always(genome):
        if random.random() < 0.5:
            mutator.remove_random_connection(genome)
        else:
            mutator.remove_random_neuron(genome)
        return True

    def _reduce(genome, probability):
        rv = random.random()
        if rv < probability:
            if rv < 0.5 * probability:
                mutator.remove_random_connection(genome)
            else:
                mutator.remove_random_neuron(genome)
            return True
        return False

    def _select_func(probability):
        if probability == 1:
            return _reduce_always
        else:
            return partial(_reduce, probability=probability)

    probas = _to_tuple_and_trunc(probas)
    _funcs = tuple(_select_func(p) for p in probas)

    def _reduce_n(genome):
        all(f(genome) for f in _funcs)
        return genome
    return _reduce_n


def crossover_two_genomes():
    def _crossover(genomes):
        genome1, genome2 = genomes
        return crossover(genome1, genome2)
    return _crossover


class InvalidConfigError(RuntimeError): pass

def _check_setting(setting_name, setting_value):
    if setting_value is None:
        raise InvalidConfigError("please provide value for {}".format(setting_name))


class Pipeline:
    def __init__(self,
        topology_mutator=None,
        topology_augmentation_proba=None,
        topology_reduction_proba=0.,
        neuron_param_mut_proba=None,
        connection_param_mut_proba=None,
        selection_sample_size=None,

        selection_step=None,
        crossover_step=None,
        parameters_mutation_step=None,
        topology_augmentation_step=None,
        topology_reduction_step=None,

        custom_reproduction_pipeline=None,
    ):
        self.selection_step = selection_step
        self.crossover_step = crossover_step
        self.parameters_mutation_step = parameters_mutation_step
        self.topology_augmentation_step = topology_augmentation_step
        self.topology_reduction_step = topology_reduction_step
        self.custom_reproduction_pipeline = custom_reproduction_pipeline

        # use default operations for reproduction pipeline if no custom pipeline provided
        if self.custom_reproduction_pipeline is None:
            if self.selection_step is None:
                _check_setting("selection_sample_size", selection_sample_size)
                if selection_sample_size < 2:
                    raise InvalidConfigError("selection_sample_size must be greater than 1")
                self.selection_step = two_best_in_sample(selection_sample_size)

            if self.crossover_step is None:
                self.crossover_step = crossover_two_genomes()

            if self.parameters_mutation_step is None:
                _check_setting("neuron_param_mut_proba", neuron_param_mut_proba)
                _check_setting("connection_param_mut_proba", connection_param_mut_proba)
                self.parameters_mutation_step = parameters_mutation(neuron_param_mut_proba, connection_param_mut_proba)

            if self.topology_augmentation_step is None:
                _check_setting("topology_mutator", topology_mutator)
                _check_setting("topology_augmentation_proba", topology_augmentation_proba)
                self.topology_augmentation_step = topology_augmentation(topology_mutator, topology_augmentation_proba)

            if self.topology_reduction_step is None:
                _check_setting("topology_mutator", topology_mutator)
                _check_setting("topology_reduction_proba", topology_reduction_proba)
                self.topology_reduction_step = topology_reduction(topology_mutator, topology_reduction_proba)


    def produce_new_genome(self, genome_and_fitness_list) -> Genome:
        if self.custom_reproduction_pipeline is None:
            value = self.selection_step(genome_and_fitness_list)
            value = self.crossover_step(value)
            value = self.parameters_mutation_step(value)
            value = self.topology_augmentation_step(value)
            value = self.topology_reduction_step(value)
            return value
        else:
            value = genome_and_fitness_list
            for step in self.custom_reproduction_pipeline:
                value = step(value)
            return value
