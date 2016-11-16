import random
from operator import itemgetter

from .genes import GeneticEncoding
from .operators import crossover


class Conf(object): pass


def neuron(neuron_type, protected, **params):
    n = Conf()
    n.protected = protected
    n.type = neuron_type
    n.params = params
    return n


def connection(connection_type, protected, src, dst, **params):
    c = Conf()
    c.protected = protected
    c.type = connection_type
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
                raise TypeError("NEAT instance: please provide value for {}".format(setting_name))



    def get_init_genome(self, **kwargs):

        connections = kwargs.pop('connections', [])
        neurons = kwargs

        neuron_map = {} # {id: hist_mark}
        genome = GeneticEncoding()
        for neuron_id, neuron_info in neurons.items():
            hmark = self.mutator.add_neuron(
                genome,
                neuron_info.type,
                **neuron_info.params
            )
            neuron_map[neuron_id] = hmark
            if neuron_info.protected: self.mutator.protect_gene(hmark)

        for conn_info in connections:
            hmark = self.mutator.add_connection(
                genome,
                conn_info.type,
                mark_from = neuron_map[conn_info.src],
                mark_to = neuron_map[conn_info.dst],
                **conn_info.params
            )
            if conn_info.protected: self.mutator.protect_gene(hmark)
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
        shared_fitness = []

        for cur_brain, cur_fitness in genomes_fitnesses:
            species_size = 1
            for other_brain, other_fitness in genomes_fitnesses:
                if not other_brain == cur_brain:
                    distance = GeneticEncoding.get_dissimilarity(other_brain, cur_brain)
                    if distance < self.speciation_threshold: species_size += 1

            shared_fitness.append((cur_brain, cur_fitness / float(species_size)))
            
        return shared_fitness



    def select_for_tournament(self, candidates):
        selected = sorted(
            random.sample(candidates, self.tournament_size),
            key = itemgetter(1),
            reverse=True)

        return list(genome for (genome, fitness) in selected)



    def produce_new_generation(self, genome_fitness_list):
        gen_fit = genome_fitness_list

        new_genomes = []

        gen_fit_shared = self.share_fitness(gen_fit)

        gen_fit = sorted(gen_fit, key=itemgetter(1), reverse=True)
        gen_fit_shared = sorted(gen_fit_shared, key=itemgetter(1), reverse=True)

        # create children:
        for _ in range(self.pop_size - self.elite_size):
            # we select genomes using their shared fitnesses:
            selected = self.select_for_tournament(gen_fit_shared)

            # select 2 best parents from the tournament and make a child:
            child_genome = self.produce_child(selected[0], selected[1])
            new_genomes.append(child_genome)


        # bringing the best parents into next generation:
        for i in range(self.elite_size):
            new_genomes.append(gen_fit[i][0])

        return new_genomes