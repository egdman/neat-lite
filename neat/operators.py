import random
from itertools import chain

from .genes import NeuronGene, ConnectionGene, Genome
from .utils import zip_with_probabilities, weighted_random


class GeneDescription(object): pass


def neuron(gene_type, non_removable=False, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    n = GeneDescription()
    n.non_removable = non_removable
    n.type = gene_type
    n.params = params
    return n


def connection(gene_type, src, dst, non_removable=False, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    c = GeneDescription()
    c.non_removable = non_removable
    c.type = gene_type
    c.src = src
    c.dst = dst
    c.params = params
    return c


class Mutator:

    def __init__(self,
        neuron_factory,
        connection_factory,
        innovation_number = 0, # starting innovation number
        pure_input_types = tuple(), # list of input-only neuron types (can't attach loopback inputs)
        pure_output_types = tuple() # list of output-only neuron types (can't attach loopback outputs)
        ):

        self.neuron_factory = neuron_factory
        self.connection_factory = connection_factory
        self.pure_input_types = pure_input_types
        self.pure_output_types = pure_output_types
        self.innovation_number = innovation_number


    def add_random_connection(self, genome, max_attempts=100):
        """
        Pick two neurons A and B at random. Make sure that connection AB does not exist.
        If that's the case, add new connection whose type is randomly selected from the set
        of allowed types and whose parameters are initialized according to spec for that type.

        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.
        """

        # TODO: rewrite this function better, get rid of attempts

        def _get_pair_neurons():
            neuron_from = random.choice(genome.neuron_genes)
            neuron_to = random.choice(genome.neuron_genes)
            mark_from = neuron_from.historical_mark
            mark_to = neuron_to.historical_mark
            are_valid = neuron_from.gene_type not in self.pure_output_types and \
                        neuron_to.gene_type not in self.pure_input_types
            return mark_from, mark_to, are_valid

        mark_from, mark_to, are_valid = _get_pair_neurons()
        num_attempts = 0

        while not are_valid or genome.has_connection(mark_from, mark_to):
            mark_from, mark_to, are_valid = _get_pair_neurons()
            num_attempts += 1
            if num_attempts >= max_attempts: return False

        new_connection_type, new_connection_params = self.connection_factory()
        self.add_connection(
            genome,
            new_connection_type,
            mark_from,
            mark_to,
            **new_connection_params)

        return True



    def _unprotected_connection_ids(self, genome):
        return list(cg_i for cg_i, cg in enumerate(genome._conn_genes) \
            if cg is not None and not cg.non_removable)



    def _unprotected_neuron_ids(self, genome):
        return list(ng_i for ng_i, ng in enumerate(genome.neuron_genes) \
            if not ng.non_removable)




    def add_random_neuron(self, genome):
        """
        Pick a connection at random from neuron A to neuron B.
        And add a neuron C in between A and B.
        Old connection AB gets deleted.
        Two new connections AC and CB are added.
        Connection AC will be a copy of AB, but with a new h-mark.
        Connection CB will be newly generated.
        """

        unprotected_conn_ids = self._unprotected_connection_ids(genome)
        if len(unprotected_conn_ids) == 0: return

        connection_to_split_id = random.choice(unprotected_conn_ids)
        connection_to_split = genome._conn_genes[connection_to_split_id]
        while connection_to_split is None:
            connection_to_split_id = random.choice(unprotected_conn_ids)
            connection_to_split = genome._conn_genes[connection_to_split_id]


        # get all the info about the old connection
        old_connection_type = connection_to_split.gene_type
        old_connection_params = connection_to_split.get_params()

        mark_from = connection_to_split.mark_from
        mark_to = connection_to_split.mark_to


        # delete the old connection from the genome
        genome.remove_connection_gene(connection_to_split_id)

        # insert new neuron
        new_neuron_type, new_neuron_params = self.neuron_factory()
        mark_middle = self.add_neuron(genome, new_neuron_type, **new_neuron_params)

        self.add_connection(
            genome,
            old_connection_type,
            mark_from,
            mark_middle,
            **old_connection_params)

        new_connection_type, new_connection_params = self.connection_factory()
        self.add_connection(
            genome,
            new_connection_type,
            mark_middle,
            mark_to,
            **new_connection_params)




    def remove_random_connection(self, genome):
        unprotected_conn_ids = self._unprotected_connection_ids(genome)
        if len(unprotected_conn_ids) == 0: return
        gene_id = random.choice(unprotected_conn_ids)
        genome.remove_connection_gene(gene_id)



    def remove_random_neuron(self, genome):
        unprotected_neuron_ids = self._unprotected_neuron_ids(genome)
        if len(unprotected_neuron_ids) == 0: return
        gene_id = random.choice(unprotected_neuron_ids)
        genome.remove_neuron_gene(gene_id)




    def add_neuron(self, genome, neuron_type, non_removable=False, **neuron_params):
        new_neuron_gene = NeuronGene(
                                gene_type = neuron_type,
                                historical_mark = self.innovation_number,
                                non_removable = non_removable,
                                **neuron_params)

        self.innovation_number += 1
        genome.add_neuron_gene(new_neuron_gene)
        return new_neuron_gene.historical_mark



    def add_connection(self, genome, connection_type, mark_from, mark_to, non_removable=False, **connection_params):
        new_conn_gene = ConnectionGene(
                                  gene_type = connection_type,
                                  mark_from = mark_from,
                                  mark_to = mark_to,
                                  historical_mark = self.innovation_number,
                                  non_removable = non_removable,
                                  **connection_params)

        self.innovation_number += 1
        genome.add_connection_gene(new_conn_gene)
        return new_conn_gene.historical_mark


    def produce_genome(self, **genes) -> Genome:
        """
        Helper function that allows to create a genome by describing its neurons and connections
        """
        connections = genes.pop('connections', ())
        neurons = genes
        genome = Genome()

        # if we want to protect a connection from removal we also
        # want to protect its 2 adjacent neurons
        # because we cannot remove a neuron without removing
        # its adjacent connections
        for conn_info in connections:
            if conn_info.non_removable:
                neurons[conn_info.src].non_removable = True
                neurons[conn_info.dst].non_removable = True

        # add neuron genes to genome using mutator
        neuron_map = {}
        for neuron_id, neuron_info in neurons.items():
            hmark = self.add_neuron(
                genome,
                neuron_info.type,
                non_removable=neuron_info.non_removable,
                **neuron_info.params
            )
            neuron_map[neuron_id] = hmark

        # add connection genes to genome using mutator
        for conn_info in connections:
            self.add_connection(
                genome,
                conn_info.type,
                mark_from = neuron_map[conn_info.src],
                mark_to = neuron_map[conn_info.dst],
                non_removable=conn_info.non_removable,
                **conn_info.params
            )

        return genome


def crossover(genome_primary, genome_secondary) -> Genome:
    '''
    Perform crossover of two genomes. The input genomes are kept unchanged.
    The first genome in the arguments will provide 100% of unpaired genes.
    '''
    neuron_pairs = Genome.get_pairs(
        genome_primary.neuron_genes,
        genome_secondary.neuron_genes)

    neuron_genes = []
    for gene0, gene1 in neuron_pairs:
        # if gene is paired, inherit one of the pair with 50/50 chance:
        if gene0 is not None and gene1 is not None:
            if random.random() < 0.5:
                neuron_genes.append(gene0)
            else:
                neuron_genes.append(gene1)

        # inherit unpaired gene from the primary parent:
        elif gene0 is not None:
            neuron_genes.append(gene0)

    connect_pairs = Genome.get_pairs(
        genome_primary.connection_genes(),
        genome_secondary.connection_genes())

    connect_genes = []
    for gene0, gene1 in connect_pairs:
        if gene0 is not None and gene1 is not None:
            if random.random() < 0.5:
                connect_genes.append(gene0)
            else:
                connect_genes.append(gene1)

        elif gene0 is not None:
            connect_genes.append(gene0)

    return Genome(neuron_genes, connect_genes)
