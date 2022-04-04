import random
from itertools import chain

from .genes import NeuronGene, ConnectionGene, GeneticEncoding
from .utils import zip_with_probabilities, weighted_random


class GeneInfo(object): pass


def neuron(gene_type, protected, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    n = GeneInfo()
    n.protected = protected
    n.type = gene_type
    n.params = params
    return n


def connection(gene_type, protected, src, dst, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    c = GeneInfo()
    c.protected = protected
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


    def add_random_connection(self, genotype, max_attempts=100):

        """
        Pick two neurons A and B at random. Make sure that connection AB does not exist.
        If that's the case, add new connection whose type is randomly selected from the set
        of allowed types and whose parameters are initialized according to spec for that type.

        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.
        """

        # TODO: rewrite this function better, get rid of attempts

        def _get_pair_neurons():
            neuron_from = random.choice(genotype.neuron_genes)
            neuron_to = random.choice(genotype.neuron_genes)
            mark_from = neuron_from.historical_mark
            mark_to = neuron_to.historical_mark
            are_valid = neuron_from.gene_type not in self.pure_output_types and \
                        neuron_to.gene_type not in self.pure_input_types
            return mark_from, mark_to, are_valid

        mark_from, mark_to, are_valid = _get_pair_neurons()
        num_attempts = 0

        while len(genotype.get_connection_genes(mark_from, mark_to)) > 0 or not are_valid:
            mark_from, mark_to, are_valid = _get_pair_neurons()
            num_attempts += 1
            if num_attempts >= max_attempts: return False

        new_connection_type, new_connection_params = self.connection_factory()
        self.add_connection(
            genotype,
            new_connection_type,
            mark_from,
            mark_to,
            **new_connection_params)

        return True



    def _unprotected_connection_ids(self, genotype):
        return list(cg_i for cg_i, cg in enumerate(genotype.connection_genes) \
            if not cg.protected)



    def _unprotected_neuron_ids(self, genotype):
        return list(ng_i for ng_i, ng in enumerate(genotype.neuron_genes) \
            if not ng.protected)




    def add_random_neuron(self, genotype):

        """
        Pick a connection at random from neuron A to neuron B.
        And add a neuron C in between A and B.
        Old connection AB gets deleted.
        Two new connections AC and CB are added.
        Connection AC will have the same type and parameters as AB.
        Connection CB will have random type (chosen from the allowed ones)
        and randomly initialized parameters.

        :type genotype: GeneticEncoding
        """

        unprotected_conn_ids = self._unprotected_connection_ids(genotype)
        # unprotected_conn_ids = range(len(genotype.connection_genes))
        if len(unprotected_conn_ids) == 0: return

        connection_to_split_id = random.choice(unprotected_conn_ids)
        connection_to_split = genotype.connection_genes[connection_to_split_id]


        # get all the info about the old connection
        old_connection_type = connection_to_split.gene_type
        old_connection_params = connection_to_split.copy_params()

        mark_from = connection_to_split.mark_from
        mark_to = connection_to_split.mark_to


        # delete the old connection from the genotype
        genotype.remove_connection_gene(connection_to_split_id)

        # insert new neuron
        new_neuron_type, new_neuron_params = self.neuron_factory()
        mark_middle = self.add_neuron(genotype, new_neuron_type, **new_neuron_params)

        self.add_connection(
            genotype,
            old_connection_type,
            mark_from,
            mark_middle,
            **old_connection_params)

        new_connection_type, new_connection_params = self.connection_factory()
        self.add_connection(
            genotype,
            new_connection_type,
            mark_middle,
            mark_to,
            **new_connection_params)




    def remove_random_connection(self, genotype):
        unprotected_conn_ids = self._unprotected_connection_ids(genotype)
        # unprotected_conn_ids = range(len(genotype.connection_genes))
        if len(unprotected_conn_ids) == 0: return
        gene_id = random.choice(unprotected_conn_ids)
        genotype.remove_connection_gene(gene_id)



    def remove_random_neuron(self, genotype):
        unprotected_neuron_ids = self._unprotected_neuron_ids(genotype)
        if len(unprotected_neuron_ids) == 0: return

        gene_id = random.choice(unprotected_neuron_ids)

        neuron_gene = genotype.neuron_genes[gene_id]
        neuron_mark = neuron_gene.historical_mark

        # find indices of attached connection genes:
        bad_connections = list(g_id for g_id, gene \
            in enumerate(genotype.connection_genes)\
            if gene.mark_from == neuron_mark or gene.mark_to == neuron_mark)


        # remove attached connection genes
        # (list is reversed because indices will be screwed up otherwise)
        for g_id in reversed(bad_connections):
            genotype.remove_connection_gene(g_id)

        # remove the neuron gene:
        genotype.remove_neuron_gene(gene_id)




    def add_neuron(self, genotype, neuron_type, protected=False, **neuron_params):
        new_neuron_gene = NeuronGene(
                                gene_type = neuron_type,
                                historical_mark = self.innovation_number,
                                protected = protected,
                                **neuron_params)

        self.innovation_number += 1
        genotype.add_neuron_gene(new_neuron_gene)
        return new_neuron_gene.historical_mark



    def add_connection(self, genotype, connection_type, mark_from, mark_to, protected=False, **connection_params):
        new_conn_gene = ConnectionGene(
                                  gene_type = connection_type,
                                  mark_from = mark_from,
                                  mark_to = mark_to,
                                  historical_mark = self.innovation_number,
                                  protected = protected,
                                  **connection_params)

        self.innovation_number += 1
        genotype.add_connection_gene(new_conn_gene)
        return new_conn_gene.historical_mark


    def produce_genome(self, **genes):
        connections = genes.pop('connections', [])
        neurons = genes
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
            hmark = self.add_neuron(
                genome,
                neuron_info.type,
                protected=neuron_info.protected,
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
                protected=conn_info.protected,
                **conn_info.params
            )

        return genome


def crossover(genotype_more_fit, genotype_less_fit):
    '''
    Perform crossover of two genotypes. The input genotypes are kept unchanged.
    The first genotype in the arguments must be the more fit one.
    '''

    # copy original genotypes to keep them intact:
    genotype_more_fit = genotype_more_fit.copy()
    genotype_less_fit = genotype_less_fit.copy()


    # sort genes by historical marks:
    genes_better = sorted(chain(
        genotype_more_fit.neuron_genes,
        genotype_more_fit.connection_genes),
        key = lambda gene: gene.historical_mark)

    genes_worse = sorted(chain(
        genotype_less_fit.neuron_genes,
        genotype_less_fit.connection_genes),
        key = lambda gene: gene.historical_mark)

    gene_pairs = GeneticEncoding.get_pairs(genes_better, genes_worse)

    child_genes = []

    for gene0, gene1 in gene_pairs:

        # if gene is paired, inherit one of the pair with 50/50 chance:
        if gene0 is not None and gene1 is not None:
            if random.random() < 0.5:
                child_genes.append(gene0)
            else:
                child_genes.append(gene1)

        # inherit unpaired gene from the more fit parent:
        elif gene0 is not None:
            child_genes.append(gene0)

    child_genotype = GeneticEncoding()
    for gene in child_genes:
        if isinstance(gene, NeuronGene):
            child_genotype.add_neuron_gene(gene)
        elif isinstance(gene, ConnectionGene):
            child_genotype.add_connection_gene(gene)

    return child_genotype