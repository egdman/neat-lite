import random
from .specs import GeneSpec
from .genes import NeuronGene, ConnectionGene, Genome

class GeneDescription: pass


def neuron(gene_spec: GeneSpec, non_removable=False, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    n = GeneDescription()
    n.non_removable = non_removable
    n.spec = gene_spec
    n.params = params
    return n


def connection(gene_spec: GeneSpec, src, dst, non_removable=False, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    c = GeneDescription()
    c.non_removable = non_removable
    c.spec = gene_spec
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
        self._non_removable_hmarks = set()


    def _iter_all_unconnected_pairs(self, genome):
        for n0 in genome.neuron_genes():
            if n0.spec in self.pure_output_types:
                continue

            downstream_set = genome.connections_index.get(n0.historical_mark, ())
            for n1 in genome.neuron_genes():
                if n1.spec not in self.pure_input_types and \
                    n1.historical_mark not in downstream_set:
                    yield n0, n1


    def add_random_connection(self, genome, max_attempts=50):
        """
        Pick two neurons n0 and n1 at random. Check that connection n0->n1 does not exist.
        If that's the case, add a new connection from n0 to n1.
        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.
        If all attempts failed, the network is most likely dense, therefore use the dense selection process.
        """
        def _connect(n0, n1):
            new_connection_spec, new_connection_params = self.connection_factory()
            self.add_connection(
                genome,
                new_connection_spec,
                new_connection_params,
                n0.historical_mark,
                n1.historical_mark)

        neuron_genes = genome._neuron_genes

        n_attempt = 0
        while n_attempt < max_attempts:
            n_attempt += 1

            n0 = random.choice(neuron_genes)
            while n0 is None:
                n0 = random.choice(neuron_genes)

            n1 = random.choice(neuron_genes)
            while n1 is None:
                n1 = random.choice(neuron_genes)

            if n0.spec in self.pure_output_types:
                continue
            if n1.spec in self.pure_input_types:
                continue
            if genome.has_connection(n0.historical_mark, n1.historical_mark):
                continue

            _connect(n0, n1)
            return True

        pairs = tuple(self._iter_all_unconnected_pairs(genome))
        if len(pairs) == 0:
            return False

        n0, n1 = random.choice(pairs)
        _connect(n0, n1)
        return True


    def _unprotected_connection_ids(self, genome):
        if len(genome._conn_genes) == genome._conn_num:
            return tuple(idx for idx, g in enumerate(genome._conn_genes) \
                if g.historical_mark not in self._non_removable_hmarks)
        else:
            return tuple(idx for idx, g in enumerate(genome._conn_genes) \
                if g is not None and g.historical_mark not in self._non_removable_hmarks)



    def _unprotected_neuron_ids(self, genome):
        if len(genome._neuron_genes) == genome._neuron_num:
            return tuple(idx for idx, g in enumerate(genome._neuron_genes) \
                if g.historical_mark not in self._non_removable_hmarks)
        else:
            return tuple(idx for idx, g in enumerate(genome._neuron_genes) \
                if g is not None and g.historical_mark not in self._non_removable_hmarks)




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

        # delete the old connection from the genome
        genome.remove_connection_gene(connection_to_split_id)

        # insert new neuron
        new_neuron_spec, new_neuron_params = self.neuron_factory()
        mark_middle = self.add_neuron(genome, new_neuron_spec, new_neuron_params)

        self.add_connection(
            genome,
            connection_to_split.spec,
            connection_to_split.params,
            connection_to_split.mark_from,
            mark_middle)

        new_connection_spec, new_connection_params = self.connection_factory()
        self.add_connection(
            genome,
            new_connection_spec,
            new_connection_params,
            mark_middle,
            connection_to_split.mark_to)




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




    def add_neuron(self, genome, neuron_spec, neuron_params, non_removable=False):
        new_neuron_gene = NeuronGene(
                                gene_spec = neuron_spec,
                                params = neuron_params,
                                historical_mark = self.innovation_number)

        self.innovation_number += 1
        genome.add_neuron_gene(new_neuron_gene)
        if non_removable:
            self._non_removable_hmarks.add(new_neuron_gene.historical_mark)
        return new_neuron_gene.historical_mark



    def add_connection(self, genome, connection_spec, connection_params, mark_from, mark_to, non_removable=False):
        new_conn_gene = ConnectionGene(
                                  gene_spec = connection_spec,
                                  params = connection_params,
                                  mark_from = mark_from,
                                  mark_to = mark_to,
                                  historical_mark = self.innovation_number)

        self.innovation_number += 1
        genome.add_connection_gene(new_conn_gene)
        if non_removable:
            self._non_removable_hmarks.add(new_neuron_gene.historical_mark)
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

        def _make_params(spec, provided_params):
            gen = spec.parameter_values_generator()
            for name in spec.iterate_param_names():
                if name in provided_params:
                    yield provided_params[name]
                else:
                    yield next(gen)

        # add neuron genes to genome using mutator
        neuron_map = {}
        for neuron_id, neuron_info in neurons.items():
            hmark = self.add_neuron(
                genome,
                neuron_info.spec,
                list(_make_params(neuron_info.spec, neuron_info.params)),
                non_removable=neuron_info.non_removable,
            )
            neuron_map[neuron_id] = hmark

        # add connection genes to genome using mutator
        for conn_info in connections:
            self.add_connection(
                genome,
                conn_info.spec,
                list(_make_params(conn_info.spec, conn_info.params)),
                mark_from = neuron_map[conn_info.src],
                mark_to = neuron_map[conn_info.dst],
                non_removable=conn_info.non_removable,
            )

        return genome


def crossover(genome_primary, genome_secondary) -> Genome:
    '''
    Perform crossover of two genomes. The input genomes are kept unchanged.
    The first genome in the arguments will provide 100% of unpaired genes.
    '''
    neuron_pairs = Genome.get_pairs(
        genome_primary.neuron_genes(),
        genome_secondary.neuron_genes())

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
