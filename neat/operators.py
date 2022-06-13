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
    n.pure_upstream = False
    n.pure_downstream = False
    return n

def upstream_neuron(gene_spec: GeneSpec, non_removable=True, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    n = GeneDescription()
    n.non_removable = non_removable
    n.spec = gene_spec
    n.params = params
    n.pure_upstream = True
    n.pure_downstream = False
    return n

def downstream_neuron(gene_spec: GeneSpec, non_removable=True, **params):
    '''
    Helper function to use with Mutator.produce_genome
    '''
    n = GeneDescription()
    n.non_removable = non_removable
    n.spec = gene_spec
    n.params = params
    n.pure_upstream = False
    n.pure_downstream = True
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
        ):

        self.neuron_factory = neuron_factory
        self.connection_factory = connection_factory
        self.innovation_number = innovation_number
        self._non_removable_hmarks = set()
        self._pure_upstream_hmarks = set()
        self._pure_downstream_hmarks = set()



    def _find_free_connection_dense(self, genome):
        n_neurons = genome.num_neuron_genes() - len(self._pure_upstream_hmarks)
        src_marks = []
        cum_weights = []
        cum_weight = 0

        for g in genome.neuron_genes():
            if g.historical_mark in self._pure_downstream_hmarks:
                continue

            downstream_set = genome.connections_index.get(g.historical_mark, ())
            weight = n_neurons - len(downstream_set)
            if weight > 0:
                cum_weight += weight
                cum_weights.append(cum_weight)
                src_marks.append(g.historical_mark)

        if len(src_marks) == 0:
            # print(genome.connections_index)
            return None, None, False

        m0, = random.choices(src_marks, k=1, cum_weights=cum_weights)

        dst_marks = (g.historical_mark for g in genome.neuron_genes() \
            if g.historical_mark not in self._pure_upstream_hmarks)

        downstream_set = genome.connections_index.get(m0, ())
        dst_marks = (m for m in dst_marks if m not in downstream_set)

        dst_marks = tuple(dst_marks)
        if len(dst_marks) == 0:
            # print(f"m0={m0}: {genome.connections_index}")
            return None, None, False

        m1 = random.choice(dst_marks)
        return m0, m1, True


    def add_random_connection(self, genome, max_attempts=20):
        """
        Pick two neurons A and B at random. Make sure that connection AB does not exist.
        If that's the case, add new connection whose type is randomly selected from the set
        of allowed types and whose parameters are initialized according to spec for that type.

        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.
        """

        # TODO: rewrite this function better, get rid of attempts

        def _get_random_neuron_pair():
            neuron_from = random.choice(genome._neuron_genes)
            if neuron_from is None:
                return None, None, False

            neuron_to = random.choice(genome._neuron_genes)
            if neuron_to is None:
                return None, None, False

            mark_from = neuron_from.historical_mark
            mark_to = neuron_to.historical_mark
            are_valid = neuron_from.historical_mark not in self._pure_downstream_hmarks and \
                        neuron_to.historical_mark not in self._pure_upstream_hmarks
            return mark_from, mark_to, are_valid


        if genome.num_neuron_genes() > 4:
            mark_from, mark_to, are_valid = _get_random_neuron_pair()
            num_attempts = 0

            while not are_valid or genome.has_connection(mark_from, mark_to):
                num_attempts += 1
                if num_attempts < max_attempts:
                    mark_from, mark_to, are_valid = _get_random_neuron_pair()
                else:
                    mark_from, mark_to, are_valid = self._find_free_connection_dense(genome)
                    if are_valid:
                        break
                    else:
                        return False
        else:
            mark_from, mark_to, are_valid = self._find_free_connection_dense(genome)
            if not are_valid:
                return False


        new_connection_spec, new_connection_params = self.connection_factory()
        self.add_connection(
            genome,
            new_connection_spec,
            new_connection_params,
            mark_from,
            mark_to)

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




    def add_neuron(
        self,
        genome,
        neuron_spec,
        neuron_params,
        non_removable=False,
        pure_upstream=False,
        pure_downstream=False):
        new_neuron_gene = NeuronGene(
                                gene_spec = neuron_spec,
                                params = neuron_params,
                                historical_mark = self.innovation_number)

        self.innovation_number += 1
        genome.add_neuron_gene(new_neuron_gene)
        if non_removable:
            self._non_removable_hmarks.add(new_neuron_gene.historical_mark)
        if pure_upstream:
            self._pure_upstream_hmarks.add(new_neuron_gene.historical_mark)
        if pure_downstream:
            self._pure_downstream_hmarks.add(new_neuron_gene.historical_mark)
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
                pure_upstream=neuron_info.pure_upstream,
                pure_downstream=neuron_info.pure_downstream,
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
