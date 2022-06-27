import random
from .specs import GeneSpec
from .genes import Gene, ConnectionGene, Genome

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
        channels,
        innovation_number=0):

        self.innovation_number = innovation_number
        self.neuron_factory = neuron_factory
        self.connection_factory = connection_factory
        self._channels = channels
        self._non_removable_hmarks = set()


    def add_random_connection(self, genome, max_attempts=50):
        """
        Pick two neurons n0 and n1 at random. Check that connection n0->n1 does not exist.
        If that's the case, add a new connection from n0 to n1.
        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.
        """

        """
        TODO:
        If all attempts failed, the channel is most likely dense, therefore use the dense selection process.
        """

        def _connect(n0, n1, channel):
            new_connection_spec, new_connection_params = self.connection_factory()
            self.add_connection(
                genome,
                new_connection_spec,
                new_connection_params,
                n0.historical_mark,
                n1.historical_mark,
                channel)

        channel_weights = []
        acc_weight = 0
        for channel in self._channels:
            acc_weight += genome.calc_channel_capacity(channel)
            channel_weights.append(acc_weight)

        if acc_weight == 0:
            return False

        # TODO: see if can implement weighted random choice more efficiently using bisect
        channel, = random.choices(self._channels, k=1, cum_weights=channel_weights)
        src_type, dst_type = channel
        src_neurons = genome.neurons_dict()[src_type]
        dst_neurons = genome.neurons_dict()[dst_type]

        n_attempt = 0
        while n_attempt < max_attempts:
            n_attempt += 1

            n0 = random.choice(src_neurons)
            while n0 is None:
                n0 = random.choice(src_neurons)

            n1 = random.choice(dst_neurons)
            while n1 is None:
                n1 = random.choice(dst_neurons)

            if genome.has_connection(n0.historical_mark, n1.historical_mark):
                continue

            _connect(n0, n1, channel)
            return True
        return False


    def add_random_neuron(self, genome):
        new_neuron_spec, new_neuron_params = self.neuron_factory()

        # find channels for this spec:
        channels = tuple(ch for ch in self._channels if new_neuron_spec in ch)
        if len(channels) == 0:
            return self.add_random_connection(genome)

        def _connect(n0, n1, channel):
            new_connection_spec, new_connection_params = self.connection_factory()
            self.add_connection(
                genome,
                new_connection_spec,
                new_connection_params,
                n0.historical_mark,
                n1.historical_mark,
                channel)

        new_n = self.add_neuron(
            genome, new_neuron_spec, new_neuron_params)

        channel_weights = []
        acc_weight = 0
        for channel in channels:
            acc_weight += genome.calc_channel_capacity(channel)
            channel_weights.append(acc_weight)

        if acc_weight == 0:
            return False

        # TODO: see if can implement weighted random choice more efficiently using bisect
        channel, = random.choices(channels, k=1, cum_weights=channel_weights)
        src_type, dst_type = channel
        src_neurons = genome.neurons_dict()[src_type]

        if src_type is dst_type or new_neuron_spec is dst_type:
            n0 = random.choice(src_neurons)
            while n0 is None:
                n0 = random.choice(src_neurons)

            _connect(n0, new_n, channel)
        else:
            dst_neurons = genome.neurons_dict()[dst_type]

            n1 = random.choice(dst_neurons)
            while n1 is None:
                n1 = random.choice(dst_neurons)

            _connect(new_n, n1, channel)

        return True


    def _unprotected_neuron_ids(self, genome):
        for spec, gs in genome.neurons_dict().items():
            for idx, g in enumerate(gs):
                if g is not None and g.historical_mark not in self._non_removable_hmarks:
                    yield spec, idx


    def _unprotected_connection_ids(self, genome):
        for channel, gs in genome.connections_dict().items():
            for idx, g in enumerate(gs):
                if g is not None and g.historical_mark not in self._non_removable_hmarks:
                    yield channel, idx


    def remove_random_connection(self, genome):
        unprotected_conn_ids = tuple(self._unprotected_connection_ids(genome))
        if len(unprotected_conn_ids) == 0: return
        channel, idx = random.choice(unprotected_conn_ids)
        genome.remove_connection_gene(channel, idx)


    def remove_random_neuron(self, genome):
        unprotected_neuron_ids = tuple(self._unprotected_neuron_ids(genome))
        if len(unprotected_neuron_ids) == 0: return
        spec, idx = random.choice(unprotected_neuron_ids)
        genome.remove_neuron_gene(spec, idx)


    def add_neuron(self, genome, neuron_spec, neuron_params, non_removable=False):
        new_neuron_gene = Gene(
            gene_spec=neuron_spec,
            params=neuron_params,
            historical_mark=self.innovation_number)

        self.innovation_number += 1
        genome.add_neuron_gene(new_neuron_gene)
        if non_removable:
            self._non_removable_hmarks.add(new_neuron_gene.historical_mark)
        return new_neuron_gene



    def add_connection(self, genome, connection_spec, connection_params, mark_from, mark_to, channel, non_removable=False):
        new_conn_gene = ConnectionGene(
          gene_spec=connection_spec,
          params=connection_params,
          mark_from=mark_from,
          mark_to=mark_to,
          historical_mark=self.innovation_number)

        self.innovation_number += 1
        genome.add_connection_gene(new_conn_gene, channel)
        if non_removable:
            self._non_removable_hmarks.add(new_neuron_gene.historical_mark)
        return new_conn_gene


    def produce_genome(self, **genes) -> Genome:
        """
        Helper function that allows to create a genome by describing its neurons and connections
        """
        connections = genes.pop('connections', ())
        neurons = genes
        genome = Genome(())

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
            neuron_gene = self.add_neuron(
                genome,
                neuron_info.spec,
                list(_make_params(neuron_info.spec, neuron_info.params)),
                non_removable=neuron_info.non_removable,
            )
            neuron_map[neuron_id] = neuron_gene

        # add connection genes to genome using mutator
        for conn_info in connections:
            n0 = neuron_map[conn_info.src]
            n1 = neuron_map[conn_info.dst]

            self.add_connection(
                genome,
                conn_info.spec,
                list(_make_params(conn_info.spec, conn_info.params)),
                mark_from=n0.historical_mark,
                mark_to=n1.historical_mark,
                channel=(n0.spec, n1.spec),
                non_removable=conn_info.non_removable,
            )

        return genome


def crossover(genome_primary, genome_secondary) -> Genome:
    '''
    Perform crossover of two genomes. The input genomes are kept unchanged.
    The first genome in the arguments will provide 100% of unpaired genes.
    '''
    def _cross_genes(gene_pairs):
        for gene_prim, gene_sec in gene_pairs:
            # inherit one of paired genes with 50/50 chance,
            # and inherit unpaired genes only from the primary parent.
            if gene_prim is None:
                continue
            elif gene_sec is None or random.random() < .5:
                yield gene_prim
            else:
                yield gene_sec

    def _cross_neurons():
        for spec, primary_neurons in genome_primary.neurons_dict().items():
            neuron_pairs = Genome.align_genes(
                primary_neurons.iter_non_empty(),
                genome_secondary.neurons_with_spec(spec))

            yield from _cross_genes(neuron_pairs)

    new_genome = Genome(_cross_neurons())

    for channel, primary_connections in genome_primary.connections_dict().items():
        connect_pairs = Genome.align_genes(
            primary_connections.iter_non_empty(),
            genome_secondary.connections_in_channel(channel))

        new_genome.add_channel(channel, _cross_genes(connect_pairs))

    return new_genome
