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
        self._channels = tuple(channels)
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
        src_neurons = genome.layers()[src_type]
        dst_neurons = genome.layers()[dst_type]

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
        if self.neuron_factory is None:
            return self.add_random_connection(genome)

        new_neuron_layer, new_neuron_params = self.neuron_factory()

        up_weights = []
        up_layers = []
        up_acc_weight = 0
        down_weights = []
        down_layers = []
        down_acc_weight = 0

        for src_l, dst_l in self._channels:
            if src_l is new_neuron_layer:
                layer_weight = genome.num_neurons_in_layer(dst_l)
                down_acc_weight += layer_weight
                down_weights.append(down_acc_weight)
                down_layers.append(dst_l)

                if src_l is dst_l:
                    up_acc_weight += layer_weight
                    up_weights.append(up_acc_weight)
                    up_layers.append(src_l)

            elif dst_l is new_neuron_layer:
                up_acc_weight += genome.num_neurons_in_layer(src_l)
                up_weights.append(up_acc_weight)
                up_layers.append(src_l)

        if up_acc_weight > 0:
            up_layer, = random.choices(up_layers, k=1, cum_weights=up_weights)
            up_neurons = genome.layers()[up_layer]
            up_neuron = random.choice(up_neurons)
            while up_neuron is None:
                up_neuron = random.choice(up_neurons)

        else:
            up_neuron = None

        if down_acc_weight > 0:
            down_layer, = random.choices(down_layers, k=1, cum_weights=down_weights)
            down_neurons = genome.layers()[down_layer]
            down_neuron = random.choice(down_neurons)
            while down_neuron is None:
                down_neuron = random.choice(down_neurons)

        else:
            down_neuron = None


        def _connect(n0, n1, channel):
            new_connection_spec, new_connection_params = self.connection_factory()
            self.add_connection(
                genome,
                new_connection_spec,
                new_connection_params,
                n0.historical_mark,
                n1.historical_mark,
                channel)

        new_neuron = self.add_neuron(
            genome, new_neuron_layer, new_neuron_params)

        if up_neuron is not None:
            _connect(up_neuron, new_neuron, (up_layer, new_neuron_layer))

        if down_neuron is not None:
            _connect(new_neuron, down_neuron, (new_neuron_layer, down_layer))

        return True


    def _unprotected_neuron_ids(self, genome):
        for layer, gs in genome.layers().items():
            for idx, g in enumerate(gs):
                if g is not None and g.historical_mark not in self._non_removable_hmarks:
                    yield layer, idx


    def _unprotected_connection_ids(self, genome):
        for channel, gs in genome.channels().items():
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
        layer, idx = random.choice(unprotected_neuron_ids)
        genome.remove_neuron_gene(layer, idx)


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
            self._non_removable_hmarks.add(new_conn_gene.historical_mark)
        return new_conn_gene


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

    new_genome = Genome()

    for layer, primary_neurons in genome_primary.layers().items():
        neuron_pairs = Genome.align_genes(
            primary_neurons.iter_non_empty(),
            genome_secondary.iterate_layer(layer))

        new_genome.add_layer(layer, _cross_genes(neuron_pairs))

    for channel, primary_connections in genome_primary.channels().items():
        connect_pairs = Genome.align_genes(
            primary_connections.iter_non_empty(),
            genome_secondary.iterate_channel(channel))

        new_genome.add_channel(channel, _cross_genes(connect_pairs))

    return new_genome
