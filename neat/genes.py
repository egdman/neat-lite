from copy import copy, deepcopy

try:
    from yaml import dump as yaml_dump
except ImportError:
    yaml_dump = None

class Gene:
    __slots__ = ('spec', 'historical_mark', 'params')

    def __init__(self, gene_spec, params, historical_mark):
        self.spec = gene_spec
        self.historical_mark = historical_mark
        self.params = params


    def get_type(self):
        return self.spec.type_name


    def get_params_with_names(self):
        return zip(self.spec.iterate_param_names(), self.params)


    def copy(self):
        c = copy(self)
        c.params = copy(self.params)
        return c


    def to_dict(self):
        d = dict(
            gene_type=self.get_type(),
            historical_mark=self.historical_mark,
        )
        d.update(self.get_params_with_names())
        return d


class NeuronGene(Gene):
    __slots__ = ()

    def __str__(self):
        s = "node, mark: {}, type: {}".format(
            self.historical_mark,
            self.get_type(),
        )
        return s




class ConnectionGene(Gene):
    __slots__ = ('mark_from', 'mark_to')

    def __init__(self, gene_spec, params, mark_from, mark_to, historical_mark):
        super(ConnectionGene, self).__init__(gene_spec, params, historical_mark)

        self.mark_from = mark_from
        self.mark_to = mark_to


    def __str__(self):
        s = "link, mark: {}, type: {}, {} -> {}".format(
            self.historical_mark,
            self.get_type(),
            self.mark_from,
            self.mark_to,
        )
        return s


    def to_dict(self):
        d = dict(
            mark_from=self.mark_from,
            mark_to=self.mark_to,
        )
        d.update(super().to_dict())
        return d


class Genome:
    def __init__(self, neuron_genes=None, connection_genes=None):
        self._neuron_genes = neuron_genes if neuron_genes else []
        self._conn_genes = connection_genes if connection_genes else []
        self._neuron_num = len(self._neuron_genes)
        self._conn_num = len(self._conn_genes)

        self.connections_index = dict()
        for g in self._conn_genes:
            m0 = g.mark_from
            m1 = g.mark_to
            downstream_set = self.connections_index.get(m0, None)
            if downstream_set is None:
                self.connections_index[m0] = {m1}
            else:
                downstream_set.add(m1)


    def copy(self):
        g = Genome.__new__(Genome)
        g._neuron_genes = self._neuron_genes[:]
        g._conn_genes = self._conn_genes[:]
        g._neuron_num = self._neuron_num
        g._conn_num = self._conn_num
        g.connections_index = deepcopy(self.connections_index)
        return g


    def num_neuron_genes(self):
        return self._neuron_num


    def num_connection_genes(self):
        return self._conn_num


    def neuron_genes(self):
        if len(self._neuron_genes) == self._neuron_num:
            return self._neuron_genes
        else:
            return (g for g in self._neuron_genes if g is not None)


    def connection_genes(self):
        if len(self._conn_genes) == self._conn_num:
            return self._conn_genes
        else:
            return (g for g in self._conn_genes if g is not None)


    def has_connection(self, mark_from, mark_to):
        downstream_set = self.connections_index.get(mark_from, ())
        return mark_to in downstream_set


    def add_neuron_gene(self, neuron_gene):
        self._neuron_genes.append(neuron_gene)
        self._neuron_num += 1



    def add_connection_gene(self, connection_gene):
        self._conn_genes.append(connection_gene)
        self._conn_num += 1

        m0, m1 = connection_gene.mark_from, connection_gene.mark_to
        downstream_set = self.connections_index.get(m0, None)
        if downstream_set is None:
            self.connections_index[m0] = {m1}
        else:
            downstream_set.add(m1)



    def remove_connection_gene(self, index):
        g = self._conn_genes[index]
        self._conn_genes[index] = None
        self._conn_num -= 1

        # collect garbage
        if len(self._conn_genes) > 2 * self._conn_num:
            self._conn_genes = list(g for g in self._conn_genes if g is not None)

        downstream_set = self.connections_index[g.mark_from]
        downstream_set.discard(g.mark_to)



    def remove_neuron_gene(self, index):
        neuron_mark = self._neuron_genes[index].historical_mark
        self._neuron_genes[index] = None
        self._neuron_num -= 1

        # collect garbage
        if len(self._neuron_genes) > 2 * self._neuron_num:
            self._neuron_genes = list(g for g in self._neuron_genes if g is not None)

        # remove all attached connection genes
        for idx, g in enumerate(self._conn_genes):
            if g is not None and neuron_mark in (g.mark_from, g.mark_to):
                self._conn_genes[idx] = None
                self._conn_num -= 1

        # collect garbage
        if len(self._conn_genes) > 2 * self._conn_num:
            self._conn_genes = list(g for g in self._conn_genes if g is not None)

        # remove all downstream connections of the neuron
        self.connections_index.pop(neuron_mark, None)

        # now remove all the upstream connections
        for downstream_set in self.connections_index.values():
            downstream_set.discard(neuron_mark)


    def check_validity(self):
        neuron_hmarks = set(gene.historical_mark for gene in self.neuron_genes())
        if self._neuron_num != len(neuron_hmarks):
            return False

        conn_hmarks = set()
        for conn_gene in self.connection_genes():
            if conn_gene.historical_mark in conn_hmarks:
                return False
            if conn_gene.historical_mark in neuron_hmarks:
                return False
            if conn_gene.mark_from not in neuron_hmarks:
                return False
            if conn_gene.mark_to not in neuron_hmarks:
                return False
            conn_hmarks.add(conn_gene.historical_mark)
        return True



    def __str__(self):
        st = "neurons:\n"
        st += "\n".join((str(ng.to_dict()) for ng in self.neuron_genes()))
        st += "\nconnections:\n"
        st += "\n".join((str(cg.to_dict()) for cg in self.connection_genes()))
        return st


    def from_yaml(self, y_desc):
        del self._neuron_genes[:]
        del self._conn_genes[:]

        y_neurons = y_desc['neurons']
        y_connections = y_desc['connections']

        for y_neuron in y_neurons:
            # y_neuron.update(y_neuron.pop("params"))
            self.add_neuron_gene(NeuronGene(**y_neuron))

        for y_connection in y_connections:
            # y_connection.update(y_connection.pop("params"))
            self.add_connection_gene(ConnectionGene(**y_connection))

        return self


    def to_yaml(self):
        if yaml_dump is None:
            raise NotImplementedError("PyYaml not installed")

        neuron_genes = list(g.to_dict() for g in self.neuron_genes())
        conn_genes = list(g.to_dict() for g in self.connection_genes())
        yaml_repr = {'neurons': neuron_genes, 'connections' : conn_genes}
        return yaml_dump(yaml_repr, default_flow_style=False)


    @staticmethod
    def count_excess_disjoint(pairs):
        def consume_(lg, rg):
            if lg is None:
                n_unpaired = 1
                for lg, rg in pairs:
                    if lg is None:
                        n_unpaired += 1
                    else:
                        return lg, rg, n_unpaired, True

            elif rg is None:
                n_unpaired = 1
                for lg, rg in pairs:
                    if rg is None:
                        n_unpaired += 1
                    else:
                        return lg, rg, n_unpaired, True

            else:
                n_unpaired = 0
                for lg, rg in pairs:
                    if lg is None or rg is None:
                        return lg, rg, n_unpaired, True

            return lg, rg, n_unpaired, False

        pairs = iter(pairs)
        try:
            lg, rg = next(pairs)
        except StopIteration:
            return 0, 0

        lg, rg, excess_num, has_more = consume_(lg, rg)
        excess_tail = 0
        disjoint_num = 0
        while has_more:
            disjoint_num += excess_tail
            lg, rg, excess_tail, has_more = consume_(lg, rg)

        return excess_num + excess_tail, disjoint_num


    @staticmethod
    def get_pairs(genes_sorted1, genes_sorted2):
        left_genes = iter(genes_sorted1)
        right_genes = iter(genes_sorted2)

        left_gene = next(left_genes, None)
        right_gene = next(right_genes, None)

        while True:
            if left_gene is None:
                if right_gene is not None:
                    yield None, right_gene
                    yield from ((None, rg) for rg in right_genes)
                break

            elif right_gene is None:
                yield left_gene, None
                yield from ((lg, None) for lg in left_genes)
                break

            elif left_gene.historical_mark < right_gene.historical_mark:
                yield left_gene, None
                left_gene = next(left_genes, None)

            elif left_gene.historical_mark > right_gene.historical_mark:
                yield None, right_gene
                right_gene = next(right_genes, None)

            else:
                yield left_gene, right_gene
                left_gene = next(left_genes, None)
                right_gene = next(right_genes, None)
