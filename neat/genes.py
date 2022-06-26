from itertools import chain, groupby

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
        c = self.__new__(Gene)
        c.spec = self.spec
        c.historical_mark = self.historical_mark
        c.params = [*self.params]
        return c


    def to_dict(self):
        d = dict(
            gene_type=self.get_type(),
            historical_mark=self.historical_mark,
        )
        d.update(self.get_params_with_names())
        return d


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


    def copy(self):
        c = self.__new__(ConnectionGene)
        c.spec = self.spec
        c.historical_mark = self.historical_mark
        c.params = [*self.params]
        c.mark_from = self.mark_from
        c.mark_to = self.mark_to
        return c


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


class ListWithEmpty:
    def __init__(self, items=()):
        self._l = list(items)
        self._n = len(self._l)

    def copy_non_empty(self):
        c = self.__new__(ListWithEmpty)
        c._l = [item for item in self._l if item is not None]
        c._n = self._n
        return c

    def iter_non_empty(self):
        return (item for item in self._l if item is not None)

    def set_empty(self, key):
        self._l[key] = None
        self._n -= 1

    def non_empty_count(self):
        return self._n

    def __len__(self):
        return len(self._l)

    def __getitem__(self, key):
        return self._l[key]

    def __setitem__(self, key, item):
        self._l[key] = item

    def __iter__(self):
        return iter(self._l)

    def append(self, item):
        # item is assumed to be not None
        self._l.append(item)
        self._n += 1


class Genome:
    def __init__(self, neuron_genes):
        '''
        neuron_genes sequence is expected to be sorted by the .spec attribute.
        '''
        self._neuron_genes = {neuron_spec: ListWithEmpty(neurons)
            for neuron_spec, neurons
            in groupby(neuron_genes, key=lambda g: g.spec)
        }
        self._conn_genes = dict()
        self.connections_index = dict()


    def copy(self):
        g = self.__new__(Genome)
        g._neuron_genes = {neuron_spec: neurons.copy_non_empty()
            for neuron_spec, neurons
            in self._neuron_genes.items()
        }
        g._conn_genes = {channel: connections.copy_non_empty()
            for channel, connections
            in self._conn_genes.items()
        }
        g.connections_index = {m0: set(downstream_set)
            for m0, downstream_set in self.connections_index.items()
        }
        return g


    def add_channel(self, channel, connection_genes):
        gs = ListWithEmpty(connection_genes)
        self._conn_genes[channel] = gs
        for g in gs:
            m0 = g.mark_from
            m1 = g.mark_to
            downstream_set = self.connections_index.get(m0, None)
            if downstream_set is None:
                self.connections_index[m0] = {m1}
            else:
                downstream_set.add(m1)


    def calc_channel_capacity(self, channel):
        src_spec, dst_spec = channel

        src_genes = self._neuron_genes.get(src_spec, None)
        if src_genes is None:
            return 0

        dst_genes = self._neuron_genes.get(dst_spec, None)
        if dst_genes is None:
            return 0

        # this is how many connections channel can have in total
        capacity = src_genes.non_empty_count()*dst_genes.non_empty_count()

        # however some of them might already exist
        connections = self._conn_genes.get((src_spec, dst_spec), None)
        if connections is None:
            return capacity

        return capacity - connections.non_empty_count()


    def connection_genes(self):
        return chain(*(gs.iter_non_empty() for gs in self._conn_genes.values()))


    def connections_in_channel(self, channel):
        cs = self._conn_genes.get(channel, None)
        if cs is None:
            return ()
        else:
            return cs.iter_non_empty()


    def has_connection(self, mark_from, mark_to):
        downstream_set = self.connections_index.get(mark_from, ())
        return mark_to in downstream_set


    def connections_dict(self):
        return self._conn_genes


    def neuron_genes(self):
        return chain(*(gs.iter_non_empty() for gs in self._neuron_genes.values()))


    def neurons_dict(self):
        return self._neuron_genes


    def neurons_with_spec(self, spec):
        gs = self._neuron_genes.get(spec, None)
        if gs is None:
            return ()
        else:
            return gs.iter_non_empty()


    def add_neuron_gene(self, neuron_gene):
        gs = self._neuron_genes.get(neuron_gene.spec, None)
        if gs is None:
            self._neuron_genes[neuron_gene.spec] = ListWithEmpty((neuron_gene,))
        else:
            gs.append(neuron_gene)


    def add_connection_gene(self, connection_gene, channel):
        gs = self._conn_genes.get(channel, None)
        if gs is None:
            self._conn_genes[channel] = ListWithEmpty((connection_gene,))
        else:
            gs.append(connection_gene)

        m0, m1 = connection_gene.mark_from, connection_gene.mark_to
        downstream_set = self.connections_index.get(m0, None)
        if downstream_set is None:
            self.connections_index[m0] = {m1}
        else:
            downstream_set.add(m1)


    def remove_connection_gene(self, channel, index):
        gs = self._conn_genes[channel]
        del_conn = gs[index]
        gs.set_empty(index)

        # collect garbage
        if len(gs) > 2 * gs.non_empty_count():
            self._conn_genes[channel] = gs.copy_non_empty()

        downstream_set = self.connections_index[del_conn.mark_from]
        downstream_set.discard(del_conn.mark_to)


    def remove_neuron_gene(self, spec, index):
        gs = self._neuron_genes[spec]
        del_neuron = gs[index]
        del_mark = del_neuron.historical_mark
        gs.set_empty(index)

        # collect garbage
        if len(gs) > 2 * gs.non_empty_count():
            self._neuron_genes[spec] = gs.copy_non_empty()

        # remove all attached connection genes
        def _enumerate_if(genes, pred):
            return (idx for idx, g in enumerate(genes)
                if g is not None and pred(g))

        modified_channels = []

        for channel, connections in self._conn_genes.items():
            src_spec, dst_spec = channel
            modified = False

            if del_neuron.spec is src_spec:
                for idx in _enumerate_if(connections, lambda g: g.mark_from == del_mark):
                    modified = True
                    connections.set_empty(idx)

            if del_neuron.spec is dst_spec:
                for idx in _enumerate_if(connections, lambda g: g.mark_to == del_mark):
                    modified = True
                    connections.set_empty(idx)

            if modified:
                modified_channels.append(channel)

        # collect garbage
        for channel in modified_channels:
            gs = self._conn_genes[channel]
            if len(gs) > 2 * gs.non_empty_count():
                self._conn_genes[channel] = gs.copy_non_empty()

        # remove all downstream connections of the neuron
        self.connections_index.pop(del_mark, None)

        # now remove all the upstream connections
        for downstream_set in self.connections_index.values():
            downstream_set.discard(del_mark)


    def check_validity(self):
        neuron_hmarks = set(g.historical_mark
            for gs in self._neuron_genes.values()
            for g in gs.iter_non_empty())

        neruon_num = sum(gs.non_empty_count() for gs in self._neuron_genes.values())

        if neruon_num != len(neuron_hmarks):
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


    # def from_yaml(self, y_desc):
    #     del self._neuron_genes[:]
    #     del self._conn_genes[:]

    #     y_neurons = y_desc['neurons']
    #     y_connections = y_desc['connections']

    #     for y_neuron in y_neurons:
    #         # y_neuron.update(y_neuron.pop("params"))
    #         self.add_neuron_gene(Gene(**y_neuron))

    #     for y_connection in y_connections:
    #         # y_connection.update(y_connection.pop("params"))
    #         self.add_connection_gene(ConnectionGene(**y_connection))

    #     return self


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
