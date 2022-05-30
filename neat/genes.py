from numbers import Real
from copy import copy, deepcopy

try:
    from yaml import dump as yaml_dump
except ImportError:
    yaml_dump = None

def unicode_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data)

def hm(gene):
    return gene.historical_mark

class Gene(object):

    _metas = ('spec', 'historical_mark', 'non_removable')

    def __init__(self, gene_spec, historical_mark=0, non_removable=False, **params):
        self.spec = gene_spec
        self.historical_mark = historical_mark
        self.non_removable = non_removable

        for key, value in params.items():
            setattr(self, key, value)


    def __getitem__(self, key):
        return self.__dict__[key]


    def __setitem__(self, key, value):
        setattr(self, key, value)


    def __contains__(self, key):
        return key in self.__dict__


    def get_params(self):
        return {key: value for key, value in self.__dict__.items() if key not in self._metas}


    def get_type(self):
        return self.spec.type_name


    def copy(self):
        return copy(self)


    def numeric_difference(self, other):
        params1 = self.get_params()
        params2 = other.get_params()
        diff = 0.
        for par_name, p1 in params1.items():
            if not isinstance(p1, Real): continue
            p2 = params2[par_name]
            diff += abs(p2 - p1)
        return diff




class NeuronGene(Gene):

    _metas = Gene._metas

    def __init__(self, gene_spec, historical_mark=0, non_removable=False, **params):
        super(NeuronGene, self).__init__(gene_spec, historical_mark, non_removable, **params)


    def __str__(self):
        s = "NEAT Neuron gene, mark: {}, type: {}".format(
            self.historical_mark,
            self.get_type(),
        )
        if self.non_removable:
            s = "{} norem".format(s)
        return s




class ConnectionGene(Gene):

    _metas = Gene._metas + ('mark_from', 'mark_to')

    def __init__(self, gene_spec, mark_from, mark_to, historical_mark=0, non_removable=False, **params):
        super(ConnectionGene, self).__init__(gene_spec, historical_mark, non_removable, **params)

        self.mark_from = mark_from
        self.mark_to = mark_to


    def __str__(self):
        s = "NEAT Connection gene, mark: {}, type: {}, {} -> {}".format(
            self.historical_mark,
            self.get_type(),
            self.mark_from,
            self.mark_to,
        )
        if self.non_removable:
            s = "{} norem".format(s)
        return s


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
        return (g for g in self._neuron_genes if g is not None)


    def connection_genes(self):
        return (g for g in self._conn_genes if g is not None)


    def has_connection(self, mark_from, mark_to):
        downstream_set = self.connections_index.get(mark_from, ())
        return mark_to in downstream_set


    @staticmethod
    def get_dissimilarity(genome1, genome2,
        excess_coef=1.,
        disjoint_coef=1.,
        neuron_diff_coef=0.,
        connection_diff_coef=0.):

        def _num_genes(g):
            return g._neuron_num + g._conn_num

        # calculate missing pair difference
        excess_num, disjoint_num = count_excess_disjoint(genome1, genome2)

        num_genes = max(_num_genes(genome1), _num_genes(genome2))
        miss_pair_diff = (disjoint_coef * disjoint_num + excess_coef * excess_num) / float(num_genes)

        # calculate difference of numeric params between genes
        neuron_diff = 0.
        connection_diff = 0.

        # if neuron_diff_coef > 0 or connection_diff_coef > 0:
        #     neuron_diff, connection_diff = Genome._calc_numeric_diff(pairs)

        return miss_pair_diff + neuron_diff_coef*neuron_diff + connection_diff_coef*connection_diff


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
    def _calc_numeric_diff(gene_pairs):
        neuron_diff = 0.
        connection_diff = 0.

        for gene1, gene2 in gene_pairs:
            if gene1 is None or gene2 is None: continue

            if isinstance(gene1, NeuronGene):
                neuron_diff += gene1.numeric_difference(gene2)

            elif isinstance(gene1, ConnectionGene):
                connection_diff += gene1.numeric_difference(gene2)
        return neuron_diff, connection_diff


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

            elif hm(left_gene) < hm(right_gene):
                yield left_gene, None
                left_gene = next(left_genes, None)

            elif hm(left_gene) > hm(right_gene):
                yield None, right_gene
                right_gene = next(right_genes, None)

            else:
                yield left_gene, right_gene
                left_gene = next(left_genes, None)
                right_gene = next(right_genes, None)


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
            self._conn_num = len(self._conn_genes)

        downstream_set = self.connections_index[g.mark_from]
        downstream_set.discard(g.mark_to)



    def remove_neuron_gene(self, index):
        neuron_mark = self._neuron_genes[index].historical_mark
        self._neuron_genes[index] = None
        self._neuron_num -= 1

        # collect garbage
        if len(self._neuron_genes) > 2 * self._neuron_num:
            self._neuron_genes = list(g for g in self._neuron_genes if g is not None)
            self._neuron_num = len(self._neuron_genes)

        # remove all attached connection genes
        for idx, g in enumerate(self._conn_genes):
            if g is not None and neuron_mark in (g.mark_from, g.mark_to):
                self._conn_genes[idx] = None
                self._conn_num -= 1

        # collect garbage
        if len(self._conn_genes) > 2 * self._conn_num:
            self._conn_genes = list(g for g in self._conn_genes if g is not None)
            self._conn_num = len(self._conn_genes)

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
        st = ''
        st += 'neurons:\n'
        for ng in self.neuron_genes(): st += str(ng.__dict__) + '\n'
        st += 'connections:\n'
        for cg in self.connection_genes():
            st += str(cg.__dict__) + '\n'
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

        def _serialize(gene):
            s = copy(gene.__dict__)
            s["gene_type"] = s.pop("spec").type_name
            return s

        neuron_genes = list(_serialize(g) for g in self.neuron_genes())
        conn_genes = list(_serialize(g) for g in self.connection_genes())
        # yaml.add_representer(unicode, unicode_representer)
        yaml_repr = {'neurons': neuron_genes, 'connections' : conn_genes}
        return yaml_dump(yaml_repr, default_flow_style=False)
