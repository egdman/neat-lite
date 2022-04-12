from numbers import Real
from copy import copy, deepcopy
from itertools import chain

try:
    from yaml import dump as yaml_dump
except ImportError:
    yaml_dump = None

def unicode_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data)

def hm(gene):
    return gene.historical_mark

class Gene(object):

    _metas = ('gene_type', 'historical_mark', 'non_removable')

    def __init__(self, gene_type, historical_mark=0, non_removable=False, **params):
        self.gene_type = gene_type
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


    def copy_params(self):
        return deepcopy(self.get_params())


    def get_type(self):
        return self.gene_type


    def copy(self):
        return deepcopy(self)


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

    def __init__(self, gene_type, historical_mark=0, non_removable=False, **params):
        super(NeuronGene, self).__init__(gene_type, historical_mark, non_removable, **params)


    def __str__(self):
        s = "NEAT Neuron gene, mark: {}, type: {}".format(
            self.historical_mark,
            self.gene_type,
        )
        if self.non_removable:
            s = "{} norem".format(s)
        return s




class ConnectionGene(Gene):

    _metas = Gene._metas + ('mark_from', 'mark_to')

    def __init__(self, gene_type, mark_from, mark_to, historical_mark=0, non_removable=False, **params):
        super(ConnectionGene, self).__init__(gene_type, historical_mark, non_removable, **params)

        self.mark_from = mark_from
        self.mark_to = mark_to


    def __str__(self):
        s = "NEAT Connection gene, mark: {}, type: {}, {} -> {}".format(
            self.historical_mark,
            self.gene_type,
            self.mark_from,
            self.mark_to,
        )
        if self.non_removable:
            s = "{} norem".format(s)
        return s


class Genome:

    def __init__(self, neuron_genes=None, connection_genes=None):
        self.neuron_genes = neuron_genes if neuron_genes else []
        self.connection_genes = connection_genes if connection_genes else []



    def num_genes(self):
        return len(self.neuron_genes) + len(self.connection_genes)



    def get_connection_genes(self, mark_from, mark_to):
        return list(c_g for c_g in self.connection_genes \
            if c_g.mark_from == mark_from and c_g.mark_to == mark_to)



    @staticmethod
    def get_dissimilarity(genome1, genome2,
        excess_coef=1.,
        disjoint_coef=1.,
        neuron_diff_coef=0.,
        connection_diff_coef=0.):

        # calculate missing pair difference
        excess_num, disjoint_num = count_excess_disjoint(genome1, genome2)

        num_genes = max(genome1.num_genes(), genome2.num_genes())
        miss_pair_diff = (disjoint_coef * disjoint_num + excess_coef * excess_num) / float(num_genes)

        # calculate difference of numeric params between genes
        neuron_diff = 0.
        connection_diff = 0.

        # if neuron_diff_coef > 0 or connection_diff_coef > 0:
        #     neuron_diff, connection_diff = Genome._calc_numeric_diff(pairs)

        return miss_pair_diff + neuron_diff_coef*neuron_diff + connection_diff_coef*connection_diff


    @staticmethod
    def count_excess_disjoint(genome1, genome2):
        genes_sorted1 = sorted(
            genome1.neuron_genes + genome1.connection_genes, key=hm)

        genes_sorted2 = sorted(
            genome2.neuron_genes + genome2.connection_genes, key=hm)

        # save ranges of historical marks for both genomes
        min_mark1 = hm(genes_sorted1[0])
        max_mark1 = hm(genes_sorted1[-1])

        min_mark2 = hm(genes_sorted2[0])
        max_mark2 = hm(genes_sorted2[-1])

        excess_num = 0
        disjoint_num = 0

        # calculate numbers of excess and disjoint genes
        for gene1, gene2 in Genome.get_pairs(genes_sorted1, genes_sorted2):
            if gene2 is None:
                if gene1.historical_mark < min_mark2 or gene1.historical_mark > max_mark2:
                    excess_num += 1
                else:
                    disjoint_num += 1

            elif gene1 is None:
                if gene2.historical_mark < min_mark1 or gene2.historical_mark > max_mark1:
                    excess_num += 1
                else:
                    disjoint_num += 1

        return excess_num, disjoint_num


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
                    for right_gene in right_genes:
                        yield None, right_gene
                break

            elif right_gene is None:
                yield left_gene, None
                for left_gene in left_genes:
                    yield left_gene, None
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
        self.neuron_genes.append(neuron_gene)



    def add_connection_gene(self, connection_gene):
        self.connection_genes.append(connection_gene)



    def remove_connection_gene(self, index):
        del self.connection_genes[index]



    def remove_neuron_gene(self, index):
        del self.neuron_genes[index]



    def copy(self):
        copy_gen = Genome()

        for n_gene in self.neuron_genes:
            copy_gen.add_neuron_gene(n_gene.copy())

        for c_gene in self.connection_genes:
            copy_gen.add_connection_gene(c_gene.copy())

        return copy_gen



    def check_validity(self):
        neuron_hmarks = set(gene.historical_mark for gene in self.neuron_genes)
        if len(self.neuron_genes) != len(neuron_hmarks):
            return False

        conn_hmarks = set()
        for conn_gene in self.connection_genes:
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
        for ng in self.neuron_genes: st += str(ng.__dict__) + '\n'
        st += 'connections:\n'
        for cg in self.connection_genes: st += str(cg.__dict__) + '\n'
        return st


    def from_yaml(self, y_desc):
        del self.neuron_genes[:]
        del self.connection_genes[:]

        y_neurons = y_desc['neurons']
        y_connections = y_desc['connections']

        for y_neuron in y_neurons:
            # y_neuron.update(y_neuron.pop("params"))
            self.neuron_genes.append(NeuronGene(**y_neuron))

        for y_connection in y_connections:
            # y_connection.update(y_connection.pop("params"))
            self.connection_genes.append(ConnectionGene(**y_connection))

        return self


    def to_yaml(self):
        if yaml_dump is None:
            raise NotImplementedError("PyYaml not installed")

        neuron_genes = list(n_g.__dict__ for n_g in self.neuron_genes)
        conn_genes = list(c_g.__dict__ for c_g in self.connection_genes)
        # yaml.add_representer(unicode, unicode_representer)
        yaml_repr = {'neurons': neuron_genes, 'connections' : conn_genes}
        return yaml_dump(yaml_repr, default_flow_style=False)
