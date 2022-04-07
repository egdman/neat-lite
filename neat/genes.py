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

        genes_sorted1 = sorted(
            genome1.neuron_genes + genome1.connection_genes, key = hm)

        genes_sorted2 = sorted(
            genome2.neuron_genes + genome2.connection_genes, key = hm)

        # save ranges of hmarks for both genomes
        min_mark1 = hm(genes_sorted1[0])
        max_mark1 = hm(genes_sorted1[-1])

        min_mark2 = hm(genes_sorted2[0])
        max_mark2 = hm(genes_sorted2[-1])

        pairs = tuple(Genome.get_pairs(genes_sorted1, genes_sorted2))

        excess_num = 0
        disjoint_num = 0


        # calculate numbers of excess and disjoint genes
        for gene0, gene1 in pairs:

            if gene0 and not gene1:
                mark = gene0.historical_mark
                if (min_mark2 - 1) < mark and mark < (max_mark2 + 1):
                    disjoint_num += 1
                else:
                    excess_num += 1

            elif gene1 and not gene0:
                mark = gene1.historical_mark
                if (min_mark1 - 1) < mark and mark < (max_mark1 + 1):
                    disjoint_num += 1
                else:
                    excess_num += 1


        # calculate missing pair difference
        num_genes = max(genome1.num_genes(), genome2.num_genes())
        miss_pair_diff = (disjoint_coef * disjoint_num + excess_coef * excess_num) / float(num_genes)


        # calculate difference of numeric params between genes
        neuron_diff = 0.
        connection_diff = 0.

        if neuron_diff_coef > 0 or connection_diff_coef > 0:
            neuron_diff, connection_diff = Genome._calc_numeric_diff(pairs)

        return miss_pair_diff + neuron_diff_coef*neuron_diff + connection_diff_coef*connection_diff



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
        rightGenes = iter(genes_sorted2)
        rightGene = next(rightGenes, None)

        for leftGene in genes_sorted1:
            while rightGene and hm(leftGene) > hm(rightGene):
                rightGene = next(rightGenes, None)
            if rightGene and hm(leftGene) == hm(rightGene):
                yield leftGene, rightGene
                rightGene = next(rightGenes, None)
            else:
                yield leftGene, None

        leftGenes = iter(genes_sorted1)
        leftGene = next(leftGenes, None)

        for rightGene in genes_sorted2:
            while leftGene and hm(leftGene) < hm(rightGene):
                leftGene = next(leftGenes, None)
            if leftGene is None or hm(leftGene) != hm(rightGene):
                yield None, rightGene



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
        for conn_gene in self.connection_genes:
            mark_from = conn_gene.mark_from
            mark_to = conn_gene.mark_to
            if not self.check_neuron_exists(mark_from):
                return False
            if not self.check_neuron_exists(mark_to):
                return False
        return True



    def check_neuron_exists(self, mark):
        for neuron_gene in self.neuron_genes:
            if mark == neuron_gene.historical_mark: return True
        return False



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
