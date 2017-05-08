import yaml
from numbers import Real
from copy import copy, deepcopy
from itertools import izip, chain


def unicode_representer(dumper, data):
    return dumper.represent_scalar(u'tag:yaml.org,2002:str', data)



class Gene(object):

    _metas = ('gene_type', 'historical_mark', 'protected')

    def __init__(self, gene_type, historical_mark=0, protected=False, **params):
        self.gene_type = gene_type
        self.historical_mark = historical_mark
        self.protected = protected

        for key, value in params.iteritems():
            setattr(self, key, value)


    def __getitem__(self, key):
        return self.__dict__[key]


    def __setitem__(self, key, value):
        setattr(self, key, value)


    def __contains__(self, key):
        return key in self.__dict__


    def _prop_names(self):
        return (name for name in self.__dict__ if name not in self._metas)


    def get_params(self):
        return {name: self.__dict__[name] for name in self._prop_names()}


    def copy_params(self):
        return {name: deepcopy(self.__dict__[name]) for name in self._prop_names()}


    def get_type(self):
        return self.gene_type


    def copy(self):
        return deepcopy(self)


    def numeric_difference(self, other):
        params1 = self.get_params()
        params2 = other.get_params()
        diff = 0.
        for par_name, p1 in params1.iteritems():
            if not isinstance(p1, Real): continue
            p2 = params2[par_name]
            diff += abs(p2 - p1)
        return diff




class NeuronGene(Gene):

    _metas = Gene._metas

    def __init__(self, gene_type, historical_mark=0, protected=False, **params):
        super(NeuronGene, self).__init__(gene_type, historical_mark, protected, **params)


    def __str__(self):
        return "NEAT Neuron gene, mark: {}, type: {}, {}".format(
            self.historical_mark,
            self.gene_type,
            'p' if self.protected else 'up'
        )





class ConnectionGene(Gene):

    _metas = Gene._metas + ('mark_from', 'mark_to')

    def __init__(self, gene_type, mark_from, mark_to, historical_mark=0, protected=False, **params):
        super(ConnectionGene, self).__init__(gene_type, historical_mark, protected, **params)

        self.mark_from = mark_from
        self.mark_to = mark_to


    def __str__(self):
        return "NEAT Connection gene, mark: {}, type: {}, from: {}, to: {}, {}".format(
            self.historical_mark,
            self.gene_type,
            self.mark_from,
            self.mark_to,
           'p' if self.protected else ''
        )



class GeneticEncoding:

    def __init__(self, neuron_genes=None, connection_genes=None):
        self.neuron_genes = neuron_genes if neuron_genes else []
        self.connection_genes = connection_genes if connection_genes else []



    def num_genes(self):
        return len(self.neuron_genes) + len(self.connection_genes)



    def get_connection_genes(self, mark_from, mark_to):
        return list(c_g for c_g in self.connection_genes \
            if c_g.mark_from == mark_from and c_g.mark_to == mark_to)



    @staticmethod
    def get_dissimilarity(genotype1, genotype2,
        excess_coef=1.,
        disjoint_coef=1.,
        neuron_diff_coef=0.,
        connection_diff_coef=0.):

        genes_sorted1 = sorted(genotype1.neuron_genes + genotype1.connection_genes,
                               key = lambda gene: gene.historical_mark)

        genes_sorted2 = sorted(genotype2.neuron_genes + genotype2.connection_genes,
                               key = lambda gene: gene.historical_mark)

        # save ranges of hmarks for both genomes
        min_mark1 = genes_sorted1[0].historical_mark
        max_mark1 = genes_sorted1[-1].historical_mark

        min_mark2 = genes_sorted2[0].historical_mark
        max_mark2 = genes_sorted2[-1].historical_mark
        
        pairs = GeneticEncoding.get_pairs(genes_sorted1, genes_sorted2)

        excess_num = 0
        disjoint_num = 0


        # calculate numbers of excess and disjoint genes
        for pair in pairs:

            if pair[0] and not pair[1]:
                mark = pair[0].historical_mark
                if mark > (min_mark2 - 1) and mark < (max_mark2 + 1):
                    disjoint_num += 1
                else:
                    excess_num += 1
                    
            elif pair[1] and not pair[0]:
                mark = pair[1].historical_mark
                if mark > (min_mark1 - 1) and mark < (max_mark1 + 1):
                    disjoint_num += 1
                else:
                    excess_num += 1


        # calculate missing pair difference
        num_genes = max(genotype1.num_genes(), genotype2.num_genes())
        miss_pair_diff = (disjoint_coef * disjoint_num + excess_coef * excess_num) / float(num_genes)


        # calculate difference of numeric params between genes
        neuron_diff = 0.
        connection_diff = 0.

        if neuron_diff_coef > 0 or connection_diff_coef > 0:
            neuron_diff, connection_diff = GeneticEncoding._calc_numeric_diff(pairs)

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
        num_genes1 = len(genes_sorted1)
        num_genes2 = len(genes_sorted2)

        min_mark1 = genes_sorted1[0].historical_mark
        max_mark1 = genes_sorted1[-1].historical_mark

        min_mark2 = genes_sorted2[0].historical_mark
        max_mark2 = genes_sorted2[-1].historical_mark

        min_mark = min(min_mark1, min_mark2)
        max_mark = max(max_mark1, max_mark2)


        gene_pairs = []

        # search for pairs of genes with equal marks:
        start_from1 = 0
        start_from2 = 0

        mark = min_mark
        while mark < max_mark + 1:

            # jump1 and jump2 are here to skip long sequences of empty historical marks

            gene1 = None
            jump1 = mark + 1
            for i in range(start_from1, num_genes1):
                if genes_sorted1[i].historical_mark == mark:
                    gene1 = genes_sorted1[i]
                    start_from1 = i
                    break

                # if there is a gap, jump over it:
                elif genes_sorted1[i].historical_mark > mark:
                    jump1 = genes_sorted1[i].historical_mark
                    start_from1 = i
                    break

                # if the end of the gene sequence is reached:
                elif i == num_genes1 - 1:
                    jump1 = max_mark + 1
                    start_from1 = i

            gene2 = None
            jump2 = mark + 1
            for i in range(start_from2, num_genes2):
                if genes_sorted2[i].historical_mark == mark:
                    gene2 = genes_sorted2[i]
                    start_from2 = i
                    break

                # if there is a gap, jump over it:
                elif genes_sorted2[i].historical_mark > mark:
                    jump2 = genes_sorted2[i].historical_mark
                    start_from2 = i
                    break

                # if the end of the gene sequence is reached:
                elif i == num_genes2 - 1:
                    jump2 = max_mark + 1
                    start_from2 = i


            # do not add a pair if both genes are None:
            if gene1 or gene2:
                gene_pairs.append((gene1, gene2))

            mark = min(jump1, jump2)

        return gene_pairs



    def get_sorted_genes(self):
        return sorted(self.neuron_genes + self.connection_genes,
            key = lambda gene: gene.historical_mark)



    def min_max_hist_mark(self):
        genes_sorted = self.get_sorted_genes()
        return genes_sorted[0].historical_mark, genes_sorted[-1].historical_mark



    def find_gene_by_mark(self, mark):
        for gene in chain(self.neuron_genes, self.connection_genes):
            if gene.historical_mark == mark:
                return gene
        return None



    def add_neuron_gene(self, neuron_gene):
        self.neuron_genes.append(neuron_gene)



    def add_connection_gene(self, connection_gene):
        self.connection_genes.append(connection_gene)



    def remove_connection_gene(self, index):
        del self.connection_genes[index]



    def remove_neuron_gene(self, index):
        del self.neuron_genes[index]



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
        neuron_genes = list(n_g.__dict__ for n_g in self.neuron_genes)
        conn_genes = list(c_g.__dict__ for c_g in self.connection_genes)
        yaml.add_representer(unicode, unicode_representer)
        yaml_repr = {'neurons': neuron_genes, 'connections' : conn_genes}
        return yaml.dump(yaml_repr, default_flow_style=False)



    def copy(self):
        copy_gen = GeneticEncoding()

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