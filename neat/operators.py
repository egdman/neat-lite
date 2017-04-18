import random

from .genes import NeuronGene, ConnectionGene, GeneticEncoding
from .specs import NumericParamSpec, NominalParamSpec
from .utils import zip_with_probabilities, weighted_random



class Mutator:

    def __init__(self,
        net_spec,

        innovation_number = 0, # starting innovation number

        allowed_neuron_types=None, # only neurons of these types can be added through mutations
                                   # otherwise, all types in the net_spec can be added

        allowed_connection_types=None, # only connections of these types can be added through mutations
                                       # otherwise, all types in the net_spec can be added

        protected_gene_marks=None, # historical marks of genes that the mutator cannot remove

        pure_input_types = None, # list of input-only neuron types (can't attach loopback inputs)
        pure_output_types = None # list of output-only neuron types (can't attach loopback outputs)
        ):

        self.net_spec = net_spec

        if protected_gene_marks is None:
            self.protected_gene_marks = set()
        else:
            self.protected_gene_marks = set(protected_gene_marks)

        # TODO check that all allowed neuron types are in the net spec
        # set types of neurons that are allowed to be added to the net
        if allowed_neuron_types is None:
            self.allowed_neuron_types = list(self.net_spec.neuron_specs.keys())
        else:
            self.allowed_neuron_types = allowed_neuron_types

        # make allowed types into a list of tuples (type, probability)
        self.allowed_neuron_types = zip_with_probabilities(self.allowed_neuron_types)


        # TODO check that all allowed connection types are in the net spec
        # set types of connections that are allowed to be added to the net
        if allowed_connection_types is None:
            self.allowed_connection_types = list(self.net_spec.connection_specs.keys())
        else:
            self.allowed_connection_types = allowed_connection_types

        # make allowed types into a list of tuples (type, probability)
        self.allowed_connection_types = zip_with_probabilities(self.allowed_connection_types)

        self.pure_input_types = pure_input_types if pure_input_types is not None else []
        self.pure_output_types = pure_output_types if pure_output_types is not None else []

        self.innovation_number = innovation_number



    def mutate_neuron_params(self, genotype, probability):
        for neuron_gene in genotype.neuron_genes:
            if random.random() < probability:
                self.mutate_gene_params(neuron_gene)




    def mutate_connection_params(self, genotype, probability):
        for connection_gene in genotype.connection_genes:
            if random.random() < probability:
                self.mutate_gene_params(connection_gene)




    def mutate_gene_params(self, gene):

        """
        Only one parameter of the gene is chosen to be mutated.
        The parameter to be mutated is chosen from the set of parameters
        with associated probabilities.

        :type gene: Gene
        """

        # gene_params = self.mutable_params[gene.gene_type]
        gene_spec = self.net_spec[gene.gene_type]
        gene_params = gene_spec.mutable_param_names()

        if len(gene_params) > 0:

            # param_name = weighted_random(gene_params)
            param_name = random.choice(gene_params)
            param_spec = gene_spec[param_name]

            if isinstance(param_spec, NumericParamSpec):
                current_value = gene[param_name]

                new_value = param_spec.mutate_value(current_value)
                gene[param_name] = new_value

            elif isinstance(param_spec, NominalParamSpec):
                gene[param_name] = param_spec.get_random_value()



    def _get_pair_neurons(self, genotype):
        neuron_from = random.choice(genotype.neuron_genes)
        neuron_to = random.choice(genotype.neuron_genes)
        mark_from = neuron_from.historical_mark
        mark_to = neuron_to.historical_mark
        destination_valid = neuron_to.gene_type not in self.pure_input_types
        return mark_from, mark_to, destination_valid



    def add_connection_mutation(self, genotype, max_attempts=100):

        """
        Pick two neurons A and B at random. Make sure that connection AB does not exist.
        If that's the case, add new connection whose type is randomly selected from the set
        of allowed types and whose parameters are initialized according to spec for that type.

        Otherwise pick two neurons again and repeat until suitable pair is found or we run out of attempts.

        :type genotype: GeneticEncoding
        :type max_attempts: int
        """

        # if there is no allowed connection types to add, cancel mutation
        if len(self.allowed_connection_types) == 0: return False


        mark_from, mark_to, dst_valid = self._get_pair_neurons(genotype)

        num_attempts = 0

        while len(genotype.get_connection_genes(mark_from, mark_to)) > 0 or not dst_valid:

            mark_from, mark_to, dst_valid = self._get_pair_neurons(genotype)
            num_attempts += 1
            if num_attempts >= max_attempts: return False


        new_connection_type = weighted_random(self.allowed_connection_types)

        new_connection_params = self.net_spec[new_connection_type].get_random_parameters()

        self.add_connection(
            genotype,
            new_connection_type,
            mark_from,
            mark_to,
            **new_connection_params)

        return True



    def _unprotected_connection_ids(self, genotype):
        return list(cg_i for cg_i, cg in enumerate(genotype.connection_genes) \
            if cg.historical_mark not in self.protected_gene_marks)



    def _unprotected_neuron_ids(self, genotype):
        return list(ng_i for ng_i, ng in enumerate(genotype.neuron_genes) \
            if ng.historical_mark not in self.protected_gene_marks)




    def add_neuron_mutation(self, genotype):

        """
        Pick a connection at random from neuron A to neuron B.
        And add a neuron C in between A and B.
        Old connection AB gets deleted.
        Two new connections AC and CB are added.
        Connection AC will have the same type and parameters as AB.
        Connection CB will have random type (chosen from the allowed ones)
        and randomly initialized parameters.

        :type genotype: GeneticEncoding
        """


        ######## INIT NEW NEURON TYPE AND PARAMS #############################################
        # if there is no allowed neuron types to add, cancel mutation
        if len(self.allowed_neuron_types) == 0: return
        
        # select new neuron type from allowed types with weights
        new_neuron_type = weighted_random(self.allowed_neuron_types)

        # initialize parameters of the new neuron at random
        new_neuron_params = self.net_spec[new_neuron_type].get_random_parameters()



        ######## INIT NEW CONNECTION TYPE AND PARAMS #########################################
        # if there is no allowed connection types to add, cancel mutation
        if len(self.allowed_connection_types) == 0: return

        # select new connection type from allowed types with weights
        new_connection_type = weighted_random(self.allowed_connection_types)

        # initialize parameters of the new connection at random
        new_connection_params = self.net_spec[new_connection_type].get_random_parameters()




        unprotected_conn_ids = self._unprotected_connection_ids(genotype)
        # unprotected_conn_ids = range(len(genotype.connection_genes))
        if len(unprotected_conn_ids) == 0: return

        connection_to_split_id = random.choice(unprotected_conn_ids)
        connection_to_split = genotype.connection_genes[connection_to_split_id]


        # get all the info about the old connection
        old_connection_type = connection_to_split.gene_type
        old_connection_params = connection_to_split.copy_params()

        mark_from = connection_to_split.mark_from
        mark_to = connection_to_split.mark_to


        # delete the old connection from the genotype
        genotype.remove_connection_gene(connection_to_split_id)

        # insert new neuron
        mark_middle = self.add_neuron(genotype, new_neuron_type, **new_neuron_params)


        self.add_connection(
            genotype,
            old_connection_type,
            mark_from,
            mark_middle,
            **old_connection_params)


        self.add_connection(
            genotype,
            new_connection_type,
            mark_middle,
            mark_to,
            **new_connection_params)




    def remove_connection_mutation(self, genotype):
        unprotected_conn_ids = self._unprotected_connection_ids(genotype)
        # unprotected_conn_ids = range(len(genotype.connection_genes))
        if len(unprotected_conn_ids) == 0: return
        gene_id = random.choice(unprotected_conn_ids)
        genotype.remove_connection_gene(gene_id)



    def remove_neuron_mutation(self, genotype):
        unprotected_neuron_ids = self._unprotected_neuron_ids(genotype)
        if len(unprotected_neuron_ids) == 0: return

        gene_id = random.choice(unprotected_neuron_ids)

        neuron_gene = genotype.neuron_genes[gene_id]
        neuron_mark = neuron_gene.historical_mark

        # find indices of attached connection genes:
        bad_connections = [g_id for g_id, gene in enumerate(genotype.connection_genes) if
                           gene.mark_from == neuron_mark or gene.mark_to == neuron_mark]


        # remove attached connection genes
        # (list is reversed because indices will be screwed up otherwise)
        for g_id in reversed(bad_connections):
            genotype.remove_connection_gene(g_id)

        # remove the neuron gene:
        genotype.remove_neuron_gene(gene_id)




    def add_neuron(self, genotype, neuron_type, **neuron_params):
        # initialize params
        init_params = self.net_spec[neuron_type].get_random_parameters()

        # overwrite params that are provided in arguments
        init_params.update(neuron_params)

        new_neuron_gene = NeuronGene(
                                gene_type = neuron_type,
                                historical_mark = self.innovation_number,
                                enabled = True,
                                **init_params)

        self.innovation_number += 1
        genotype.add_neuron_gene(new_neuron_gene)
        return new_neuron_gene.historical_mark



    def add_connection(self, genotype, connection_type, mark_from, mark_to, **connection_params):
        # initialize params
        init_params = self.net_spec[connection_type].get_random_parameters()

        # overwrite params that are provided in arguments
        init_params.update(connection_params)

        new_conn_gene = ConnectionGene(
                                  gene_type = connection_type,
                                  mark_from = mark_from,
                                  mark_to = mark_to,
                                  historical_mark = self.innovation_number,
                                  enabled = True,
                                  **init_params)

        self.innovation_number += 1
        genotype.add_connection_gene(new_conn_gene)
        return new_conn_gene.historical_mark



    
    def protect_gene(self, historical_mark):
        '''
        protect gene with the given historical_mark from being deleted 
        '''
        self.protected_gene_marks.add(historical_mark)




def crossover(genotype_more_fit, genotype_less_fit):
    '''
    Perform crossover of two genotypes. The input genotypes are kept unchanged.
    The first genotype in the arguments must be the more fit one.
    '''

    # copy original genotypes to keep them intact:
    genotype_more_fit = genotype_more_fit.copy()
    genotype_less_fit = genotype_less_fit.copy()


    # sort genes by historical marks:
    genes_better = sorted(genotype_more_fit.neuron_genes + genotype_more_fit.connection_genes,
                    key = lambda gene: gene.historical_mark)

    genes_worse = sorted(genotype_less_fit.neuron_genes + genotype_less_fit.connection_genes,
                    key = lambda gene: gene.historical_mark)

    gene_pairs = GeneticEncoding.get_pairs(genes_better, genes_worse)

    child_genes = []

    for pair in gene_pairs:

        # if gene is paired, inherit one of the pair with 50/50 chance:
        if pair[0] is not None and pair[1] is not None:
            if random.random() < 0.5:
                child_genes.append(pair[0])
            else:
                child_genes.append(pair[1])

        # inherit unpaired gene from the more fit parent:
        elif pair[0] is not None:
            child_genes.append(pair[0])

    child_genotype = GeneticEncoding()
    for gene in child_genes:
        if isinstance(gene, NeuronGene):
            child_genotype.add_neuron_gene(gene)
        elif isinstance(gene, ConnectionGene):
            child_genotype.add_connection_gene(gene)

    return child_genotype