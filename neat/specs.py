import random
from itertools import chain
from .utils import zip_with_probabilities, weighted_random


class NumericParamSpec(object):

    '''
    Specification for a mutable numerical parameter of a gene.

    An instance of this class defines restrictions on a single mutable parameter
    of a gene, namely its max and min bounds, or lack thereof. It also contains the name
    of the parameter.

    Additionally, it contains the mean value (mean_value) and the standard deviation (mutation_sigma)
    that are used for mutation and random initialization of the parameter value.

    The mutation happens as follows:
    If both bounds are set, then the mutation_sigma is treated as fraction of the total range
    of possible values. The new value is drawn from a normal distribution centered around
    the current value with st.deviation = (max_value - min_value)*mutation_sigma. The resulting
    value is checked agains the bounds.

    If less than two bouns are set, then the mutation_sigma is treated as absolute value. The new value
    is drawn from a normal distribution centered around the current value with
    st.deviation = mutation_sigma. If one bound is set, the resulting value is checked against it
    (i.e. if new_value < min_value then new_value = min_value).


    The process of getting a random value for the parameter (using the get_random_value() method)
    is as follows:

    If both bounds are set, then the value is drawn from a uniform distribution within the range of possible
    values.

    If less than two bouns are set, then the random value is drawn from a normal distribution with
    mean = mean_value and st.deviation = mutation_sigma. If one bound is set, the resulting value
    is check against it.
    '''


    def __init__(self, param_name,
                 min_value=None, max_value=None,
                 mutation_sigma=1., mean_value=0.,
                 mutable=True):

        if max_value is not None and min_value is not None and max_value < min_value:
            raise ValueError("max_value value should not be smaller than min_value")

        self.param_name = param_name
        self.mutable = mutable
        self.min_value = min_value
        self.max_value = max_value
        self.mean_value = mean_value
        self.mutation_sigma = mutation_sigma



    def get_random_value(self):

        if self.min_value is not None and self.max_value is not None:
            return random.uniform(self.min_value, self.max_value)

        new_value = random.gauss(self.mean_value, self.mutation_sigma)
        return self.put_within_bounds(new_value)
        


    def put_within_bounds(self, value):
        if self.min_value is not None and value < self.min_value:
            value = self.min_value

        if self.max_value is not None and value > self.max_value:
            value = self.max_value
        return value



    def mutate_value(self, current_value):

        if self.min_value is not None and self.max_value is not None:
            abs_sigma = self.mutation_sigma*(self.max_value - self.min_value)

        else:
            abs_sigma = self.mutation_sigma

        new_value = current_value + random.gauss(0, abs_sigma)
        return self.put_within_bounds(new_value)




class NominalParamSpec(object):

    '''
    Specification for a nominal (non-numeric) parameter of a gene.

    An instance of this class defines a set of possible values of a single mutable
    nominal parameter of a gene. It also contains the name of the parameter.
    '''

    def __init__(self, param_name, set_of_values, mutable=True):
        self.param_name = param_name
        self.mutable = mutable
        self.set_of_values = zip_with_probabilities(set_of_values)


    def get_random_value(self):
        if len(self.set_of_values) == 0: return None
        return weighted_random(self.set_of_values)





class GeneSpec(object):
    '''
    A collection of parameter specifications for a gene.

    An instance of this class contains specifications of all parameters
    of a gene.
    '''

    def __init__(self, type_name, *param_specs):
        self.type_name = type_name
        self.param_specs = {param_spec.param_name: param_spec for param_spec in param_specs}


    def __getitem__(self, key):
        return self.param_specs[key]
    

    def __iter__(self):
        return self.param_specs.__iter__()


    def get(self, key, default=None):
        return self.param_specs.get(key, default)


    def param_names(self):
        return list(self.param_specs.keys())


    def mutable_param_names(self):
        return list(pname for pname, pspec in self.param_specs.items() if pspec.mutable)


    def get_random_parameters(self):
        '''
        Return a dictionary where keys are parameter names and values are random parameter values.
        '''
        return {param_name: self.param_specs[param_name].get_random_value() \
                for param_name in self.param_specs}




class NetworkSpec(object):

    def __init__(self, neuron_specs, connection_specs):
        self.neuron_specs     = {nspec.type_name: nspec for nspec in neuron_specs}
        self.connection_specs = {cspec.type_name: cspec for cspec in connection_specs}


    def gene_types(self):
        return self.neuron_specs.keys() + self.connection_specs.keys()


    def __getitem__(self, key):
        nspec = self.neuron_specs.get(key, None)
        cspec = self.connection_specs.get(key, None)
        if nspec is None and cspec is None: raise KeyError(key)

        return nspec or cspec


    def __iter__(self):
        return chain(self.neuron_specs.__iter__(), self.connection_specs.__iter__())