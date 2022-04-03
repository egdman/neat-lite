import random
from itertools import chain
from .utils import zip_with_probabilities, weighted_random


def clamp(min_value, value, max_value):
    if min_value is not None and value < min_value:
        value = min_value

    if max_value is not None and value > max_value:
        value = max_value
    return value


def gen_uniform():
    def _sample(min_value, max_value):
        return random.uniform(min_value, max_value)
    return _sample


def gen_random_choice(values):
    def _pick(*_):
        return random.choice(values)
    return _pick


def gen_gauss(mean, sigma):
    def _sample(min_value, max_value):
        return clamp(min_value, random.gauss(mean, sigma), max_value)
    return _sample


def mut_gauss(sigma):
    def _sample(min_value, current_value, max_value):
        return clamp(min_value, random.gauss(current_value, sigma), max_value)
    return _sample


class ParamSpec:
    def __init__(self, name: str, value_generator, value_mutator=None):
        self.name = name
        self.min_value = None
        self.max_value = None
        self.generator = value_generator
        self.mutator = value_mutator


    def with_bounds(self, min_value, max_value):
        self.min_value = min_value
        self.max_value = max_value
        return self


    def get_random_value(self):
        return self.generator(self.min_value, self.max_value)


    def mutate_value(self, current_value):
        if self.mutator is None:
            return current_value
        else:
            return self.mutator(self.min_value, current_value, self.max_value)


class GeneSpec(object):
    '''
    A collection of parameter specifications for a gene.

    An instance of this class contains specifications of all parameters
    of a gene.
    '''

    def __init__(self, type_name, *param_specs):
        self.type_name = type_name
        self.param_specs = {param_spec.name: param_spec for param_spec in param_specs}


    def __getitem__(self, key):
        return self.param_specs[key]
    

    def __iter__(self):
        return self.param_specs.__iter__()


    def get(self, key, default=None):
        return self.param_specs.get(key, default)


    def param_names(self):
        return list(self.param_specs.keys())


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