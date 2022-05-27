import random


def clamp(min_value, value, max_value):
    if min_value is not None and value < min_value:
        value = min_value

    if max_value is not None and value > max_value:
        value = max_value
    return value

class bound_required: pass
def require_missing_bounds(spec):
    if spec.min_value is None:
        spec.min_value = bound_required
    if spec.max_value is None:
        spec.max_value = bound_required

class bounds:
    def __init__(self, min_value, max_value):
        def _apply(spec):
            spec.min_value, spec.max_value = min_value, max_value
        self._apply = _apply

class gen:
    def __init__(self, func, require_bounds=False):
        def _apply(spec):
            spec.generator = func
            if require_bounds:
                require_missing_bounds(spec)
        self._apply = _apply

    def uniform():
        def _sample(min_value, max_value):
            return random.uniform(min_value, max_value)
        return gen(_sample, require_bounds=True)

    def gauss(mean, sigma):
        def _sample(min_value, max_value):
            return clamp(min_value, random.gauss(mean, sigma), max_value)
        return gen(_sample)

    def random_choice(values):
        def _sample(_0, _1):
            return random.choice(values)
        return gen(_sample)

    def const(value):
        def _sample(_0, _1):
            return value
        return gen(_sample)

class mut:
    def __init__(self, func, require_bounds=False):
        def _apply(spec):
            spec.mutator = func
            if require_bounds:
                require_missing_bounds(spec)
        self._apply = _apply

    def uniform():
        def _sample(min_value, _1, max_value):
            return random.uniform(min_value, max_value)
        return mut(_sample, require_bounds=True)

    def gauss(sigma):
        def _sample(min_value, current_value, max_value):
            return clamp(min_value, random.gauss(current_value, sigma), max_value)
        return mut(_sample)

    def random_choice(values):
        def _sample(_0, _1, _2):
            return random.choice(values)
        return mut(_sample)


class InvalidSpecError(RuntimeError): pass

class ParamSpec:
    def __init__(self, name: str, *options):
        self.name = name
        self.min_value = None
        self.max_value = None
        self.generator = None
        self.mutator = None
        for option in options:
            option._apply(self)

        if self.generator is None:
            raise InvalidSpecError(
                f"'gen' argument is required for ParamSpec '{name}'")

        if self.min_value == bound_required or self.max_value == bound_required:
            raise InvalidSpecError(
                f"'bounds' argument with both min and max values is required for ParamSpec '{name}'")


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
