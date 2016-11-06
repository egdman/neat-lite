import random

class NumericParamSpec(object):

	'''
	Specification for a mutable numerical parameter of a neuron.

	An instance of this class defines restrictions on a single mutable parameter
	of a neuron, namely its max and min bounds, or lack thereof.
	It also contains the mean value and the standard deviation that are used
	for random initialization of the parameter value from normal distribution.

	The random distribution is used only when one or both bounds are not set.
	When both bounds are set, the random value is drawn from the uniform distribution
	within the range.
	When only one bound is set, th new gaussian random value will be checked against it.
	'''


	def __init__(self, param_name, min_value=None, max_value=None, mean_value=0., sigma=1.):
		if max_value is not None and min_value is not None and max_value < min_value:
			raise ValueError("max_value value should not be smaller than min_value")

		self.param_name = param_name
		self.min_value = min_value
		self.max_value = max_value
		self.mean_value = mean_value
		self.sigma = sigma



	def get_random_value(self):

		if self.min_value is not None and self.max_value is not None:
			return random.uniform(self.min_value, self.max_value) + self.min_value

		new_value = random.gauss(self.mean_value, self.sigma)

		# check against bounds:
		if self.min_value is not None and new_value < self.min_value:
			new_value = self.min_value

		if self.max_value is not None and new_value > self.max_value:
			new_value = self.max_value

		return new_value
		






class NominalParamSpec(object):

	'''
	Specification for a nominal (non-numeric) parameter of a neuron.

	An instance of this class defines a set of possible values of a single mutable
	nominal parameter of a neuron. It also contains the name of the parameter.
	'''

	def __init__(self, param_name, set_of_values):
		self.param_name = param_name
		self.set_of_values = set_of_values


	def get_random_value(self):
		if len(self.set_of_values) == 0: return None
		return random.choice(self.set_of_values)





class NeuronSpec(object):
	'''
	A collection of parameter specifications for a neuron.

	An instance of this class contains specifications of all parameters
	of a neuron.
	'''

	def __init__(self, neuron_name, numeric_specs, nominal_specs):
		self.neuron_name = neuron_name
		self.numeric_specs = {param_spec.param_name: param_spec for param_spec in numeric_specs}
		self.nominal_specs = {param_spec.param_name: param_spec for param_spec in nominal_specs}



	def param_names(self):
		return list(self.numeric_specs.keys()) + list(self.nominal_specs.keys())	



	def get_random_parameters(self):
		'''
		Return a dictionary where keys are parameter names and values are random parameter values.
		'''

		param_values = {param_name: self.numeric_specs[param_name].get_random_value() \
				for param_name in self.numeric_specs}

		param_values.update({param_name: self.nominal_specs[param_name].get_random_value() \
					for param_name in self.nominal_specs})
		return param_values