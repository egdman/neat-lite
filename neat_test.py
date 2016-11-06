import yaml
from neat import NumericParamSpec, NominalParamSpec, NeuronSpec


linear_spec = NeuronSpec('linear',
	[NumericParamSpec('bias', -1., 1.),
	 NumericParamSpec('gain', 0., 1.)],
	[NominalParamSpec('some_thing', ['surprise', 'motherfucker'])]
	)


print(linear_spec.param_names())

