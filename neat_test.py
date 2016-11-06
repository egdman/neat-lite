import yaml
from neat import NumericParamSpec, NominalParamSpec, NeuronSpec, NeuronGene, ConnectionGene


linear_spec = NeuronSpec('linear',
	[NumericParamSpec('bias', -1., 1.),
	 NumericParamSpec('gain', 0., 1.)],
	[NominalParamSpec('some_thing', ['surprise', 'motherfucker'])]
	)


print(linear_spec.param_names())


ng = NeuronGene('my neuron is linear', gain=0.44, bias=99.5, somethingelse='hello')

print(ng.neuron_type)
print(ng.somethingelse)
print(ng['somethingelse'])

print(ng.gain)
print(ng['gain'])


cg = ConnectionGene(5, 15, 4.77, socket='mysocket', default_socket='default')
print(cg.socket)
print(cg.default_socket)

cg.default_socket = 'new_default_value!!'
print(cg.default_socket)
