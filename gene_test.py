from neat import NeuronGene, ConnectionGene


print("\nNeuronGene tests")
ng = NeuronGene('sigmoid', bias=0.3, gain=0.96)


res = ng.neuron_type
print("ng.neuron_type = {}".format(res))

res = ng.gene_type
print("ng.gene_type = {}".format(res))

res = ng.gain
print('ng.gain = {}'.format(res))


res = ng.neuron_params
print('ng.neuron_params = {}'.format(res))

res = ng.gene_params
print('ng.gene_params = {}'.format(res))

res = ng.get_params(['bias', 'gain'])
print("ng.get_params(['bias', 'gain']) = {}".format(res))

res = ng.copy_params(['bias', 'gain'])
print("ng.copy_params(['bias', 'gain']) = {}".format(res))

print("ng.get_params()['bias'] = 999.888")
ng.get_params()['bias'] = 999.888
res = ng.get_params(['bias', 'gain'])
print("ng.get_params(['bias', 'gain']) = {}".format(res))

res = hasattr(ng, 'id')
print("hasattr(ng, 'id') = {}".format(res))

print("if not hasattr(ng, 'id'): ng.id = 'IIIDDD'")
if not hasattr(ng, 'id'): ng.id = "IIIDDD"

res = hasattr(ng, 'id')
print("hasattr(ng, 'id') = {}".format(res))


res = ng.id
print("ng.id = {}".format(res))

res = ng.neuron_params
print('ng.neuron_params = {}'.format(res))

res = ng.gene_params
print("ng.gene_params = {}".format(res))

res = ng.copy_params(['bias', 'gain', 'id'])
print("ng.copy_params(['bias', 'gain', 'id']) = {}".format(res))

print("assigning ng.gain = -555777.")
ng.gain = -555777.

res = ng.gain
print('ng.gain = {}'.format(res))
