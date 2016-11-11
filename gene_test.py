from neat import NeuronGene, ConnectionGene


print("\nNeuronGene tests")
ng = NeuronGene('sigmoid', bias=0.3, gain=0.96)

res = ng.__dict__
print("ng.__dict__ : {}".format(res))


res = ng.neuron_type
print("ng.neuron_type : {}".format(res))

res = ng.gene_type
print("ng.gene_type : {}".format(res))

res = ng.gain
print('ng.gain : {}'.format(res))


res = ng.neuron_params
print('ng.neuron_params : {}'.format(res))

res = ng.gene_params
print('ng.gene_params : {}'.format(res))

res = ng.get_params()
print("ng.get_params() : {}".format(res))

res = ng.copy_params()
print("ng.copy_params() : {}".format(res))

print("ng.get_params()['bias'] <- 999.888")
ng.get_params()['bias'] = 999.888
res = ng.get_params()
print("ng.get_params() : {}".format(res))

print("ng.gain <- 555.444")
ng.gain = 555.444
res = ng.gain
print('ng.gain : {}'.format(res))

res = ng.get_params()
print("ng.get_params() : {}".format(res))

res = hasattr(ng, 'id')
print("hasattr(ng, 'id') : {}".format(res))

print("if not hasattr(ng, 'id'): ng.id <- 'IIIDDD'")
if not hasattr(ng, 'id'): ng.id = "IIIDDD"

res = hasattr(ng, 'id')
print("hasattr(ng, 'id') : {}".format(res))


res = ng.id
print("ng.id : {}".format(res))

res = ng.neuron_params
print('ng.neuron_params : {}'.format(res))

res = ng.gene_params
print("ng.gene_params : {}".format(res))

res = ng.copy_params()
print("ng.copy_params() : {}".format(res))

print("ng.new_param <- 22.22")
ng.new_param = 22.22

res = ng.new_param
print("ng.new_param : {}".format(res))


print("ng['newer_param'] <- 44.44")
ng['newer_param'] = 44.44

res = ng['newer_param']
print("ng['newer_param'] : {}".format(res))

res = ng.neuron_params
print('ng.neuron_params : {}'.format(res))

res = ng.__dict__
print("ng.__dict__ : {}".format(res))



print("\nConnectionGene tests")
cg = ConnectionGene('def_con', 888, 999, weight=0.3)

res = cg.__dict__
print("cg.__dict__ : {}".format(res))

res = cg.connection_type
print("cg.connection_type : {}".format(res))

res = cg.mark_from
print("cg.mark_from : {}".format(res))

print("cg.mark_from <- 1919")
cg.mark_from = 1919
res = cg.mark_from
print("cg.mark_from : {}".format(res))

res = cg.__dict__
print("cg.__dict__ : {}".format(res))
