from neat import (NetworkSpec, GeneSpec,
                NumericParamSpec as PS, NominalParamSpec as NPS,
                Mutator, NEAT, neuron, connection)


#### CONFIG #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####
neuron_sigma = 0.25                 # mutation sigma for neuron params
conn_sigma = 1.0                    # mutation sigma for connection params

conf = dict(
pop_size = 5,                       # population size
elite_size = 1,                     # size of the elite club
tournament_size = 4,                # size of the selection subsample (must be in the range [2, pop_size])
neuron_param_mut_proba = 0.5,       # probability to mutate each single neuron in the genome
connection_param_mut_proba = 0.5,   # probability to mutate each single connection in the genome
structural_augmentation_proba = 0.8,# probability to augment the topology of a newly created genome 
structural_removal_proba = 0.0,     # probability to diminish the topology of a newly created genome
speciation_threshold = 0.005        # genomes that are more similar than this value will be considered the same species
)
#### ###### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####


net_spec = NetworkSpec(
    [
        GeneSpec('input',
            NPS('layer', ['input'], mutable=False)
        ),
        GeneSpec('sigmoid',
            PS('bias', -1., 1., neuron_sigma, mutable=True),
            PS('gain', 0, 1., neuron_sigma, mutable=True),
            NPS('layer', ['hidden'], mutable=False)
        )
    ],
    [
        GeneSpec('default',
            PS('weight', mutation_sigma=conn_sigma, mean_value = 0., mutable=True))
    ]
)


mut = Mutator(net_spec, allowed_neuron_types = ['sigmoid'])


neat_obj = NEAT(mutator = mut, **conf)

genome = neat_obj.get_init_genome(
        in1=neuron('input', protected=True, layer='input'),
        in2=neuron('input', protected=True, layer='input'),
        out1=neuron('sigmoid', protected=True, layer='output'),
        connections=[
            connection('default', protected=False, src='in1', dst='out1', weight = 0.33433),
            connection('default', protected=False, src='in2', dst='out1', weight = -0.77277)
        ]
    )


with open('init_genome.yaml', 'w+') as outfile:
    outfile.write(genome.to_yaml())