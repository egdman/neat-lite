__author__ = 'Dmitry Egorov'
__version__ = '0.4.0'

from .specs import GeneSpec, ParamSpec, bounds, gen, mut
from .genes import NeuronGene, ConnectionGene, Genome
from .operators import Mutator, crossover, neuron, upstream_neuron, downstream_neuron, connection
from .pipeline import Pipeline, validate_genome, default_gene_factory, get_TIMING, get_ARC_STATS
