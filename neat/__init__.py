__author__ = 'Dmitry Egorov'
__version__ = '0.4.1'

from .specs import GeneSpec, ParamSpec, bounds, gen, mut
from .genes import NeuronGene, ConnectionGene, Genome
from .operators import Mutator, crossover, neuron, connection
from .pipeline import Pipeline, validate_genome, default_gene_factory
