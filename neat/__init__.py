__author__ = 'Dmitry Egorov'
__version__ = '0.3.0'

from .specs import GeneSpec, ParamSpec, gen_gauss, gen_uniform, mut_gauss
from .genes import NeuronGene, ConnectionGene, GeneticEncoding
from .operators import Mutator, crossover, neuron, connection
from .utils import zip_with_probabilities, weighted_random
from .neat import NEAT, validate_genotype, default_gene_factory