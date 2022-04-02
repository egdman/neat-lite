__author__ = 'Dmitry Egorov'
__version__ = '0.3.0'

from .specs import GeneSpec, ParamSpec, NetworkSpec, gen_gauss, gen_uniform, mut_gauss
from .genes import NeuronGene, ConnectionGene, GeneticEncoding
from .operators import Mutator, crossover
from .utils import zip_with_probabilities, weighted_random
from .neat import NEAT, neuron, connection, validate_genotype