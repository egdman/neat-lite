__author__ = 'Dmitry Egorov'
__version__ = '0.2.0'

from .specs import GeneSpec, NumericParamSpec, NominalParamSpec, NetworkSpec
from .genes import NeuronGene, ConnectionGene, GeneticEncoding
from .operators import Mutator, crossover
from .utils import zip_with_probabilities, weighted_random
from .neat import NEAT, neuron, connection, validate_genotype