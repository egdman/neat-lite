import unittest
from neat import GeneticEncoding
import os
import sys
import yaml
import math

here = os.path.abspath(os.path.dirname(__file__))

nn_impl_path = os.path.join(here, "../examples")
sys.path.append(nn_impl_path)
from nn import NN


class TestGenome(unittest.TestCase):
    def test_neuron_gene(self):
        print("Testing GeneticEncoding")

        with open(os.path.join(here, "xor_genome.yaml"), 'r') as in_file:
            genome = GeneticEncoding().from_yaml(yaml.load(in_file.read()))


        self.assertEqual(
            0.650586,
            round(genome.neuron_genes[-1].bias, 6),
            msg="Got wrong attrribute value when converted from YAML"
            )

        self.assertEqual(
            -1.0,
            round(genome.neuron_genes[-3].bias, 1),
            msg="Got wrong attrribute value when converted from YAML"
            )

        self.assertEqual(
            53,
            genome.neuron_genes[-3].historical_mark,
            msg="Got wrong attrribute value when converted from YAML"
            )


        # xor_net = NN().from_genome(genome)

        # inputs = ((0., 0.), (0., 1.), (1., 0.), (1., 1.))
        # true_outputs = (0., 1., 1., 0.)

        # nn_outputs = list(xor_net.compute(inp)[0] for inp in inputs)

        # def rmse(X, Y):
        #     return math.sqrt( sum( (x - y)**2 for x, y in zip(X, Y) ) )

        # self.assertEqual(
        #     round(rmse(nn_outputs, true_outputs), 6),
        #     0.,
        #     msg="Test network gives wrong results")
