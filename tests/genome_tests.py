import unittest
from neat.genes import Genome, NeuronGene
# import os
import sys
# import yaml
# import math

# here = os.path.abspath(os.path.dirname(__file__))

# nn_impl_path = os.path.join(here, "../examples")
# sys.path.append(nn_impl_path)
# from nn_impl import NN



# no overlap
hm1 = (1, 2, 3, 5, 6, 8, 9, 10, 14, 15, 17, 18, 19, 20, 21, 23, 24, 26, 27, 29, 32, 33, 35, 38, 39, 40, 41, 42, 43, 44, 45, 48)
hm2 = (51, 52, 53, 54, 55, 57, 58, 59, 61, 63, 65, 67, 68, 69, 70, 71, 72, 73, 74, 76, 80, 81, 82, 84, 85, 87, 88, 89, 91, 92, 95, 96)


# overlap
hm1 = (5, 6, 7, 9, 11, 15, 19, 20, 21, 25, 26, 29, 30, 33, 34, 36, 37, 38, 43, 44, 46, 47, 48, 49, 52, 54, 64, 65, 66, 68, 69, 70)
hm2 = (25, 27, 28, 30, 32, 34, 36, 39, 43, 44, 52, 56, 57, 58, 59, 61, 63, 67, 68, 69, 70, 74, 76, 77, 79, 80, 84, 85, 89, 92, 94, 96)
# 25 25
# 30 30
# 34 34
# 36 36
# 43 43
# 44 44
# 52 52
# 68 68
# 69 69
# 70 70

def hm(gene):
    return gene.historical_mark if gene is not None else None

class TestGenome(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.genome1 = Genome()
        for hm in hm1:
            cls.genome1.neuron_genes.append(NeuronGene('1', hm))

        cls.genome2 = Genome()
        for hm in hm2:
            cls.genome2.neuron_genes.append(NeuronGene('2', hm))


    def test_gene_pairing(self):
        genes1 = sorted(
            self.genome1.neuron_genes + self.genome1.connection_genes, key=hm)

        genes2 = sorted(
            self.genome2.neuron_genes + self.genome2.connection_genes, key=hm)

        for g1, g2 in Genome.get_pairs(genes1, genes2):
            print("{} {}".format(hm(g1), hm(g2)))


    # def test_neuron_gene(self):
        
    #     genome = self.genome
    #     # print(genome)

    #     self.assertEqual(
    #         -0.09032,
    #         round(genome.neuron_genes[-1].bias, 6),
    #         msg="Got wrong attrribute value when converted from YAML"
    #         )

    #     self.assertEqual(
    #         0.774753,
    #         round(genome.neuron_genes[-3].gain, 6),
    #         msg="Got wrong attrribute value when converted from YAML"
    #         )

    #     self.assertEqual(
    #         535,
    #         genome.neuron_genes[-3].historical_mark,
    #         msg="Got wrong attribute value when converted from YAML"
    #         )
