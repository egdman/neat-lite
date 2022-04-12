import unittest
from neat.genes import Genome, NeuronGene
# import os
import sys
from operator import itemgetter
# import yaml
# import math

# here = os.path.abspath(os.path.dirname(__file__))

# nn_impl_path = os.path.join(here, "../examples")
# sys.path.append(nn_impl_path)
# from nn_impl import NN


def hm(gene1, gene2=None):
    if gene1 is None:
        return None if gene2 is None else gene2.historical_mark
    else:
        return gene1.historical_mark


def make_genomes(marks1, marks2):
    g1, g2 = [NeuronGene('1', m) for m in marks1], [NeuronGene('2', m) for m in marks2]
    return sorted(g1, key=hm), sorted(g2, key=hm)


class TestGenome(unittest.TestCase):
    def assertGenePair(self, gene1, gene2, marks1, marks2):
        if gene1 is not None and gene2 is not None:
            self.assertEqual(hm(gene1), hm(gene2))
            self.assertIn(hm(gene1), marks1)
            self.assertIn(hm(gene1), marks2)
        elif gene1 is not None:
            self.assertIn(hm(gene1), marks1)
        else:
            self.assertIsNotNone(gene2)
            self.assertIn(hm(gene2), marks2)


    def assertUniqueAndSorted(self, elems):
        if len(elems) == 0:
            return
        self.assertEqual(len(elems), len(set(elems)))
        elems = iter(elems)
        e1 = next(elems)
        for e2 in elems:
            self.assertLess(e1, e2)
            e1 = e2


    def assertSwapped(self, genes1, genes2):
        pairs = tuple(Genome.get_pairs(genes1, genes2))
        expected1 = tuple(hm(itemgetter(0)(p)) for p in pairs)
        expected2 = tuple(hm(itemgetter(1)(p)) for p in pairs)

        pairs = tuple(Genome.get_pairs(genes2, genes1))
        received1 = tuple(hm(itemgetter(0)(p)) for p in pairs)
        received2 = tuple(hm(itemgetter(1)(p)) for p in pairs)
        self.assertEqual(received1, expected2)
        self.assertEqual(received2, expected1)


    def test_gene_pairing(self):
        # genes without overlap
        marks1 = (1, 2, 3, 5, 6, 8, 9, 10, 14, 15, 17, 18, 19, 20, 21, 23)
        marks2 = (24, 26, 27, 29, 32, 33, 35, 38, 39, 40, 41, 42, 43, 44, 45, 48)
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertGenePair(g1, g2, marks1, marks2)
            combined_marks.append(hm(g1, g2))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(set(marks1) | set(marks2)))
        self.assertSwapped(genes1, genes2)

        # genes with partial overlap, last is unpaired
        marks1 = (1, 2, 4, 6, 7, 8, 10, 14, 16, 17, 22, 26, 29, 32, 33, 34)
        marks2 = (14, 15, 21, 23, 24, 25, 26, 28, 30, 31, 32, 36, 39, 41, 43, 48)
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertGenePair(g1, g2, marks1, marks2)
            combined_marks.append(hm(g1, g2))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(set(marks1) | set(marks2)))
        self.assertSwapped(genes1, genes2)


        # genes with partial overlap, last is paired
        marks1 = (1, 2, 4, 6, 7, 8, 10, 14, 16, 17, 22, 26, 29, 32, 33, 36)
        marks2 = (14, 15, 21, 23, 24, 25, 26, 28, 30, 31, 32, 36, 39, 41, 43, 48)
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertGenePair(g1, g2, marks1, marks2)
            combined_marks.append(hm(g1, g2))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(set(marks1) | set(marks2)))
        self.assertSwapped(genes1, genes2)


        # genes with full overlap, last is unpaired
        marks1 = (1, 2, 4, 6, 7, 8, 10, 14, 16, 17, 22, 26, 29, 32, 33, 34, 39, 41, 43, 48)
        marks2 = (12, 16, 21, 22, 25, 26, 28, 30, 31, 32, 36)
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertGenePair(g1, g2, marks1, marks2)
            combined_marks.append(hm(g1, g2))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(set(marks1) | set(marks2)))
        self.assertSwapped(genes1, genes2)


        # genes with full overlap, last is paired
        marks1 = (1, 2, 4, 6, 7, 8, 10, 14, 16, 17, 22, 26, 29, 32, 33, 34, 39, 41, 43, 48)
        marks2 = (12, 16, 21, 22, 25, 26, 28, 30, 31, 32, 34)
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertGenePair(g1, g2, marks1, marks2)
            combined_marks.append(hm(g1, g2))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(set(marks1) | set(marks2)))
        self.assertSwapped(genes1, genes2)


        # one genome is empty
        marks1 = (8, 9, 15, 17, 18, 19, 21, 23, 25, 28, 29, 30, 31, 32, 33, 34)
        marks2 = ()
        genes1, genes2 = make_genomes(marks1, marks2)

        combined_marks = []
        for g1, g2 in Genome.get_pairs(genes1, genes2):
            self.assertIsNotNone(g1)
            self.assertIsNone(g2)
            self.assertIn(hm(g1), marks1)
            combined_marks.append(hm(g1))
        self.assertUniqueAndSorted(combined_marks)
        self.assertEqual(len(combined_marks), len(marks1))
        self.assertSwapped(genes1, genes2)

        # both genomes are empty
        self.assertEqual(tuple(Genome.get_pairs((), ())), ())


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
