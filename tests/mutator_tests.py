import unittest
from neat import GeneticEncoding, Mutator, NetworkSpec
import os
import sys
import yaml
import math

here = os.path.abspath(os.path.dirname(__file__))


class TestMutator(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("Testing Mutator")
        with open(os.path.join(here, "loopback_test_genome.yaml"), 'r') as in_file:
            cls.loopback_test_genome = GeneticEncoding().from_yaml(yaml.load(in_file.read()))



    def test_loopback_protection(self):
        
        genome = self.loopback_test_genome
        print(genome)

        mutator = Mutator(
            net_spec=NetworkSpec([], []),
        )