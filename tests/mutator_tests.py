import unittest
from neat import GeneticEncoding, Mutator, NetworkSpec, GeneSpec
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
        
        genome = self.loopback_test_genome.copy()

        net_spec = NetworkSpec(
            [],
            [GeneSpec('new_connection')]
        )

        mutator = Mutator(
            net_spec=net_spec            
        )


        con_added = False
        con_added = mutator.add_connection_mutation(genome)
        # print("---------------- ----------------")
        # print(genome)
        self.assertTrue(con_added, msg="Connection should have been added")

        genome = self.loopback_test_genome.copy()

        mutator = Mutator(
            net_spec=net_spec,
            pure_input_types=('input_type_1, input_type_2')
        )

        con_added = False
        con_added = mutator.add_connection_mutation(genome)
        # print("---------------- ----------------")
        # print(genome)
        self.assertFalse(con_added, msg="Connection should have not been added")
