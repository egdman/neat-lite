import unittest
from neat.genes import NeuronGene, ConnectionGene
from neat.specs import GeneSpec

class TestGenes(unittest.TestCase):
    def test_neuron_gene(self):
        spec = GeneSpec('sigmoid')
        ng = NeuronGene(spec, bias=0.3, gain=0.96)

        self.assertEqual(
            'sigmoid',
            ng.get_type(),
            msg="ng.get_type() is wrong: {}".format(ng.get_type())
            )

        self.assertEqual(
            0.96,
            ng.gain,
            msg="ng.gain is wrong"
            )


        self.assertEqual(
            {'bias': 0.3, 'gain': 0.96},
            ng.get_params(),
            msg="ng.get_params() is wrong: {}".format(ng.get_params())
            )

        copied_params = ng.get_params()
        self.assertEqual(
            {'bias': 0.3, 'gain': 0.96},
            copied_params,
            msg="ng.get_params() is wrong"
            )
        copied_params['bias'] = 1111.2222

        self.assertEqual(
            {'bias': 0.3, 'gain': 0.96},
            ng.get_params(),
            msg="ng's params should not have changed after we make changes to their copy"
            )


        self.assertEqual(
            0.96,
            ng['gain'],
            msg="ng['gain'] is wrong"
            )


        ng.gain = 0.15
        self.assertEqual(
            0.15,
            ng['gain'],
            msg="ng.gain <- 0.5 assignment did not work"
            )


        ng['bias'] = 0.95
        self.assertEqual(
            0.95,
            ng['bias'],
            msg="ng['bias'] <- 0.95 assignment did not work"
            )


        # check that new attribute assignment with dot notation works
        ng.new_param = 'foo'
        self.assertEqual(
            'foo',
            ng['new_param'],
            msg="ng.new_param <- 'foo' : addition of a new parameter did not work"
            )


        # check that new attribute assignment with [] notation works
        ng['newer_param'] = 'bar'
        self.assertEqual(
            'bar',
            ng['newer_param'],
            msg="ng['newer_param'] <- 'bar' : addition of a new parameter did not work"
            )

        # check that __getattr__ throws correct exception for missing attributes
        with self.assertRaises(AttributeError):
            ng.does_not_exist


        # check that 'in' operator works correctly (it uses __contains__() method)
        self.assertEqual(
            False,
            'does_not_exist' in ng,
            msg="\"'does_not_exist' in ng\" should return False"
            )

        self.assertEqual(
            True,
            'new_param' in ng,
            msg="\"'new_param' in ng\" should return True"
            )


       # check that 'hasattr' works correctly
        self.assertEqual(
            False,
            hasattr(ng, 'surprise'),
            msg="hasattr(ng, 'surprise') should be False"
            )


        ng.surprise = 'foo'
        self.assertEqual(
            True,
            hasattr(ng, 'surprise'),
            msg="hasattr(ng, 'surprise') should be False"
            )




    def test_connection_gene(self):
        spec = GeneSpec('connection')
        cg = ConnectionGene(spec, 888, 999, weight=0.8)

        self.assertEqual(
            'connection',
            cg.get_type(),
            msg="ng.get_type() is wrong: {}".format(cg.get_type())
            )


        self.assertEqual(
            888,
            cg.mark_from,
            msg="cg.mark_from is wrong"
            )


        self.assertEqual(
            0.8,
            cg.weight,
            msg="cg.weight is wrong"
            )


        cg.mark_to = 556

        self.assertEqual(
            556,
            cg.mark_to,
            msg="cg.mark_to <- 556 : assignment did not work"
            )


        cg.second_param = 62.

        self.assertEqual(
            62.,
            cg.second_param,
            msg="cg.second_param <- 62 : addition of a new parameter did not work"
            )


        self.assertEqual(
            False,
            hasattr(cg, 'surprise'),
            msg="hasattr(cg, 'surprise') should be False"
            )


        cg.surprise = 'foo'
        self.assertEqual(
            True,
            hasattr(cg, 'surprise'),
            msg="hasattr(cg, 'surprise') should be False"
            )


        self.assertEqual(
            {'surprise': 'foo', 'second_param': 62.0, 'weight': 0.8},
            cg.get_params(),
            msg="cg.get_params() is wrong: {}".format(cg.get_params())
            )
