import unittest
from neat import NeuronGene, ConnectionGene



class TestGenes(unittest.TestCase):
    def test_neuron_gene(self):
        print("Testing NeuronGene")
        
        ng = NeuronGene('sigmoid', bias=0.3, gain=0.96)

        self.assertEquals(
            'sigmoid',
            ng.gene_type,
            msg="ng.gene_type is wrong"
            )

        self.assertEquals(
            0.96,
            ng.gain,
            msg="ng.gain is wrong"
            )

        
        self.assertEquals(
            {'bias': 0.3, 'gain': 0.96},
            ng.get_params(),
            msg="ng.get_params() is wrong"
            )

        # params = ng.get_params()
        # params['gain'] = 4444.8888

        # self.assertEquals(
        #     {'bias': 0.3, 'gain': 4444.8888},
        #     ng.get_params(),
        #     msg="ng's params should have changed after we make changes to their reference" 
        #     )


        copied_params = ng.copy_params()
        self.assertEquals(
            {'bias': 0.3, 'gain': 0.96},
            copied_params,
            msg="ng.copy_params() is wrong"
            )
        copied_params['bias'] = 1111.2222

        self.assertEquals(
            {'bias': 0.3, 'gain': 0.96},
            ng.get_params(),
            msg="ng's params should not have changed after we make changes to their copy"
            )


        self.assertEquals(
            0.96,
            ng['gain'],
            msg="ng['gain'] is wrong"
            )


        ng.gain = 0.15
        self.assertEquals(
            0.15,
            ng['gain'],
            msg="ng.gain <- 0.5 assignment did not work"
            )


        ng['bias'] = 0.95
        self.assertEquals(
            0.95,
            ng['bias'],
            msg="ng['bias'] <- 0.95 assignment did not work"
            )


        # check that new attribute assignment with dot notation works
        ng.new_param = 'foo'
        self.assertEquals(
            'foo',
            ng['new_param'],
            msg="ng.new_param <- 'foo' : addition of a new parameter did not work"
            )


        # check that new attribute assignment with [] notation works
        ng['newer_param'] = 'bar'
        self.assertEquals(
            'bar',
            ng['newer_param'],
            msg="ng['newer_param'] <- 'bar' : addition of a new parameter did not work"
            )

        # check that __getattr__ throws correct exception for missing attributes
        with self.assertRaises(AttributeError):
            ng.does_not_exist


        # check that 'in' operator works correctly (it uses __contains__() method)
        self.assertEquals(
            False,
            'does_not_exist' in ng,
            msg="\"'does_not_exist' in ng\" should return False"
            )

        self.assertEquals(
            True,
            'new_param' in ng,
            msg="\"'new_param' in ng\" should return True"
            )


       # check that 'hasattr' works correctly
        self.assertEquals(
            False,
            hasattr(ng, 'surprise'),
            msg="hasattr(ng, 'surprise') should be False"
            )


        ng.surprise = 'foo'
        self.assertEquals(
            True,
            hasattr(ng, 'surprise'),
            msg="hasattr(ng, 'surprise') should be False"
            )



        
    def test_connection_gene(self):
        print("Testing ConnectionGene")
        
        cg = ConnectionGene('def_con', 888, 999, weight=0.8)


        self.assertEquals(
            'def_con',
            cg.gene_type,
            msg="ng.gene_type is wrong"
            )


        self.assertEquals(
            888,
            cg.mark_from,
            msg="cg.mark_from is wrong"
            )


        self.assertEquals(
            0.8,
            cg.weight,
            msg="cg.weight is wrong"
            )


        cg.mark_to = 556

        self.assertEquals(
            556,
            cg.mark_to,
            msg="cg.mark_to <- 556 : assignment did not work"
            )


        cg.second_param = 62.

        self.assertEquals(
            62.,
            cg.second_param,
            msg="cg.second_param <- 62 : addition of a new parameter did not work"
            )


        self.assertEquals(
            False,
            hasattr(cg, 'surprise'),
            msg="hasattr(cg, 'surprise') should be False"
            )


        cg.surprise = 'foo'
        self.assertEquals(
            True,
            hasattr(cg, 'surprise'),
            msg="hasattr(cg, 'surprise') should be False"
            )


        print cg.get_params()
        self.assertEquals(
            {'surprise': 'foo', 'second_param': 62.0, 'weight': 0.8},
            cg.get_params(),
            msg="cg.get_params() is wrong"
            )
