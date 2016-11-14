import unittest
from neat import NeuronGene, ConnectionGene



class TestGenes(unittest.TestCase):
    def test_neuron_gene(self):
        print("Testing NeuronGene")
        
        ng = NeuronGene('sigmoid', bias=0.3, gain=0.96)

        self.assertEquals(
            'sigmoid',
            ng.neuron_type,
            msg="ng.neuron_type is wrong"
            )

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
            ng.gene_params,
            msg="ng.gene_params is wrong"
            )

        self.assertEquals(
            {'bias': 0.3, 'gain': 0.96},
            ng.neuron_params,
            msg="ng.neuron_params is wrong"
            )

        params = ng.get_params()
        self.assertEquals(
            {'bias': 0.3, 'gain': 0.96},
            params,
            msg="ng.get_params() is wrong"
            )

        params['gain'] = 4444.8888

        self.assertEquals(
            {'bias': 0.3, 'gain': 4444.8888},
            ng.get_params(),
            msg="ng's params should have changed after we make changes to their reference" 
            )


        copied_params = ng.copy_params()
        self.assertEquals(
            {'bias': 0.3, 'gain': 4444.8888},
            copied_params,
            msg="ng.copy_params() is wrong"
            )
        copied_params['bias'] = 1111.2222

        self.assertEquals(
            {'bias': 0.3, 'gain': 4444.8888},
            ng.get_params(),
            msg="ng's params should not have changed after we make changes to their copy"
            )


        self.assertEquals(
            4444.8888,
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

        ng.new_param = 'foo'
        self.assertEquals(
            'foo',
            ng['new_param'],
            msg="ng.new_param <- 'foo' : addition of a new parameter did not work"
            )


        ng['newer_param'] = 'bar'
        self.assertEquals(
            'bar',
            ng['newer_param'],
            msg="ng['newer_param'] <- 'bar' : addition of a new parameter did not work"
            )


        with self.assertRaises(AttributeError):
            ng.does_not_exist


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
            cg.connection_type,
            msg="ng.connection_type is wrong"
            )

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