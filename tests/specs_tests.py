import unittest
from neat import ParamSpec as PS, gen, mut, bounds, GeneSpec

class TestSpecs(unittest.TestCase):
    def test_gene_spec(self):
        p1 = PS('param1', bounds(-1., 1.), gen.uniform(), mut.gauss(0.5))
        p2 = PS('param2', gen.random_choice(("x",)), mut.random_choice(("x",)))
        p3 = PS('param3', gen.const(53))

        self.assertEqual(p1.min_value, -1)
        self.assertEqual(p1.max_value, 1)
        self.assertIsNotNone(p1.generator)
        self.assertIsNotNone(p1.mutator)

        self.assertIsNone(p2.min_value)
        self.assertIsNone(p2.max_value)
        self.assertIsNotNone(p2.generator)
        self.assertIsNotNone(p2.mutator)
        self.assertEqual(p2.generate_value(), "x")
        self.assertEqual(p2.mutate_value("irrelevant_value"), "x")

        self.assertIsNone(p3.min_value)
        self.assertIsNone(p3.max_value)
        self.assertIsNotNone(p3.generator)
        self.assertIsNone(p3.mutator)
        self.assertEqual(p3.generate_value(), 53)
        self.assertEqual(p3.mutate_value(257), 257) # same value because no mutator is set for param3
