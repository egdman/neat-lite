import unittest
from neat.utils import _has_probability, zip_with_probabilities, weighted_random



class TestUtils(unittest.TestCase):
	def test_has_probability(self):
		print("Testing _has_probability()")
		hp = _has_probability

		self.assertEquals(False, hp('word'), msg="_has_probability('word') : Failed")
		self.assertEquals(False, hp(('word', '0.5')), msg="_has_probability(('word', '0.5')) : Failed")
		self.assertEquals(True,  hp(('word', 0.5)), msg="_has_probability(('word', 0.5)) : Failed")
		self.assertEquals(True,  hp(['word', 0.5]), msg="_has_probability(['word', 0.5]) : Failed")


	def test_zip_with_probabilities(self):
		print("Testing zip_with_probabilities()")
		zp = zip_with_probabilities

		self.assertEquals(
			[('a', .25), ('b', .25), ('c', .25), ('d', .25)],
			zp(['a', 'b', 'c', 'd']),
			msg="zip_with_probabilities(['a', 'b', 'c', 'd']) : Failed")

		self.assertEquals(
			[('a', .25), ('b', .25), ('c', .5)],
			zp([('a', .25), ('b', .25), 'c']),
			msg="zip_with_probabilities([('a', .25), ('b', .25), 'c']) : Failed")

		self.assertEquals(
			[('a', .25), ('b', .25), ('c', .5)],
			zp([('a', .25), ('b', .25), 'c']),
			msg="zip_with_probabilities([('a', .25), ('b', .25), 'c']) : Failed")

		self.assertEquals(
			[('a', .25), ('b', .25), ('c', .5)],
			zp([('a', .25), ('b', .25), ('c', .5)]),
			msg="zip_with_probabilities([('a', .25), ('b', .25), ('c', .5)]) : Failed")