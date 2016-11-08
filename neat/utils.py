import numbers
import random
import numpy as np


def _has_probability(obj):
	try:
		proba = obj[-1]
		return isinstance(proba, numbers.Number)
	except (IndexError, TypeError):
		return False



def zip_with_probabilities(items):
	have_probas = list(elem for elem in items if _has_probability(elem))
	have_no_probas = list(elem for elem in items if elem not in have_probas)

	total_proba = sum(elem[1] for elem in have_probas)

	remain_proba = 1. - total_proba
	remain_proba = max(0. , remain_proba)

	num_elem = len(have_no_probas)
	if num_elem == 0: return have_probas

	proba = remain_proba / (1.*num_elem)
	have_no_probas = list((elem, proba) for elem in have_no_probas)
	return have_probas + have_no_probas



# TODO: write own implementation to remove dependency from numpy
def weighted_random(items_with_probas):
	items, probas = zip(*items_with_probas)
	return np.random.choice(items, p = probas)
