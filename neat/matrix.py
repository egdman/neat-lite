m_source = """
_ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _
_ _ 8 _ _ _ _ _ _
_ 2 5 _ _ _ _ _ 7
_ _ _ _ _ _ _ _ _
_ 8 _ 1 _ _ _ _ _
_ _ _ _ _ _ _ _ _
_ _ _ _ _ _ _ _ _
"""

def from_str(s):
	row_idx, col_idx = -1, -1
	rows_arr = []

	for row_idx, row in enumerate(s.strip().splitlines()):
		row_arr = []
		for col_idx, item in enumerate(row.strip().split(' ')):
			if item != '_':
				row_arr.append(int(item))
		rows_arr.append(row_arr)

	return rows_arr, (row_idx + 1, col_idx + 1)


mtx, shape = from_str(m_source)
# print(shape)
# for row in mtx:
# 	print(row)


def sparse_list(ls):
	n_empty = 0
	p_empty = False
	heads = []
	prefixes = []
	for idx, e in enumerate(ls):
		if e is None:
			if p_empty:
				pass
			else:
				p_empty = True
				heads.append(idx)
				prefixes.append(n_empty)
			n_empty += 1
		else:
			p_empty = False
	return heads, prefixes, n_empty


s = "0 0 0 0 0 8 0 0 0 0 3 0 0 6 0 0"
# s = "0 0 0 0 0 8 0 0 0 1 3 0 0 6 0 0"
# s = "0 0 0 0 0 8 0 0 0 1 3 0 1 6 0 0"
# s = "0 1 1 0 1 8 0 1 1 1 3 0 1 6 0 0"



# s = "0 0 0 0 0 8 0 1 0 0 3 0 0 6 0 0"
# s = "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15"
ls = tuple((None if e == '0' else int(e) for e in s.split(' ')))
# print(ls)

z_heads, z_prefixes, z_num = sparse_list(ls)
sls = tuple(zip(z_heads, z_prefixes))
# print(sls)
# print(f"num zeros = {z_num}")



# pick random zero from the sparse list:

from bisect import bisect
from random import randint

z_heads, z_prefixes, z_num = [0], [0], 10000

while z_num > 0:
	# pick a random zero element:
	z_elem_idx = randint(0, z_num - 1)
	z_block_idx = bisect(z_prefixes, z_elem_idx) - 1
	z_idx = z_heads[z_block_idx] - z_prefixes[z_block_idx] + z_elem_idx

	# print(f"random zero picked at {z_idx}, value = {ls[z_idx]}")
	# if ls[z_idx] is not None:
	# 	raise RuntimeError(f"random zero picked at {z_idx}, value = {ls[z_idx]}")

	# ### ### ### ### ### ### ### ### ### ### ### ### ###
	# insertion of a non-zero element:

	# calculate block length
	if z_block_idx + 1 == len(z_prefixes):
		block_len = z_num - z_prefixes[z_block_idx]
	else:
		block_len = z_prefixes[z_block_idx + 1] - z_prefixes[z_block_idx]

	if block_len == 1: # delete this block
		z_heads = z_heads[:z_block_idx] + z_heads[z_block_idx + 1:]
		z_prefixes = z_prefixes[:z_block_idx] + z_prefixes[z_block_idx + 1:]
		z_block_idx -= 1

	elif z_idx == z_heads[z_block_idx]: # at block head
		z_heads[z_block_idx] += 1

	elif z_idx + 1 == z_heads[z_block_idx] + block_len: # at block tail
		pass

	else: # neither head nor tail, split the block in two
		new_prefix = z_idx - z_heads[z_block_idx] + z_prefixes[z_block_idx]
		# if z_block_idx > 0:
		# 	new_prefix += z_prefixes[z_block_idx - 1]	

		z_heads = z_heads[:z_block_idx + 1] + [z_idx + 1] + z_heads[z_block_idx + 1:]
		z_prefixes = z_prefixes[:z_block_idx + 1] + [new_prefix] + z_prefixes[z_block_idx + 1:]
		z_block_idx += 1 # skip the new block when updating prefixes

	# update prefixes:
	for idx in range(z_block_idx + 1, len(z_prefixes)):
		z_prefixes[idx] = z_prefixes[idx] - 1
	z_num -= 1


print("no more zeros")
sls = tuple(zip(z_heads, z_prefixes))
print(sls)
