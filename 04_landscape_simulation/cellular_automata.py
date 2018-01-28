# --------------------------------------------------------------------
# Game of life using explicit rules.
# --------------------------------------------------------------------

import pylab as plt
import numpy as np

# Game of life rules:
# 1.Any live cell with fewer than two live neighbours dies, as if caused by under-population.
# 2.Any live cell with two or three live neighbours lives on to the next generation.
# 3.Any live cell with more than three live neighbours dies, as if by overcrowding.
# 4.Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

def new_state(t, x, y):
	numAdjacentOn = 0
	if (x > 0) and (x < t.shape[1]-1) and (y > 0) and (y < t.shape[0]-1):
		for i in range(x-1, x+2):
			for j in range(y-1, y+2):
				if (i != x) or (j != y):
					numAdjacentOn += t[i, j]

		if t[x, y] == 1: # This is a live cell
			if numAdjacentOn < 2:
				return 0
			if numAdjacentOn > 3:
				return 0

		if t[x, y] == 0: # this is a dead cell
			if numAdjacentOn == 3:
				return 1

	return t[x, y]

# Generate inputs in 2 dimensions
dim = 50
t0 = np.zeros((dim, dim))
t1 = np.zeros((dim, dim))

# Initialize to random on/off states
t0 = np.random.randint(2, size=(dim, dim))

for i in range(20):
	for x in range(dim):
		for y in range(dim):
			t1[x, y] = new_state(t0, x, y)
	t0 = t1.copy()  # assignment is by reference; we want a separate copy.

	plt.matshow(t1,fignum=100,cmap=plt.cm.gray)

	plt.savefig('C:/dev/code/output/{0}.png'.format(i))

print "Done!"