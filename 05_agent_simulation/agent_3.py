#-------------------------------------------------------------------------------
# Name:		module1
# Purpose:
#
# Author:	  Administrator
#
# Created:	 08/03/2013
# Copyright:   (c) Administrator 2013
# Licence:	 <your licence>
#-------------------------------------------------------------------------------

import pylab as plt
import numpy as np

'''
Agent version of game of life
# Game of life rules:
# 1.Any live cell with fewer than two live neighbours dies, as if caused by under-population.
# 2.Any live cell with two or three live neighbours lives on to the next generation.
# 3.Any live cell with more than three live neighbours dies, as if by overcrowding.
# 4.Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.
'''

class agent:
	def __init__(self, x, y):
		self.state = 0
		self.x = x
		self.y = y
		if ( np.random.random() < 0.5 ):
			self.state = 1

	def getNextState(self, agentGrid, numX, numY):
		numAdjacentOn = 0
		if (self.x > 0) and (self.x < numX-1) and (self.y > 0) and (self.y < numY-1):
			  for j in range(self.y-1, self.y+2):
				for i in range(self.x-1, self.x+2):
					if (i != self.x) or (j != self.y):
						numAdjacentOn += agentGrid[i][j].state

		self.nextState = self.state

		if self.state == 1: # This is a live cell
			if numAdjacentOn < 2:
				self.nextState = 0
			if numAdjacentOn > 3:
				self.nextState = 0

		if self.state == 0: # this is a dead cell
			if numAdjacentOn == 3:
				self.nextState = 1

	def update(self):
		self.state = self.nextState

class context:
	def __init__(self, numX, numY):
		self.stateGrid = np.zeros((numX, numY))
		self.agentGrid = []
		self.width = numX
		self.height = numY
		for j in range(numY):
			self.agentGrid.append([])
			self.agentGrid[j] = []
			for i in range(numX):
				a = agent(i, j)
				self.stateGrid[j, i] = a.state
				self.agentGrid[j].append(a)

	def run(self, iterations):
		for k in range(iterations):
			for j in range(self.height):
				for i in range(self.width):
					self.agentGrid[i][j].getNextState(self.agentGrid, self.width, self.height)
			for j in range(self.height):
				for i in range(self.width):
					self.agentGrid[i][j].update()
					self.stateGrid[j, i] = self.agentGrid[i][j].state
			plt.matshow(self.stateGrid, cmap=plt.cm.gray)
			plt.savefig('C:/dev/code/output/{0}.png'.format(k))

def main():
	c = context(25, 25)
	c.run(100)
	print "Done!"

if __name__ == '__main__':
	main()
