# This module includes a few different algorithms for converting vector geometry to raster.
# Line drawing - digital differential analyzer and Bresenham's algorithm.
# Polygon fill -

import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import math


def DDA(coords):
    """Algorithm:
        Calculate the difference between endpoints
        Get the number of steps in each direction
        Get the x,y increment
        Step through increments and turn on pixels
      From: https://www.tutorialspoint.com/computer_graphics/pdf/line_generation_algorithm.pdf
      :param coords: Starting and ending points of line
      :return: list of integer points on the line
      """
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]

    if dx > dy:
        steps = int(round(math.fabs(dx)))
    else:
        steps = int(round(math.fabs(dy)))

    x_increment = 0 if steps == 0 else dx / float(steps)
    y_increment = 0 if steps == 0 else dy / float(steps)
    result = []
    x, y = coords[0][0], coords[0][1]
    for i in range(steps):
        x += x_increment;
        y += y_increment;
        result.append([int(round(x)), int(round(y))])
    return result


def bresenham(coords):
    """Algorithm - involves integer calculations only - should be faster:
        Calculate the difference between endpoints
        Get the number of steps in each direction
        Get the x,y increment
        Step through increments and turn on pixels
      From: https://github.com/encukou/bresenham/blob/master/bresenham.py
      :param coords: Starting and ending points of line
      :return: list of integer points on the line
      """

    x0, y0 = coords[0]
    dx = coords[1, 0] - coords[0, 0]
    dy = coords[1, 1] - coords[0, 1]

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2*dy - dx
    y = 0

    result = []
    for x in range(dx + 1):
        # yield x0 + x*xx + y*yx, y0 + x*xy + y*yy
        result.append([x0 + x*xx + y*yx, y0 + x*xy + y*yy])
        if D > 0:
            y += 1
            D -= dx
        D += dy
    return result


grid_size = 100
grid = np.zeros((grid_size, grid_size))

lines = [randint(100, size=(2, 2)) for i in range(10)]

for line in lines:
    # points_in_line = DDA(line)
    points_in_line = bresenham(line)
    for c in points_in_line:
        grid[c[0], c[1]] = 1
    grid[line[0][0], line[0][1]] = 2
    grid[line[1][0], line[1][1]] = 2

plt.imshow(grid)
plt.show(block=True)