import math
import numpy as np
from numpy.random import randint
from scipy.spatial import distance
import matplotlib.pyplot as plt


def IDW(Z, b):
    """
    Inverse distance weighted interpolation, from Xiao GIS Algorithms, https://github.com/gisalgs/interpolation.
    Input
      Z: a list of lists where each element list contains
         four values: X, Y, Value, and Distance to target
          point. Z can also be a NumPy 2-D array.
      b: power of distance
    Output
      Estimated value at the target location.
    """
    zw = 0.0                # sum of weighted z
    sw = 0.0                # sum of weights
    N = len(Z)              # number of points in the data
    for i in range(N):
        d = Z[i][3]
        if d == 0:
            return Z[i][2]
        w = 1.0/d**b
        sw += w
        zw += w*Z[i][2]
    return zw/sw


def d1(p1, p2):
    """2-dimensional Euclidean distance"""
    d = 0
    for i in range(len(p1)):
        d += (p1[i] - p2[i]) ** 2
    return math.sqrt(d)


def d2(p1, p2):
    """Same thing using dot product"""
    x = p1 - p2
    return np.sqrt(x.dot(x))


def d3(p1, p2):
    """Or 2-norm, magnitude of vector from linear algebra library"""
    x = p1 - p2
    return np.linalg.norm(x)


def d4(p1, p2):
    """Or distance from distance library"""
    return distance.euclidean(p1, p2)


def test_distances(p1, p2):
    """Tests to make sure the distance functions are all the same"""
    p1 = randint(100, size=(100,))
    p2 = randint(100, size=(100,))
    print("Simple calculation")
    print(d1(p1, p2))
    print("Dot product")
    print(d2(p1, p2))
    print("Linear algebra 2-norm")
    print(d3(p1, p2))
    print("Scipy Euclidean distance")
    print(d4(p1, p2))

def sample_distance_classifier():
    """Show that you can use distance to classify points; example in 2 dimensions for grid display"""
    grid = np.zeros((100, 100))
    class_pts = [(25, 25), (75, 75)]
    candidate_pt = (93, 27)
    grid[class_pts[0]] =1
    grid[class_pts[1]] =2
    if d3(np.array(candidate_pt), np.array(class_pts[0])) < d3(np.array(candidate_pt), np.array(class_pts[1])):
        grid[candidate_pt] = 1
    else:
        grid[candidate_pt] = 2
    plt.imshow(grid)
    plt.show(block=True)

# A few random points, with extra dimensions for z and distance
pts = [randint(100, size=(4,)) for i in range(100)]
grid_size = 100
num_points_list = [1, 5, 10]

# Trying for different values of numpoints and exponent for IDW surface
for i in range(len(num_points_list)):
    num_points = num_points_list[i]
    for power in range(1, 4):
        # For every cell
        grid = np.zeros((grid_size, grid_size))
        for r in range(grid.shape[0]):
            for c in range(grid.shape[1]):
                p = np.array([c, r])
                # Set the distance of each point to the current row and column
                for j in range(len(pts)):
                    pts[j][3] = d3(pts[j][:2], p)
                pts.sort(key=lambda x: x[3])
                val = IDW(pts[:num_points], power)
                grid[r, c] = val
        plt.subplot(3, 3, power + i * 3)
        plt.title("Power: {0}, NumPoints: {1}".format(power, num_points))
        plt.imshow(grid)

plt.show(block=True)