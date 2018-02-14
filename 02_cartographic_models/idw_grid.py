import math
import numpy as np
from numpy.random import randint
from scipy.spatial import distance
import matplotlib.pyplot as plt


def idw(Z, b):
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
    a = p1 - p2
    return np.sqrt(a.dot(a))


def d3(p1, p2):
    """Or 2-norm, or euclidean norm or magnitude of vector from linear algebra library"""
    a = p1 - p2
    return np.linalg.norm(a)


def d4(p1, p2):
    """Or distance from distance library"""
    return distance.euclidean(p1, p2)


def test_distances():
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

def vector2d():
    """Aside to show vector magnitude and direction"""
    p1 = np.array([10, 10])
    p2 = np.array([50, 90])
    a = p2 - p1
    print(a)
    v = np.sqrt(a.dot(a))
    print(v)
    theta = math.atan(a[0] / a[1])
    print(math.degrees(theta))
    print(v * math.sin(theta), v * math.cos(theta))  # Row, columnsss

def distance_as_rank():
    """Shows that generic distance can be used to rank items by similarity (inverse of distance)"""
    pts = np.array([randint(100, size=(2,)) for i in range(5)])
    ref_pt = randint(100, size=(2,))
    a = pts - ref_pt
    b = a ** 2
    c = b.sum(axis=1)
    d = np.sqrt(c)
    sorted_indices = d.argsort()
    print("Reference point:", ref_pt)
    print("10 closest points:", pts[sorted_indices])

def distance_as_classification():
    """Shows this can be used as a simple classifier - e.g., eight satellite bands"""
    pts = np.array([randint(100, size=(8,)) for i in range(5)])
    ref_pt = randint(100, size=(8,))
    a = pts - ref_pt
    b = a ** 2
    c = b.sum(axis=1)
    d = np.sqrt(c)
    sorted_indices = d.argsort()
    print("Reference point:", ref_pt)
    print("Class is that of the closest point:", pts[sorted_indices[0]])

def distance_as_regression():
    """Shows this can be used as a simple k nearest-neighbor regression, e.g. 5 predictor variables"""
    pts = np.array([randint(100, size=(5,)) for i in range(10)])
    z = np.random.rand(10)
    ref_pt = randint(100, size=(5,))
    a = pts - ref_pt
    b = a ** 2
    c = b.sum(axis=1)
    d = np.sqrt(c)
    sorted_indices = d.argsort()
    sorted_z = z[sorted_indices]
    weights = 1 / d[sorted_indices] ** 2
    zw = (sorted_z* weights).sum()
    sw = weights.sum()
    print("Sorted points:", pts[sorted_indices])
    print("Sorted values:", sorted_z)
    print("Reference point:", ref_pt)
    print("Estimated value:", zw/sw)


def idw_grid():
    """Runds IDW over a grid to generate a model input surface"""
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
                    val = idw(pts[:num_points], power)
                    grid[r, c] = val
            plt.subplot(3, 3, power + i * 3)
            plt.title("Power: {0}, NumPoints: {1}".format(power, num_points))
            plt.imshow(grid)

    plt.show(block=True)

if __name__ == "__main__":
    print("----------------------------------------------")
    print("Equivalent representations of distance")
    test_distances()
    print("\n----------------------------------------------")
    print("Distance as vector magnitude")
    vector2d()
    print("\n----------------------------------------------")
    print("Distance used to rank by similarity")
    distance_as_rank()
    print("\n----------------------------------------------")
    print("Using nearest-neighbor distance to classify")
    distance_as_classification()
    print("\n----------------------------------------------")
    print("Using distance for k-nearest-neighbor regression")
    distance_as_regression()
    print("\n----------------------------------------------")
    idw_grid()