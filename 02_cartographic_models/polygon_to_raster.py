# This module includes a few different algorithms for converting vector geometry to raster.
# Line drawing - digital differential analyzer and Bresenham's algorithm.
# Polygon fill -

import matplotlib.pyplot as plt
import json
import numpy as np

def polygon_fill(coords):
    """Algorithm - involves integer calculations only - should be faster:
        Find y min and y max of polygon
        Find the intersection point of of each scan line
        Sort by x coordinates
        Fill the alternate pairs
      From: https://www.geeksforgeeks.org/scan-line-polygon-filling-using-opengl-c/
      :param coords: Polygon points, with first point equal to last
      :return: list of integer points in the polygon
      """
    edges = []
    for i in range(len(coords) - 1):
        if coords[i][1] < coords[i+1][1]:
            pt1, pt2 = coords[i], coords[i+1]
        else:
            pt1, pt2 = coords[i+1], coords[i]
        slope_inv = 0
        if (pt2[0] - pt1[0]) != 0:
            slope_inv = 1.0 / ((pt2[1] - pt1[1]) / (pt2[0] - pt1[0]))
        edges.append([pt1[0], pt1[1], pt2[1], slope_inv])

    edges_sorted = sorted(edges, key=lambda p: p[1])

    y_min = edges_sorted[0][1]
    y_max = edges_sorted[len(edges_sorted)-1][2]
    xs, ys = [], []
    for y in range(y_min, y_max):
        # Does scan line intersect the edge?
        active_edges = []
        for e in edges:
            if e[1] < y < e[2]:
                active_edges.append(e)
        active_edges.sort(key=lambda a: a[0])
        intersections = []
        for e in active_edges:
            x = e[0] + (y - e[1]) * e[3]
            intersections.append(int(round(x)))
        for j in range(len(intersections)-1):
            if j % 2 == 0:
                xs.append(intersections[j])
                ys.append(y)
                for x in range(intersections[j]+1, intersections[j+1]):
                    xs.append(x)
                    ys.append(y)
    return xs, ys

with open("hunter_north_3857.geojson") as f:
    feature_collection = json.loads(f.read())
    coords = feature_collection["features"][0]["geometry"]["coordinates"][0]
    x, y = [], []
    for c in coords:
        x.append(int(round(c[0])))
        y.append(int(round(c[1])))
    x_min = min(x)
    y_min = min(y)
    x_max = max(x)
    y_max = max(y)
    grid = np.zeros((y_max - y_min, x_max - x_min))
    xs, ys = polygon_fill(list(zip(x, y)))
    plt.scatter(xs, ys)
    plt.plot(x, y, color='red')
    plt.show()

plt.show(block=True)