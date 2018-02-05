# #############################################################################
# # Flood fill without recursion
# #############################################################################
import matplotlib.pyplot as plt
import numpy as np

# Land cover classes from https://www.mrlc.gov/nlcd01_leg.php.  See
# the legend here:  https://www.mrlc.gov/nlcd11_leg.php
file_name = "nlcd_subset_1.npy"
class_raster = np.load(file_name)
class_raster[class_raster != 41] = 1  # Class 1 - everything else.
class_raster[class_raster == 41] = 2  # Class 2 - deciduous forest.
patch_id = 0
fill = set()
height, width = class_raster.shape

def flood_fill():
    while fill:
        r, c = fill.pop()
        if r == height or c == width or r < 0 or c < 0 or fill_raster[r, c] > 0:
            continue
        v = class_raster[r, c]
        fill_raster[r, c] = patch_id
        if r > 0 and class_raster[r-1, c] == v:
            fill.add((r-1, c))
        if r < fill_raster.shape[0] - 1 and class_raster[r+1, c] == v:
            fill.add((r+1, c))
        if c > 0 and class_raster[r, c-1] == v:
            fill.add((r, c-1))
        if c < fill_raster.shape[1] - 1 and class_raster[r, c+1] == v:
            fill.add((r, c+1))

# Create an empty target raster.
fill_raster = np.zeros((class_raster.shape), np.uint32)

# Go through the cells and assign a unique patch id to each contiguous set of pixels.
for r in range(class_raster.shape[0]):
    for c in range(class_raster.shape[1]):
        if fill_raster[r, c] == 0:
            patch_id += 1
            print("Processing patch " + str(patch_id))
            fill.add((r, c))
            flood_fill()

# An example of what you can do with this data - plot a histogram of patch sizes.
patch_sizes = []
for i in range(fill_raster.max()):
    patch_sizes.append((fill_raster == i).sum())
patch_sizes.sort(reverse=True)
plt.hist(patch_sizes, bins=10)
plt.show()

# Compare original, class raster, target raster
original = np.load(file_name)

# plt.subplot(211)
# plt.title("Original")
# plt.imshow(original)

plt.subplot(211)
plt.title("Deciduous Forest / Other")
plt.imshow(class_raster)

# plt.subplot(212)
# plt.title("Patch Identifiers")
# plt.imshow(fill_raster)


plt.subplot(212)
plt.title("Patch Identifiers")
plt.imshow(fill_raster * (class_raster == 2))

# Skewed distribution of patch sizes.
plt.show(block=True)
print("All done!")

"""
#############################################################################
# Flood fill with recursion - note that it causes a stack overflow.  The
# version above avoids recursion using by storing pending data in a set.
#############################################################################
import matplotlib.pyplot as plt
import numpy as np

patch_id = 0
class_raster = np.load("nlcd_subset_1.npy")
class_raster[class_raster <> 41] = 0
class_raster[class_raster == 41] = 1
plt.imshow(class_raster)
plt.show()

fill_raster = np.zeros((class_raster.shape), class_raster.dtype)

def flood_fill(fill_raster, r, c, target_class, depth):
    if fill_raster[r, c] == 0 and class_raster[r, c] == target_class:
        fill_raster[r, c] = patch_id
        if r > 0:
            flood_fill(fill_raster, r-1, c, target_class, depth + 1)
        if r < fill_raster.shape[0] - 1:
            flood_fill(fill_raster, r+1, c, target_class, depth + 1)
        if c > 0:
            flood_fill(fill_raster, r, c-1, target_class, depth + 1)
        if c < fill_raster.shape[1] - 1:
            flood_fill(fill_raster, r, c+1, target_class, depth + 1)

for r in range(class_raster.shape[0]):
    for c in range(class_raster.shape[1]):
        if fill_raster[r, c] == 0:
            patch_id += 1
        depth = 0
        flood_fill(fill_raster, r, c, class_raster[r, c], depth)

plt.subplot(211)
plt.imshow(class_raster)
plt.subplot(212)
plt.imshow(fill_raster)
plt.show()
"""