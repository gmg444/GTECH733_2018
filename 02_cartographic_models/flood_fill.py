# #############################################################################
# # Flood fill without recursion
# #############################################################################
import matplotlib.pyplot as plt
import numpy as np
import json

# Land cover classes from https://www.mrlc.gov/nlcd01_leg.php.  See
# the legend here:  https://www.mrlc.gov/nlcd11_leg.php
file_name = "nlcd_subset_5.npy"
class_raster = np.load(file_name)
# class_raster[class_raster != 41] = 1  # Class 1 - everything else.
# class_raster[class_raster == 41] = 2  # Class 2 - deciduous forest.
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
patch_list = []
for r in range(class_raster.shape[0]):
    for c in range(class_raster.shape[1]):
        if fill_raster[r, c] == 0:
            patch_id += 1
            print("Processing patch " + str(patch_id))
            fill.add((r, c))
            flood_fill()
            v = class_raster[r, c]
            patch_list.append({
                "id": patch_id,
                "size": int((fill_raster == patch_id).sum()),
                "class": int(v),
                "class_size": int((class_raster == v).sum())
            })

patch_list.sort(key=lambda x: x["size"], reverse=True)
with open("patch_list.json", "w") as f:
    f.write(json.dumps(patch_list))

# Now we can use this data to get information from the raster, for example, the largest forest class (v = 41)
out_arr = np.zeros(class_raster.shape)
for d in patch_list:
    if d["class"] == 41:
        patch_id = d["id"]
        out_arr[fill_raster == patch_id] = 1
        plt.imshow(out_arr)
        plt.show(block=True)
        break

# Compare original, class raster, target raster
original = np.load(file_name)

# plt.subplot(211)
# plt.title("Original")
# plt.imshow(original)

plt.subplot(211)
plt.title("Land Cover")
plt.imshow(class_raster)

plt.subplot(212)
plt.title("Patch Identifiers")
plt.imshow(fill_raster)


# plt.subplot(212)
# plt.title("Patch Identifiers")
# plt.imshow(fill_raster * (class_raster == 2))

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