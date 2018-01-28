import numpy as np
import time as tm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import psycopg2

'''
Climatic data from WorldClim / BioClim rasters:

BIO1 = Annual Mean Temperature
BIO2 = Mean Diurnal Range (Mean of monthly (max temp - min temp))
BIO3 = Isothermality (BIO2/BIO7) (* 100)
BIO4 = Temperature Seasonality (standard deviation *100)
BIO5 = Max Temperature of Warmest Month
BIO6 = Min Temperature of Coldest Month
BIO7 = Temperature Annual Range (BIO5-BIO6)
BIO8 = Mean Temperature of Wettest Quarter
BIO9 = Mean Temperature of Driest Quarter
BIO10 = Mean Temperature of Warmest Quarter
BIO11 = Mean Temperature of Coldest Quarter
BIO12 = Annual Precipitation
BIO13 = Precipitation of Wettest Month
BIO14 = Precipitation of Driest Month
BIO15 = Precipitation Seasonality (Coefficient of Variation)
BIO16 = Precipitation of Wettest Quarter
BIO17 = Precipitation of Driest Quarter
BIO18 = Precipitation of Warmest Quarter
BIO19 = Precipitation of Coldest Quarter
'''

# Connect to database
conn = psycopg2.connect("dbname='gtech' user='postgres' host='localhost' password='walrus'")

# Get a cursor
cur = conn.cursor()

# Try getting data. Note calls to Coalesce, which provides a substitute value
# for cases when the database column is null.  Ordering by random()
# and limiting to 1000 draws 1000 random samples from the table.
cur.execute("select bio1/10, bio12/10, bio15/100 from fia_plot_attrib " +
	"where coalesce(bio1, -9999) <> -9999 " +
	"and coalesce(bio12, -9999) <> -9999 " +
	"and coalesce(bio15, -9999) <> -9999 order by random() limit 1000")

# Get all rows.
rows = cur.fetchall()

# Initialize a numpy array the same dimensions as the data
data = np.zeros((cur.rowcount, 3), dtype=np.float64)

# Close the cursor
cur.close()

# Index is current item
index = 0

# Initialize the array
for row in rows:
	data[index, 0] = np.float64(row[0])
	data[index, 1] = np.float64(row[1])
	data[index, 2] = np.float64(row[2])
	index += 1

# Set up variables for the k-means calculation
n = cur.rowcount  # Number of elements
k = 3   # Number of means to find
dim = np.shape(data)[1]   # Dimensions in the data (3)
centers = np.random.rand(k, dim)   # Creates k random centers
oldCenters = np.random.rand(k, dim)  # Placeholder for old centers
numIterations = 0   # Counter for the number of iterations
MAX_ITERATIONS = 100   # Maximum number of itertions (constant)

# Normalize data based on minimum and maximum
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

# Keep recalculating centers until there is no change or we hit the maximum
while sum(sum(oldCenters-centers))!= 0 and numIterations<MAX_ITERATIONS:

    # Make a copy of the the k centers
	oldCenters = centers.copy()

    # Bump up the iteration counter
	numIterations += 1

	# Compute distances between each point and the first set of centers
	distances = np.ones((1, n))*np.sum((data-centers[0,:])**2,axis=1)

    # Calculate a distance between each point and the remaining k-1 centers
	for j in range(k-1):
		distances = np.append(distances, np.ones((1, n))*np.sum((data-centers[j+1,:])**2,axis=1), axis=0)

	# Identify the closest cluster for each data element
	closestClusters = distances.argmin(axis=0) # argmin give you the index of the minimum.
	closestClusters = np.transpose(closestClusters*np.ones((1, n)))

	# Update the cluster centres
	for j in range(k):
       # Find the points closest to the current cluster
	   currentCluster = np.where(closestClusters==j,1,0)
       # If any points have been found
	   if sum(currentCluster)>0:
           # Recalculate the cluster centers as the center of the closest points
		   centers[j,:] = np.sum(data*currentCluster,axis=0)/np.sum(currentCluster)

# Plot the results
color = closestClusters/float(k) # Array of colors for the plost

# Create a new 3d scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the 3 dimensions of the data
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color, s=20)
ax.set_xlabel('Temperature')
ax.set_ylabel('Precipitation')
ax.set_zlabel('Coefficient of Variation of Precipitation')

# Plot the found centers
ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c='#FF0000', marker="^", s=100)
plt.show()