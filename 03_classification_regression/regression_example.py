import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Linear regression in plain python
# From https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
def regression_with_lists(data):
    # Calculate Coefficients

    # Calculate the mean value of a list of numbers
    def mean(values):
        return sum(values) / float(len(values))

    # Calculate covariance between x and y
    def covariance(x, mean_x, y, mean_y):
        covar = 0.0
        for i in range(len(x)):
            covar += (x[i] - mean_x) * (y[i] - mean_y)
        return covar

    # Calculate the variance of a list of numbers
    def variance(values, mean):
        return sum([(x-mean)**2 for x in values])

    # Calculate coefficients
    def coefficients(dataset):
        x = [row[0] for row in dataset]
        y = [row[1] for row in dataset]
        x_mean, y_mean = mean(x), mean(y)
        b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    # calculate coefficients
    b0, b1 = coefficients(data)
    print('Coefficients: B0=%.3f, B1=%.3f' % (b0, b1))

def regression_with_matrices(data):
    # Linear regression in numpy
    # From https://thetarzan.wordpress.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/

    ## read data into a Numpy array
    b1 = np.array(list(data))[1:, 3:5].astype('float')

    nrow = b1.shape[0]

    intercept = np.ones((nrow, 1))
    b2 = b1[:, 0].reshape(-1, 1)

    X = np.concatenate((intercept, b2), axis=1)
    Y = b1[:, 1].T

    ## X and Y arrays must have the same number of columns for the matrix multiplication to work:
    print(X.shape)
    print(Y.shape)

    ## Use the equation above (X'X)^(-1)X'Y to calculate OLS coefficient estimates:
    bh = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    print (bh)

df_all = pd.read_csv("biomass_predictors_continuous.csv")
df_all = df_all.sample(frac=1.0, random_state=0)

data = df_all[:10000]

y_train = data["NCBD_30M"].values
x_train = data.iloc[:, 0:1].values  #Try different columns here

## check your work with Numpy's built in OLS function:
regression_with_lists(data)
regression_with_matrices(data)
z, resid, rank, sigma = np.linalg.lstsq(x_train, y_train)
print(z)