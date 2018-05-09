import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# Linear regression in plain python
# From https://machinelearningmastery.com/implement-simple-linear-regression-scratch-python/
def linear_regression_with_lists(X, y):
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
    def coefficients(X, y):
        x = X.tolist()
        y = y.tolist()
        x_mean, y_mean = mean(x), mean(y)
        b1 = covariance(x, x_mean, y, y_mean) / variance(x, x_mean)
        b0 = y_mean - b1 * x_mean
        return [b0, b1]

    # calculate coefficients
    b0, b1 = coefficients(X, y)
    return b0, b1

def linear_regression_with_matrices(X, y):
    # Linear regression in numpy
    # From https://thetarzan.wordpress.com/2012/10/27/calculate-ols-regression-manually-in-python-using-numpy/
    nrow = y.shape[0]
    intercept = np.ones((nrow, 1))

    X = np.concatenate((intercept, X), axis=1)
    Y = y.T

    # Use the equation above (X'X)^(-1)X'Y to calculate OLS coefficient estimates:
    bh = np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, Y))
    return(bh)


def linear_regression_with_gradient_descent(X, y, m_current=0, b_current=0, epochs=10000, learning_rate=0.0001):
    # Linear regression with gradient descent.
    # https://towardsdatascience.com/linear-regression-using-gradient-descent-in-10-lines-of-code-642f995339c0
    N = float(len(y))
    for i in range(epochs):
        y_current = (m_current * X) + b_current
        cost = sum([data ** 2 for data in (y - y_current)]) / N
        m_gradient = -(2 / N) * sum(X * (y - y_current))
        b_gradient = -(2 / N) * sum(y - y_current)
        m_current = m_current - (learning_rate * m_gradient)
        b_current = b_current - (learning_rate * b_gradient)
    return m_current, b_current, cost

df_all = pd.read_csv("biomass_predictors_continuous.csv")
df_all = df_all.sample(frac=1.0, random_state=0)
data = df_all[:10000]

y_train = data["NCBD_30M"].values
x_train_2d = data.iloc[:, 0:1].values
x_train_1d = data["HEIGHT"].values

print("\nLinear regression with lists")
print(linear_regression_with_lists(x_train_1d, y_train))

print("\nLinear regression with matrices")
print(linear_regression_with_matrices(x_train_2d, y_train))

print("\nLinear regression with scikit learn")
reg = linear_model.LinearRegression(fit_intercept=True)
reg.fit(x_train_2d, y_train)
print(reg.intercept_, reg.coef_)

print("\nLinear regression with gradient descent")
print(linear_regression_with_gradient_descent(x_train_1d, y_train))

