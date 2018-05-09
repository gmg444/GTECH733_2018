import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

df_all = pd.read_csv("biomass_predictors_discrete.csv")
df_all = df_all.sample(frac=1.0, random_state=0)
df_train = df_all[:10000]
y_train = df_train["NCBD_30M"].values
x_train = df_train.iloc[:, 0:13].values
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x_train, y_train)

df_test = df_all[:-10000]
y_test = df_test["NCBD_30M"].values
x_test = df_test.iloc[:, 0:13].values
y_est = clf.predict(x_test)

c = confusion_matrix(y_test, y_est)
print(c)

clf = RandomForestClassifier()
clf = clf.fit(x_train, y_train)

df_test = df_all[:-10000]
y_test = df_test["NCBD_30M"].values
x_test = df_test.iloc[:, 0:13].values
y_est = clf.predict(x_test)

c = confusion_matrix(y_test, y_est)
print(c)

"""
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
regr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)
print(regr.coef_)

print(np.mean((regr.predict(x_test)-y_test)**2))

# Explained variance score: 1 is perfect prediction
# and 0 means that there is no linear relationship
# between X and Y.
regr.score(x_test, y_test)

import numpy as np
>>> X_folds = np.array_split(X_digits, 3)
>>> y_folds = np.array_split(y_digits, 3)
>>> scores = list()
>>> for k in range(3):
...     # We use 'list' to copy, in order to 'pop' later on
...     X_train = list(X_folds)
...     X_test  = X_train.pop(k)
...     X_train = np.concatenate(X_train)
...     y_train = list(y_folds)
...     y_test  = y_train.pop(k)
...     y_train = np.concatenate(y_train)
...     scores.append(svc.fit(X_train, y_train).score(X_test, y_test))
>>> print(scores)
[0.93489148580968284, 0.95659432387312182, 0.93989983305509184]

 from sklearn.neighbors import NearestNeighbors
>>> import numpy as np
>>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
>>> nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X)
>>> distances, indices = nbrs.kneighbors(X)
>>> indices    

from sklearn.ensemble import RandomForestClassifier
X = [[0, 0], [1, 1]]
Y = [0, 1]
 clf = RandomForestClassifier(n_estimators=10)
>>> clf = clf.fit(X, Y)
"""