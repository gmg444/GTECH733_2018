import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df_all = pd.read_csv("biomass_predictors_continuous.csv")
df_all = df_all.sample(frac=1.0, random_state=0)

df_train = df_all[:10000]
y_train = df_train["NCBD_30M"].values
x_train = df_train.iloc[:, 0:1].values  #Try different columns here

df_test = df_all[:-10000]
y_test = df_test["NCBD_30M"].values
x_test = df_test.iloc[:, 0:1].values

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(x_train, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test)

# The coefficients
print('Coefficients: \n', regr.coef_)

# The mean squared error
print("Mean squared error: %.2f"  % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

# Plot outputs
plt.scatter(x_test[:, 0], y_test,  color='black')
plt.plot(x_test[:, 0], y_pred, color='blue', linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show(block=True)
