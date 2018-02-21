import Orange
from random import randint
import numpy as np

# Get the data - note that the .tab file must
# include column headings as well as discrete/continuous
# indicators, and the class label indicator
data = Orange.data.Table("biomass_predictors_continuous.tab")[:5000]

lin = Orange.regression.linear.LinearRegressionLearner()
rf = Orange.regression.random_forest.RandomForestRegressionLearner()
rf.name = "Random forest"
ridge = Orange.regression.RidgeRegressionLearner()
mean = Orange.regression.MeanLearner()
mean.neam = mean.value


learners = [lin, rf, ridge, mean]

res = Orange.evaluation.CrossValidation(data, learners, k=5)
rmse = Orange.evaluation.RMSE(res)
r2 = Orange.evaluation.R2(res)

print("Learner  RMSE  R2")
for i in range(len(learners)):
    print("{:8s} {:.2f} {:5.2f}".format(learners[i].name, rmse[i], r2[i]))