import Orange
from random import randint
import numpy as np

# Get the data - note that the .tab file must
# include column headings as well as discrete/continuous
# indicators, and the class label indicator
data = Orange.data.Table("biomass_predictors_discrete.tab")
random_indices = [randint(0, 1) for i in range(len(data))]

# Get test and train randomized subsets
train = Orange.data.Table(data.domain,
                           [data[i] for i in range(len(data)) if random_indices[i] == 0])
test = Orange.data.Table(data.domain,
                           [data[i] for i in range(len(data)) if random_indices[i] == 1])

# Create and train the learner
learner = Orange.classification.RandomForestLearner()
clf = learner(train)

# Get predictions
predicted = clf(test)
confusion_matrix = np.zeros((3, 3))
for i in range(len(predicted)):
    confusion_matrix[
        int(predicted[i]),
        int(test[i].get_class().value)-1] += 1

print(confusion_matrix)

res = Orange.evaluation.CrossValidation(data, [learner])
ca = Orange.evaluation.scoring.CA(res)
precision = Orange.evaluation.Precision(res)
recall = Orange.evaluation.Recall(res)


print(ca)
print(precision)
print(recall)
