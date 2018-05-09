import numpy as np


num_classes = 3

# i = predicted; j = refererence
def overall_accuracy(c):
    # Overall accuracy equals summ[ii] / sum[all]
    num_all = 0
    num_ii = 0
    for i in range(num_classes):
        num_ii += c[i, i]
        for j in range(num_classes):
            num_all += c[i, j]
    return num_ii / num_all

def users_accuracy(c):
    # User's accuracy - number of correct that were guessed
    num_jj = np.zeros((num_classes, ))
    num_j = np.zeros((num_classes, ))
    for j in range(num_classes):
        num_jj[j] += c[j, j]
        for i in range(num_classes):
            num_j[i] += c[i, j]
    return num_jj / num_j

def producers_accuracy(c):
    # Producer's accuracy - number of correct that were guessed
    num_ii = np.zeros((num_classes, ))
    num_i = np.zeros((num_classes, ))
    for i in range(num_classes):
        num_ii[i] += c[i, i]
        for j in range(num_classes):
            num_i[j] += c[i, j]
    return num_ii / num_i

def kappa(con):
    row_total = np.zeros((num_classes, ))
    col_total = np.zeros((num_classes, ))
    eye = np.eye(num_classes, num_classes)
    [rows, cols] = con.shape

    total = con.sum()
    total_match = (con * eye).sum()

    for c in range(cols):
        for r in range(rows):
            row_total[r] += con[r, c]
            col_total[c] += con[r, c]

    marg_total = 0.0
    for i in range(num_classes):
        marg_total += row_total[i] * col_total[i]

    result = ((total * total_match) - marg_total)  / ((total ** 2) - marg_total)
    return result

# Sample confusion matrix from Orange classification example
c = np.array([[ 4527.0,   842.0,    97.0],
              [ 455.0,   1865.0,   632.0],
              [  77.0,    483.0,  1445.0]])
print()
print(c)
print("Overall accuracy: " + str(overall_accuracy(c)))
print("User's accuracy - precision - proportion of guessed that were correct: " + str(users_accuracy(c)))
print("Producer's accuracy - recall - proportion of correct that were guessed: " + str(producers_accuracy(c)))
print("Kappa index of agreement - observed versus expected: " + str(kappa(c)))

# Extreme case - total agreement
c = np.array([[ 1000,   0,    0],
              [ 0,   1000,   0],
              [  0,    0,  1000]])
print()
print(c)
print("Overall accuracy: " + str(overall_accuracy(c)))
print("User's accuracy - precision - proportion of guessed that were correct: " + str(users_accuracy(c)))
print("Producer's accuracy - recall - proportion of correct that were guessed: " + str(producers_accuracy(c)))
print("Kappa index of agreement - observed versus expected: " + str(kappa(c)))

# Extreme case - total disagreement
c = np.array([[ 0,   500,    500],
              [ 500,   0,   500],
              [  500,    500,  0]])
print()
print(c)
print("Overall accuracy: " + str(overall_accuracy(c)))
print("User's accuracy - precision - proportion of guessed that were correct: " + str(users_accuracy(c)))
print("Producer's accuracy - recall - proportion of correct that were guessed: " + str(producers_accuracy(c)))
print("Kappa index of agreement - observed versus expected: " + str(kappa(c)))

# Extreme case - unbalanced class distribution
c = np.array([[ 10,   0,    0],
              [ 0,   1,   1],
              [  0,   100,  1000]])
print()
print(c)
print("Overall accuracy: " + str(overall_accuracy(c)))
print("User's accuracy - precision - proportion of guessed that were correct: " + str(users_accuracy(c)))
print("Producer's accuracy - recall - proportion of correct that were guessed: " + str(producers_accuracy(c)))
print("Kappa index of agreement - observed versus expected: " + str(kappa(c)))