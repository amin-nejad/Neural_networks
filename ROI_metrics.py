## This file calculates metrics from a confusion matrix

import numpy as np


# Populates the confusion matrix (predicted x expected) based on y_true and y_pred
def populate_confusion_matrix(y_true, y_pred):
    # Create an empty confusion matrix filled with 0s
    numOfRows = y_pred.shape[1]
    numOfColumns = y_true.shape[1]
    confusionMatrix = create_confusion_matrix(numOfRows, numOfColumns)
    
    # Start populating the confusion matrix
    row = extract_indices(y_pred)
    col = extract_indices(y_true)
    for i in range(y_true.shape[0]):
        confusionMatrix[row[i], col[i]] += 1

    return confusionMatrix

# Create confusion matrix (predicted x expected) with all 0s
def create_confusion_matrix(numOfRows, numOfColumns):
    matrix = np.zeros(shape = (numOfRows, numOfColumns))

    return matrix

# Given a matrix of values, returns as an array the indices of the maximum value for each row
# E.g. [0, 1, 0, 0] returns 1
def extract_indices(data):
    indices = np.argmax(data, axis=1)

    return indices

# Performs one-hot encoding on the output
def one_hot_encode(output, numOfPossibleLabels):
    indices = extract_indices(output)
    encodedOutput = np.empty([0, numOfPossibleLabels])

    for i in range(indices.shape[0]):
        newRow = np.zeros((1, numOfPossibleLabels))
        newRow[:, indices[i]] = 1
        encodedOutput = np.append(encodedOutput, newRow, axis=0)

    return encodedOutput    
    
# Metric: true positive, false positive, false negative
def calculate_metrics(matrix, index):
    numOfRows = matrix.shape[0]
    truePositive = matrix[index, index]
    falsePositive = 0
    falseNegative = 0

    # Calculate false positive and false negative
    for currentIndex in range(numOfRows):
        if currentIndex == index:
            continue
        falsePositive += matrix[index, currentIndex]
        falseNegative += matrix[currentIndex, index]    

    return truePositive, falsePositive, falseNegative

# Metric: recall = true pos / (true pos + false neg)
def calculate_recall(truePositive, falseNegative):
    if truePositive + falseNegative == 0:
        return 0
    return truePositive / (truePositive + falseNegative)

# Metric: precision = true pos / (true pos + false pos)
def calculate_precision(truePositive, falsePositive):
    if truePositive + falsePositive == 0:
        return 0
    return truePositive / (truePositive + falsePositive)

# Metric: F1 = 2 * (prec * rec) / (prec + rec)
def calculate_f1(precision, recall): 
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)

# Metric: classification rate = 1 - classification error
def calculate_classification_rate(numOfRows, totalErrors):
    if numOfRows == 0:
        return 0
    return (numOfRows - totalErrors) / numOfRows