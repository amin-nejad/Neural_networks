## This file augments the provided dataset

import numpy as np


# Augments the data into the desired proportion. The size of the new dataset will be the same as the input dataset
def augment_data_balanced(dataset, inputDim, label1=0.25, label2=0.25, label3=0.25, label4=0.25):
    # Calculate the relative proportions based on the input arguments
    label1 /= (label1 + label2 + label3 + label4)
    label2 /= (label1 + label2 + label3 + label4)
    label3 /= (label1 + label2 + label3 + label4)
    label4 /= (label1 + label2 + label3 + label4)
    listOfLabelProportions = [label1, label2, label3, label4]

    # Get the counts of each label in the input dataset and store as a dictionary (key = index, value = count)
    indices = np.argmax(dataset[:, inputDim:], axis=1)
    unique, counts = np.unique(indices, return_counts=True)
    countsDict = dict(zip(unique, counts)) 

    # Segregate the dataset according to the label
    numOfRows = dataset.shape[0]
    numOfColumns = dataset.shape[1]
    labelData = np.empty([0, numOfColumns])
    listOfLabelData = [labelData, labelData, labelData, labelData]      # Index 0 = label1 data, index 1 = label2 data, etc.
    for i in range(numOfRows):
        labelIndex = np.argmax(dataset[i, inputDim:])       # Get the index with the maximum value out of indices 0 to 3
        listOfLabelData[labelIndex] = np.append(listOfLabelData[labelIndex], [dataset[i, :]], axis=0)

    # Augment the dataset
    newDataset = np.empty([0, numOfColumns])
    for i in range(len(listOfLabelData)):
        numOfDataNeeded = int(listOfLabelProportions[i] * numOfRows)
        if numOfDataNeeded <= listOfLabelData[i].shape[0]:
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfDataNeeded, :], axis=0)
        else:
            numOfDuplicationsNeeded = int(numOfDataNeeded / listOfLabelData[i].shape[0])
            numOfRemaindersNeeded = int(numOfDataNeeded % listOfLabelData[i].shape[0])
            for j in range(numOfDuplicationsNeeded):
                newDataset = np.append(newDataset, listOfLabelData[i][:, :], axis=0)
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfRemaindersNeeded, :], axis=0)

    return newDataset   

# Augments the data into the desired proportion. The size of the new dataset will be larger than the input dataset
# Each and every label will be oversampled relative to the size as that of the label with the largest sample size
# No data will be shrunk/discarded
def augment_data_oversample(dataset, inputDim, label1=0.25, label2=0.25, label3=0.25, label4=0.25):
    # Calculate the relative proportions based on the input arguments
    label1 /= (label1 + label2 + label3 + label4)
    label2 /= (label1 + label2 + label3 + label4)
    label3 /= (label1 + label2 + label3 + label4)
    label4 /= (label1 + label2 + label3 + label4)
    listOfLabelProportions = [label1, label2, label3, label4]

    # Get the counts of each label in the input dataset and store as a dictionary (key = index, value = count)
    indices = np.argmax(dataset[:, inputDim:], axis=1)
    unique, counts = np.unique(indices, return_counts=True)
    countsDict = dict(zip(unique, counts)) 

    # Segregate the dataset according to the label
    numOfRows = dataset.shape[0]
    numOfColumns = dataset.shape[1]
    labelData = np.empty([0, numOfColumns])
    listOfLabelData = [labelData, labelData, labelData, labelData]      # Index 0 = label1 data, index 1 = label2 data, etc.
    for i in range(numOfRows):
        labelIndex = np.argmax(dataset[i, inputDim:])       # Get the index with the maximum value out of indices 0 to 3
        listOfLabelData[labelIndex] = np.append(listOfLabelData[labelIndex], [dataset[i, :]], axis=0)

    # Augment the dataset
    minNumberOfDataKey = max(countsDict, key=lambda i: countsDict[i])   # Get the label with the most data
    minNumberOfDataValue = countsDict[minNumberOfDataKey]
    newDataset = np.empty([0, numOfColumns])

    print (np.min(abs(dataset), axis = 0))
    for i in range(len(listOfLabelData)):
        # mean is set to be all zero
        mu = [0,0,0,0,0,0,0]
        # standard deviation is set to be 0.1 for the input variable
        sigma = [0.1,0.1,0.1,0,0,0,0]
        # creating a gaussian noise based on the mean and standard deviation
        noise = np.random.normal(mu, sigma, listOfLabelData[i].shape)*0.05
        # don't want to add noise to label
        noise[:, 3:] = 0
        numOfDataNeeded = int((listOfLabelProportions[i] / listOfLabelProportions[minNumberOfDataKey]) * countsDict[minNumberOfDataKey])
        if numOfDataNeeded <= listOfLabelData[i].shape[0]:
            newDataset = np.append(newDataset, listOfLabelData[i][:numOfDataNeeded, :], axis=0)
        else:
            numOfDuplicationsNeeded = int(numOfDataNeeded / listOfLabelData[i].shape[0])
            numOfRemaindersNeeded = int(numOfDataNeeded % listOfLabelData[i].shape[0])
            for j in range(numOfDuplicationsNeeded):
                add_noisy_dataset = listOfLabelData[i][:, :] + noise
                newDataset = np.append(newDataset, add_noisy_dataset, axis=0)
            # create gaussian noise with different shape.
            noise = np.random.normal(mu, sigma, listOfLabelData[i][:numOfRemaindersNeeded, :].shape)*0.05
            noise[:, 3:] = 0
            add_noisy_dataset = listOfLabelData[i][:numOfRemaindersNeeded, :] + noise
            newDataset = np.append(newDataset, add_noisy_dataset, axis=0)
    return newDataset 