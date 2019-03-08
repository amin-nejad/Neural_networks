import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)

from illustrate import illustrate_results_ROI
from ROI_metrics import *
from ROI_augment_data import *
import matplotlib.pyplot as plt

# Global variables to stores values that will be used for plotting
xValues = []
yValues = [[], [], [], [], []]      # label1(f1), label2(f1), label3(f1), label4(f1), accuracy

# Main function that calls other functions to train and evaluate the neural network
def main(_neurons, _activationFunctionHidden, _activationFunctionOutput, _lossFunction, _batchSize, _learningRate, _numberOfEpochs, _writeToCSV = False):
    dataset = np.loadtxt("ROI_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    # Setup hyperparameters and neural network
    input_dim = 3       # CONSTANT: Stated in specification
    neurons = _neurons
    activations = _activationFunctionHidden
    net = MultiLayerNetwork(input_dim, neurons, activations)

    np.random.shuffle(dataset)

    # Separate data columns into x (input features) and y (output)
    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x))

    # Split data by rows into a training set and a validation set. We then augment the training data into the desired proportions
    # Use this for original dataset (training)
    # x_train = x[:split_idx]       
    # y_train = y[:split_idx]

    # Use this for augmented dataset (training)
    augmentedTrainingData = augment_data_oversample(dataset[:split_idx, :], input_dim, label1=0.25, label2=0.25, label3=0.25, label4=0.25)
    x_train = augmentedTrainingData[:, :input_dim]
    y_train = augmentedTrainingData[:, input_dim:]

    # Validation dataset
    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Apply preprocessing to the data
    prep_input = Preprocessor(x_train)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    trainer = Trainer(
        network=net,
        batch_size=_batchSize,
        nb_epoch=_numberOfEpochs,
        learning_rate=_learningRate,
        loss_fun=_lossFunction,
        shuffle_flag=True,
    )

    # Train the neural network
    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    # Evaluate the neural network
    preds = net(x_val_pre)
    targets = y_val
    accuracy, confusionMatrix, labelDict = evaluate_architecture(targets, preds)

    # Optional: Print results
    print_results(confusionMatrix, labelDict, accuracy)

    # Optional: Append x and y values, to be plotted at the end
    global xValues, yValues
    xValues.append(neurons[0])
    for i in range(len(labelDict)):
        key = "label" + str(i + 1)
        metric = "f1"
        yValues[i].append(labelDict[key][metric])
    yValues[len(yValues) - 1].append(accuracy)

    # Optional: Write results to a csv file
    # if _writeToCSV:
    #     with open('ROI_results.csv','a') as file:
    #         # No. of hidden layers, no. of neurons per hidden layer, activation in hidden layer, activation in output layer, 
    #         # batch size, learning rate, number of epochs, accuracy, confusionMatrix, labelDict
    #         csvList = [len(neurons) - 1, neurons[0], activations[0], _activationFunctionOutput, _batchSize, 
    #             _learningRate, _numberOfEpochs, accuracy, confusionMatrix, labelDict]
    #         csvRow = str(csvList).strip("[]")
    #         csvRow += "\n"
    #         file.write(csvRow)

    # Optional: Save the network
    save_network(net, "trained_ROI")
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    # illustrate_results_ROI(network, prep)

# First create the confusion matrix (predicted x expected)
# Then, evaluate the architecture using accuracy, precision, recall and F1 score
def evaluate_architecture(y_true, y_pred):
    # Generate and populate the confusion matrix
    confusionMatrix = populate_confusion_matrix(y_true, y_pred)

    # Stores data on recall, precision and f1 for each label
    labelDict = dict.fromkeys({"label1", "label2", "label3", "label4"}) 

    # Compute and store the metrics 
    index = 0
    totalErrors = 0
    numOfRows = y_true.shape[0]
    for i in range(len(labelDict)):
        truePositive, falsePositive, falseNegative = calculate_metrics(confusionMatrix, index)
        recall = calculate_recall(truePositive, falseNegative)
        precision = calculate_precision(truePositive, falsePositive)
        f1 = calculate_f1(recall, precision)
        totalErrors += falsePositive

        key = "label" + str(index + 1)
        labelDict[key] = {"recall": recall, "precision": precision, "f1": f1}
        index += 1

    accuracy = calculate_classification_rate(numOfRows, totalErrors)

    # Return metrics
    return accuracy, confusionMatrix, labelDict

# Prints the metrics
def print_results(confusionMatrix, labelDict, accuracy):
    print(confusionMatrix)
    for i in range(len(labelDict)):
        key = "label" + str(i + 1)
        print(key, labelDict[key])
    print("Accuracy: ", accuracy)

# Plot a line graph of y against x
def plot_data(x, y):
    # Define the box area for the main plot
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Set the data we want to be plotted
    for i in range(len(y)):
        labelName = "Label " + str(i + 1) + " (f1)"
        if i == len(y) - 1:
            plt.plot(x, y[i], marker="x", label="Accuracy")
        else:
            plt.plot(x, y[i], marker="x", label=labelName)
 
    # Set axes scales
    plt.ylim(0.0, 1.0)

    # Set label and title names
    xLabel = "Number of neurons per hidden layer (4 hidden layers)"
    yLabel = "F1 + Accuracy"
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.title(yLabel + " vs " + xLabel)
    plt.grid(True)

    # Set legend to be outside of the plot
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()







# Given a set of input data, the function uses the trained neural network to generate a set of predictions
def predict_hidden(inputData):
    filePath = "trained_ROI"
    input_dim = 3

    # Pre-process data
    xData = inputData[:, :input_dim]
    yData = inputData[:, input_dim:]
    prep_input = Preprocessor(xData)
    x_data_pre = prep_input.apply(xData)

    # Load the network
    net = load_network(filePath)

    # Generate the output
    preds = net(x_data_pre)
    oneHotEncoding = one_hot_encode(preds, preds.shape[1])

    # Optional: Evaluate the output
    # accuracy, confusionMatrix, labelDict = evaluate_architecture(yData, oneHotEncoding)
    # print_results(confusionMatrix, labelDict, accuracy)

    return oneHotEncoding


    


if __name__ == "__main__":
    # Setup for the hyperparameters for main()
    neurons = []
    activationFunctions = [] 
    outputDimension = 4

    # Modify any of the following hyperparameters     
    numOfHiddenLayers = 4              # Does not count input/output layer
    numOfNeuronsPerHiddenLayer = 35      # Configures all hidden layers to have the same number of neurons
    activationHidden = "relu"          # Does not apply for input/output layer
    activationOutput = "sigmoid"
    lossFunction = "mse"
    batchSize = 64
    learningRate = 1e-3
    numberOfEpochs = 1500

    # Optional: Write results to csv
    writeToCSV = False

    # Optional: Set number of neurons in hidden layers based on hyperparameters
    # This results in all hidden layers having the same number of neurons (except output layer)
    for i in range(numOfHiddenLayers):
        neurons.append(numOfNeuronsPerHiddenLayer)
    neurons.append(outputDimension)       # CONSTANT: For the output layer

    # Optional: Set activation functions in hidden layers based on hyperparameters
    # This results in all hidden layers having the same activation functions (except output layer)
    for i in range(numOfHiddenLayers):
        activationFunctions.append(activationHidden)
    activationFunctions.append(activationOutput)       # For the output layer

    # Call the main function to train and evaluate the neural network
    main(neurons, activationFunctions, activationOutput, lossFunction, batchSize, learningRate, numberOfEpochs, writeToCSV)

    # Optional: Plot the results in a line graph
    # plot_data(xValues, yValues)