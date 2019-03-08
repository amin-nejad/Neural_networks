import numpy as np

from nn_lib import (
    MultiLayerNetwork,
    Trainer,
    Preprocessor,
    save_network,
    load_network,
)
import matplotlib.pyplot as plt

from illustrate import illustrate_results_FM
import sklearn.metrics as skmetrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def main():
    dataset = np.loadtxt("FM_dataset.dat")
    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    # SETUP THE HYPERPARAMETERS
    neurons = []
    activation_functions = [] 
    input_dim = output_dim = 3 # constant outlined in specification
    
    # Modify any of the following hyperparameters     
    num_of_hidden_layers = 3  # Does not count input/output layer
    neurons_per_hidden_layer = 8 # Configures all hidden layers to have the same number of neurons
    activation_hidden = "relu"
    activation_output = "identity"
    loss_function = "mse"
    batch_size = 64
    learning_rate = 0.001
    number_of_epochs = 1000

    # Setup neurons and activations

    for i in range(num_of_hidden_layers):
        neurons.append(neurons_per_hidden_layer)
    neurons.append(output_dim)       # CONSTANT: For the output layer

    for i in range(num_of_hidden_layers):
        activation_functions.append(activation_hidden)
    activation_functions.append(activation_output)       # For the output layer

    # Initiate the neural network
    network = MultiLayerNetwork(input_dim, neurons, activation_functions)

    np.random.shuffle(dataset) # shuffle dataset before splitting into training and test

    # Separate data columns into x (input features) and y (output)
    x = dataset[:, :input_dim]
    y = dataset[:, input_dim:]

    split_idx = int(0.8 * len(x)) # 80% training, 20% test

    # Split data by rows into a training set and a validation set
    train = dataset[:split_idx,:]
    eva = dataset[split_idx:,:]
    y_val = eva[:, input_dim:]

    # Apply preprocessing to the data
    prep = Preprocessor(train)
    x_train_pre = prep.apply(train)[:,:input_dim]
    y_train_pre = prep.apply(train)[:,input_dim:]
    x_val_pre = prep.apply(eva)[:,:input_dim]
    y_val_pre = prep.apply(eva)[:,input_dim:]

    trainer = Trainer(
        network = network,
        batch_size = batch_size,
        nb_epoch = number_of_epochs,
        learning_rate = learning_rate,
        loss_fun = loss_function,
        shuffle_flag = True,
    )

    trainer.train(x_train_pre, y_train_pre)

    y_val_result = network.forward(x_val_pre)

    dataset_result = np.ones(eva.shape)
    dataset_result[:, input_dim:] = y_val_result
    prediction = prep.revert(dataset_result)[:,3:]

    mean_absolute_error = skmetrics.mean_absolute_error(y_val, prediction)
    coefficient_of_dermination = r2_score(y_val, prediction)
    print("R square = ", coefficient_of_dermination)
    print("MAE = ", mean_absolute_error)

    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train_pre))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val_pre))    
    
    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################
    
    illustrate_results_FM(network, prep)


if __name__ == "__main__":
    main()
