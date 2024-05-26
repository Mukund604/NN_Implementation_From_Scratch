import numpy as np
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU
from ActivationSoftmax import ActivationSoftmax
from LossCategoricalCrossEntropy import LossCategoricalCrossEntropy
from SyntheticData import createSyntheticData

np.random.seed(0)

#Generating synthetic data
X, y = createSyntheticData(100, 3)

#Creating 1st Layer
Dense1 = LayerDense(2, 3)
Activation1 = ActivationReLU()

#Creating 2nd Layer
Dense2 = LayerDense(3, 3)
Activation2 = ActivationSoftmax()

#Initializing the loss function
lossFunction = LossCategoricalCrossEntropy()

lowest_loss = 9999999  # some initial value
best_dense1_weights = Dense1.weights.copy()
best_dense1_biases = Dense1.biases.copy()
best_dense2_weights = Dense2.weights.copy()
best_dense2_biases = Dense2.biases.copy()


for iteration in range(10000):

    # Update weights with some small random values
    Dense1.weights += 0.05 * np.random.randn(2, 3)
    Dense1.biases += 0.05 * np.random.randn(1, 3)
    Dense2.weights += 0.05 * np.random.randn(3, 3)
    Dense2.biases += 0.05 * np.random.randn(1, 3)

    # Perform a forward pass of our training data through this layer
    Dense1.forward(X)
    Activation1.forward(Dense1.output)
    Dense2.forward(Activation1.output)
    Activation2.forward(Dense2.output)

    # Perform a forward pass through activation function
    # it takes the output of second dense layer here and returns loss
    loss = lossFunction.calculate(Activation2.output, y)


    # Calculate accuracy from output of Activation2 and targets
    # calculate values along first axis
    predictions = np.argmax(Activation2.output, axis=1)
    accuracy = np.mean(predictions==y)

    # If loss is smaller - print and save weights and biases aside
    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = Dense1.weights.copy()
        best_dense1_biases = Dense1.biases.copy()
        best_dense2_weights = Dense2.weights.copy()
        best_dense2_biases = Dense2.biases.copy()
        lowest_loss = loss
    # Revert weights and biases
    else:
        Dense1.weights = best_dense1_weights.copy()
        Dense1.biases = best_dense1_biases.copy()
        Dense2.weights = best_dense2_weights.copy()
        Dense2.biases = best_dense2_biases.copy()



#Adding the Categorical cross entropy to calculate the loss 


