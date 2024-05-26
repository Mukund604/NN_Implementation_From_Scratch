# Convolutional Neural Network from Scratch

This repository contains the implementation of a simple neural network from scratch using NumPy. The neural network is designed to classify data points in a spiral pattern. The network consists of a dense layer followed by a ReLU activation function.

## Files

- `DenseLayer.py`: The main script that sets up and runs the neural network.
- `SyntheticData.py`: A module that contains the `spiral_data` function to generate synthetic spiral data for training.

## Installation

To run this project, you need to have Python installed along with NumPy. You can install the required package using pip:

```bash
pip install numpy
```

## Usage

1. **Generate Synthetic Data**: The `spiral_data` function generates synthetic data for training the neural network.

2. **LayerDense Class**: Initializes a dense layer with random weights and biases, and computes the forward pass.

3. **ActivationReLU Class**: Applies the ReLU activation function on the output of the dense layer.

### Example

The following code demonstrates how to generate synthetic data, create a neural network with one dense layer and ReLU activation, and compute the forward pass:

```python
import numpy as np
from SyntheticData import spiral_data

np.random.seed(0)

# Generating synthetic data
X, y = spiral_data(100, 3)

class LayerDense:
    # Initializing the weights and biases
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    
    def forward(self, inputs):
        # Compute the forward pass
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Initializing layers
layer1 = LayerDense(2, 5) # Input layer with 2 features and output layer with 5 neurons
activation1 = ActivationReLU() 

# Passing the data into the first layer
layer1.forward(X)

# Applying ReLU activation function
activation1.forward(layer1.output)

# Print the output of the activation function
print(activation1.output)
```

### Output

The output of the activation function will be printed to the console, showing the result of the forward pass through the dense layer followed by the ReLU activation.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This project is inspired by the concept of building neural networks from scratch to understand the inner workings of machine learning algorithms.
