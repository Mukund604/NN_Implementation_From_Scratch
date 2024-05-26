# Neural Network Implementation from Scratch

This project demonstrates how to build a Convolutional Neural Network (CNN) from scratch using NumPy. It covers essential components such as dense layers, activation functions, and loss calculations.

## Files
- `LayerDense.py`: Implements dense (fully connected) layers.
- `ActivationReLU.py`: Implements ReLU activation function.
- `ActivationSoftmax.py`: Implements Softmax activation function.
- `LossCategoricalCrossEntropy.py`: Implements categorical cross-entropy loss function.
- `SyntheticData.py`: Generates synthetic data for training.
- `NN.py`: Main script to set up and train the neural network.

## Installation
To run this project, you need Python and NumPy installed. Install the required package with:
```bash
pip install numpy
```


## Usage
### 1. Generate Synthetic Data:
```python
from SyntheticData import spiral_data
X, y = spiral_data(samples, classes)
```
### 2. Initialize Layers:
```python
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU

layer1 = LayerDense(n_inputs, n_neurons)
activation1 = ActivationReLU()
```
### 3. Forward Pass:
```python
layer1.forward(X)
activation1.forward(layer1.output)

```
## Example: 
### Here's a simple example to demonstrate the usage:
```python
import numpy as np
from SyntheticData import spiral_data
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU

np.random.seed(0)
X, y = spiral_data(100, 3)
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

```
