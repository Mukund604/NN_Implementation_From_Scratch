# Neural Network Implementation from Scratch

A small dense neural network written in plain NumPy, with no framework underneath. It covers dense layers, activation functions and cross entropy loss, and classifies 3 class synthetic spiral data.

`NN.py` trains by random search: it nudges the weights and keeps a change only when the loss improves. There's no backpropagation yet, so this is a study of the building blocks and not a production training loop.

## Files
- `LayerDense.py`: Implements dense (fully connected) layers.
- `ActivationReLU.py`: Implements ReLU activation function.
- `ActivationSoftmax.py`: Implements Softmax activation function.
- `LossCategoricalCrossEntropy.py`: Implements categorical cross-entropy loss function.
- `SyntheticData.py`: Generates synthetic data for training.
- `NN.py`: Main script to set up and train the neural network.

## Installation
To run this project, you need Python, NumPy and Matplotlib installed. Install them with:
```bash
pip install numpy matplotlib
```


## Usage
### 1. Generate Synthetic Data:
```python
from SyntheticData import createSyntheticData
X, y = createSyntheticData(samples, classes)
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
from SyntheticData import createSyntheticData
from LayerDense import LayerDense
from ActivationReLU import ActivationReLU

np.random.seed(0)
X, y = createSyntheticData(100, 3)
layer1 = LayerDense(2, 5)
activation1 = ActivationReLU()

layer1.forward(X)
activation1.forward(layer1.output)
print(activation1.output)

```
