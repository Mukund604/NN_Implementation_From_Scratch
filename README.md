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

## Usage
```python
from SyntheticData import spiral_data
X, y = spiral_data(samples, classes)
