import numpy as np

class ActivationSoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probability = exp_values / np.sum(exp_values, axis=1, keepdims=True) #Axis 1 becuase we need to sum each row, and axis 0 if we want to sum the each column.
        self.output = probability