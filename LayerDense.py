import numpy as np

class LayerDense:

    #initalizing the weights and baises
    def __init__(self, n_inputs, n_nuerons):
        self.weights = 0.10 * np.random.randn(n_inputs,n_nuerons)
        self.biases = np.zeros((1, n_nuerons))
    
    def forward(self, inputs):
        #We need to pass the output from the previous nureon to the next nueron (Feed Forward)
        self.output = np.dot(inputs, self.weights) + self.biases