import numpy as np 

class Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    # forward pass 
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # backward pass (backpropagation)
    def backward_pass(self, dvalues):
        # parametrelerin graydanları
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # girdilerin gradyanları 
        self.dinputs = np.dot(dvalues, self.weights.T)