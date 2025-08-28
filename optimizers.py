import numpy as np 

class OptimizerSDG():
    def __init__(self, learning_rate=1):
        self.learning_rate = learning_rate
        
    def uptade_params(self, layer):
        layer.weights += -self.learning_rate * layer.dweights
        layer.biases += -self.learning_rate * layer.dbiases 
        
        
        