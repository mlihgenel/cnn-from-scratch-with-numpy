import numpy as np 
from scipy import signal

class Input():
    def forward_pass(self, inputs, training):
        self.output = inputs
        
class Dense():
    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
     
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases 
        
    # forward pass 
    def forward_pass(self, inputs, training):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        
    # backward pass (backpropagation)
    def backward_pass(self, dvalues):
        # parametrelerin graydanları
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # regularizasyonların gradyanları 
        # Ağırlıkların L1 regularizasyonu 
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1 
            self.dweights += self.weight_regularizer_l1 * dL1
        # Ağırlıkların L2 regularizasyonu 
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights
        
        # Biasların L1 regularizasyonu 
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.biases += self.bias_regularizer_l1 * dL1 
        # Biasların L2 regularizasyonu 
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases
               
        # girdilerin gradyanları 
        self.dinputs = np.dot(dvalues, self.weights.T)
        
class Dropout():
    def __init__(self, rate):
        self.rate = 1 - rate 
    
    def forward_pass(self, inputs, training=True):
        self.inputs = inputs
        if not training:
            self.output = inputs.copy()
            return 
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate
        self.output = inputs * self.binary_mask
        
    def backward_pass(self, dvalues):
        self.dinputs = dvalues * self.binary_mask
        
class Conv():
    def __init__(self, input_shape, kernel_size, filters, stride=1, padding=0):
        input_depth, input_height, input_width = input_shape 
        self.filters = filters
        self.input_shape = input_shape
        self.kernel_size = kernel_size
        self.input_depth = input_depth
        self.stride = stride
        self.padding = padding 
        
        self.output_height = int((input_height - kernel_size + (2 * padding)) / stride) + 1 
        self.output_width = int((input_width - kernel_size + (2 * padding)) / stride) + 1
        
        self.output_shape = (filters, self.output_height, self.output_width)
        
        self.kernels_shape = (filters, input_depth, kernel_size, kernel_size)
        self.weights = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(filters)
        
    def forward_pass(self, inputs, training=True):
        self.inputs = inputs
        
        if self.padding > 0:
            self.input_padded = np.pad(
                input, 
                ((0,0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            self.input_padded = inputs
            
        self.output = np.zeros(self.output_shape)
        
        for f in range(self.filters):
            for c in range(self.input_depth):
                # forward as stride steps
                for y in range(0, self.output_height):
                    for x in range(0, self.output_width):
                        y_start = y * self.stride
                        x_start = x * self.stride 
                        region = self.input_padded[c, y_start:y_start + self.kernel_size, x_start:x_start + self.kernel_size]
                        self.output[f, y, x] += np.sum(region * self.weights[f, c])
            
            self.output[f] += self.biases[f]
        
        return self.output 
    
    def backward_pass(self, dvalues):
        self.dweights = np.zeros(self.kernels_shape)
        self.dinputs = np.zeros(self.input_shape)
        
        for f in range(self.filters):
            for c in range(self.input_depth):
                self.dweights[f, c] = signal.correlate2d(self.inputs[c], dvalues[f], mode='valid')
                self.dinputs[c] += signal.convolve2d(dvalues[f], np.flip(self.weights[f, c]), mode='full')
        
        if self.padding > 0:
            self.dinputs = self.dinputs[:, self.padding:-self.padding, self.padding:-self.padding]
            
        self.dbiases = np.sum(dvalues, axis=(1, 2))
        
    def get_parameters(self):
        return self.weights, self.biases
    
    def set_parameters(self, weights, biases):
        self.weights = weights
        self.biases = biases 
        

                
