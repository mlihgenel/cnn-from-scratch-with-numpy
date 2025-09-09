import numpy as np

class ReLu():
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)
    
    def backward_pass(self, dvalues):
        self.dinputs = dvalues.copy()
        
        # değerlerin negatif olduğu yerlerde gradyan 0 olur 
        self.dinputs[self.inputs <= 0] = 0
        
class Softmax():
    def forward_pass(self, inputs):
        self.inputs = inputs
        # normalize edilmemiş olasılıklar
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # olasılıkları normalize et
        probilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probilities
        
    def backward_pass(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        
        # çıktıları ve gradyanları numaralandır
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # çıktıyı düzleştir (flatten)
            single_output = single_output.reshape(-1, 1)
            # çıktının jacobian matrisini hesapla 
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            
class Sigmoid():
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))
        
    def backward_pass(self, dvalues):
        self.dinputs = dvalues * (1 - self.output) * self.output
        
        
class Linear():
    def forward_pass(self, inputs):
        self.inputs = inputs
        self.output = inputs
        
    def backward_pass(self, dvalues):
        self.dinputs = dvalues.copy()