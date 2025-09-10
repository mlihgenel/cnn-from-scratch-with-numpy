import numpy as np 
from activations import Softmax

class Loss():
    
    def regularization_loss(self):
        regularization_loss = 0
        
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))
                
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)
                
            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))
            
            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)
                
            return regularization_loss 
    
    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers
        
    def calculate(self, output, y, *, include_regularization=False):
        sample_loss = self.forward_pass(output, y)
        data_loss = np.mean(sample_loss)
        
        if not include_regularization:
            return data_loss
        
        return data_loss, self.regularization_loss()
    
class CategoricalCrossEntropy(Loss):
    def forward_pass(self, y_pred, y_true):
        samples = len(y_pred)
        
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]
            
        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)
            
        negative_log = -np.log(correct_confidence)
        return negative_log 
    
    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)
        labels = len(dvalues[0])
        
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        # gradyan hesaplama 
        self.dinputs = -y_true / dvalues
        # gradyanı normalize etme
        self.dinputs = self.dinputs / samples 

class BinaryCrossEntropy(Loss):
    def forward_pass(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        
        sample_losses = -((y_true * np.log(y_pred_clipped)) + ((1 - y_true) * np.log(1 - y_pred_clipped)))
        sample_losses = np.mean(sample_losses)
        
        return sample_losses
    
    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        clipped_values = np.clip(dvalues, 1e-7, 1 - 1e-7)
        self.dinputs = -((y_true / clipped_values) - ((1 - y_true) / (1 - clipped_values)))
        
        self.dinputs = self.dinputs / samples 
                
class ActivationSoftmaxCategoricalCrossEntropy():
    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)
        # eğer etiketler one-hot encode ise ayrık değerlere dönüştür
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)
            
        # güvenlik için kopyalama
        self.dinputs = dvalues.copy()
        # gradyan hesapla 
        self.dinputs[range(samples), y_true] -= 1
        # gradyanı normalize et
        self.dinputs = self.dinputs / samples
  
class MeanSquaredError(Loss): #L2
    def forward_pass(self, y_pred, y_true):
        sample_losses = np.mean((y_pred - y_true)**2, axis=1)
        return sample_losses
    
    def backward_pass(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
  
class MeanAbsoluteError(Loss):
    def forward_pass(self, y_pred, y_true):
        sample_loss = np.mean(np.abs(y_pred - y_true), axis=-1)
        return sample_loss
    
    def backward_loss(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])
        
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples 
        
        
