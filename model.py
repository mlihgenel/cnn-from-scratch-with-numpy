from activations import Softmax
from layers import Input
from losses import CategoricalCrossEntropy, ActivationSoftmaxCategoricalCrossEntropy

class Model():
    def __init__(self):
        self.layers = []
        
    def forward_pass(self, X, training):
        self.input_layer.forward_pass(X, training)
        
        for layer in self.layers:
            layer.forward_pass(layer.prev.output, training)
        
        return layer.output
    
    def backward_pass(self, output, y):
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward_pass(output, y)
            
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
        
            for layer in reversed(self.layers[:-1]):
                layer.backward_pass(layer.next.dinputs)  
            return 

        self.loss.backward_pass(output, y)
        
        for layer in reversed(self.layers):
            layer.backward_pass(layer.next.dinputs)
            
    def add(self, layer):
        self.layers.append(layer)
        
    def compile(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy
    
    def finalize(self):
        self.input_layer = Input()
        layer_count = len(self.layers)
        
        self.trainable_layers = []
        
        for i in range(layer_count):
            # eğer ilk katman ise önceki katman girdi katmanıdır
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            
            # çıkış katmanı haricindeki katmanlar 
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
                
            # eğer son katman ise sonraki nesne loss'tur
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]
                
            # eğer katmanlar 'weight' değişkeni içeriyorsa
            # eğitirebilir bir katmandır 
            # eğitebilir katmanlara ekleriz sadece weightleri kontrol etmemiz yeterli olacaktır 
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                
            self.loss.remember_trainable_layers(self.trainable_layers)
            
        if isinstance(self.layers[-1], Softmax) and isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = ActivationSoftmaxCategoricalCrossEntropy()
    
                    
    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)
        
        for epoch in range(1, epochs+1):
            output = self.forward_pass(X, training=True)
            data_loss, regularization_loss = self.loss.calculate(output, y, include_regularization=True) 
            loss = data_loss + regularization_loss
            
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)
            
            self.backward_pass(output, y)
            self.optimizer.pre_update_params()
            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()
            
            if not epoch % 100:
                print(f'epoch: {epoch}, ' + 
                f'acc: {accuracy:.3f}, ' + 
                f'loss: {loss:.3f} (' +
                f'data_loss: {data_loss:.3f}, ' + 
                f'reg_loss: {regularization_loss:.3f} ), ' + 
                f'lr: {self.optimizer.current_learning_rate}')
                
        if validation_data is not None:
                X_val, y_val = validation_data
                output = self.forward_pass(X_val, training=False)
                
                loss = self.loss.calculate(output, y_val)
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)
                
                print(f'validation -> | acc: {accuracy:.3f}, loss: {loss:.3f}')
                