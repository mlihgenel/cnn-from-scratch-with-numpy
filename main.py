import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
from layers import Dense
from activations import ReLu, Softmax
from losses import CategoricalCrossEntropy, ActivationSoftmaxCategoricalCrossEntropy
from optimizers import OptimizerSDG

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 64)
activation1 = ReLu()
dense2 = Dense(64, 3)

loss_activation = ActivationSoftmaxCategoricalCrossEntropy()
optimizer = OptimizerSDG()

for epoch in range(10001):
    
    dense1.forward_pass(X)
    activation1.forward_pass(dense1.output)
    dense2.forward_pass(activation1.output)

    loss = loss_activation.forward_pass(dense2.output, y)

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}')
        
    # ===== backpropagation ===== 
    loss_activation.backward_pass(loss_activation.output, y)
    dense2.backward_pass(loss_activation.dinputs)
    activation1.backward_pass(dense2.dinputs)
    dense1.backward_pass(activation1.dinputs)

    optimizer.uptade_params(dense1)
    optimizer.uptade_params(dense2)



