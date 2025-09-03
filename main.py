import numpy as np
import nnfs 
from nnfs.datasets import spiral_data
from layers import Dense
from activations import ReLu, Softmax
from losses import CategoricalCrossEntropy, ActivationSoftmaxCategoricalCrossEntropy
from optimizers import SDG, AdaGrad, RMSProb, Adam

nnfs.init()

X, y = spiral_data(samples=100, classes=3)

dense1 = Dense(2, 512, weight_regularizer_l2=5e-4,
                      bias_regularizer_l2=5e-4)
activation1 = ReLu()
dense2 = Dense(512, 3)

loss_activation = ActivationSoftmaxCategoricalCrossEntropy()
# optimizer = RMSProb(learning_rate=0.02, decay=1e-5, rho=0.999)
optimizer = Adam(learning_rate=0.05, decay=5e-7)

for epoch in range(10001):
    
    dense1.forward_pass(X)
    activation1.forward_pass(dense1.output)
    dense2.forward_pass(activation1.output)

    data_loss = loss_activation.forward_pass(dense2.output, y)
    regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
    loss = data_loss + regularization_loss

    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    
    accuracy = np.mean(predictions == y)
    if not epoch % 100:
        print(f'epoch: {epoch}, ' + 
              f'acc: {accuracy:.3f}, ' + 
              f'loss: {loss:.3f} (' +
              f'data_loss: {data_loss:.3f}, ' + 
              f'reg_loss: {regularization_loss:.3f} ), ' + 
              f'lr: {optimizer.current_learning_rate}')
        
    # ===== backpropagation ===== 
    loss_activation.backward_pass(loss_activation.output, y)
    dense2.backward_pass(loss_activation.dinputs)
    activation1.backward_pass(dense2.dinputs)
    dense1.backward_pass(activation1.dinputs)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()
    
    
X_test, y_test = spiral_data(samples=100, classes=3)

dense1.forward_pass(X_test)
activation1.forward_pass(dense1.output)

dense2.forward_pass(activation1.output)

loss = loss_activation.forward_pass(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)

print(f'validation, acc: {accuracy :.3f} , loss: {loss :.3f} ')

