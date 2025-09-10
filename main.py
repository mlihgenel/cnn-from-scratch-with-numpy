from model import Model 
from layers import Dense, Dropout
from activations import ReLu, Linear, Sigmoid, Softmax
from losses import CategoricalCrossEntropy, MeanSquaredError as MSE
from losses import BinaryCrossEntropy
from accuracy import Accuracy_Regression, Accuracy_Categorical
from optimizers import Adam

import numpy as np

from nnfs.datasets import spiral_data

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Dense(2, 64, weight_regularizer_l2=5e-4,
                       bias_regularizer_l2=5e-4))

model.add(ReLu())
model.add(Dropout(0.1))
model.add(Dense(64, 3))
model.add(Softmax())

model.compile(loss=CategoricalCrossEntropy(),
          optimizer=Adam(learning_rate=0.05, decay=5e-5),
          accuracy=Accuracy_Categorical())

model.finalize()

model.train(X, y,
            validation_data=(X_test, y_test),
            epochs=10000,
            print_every=100)
