import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import Model 
from src.core.layers import Dense, Dropout
from src.core.activations import ReLu, Linear, Sigmoid, Softmax
from src.core.losses import CategoricalCrossEntropy, MeanSquaredError as MSE
from src.core.losses import BinaryCrossEntropy
from src.utils.accuracy import Accuracy_Regression, Accuracy_Categorical
from src.core.optimizers import Adam
from src.utils.dataset import load_minst_dataset, create_data_mnist
import numpy as np
import cv2

np.set_printoptions(linewidth=200)

# Data preproccessing 
X, y, X_test, y_test = create_data_mnist('../data/fashion_mnist_images')

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X = X.reshape(X.shape[0], -1).astype(np.float32) / 255
X_test = X_test.reshape(X_test.shape[0], -1).astype(np.float32) / 255

# Create Model 
model = Model()

model.add(Dense(X.shape[1], 128))
model.add(ReLu())
model.add(Dense(128, 128))
model.add(ReLu())
model.add(Dense(128, 10))
model.add(Softmax())

model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(decay=1e-3),
    accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=10, batch_size=128, print_every=100)

model.evaluate(X_test, y_test)

model.save('../models/fm.model')
