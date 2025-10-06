# Neural Network from Scratch in NumPy
---
A complete implementation of neural networks using only NumPy, designed for educational purposes and understanding the fundamentals of deep learning.

## 🚀 Features

- **Complete Neural Network Implementation**: Built from scratch using only NumPy
- **Multiple Layer Types**: Dense layers, Dropout regularization
- **Various Activation Functions**: ReLU, Softmax, Sigmoid, Linear
- **Multiple Loss Functions**: Categorical Cross-Entropy, Binary Cross-Entropy, Mean Squared Error, Mean Absolute Error
- **Advanced Optimizers**: SGD, Adam, AdaGrad, RMSProp
- **Accuracy Metrics**: Support for both regression and classification tasks
- **Model Persistence**: Save and load trained models

## 📁 Project Structure

```
cnn-from-scratch-numpy/
├── src/                          # Source code
│   ├── core/                     # Core neural network components
│   │   ├── layers.py            # Layer implementations (Dense, Dropout, Input)
│   │   ├── activations.py       # Activation functions (ReLU, Softmax, etc.)
│   │   ├── losses.py            # Loss functions
│   │   └── optimizers.py        # Optimization algorithms
│   ├── utils/                    # Utility modules
│   │   ├── accuracy.py          # Accuracy calculation utilities
│   │   └── dataset.py           # Dataset loading and preprocessing
│   └── model/                    # Model implementation
│       └── model.py             # Main Model class
├── examples/                     # Example scripts
│   └── main.py                  # Fashion-MNIST training example
├── tests/                        # Unit tests
├── data/                         # Dataset files
│   └── fashion_mnist_images/    # Fashion-MNIST dataset
├── models/                       # Saved model files
├── docs/                         # Documentation
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd cnn-from-scratch-numpy
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🎯 Quick Start

Run the Fashion-MNIST example:

```bash
cd examples
python main.py
```

## 📚 Usage

### Basic Model Creation

```python
from src.model import Model
from src.core.layers import Dense, Dropout
from src.core.activations import ReLu, Softmax
from src.core.losses import CategoricalCrossEntropy
from src.core.optimizers import Adam
from src.utils.accuracy import Accuracy_Categorical

# Create model
model = Model()

# Add layers
model.add(Dense(784, 128))
model.add(ReLu())
model.add(Dropout(0.2))
model.add(Dense(128, 10))
model.add(Softmax())

# Compile model
model.compile(
    loss=CategoricalCrossEntropy(),
    optimizer=Adam(learning_rate=0.001),
    accuracy=Accuracy_Categorical()
)

# Finalize and train
model.finalize()
model.train(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
```

### Available Components

#### Layers
- `Dense`: Fully connected layer with optional L1/L2 regularization
- `Dropout`: Regularization layer that randomly sets inputs to zero
- `Input`: Input layer placeholder

#### Activation Functions
- `ReLu`: Rectified Linear Unit activation
- `Softmax`: Softmax activation for multi-class classification
- `Sigmoid`: Sigmoid activation for binary classification
- `Linear`: Linear (identity) activation

#### Loss Functions
- `CategoricalCrossEntropy`: For multi-class classification
- `BinaryCrossEntropy`: For binary classification
- `MeanSquaredError`: For regression tasks
- `MeanAbsoluteError`: For regression tasks

#### Optimizers
- `SGD`: Stochastic Gradient Descent with optional momentum
- `Adam`: Adaptive Moment Estimation
- `AdaGrad`: Adaptive Gradient Algorithm
- `RMSProp`: Root Mean Square Propagation

## 🧪 Testing

Run tests to verify the implementation:

```bash
python -m pytest tests/
```

## 📊 Example Results

The Fashion-MNIST example typically achieves:
- Training accuracy: ~85-90%
- Validation accuracy: ~82-87%
- Training time: ~5-10 minutes (depending on hardware)
---
