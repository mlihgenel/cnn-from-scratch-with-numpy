"""
Tests for layer implementations.
"""

import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.layers import Dense, Dropout, Input
from core.activations import ReLu


def test_dense_layer():
    """Test Dense layer forward and backward pass."""
    # Create a simple dense layer
    dense = Dense(4, 3)
    
    # Test forward pass
    inputs = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
    dense.forward_pass(inputs, training=True)
    
    # Check output shape
    assert dense.output.shape == (2, 3), f"Expected shape (2, 3), got {dense.output.shape}"
    
    # Test backward pass
    dvalues = np.ones_like(dense.output)
    dense.backward_pass(dvalues)
    
    # Check gradient shapes
    assert dense.dweights.shape == dense.weights.shape, "Weight gradients shape mismatch"
    assert dense.dbiases.shape == dense.biases.shape, "Bias gradients shape mismatch"
    assert dense.dinputs.shape == inputs.shape, "Input gradients shape mismatch"


def test_dropout_layer():
    """Test Dropout layer functionality."""
    dropout = Dropout(0.5)
    inputs = np.ones((10, 5))
    
    # Test training mode
    dropout.forward_pass(inputs, training=True)
    assert dropout.output.shape == inputs.shape, "Dropout output shape mismatch"
    
    # Test inference mode
    dropout.forward_pass(inputs, training=False)
    assert np.array_equal(dropout.output, inputs), "Dropout should not modify inputs in inference mode"


def test_relu_activation():
    """Test ReLU activation function."""
    relu = ReLu()
    inputs = np.array([[-2, -1, 0, 1, 2]])
    
    relu.forward_pass(inputs, training=True)
    expected = np.array([[0, 0, 0, 1, 2]])
    
    assert np.array_equal(relu.output, expected), f"ReLU output incorrect: {relu.output}"


if __name__ == "__main__":
    print("Running layer tests...")
    test_dense_layer()
    print("✓ Dense layer test passed")
    
    test_dropout_layer()
    print("✓ Dropout layer test passed")
    
    test_relu_activation()
    print("✓ ReLU activation test passed")
    
    print("All tests passed!")
