import numpy as np

from src.core.activations import Linear, Softmax
from src.core.layers import Conv, Dense, Dropout, Flatten
from src.core.losses import BinaryCrossEntropy, CategoricalCrossEntropy, MeanAbsoluteError, MeanSquaredError
from src.core.optimizers import SGD
from src.model import Model
from src.utils.accuracy import Accuracy_Categorical, Accuracy_Regression


class TrackingDropout(Dropout):
    def __init__(self, rate):
        super().__init__(rate)
        self.training_flags = []

    def forward_pass(self, inputs, training=True):
        self.training_flags.append(training)
        super().forward_pass(inputs, training)


def test_dense_bias_l1_updates_gradient_not_parameter():
    dense = Dense(3, 2, bias_regularizer_l1=0.5)
    dense.biases = np.array([[1.0, -2.0]])

    dense.forward_pass(np.ones((4, 3)), training=True)
    original_biases = dense.biases.copy()
    dense.backward_pass(np.zeros((4, 2)))

    assert np.array_equal(dense.biases, original_biases)
    assert np.allclose(dense.dbiases, np.array([[0.5, -0.5]]))


def test_flatten_preserves_batch_dimension_for_2d_inputs():
    flatten = Flatten()
    inputs = np.arange(6).reshape(2, 3)
    flatten.forward_pass(inputs, training=True)

    assert flatten.output.shape == (2, 3)
    assert np.array_equal(flatten.output, inputs)


def test_binary_cross_entropy_forward_returns_per_sample_losses():
    loss = BinaryCrossEntropy()
    y_pred = np.array([[0.8, 0.2], [0.3, 0.7], [0.9, 0.1]])
    y_true = np.array([[1, 0], [0, 1], [1, 0]])

    sample_losses = loss.forward_pass(y_pred, y_true)
    assert sample_losses.shape == (3,)

    loss.new_pass()
    total_loss = loss.calculate(y_pred, y_true)
    assert np.isscalar(total_loss)
    assert loss.accumulated_count == 3


def test_mean_absolute_error_exposes_backward_pass():
    loss = MeanAbsoluteError()
    y_pred = np.array([[0.2, 0.8], [0.3, 0.7]])
    y_true = np.array([[0.0, 1.0], [1.0, 0.0]])

    loss.backward_pass(y_pred, y_true)
    assert loss.dinputs.shape == y_pred.shape
    assert np.all(np.isfinite(loss.dinputs))


def test_evaluate_uses_inference_mode():
    model = Model()
    tracking_dropout = TrackingDropout(0.5)

    model.add(Dense(4, 8))
    model.add(tracking_dropout)
    model.add(Dense(8, 3))
    model.add(Softmax())
    model.compile(
        loss=CategoricalCrossEntropy(),
        optimizer=SGD(),
        accuracy=Accuracy_Categorical(),
    )
    model.finalize()

    X = np.random.randn(6, 4)
    y = np.array([0, 1, 2, 0, 1, 2])
    model.evaluate(X, y, batch_size=2)

    assert tracking_dropout.training_flags
    assert all(flag is False for flag in tracking_dropout.training_flags)


def test_finalize_sets_softmax_classifier_output_for_other_loss_types():
    model = Model()
    model.add(Dense(3, 1))
    model.add(Linear())
    model.compile(
        loss=MeanSquaredError(),
        optimizer=SGD(),
        accuracy=Accuracy_Regression(),
    )
    model.finalize()

    assert model.softmax_classifier_output is None


def test_conv_backward_vectorized_matches_reference_and_regularization():
    np.random.seed(42)
    conv = Conv(
        input_shape=(2, 5, 5),
        kernel_size=3,
        filters=3,
        stride=1,
        padding=1,
        weight_regularizer_l1=0.1,
        weight_regularizer_l2=0.05,
        bias_regularizer_l1=0.2,
        bias_regularizer_l2=0.03,
    )

    inputs = np.random.randn(4, 2, 5, 5)
    dvalues = np.random.randn(4, 3, 5, 5)
    conv.forward_pass(inputs, training=True)
    conv.backward_pass(dvalues)

    dOut_col = dvalues.reshape(4, 3, -1)
    W_col = conv.weights.reshape(3, -1)
    X_col = conv.X_col

    # Reference dW using explicit loops
    ref_dW_col = np.zeros_like(W_col)
    for b in range(4):
        ref_dW_col += dOut_col[b] @ X_col[b].T
    ref_dweights = ref_dW_col.reshape(conv.weights.shape)

    dL1_w = np.ones_like(conv.weights)
    dL1_w[conv.weights < 0] = -1
    ref_dweights += 0.1 * dL1_w
    ref_dweights += 2 * 0.05 * conv.weights

    ref_dbiases = np.sum(dvalues, axis=(0, 2, 3))
    dL1_b = np.ones_like(conv.biases)
    dL1_b[conv.biases < 0] = -1
    ref_dbiases += 0.2 * dL1_b
    ref_dbiases += 2 * 0.03 * conv.biases

    # Reference dX using explicit loop
    ref_dX_col = np.zeros_like(X_col)
    WT_col = W_col.T
    for b in range(4):
        ref_dX_col[b] = WT_col @ dOut_col[b]
    ref_dinputs = conv.col2im(
        ref_dX_col,
        inputs.shape,
        conv.kernel_size,
        conv.stride,
        conv.padding,
        conv.out_h,
        conv.out_w,
    )

    assert np.allclose(conv.dweights, ref_dweights)
    assert np.allclose(conv.dbiases, ref_dbiases)
    assert np.allclose(conv.dinputs, ref_dinputs)
