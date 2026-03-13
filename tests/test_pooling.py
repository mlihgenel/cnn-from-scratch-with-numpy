import numpy as np

from src.core.layers import MaxPool2D


def test_maxpool2d_forward():
    pool = MaxPool2D(pool_size=2, stride=2)
    inputs = np.array(
        [[[[1, 3, 2, 4],
           [5, 6, 7, 8],
           [9, 10, 11, 12],
           [13, 14, 15, 16]]]],
        dtype=np.float32
    )

    pool.forward_pass(inputs, training=True)
    expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)

    assert pool.output.shape == (1, 1, 2, 2)
    assert np.array_equal(pool.output, expected)


def test_maxpool2d_backward_with_overlap_accumulates_gradients():
    pool = MaxPool2D(pool_size=2, stride=1)
    inputs = np.array(
        [[[[1, 2, 3],
           [4, 9, 6],
           [7, 8, 5]]]],
        dtype=np.float32
    )
    pool.forward_pass(inputs, training=True)

    dvalues = np.ones((1, 1, 2, 2), dtype=np.float32)
    pool.backward_pass(dvalues)

    expected_dinputs = np.zeros_like(inputs)
    expected_dinputs[0, 0, 1, 1] = 4.0

    assert np.array_equal(pool.dinputs, expected_dinputs)
