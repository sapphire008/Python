"""Utilities for tensorflow sparse operations."""

import numpy as np
import scipy.sparse

import tensorflow as tf
from tensorflow.python.framework import sparse_tensor


def map_values(op, *args):
    """
    Applies the `op` to the `.values` tensor of one or more `SparseTensor`s.
    For tensorflow versions below 2.4. For versions above, use function
    `tf.sparse.map_values`.
    """
    return sparse_tensor.SparseTensor(args[0].indices, 
                                      op(*[a.values for a in args]),
                                      args[0].dense_shape)

def test_map_values():
    w = scipy.sparse.random(10, 10, density=0.1, format="coo", random_state=42)
    indices = np.c_[w.row, w.col]
    values = w.data
    W = tf.sparse.SparseTensor(indices, tf.cast(values, "float32"), dense_shape=w.shape)
    W = tf.sparse.reorder(W)
    
    W_mapped = tf.map_fn(
        lambda x: map_values(tf.cumsum, x), W)
    
    tf_out = tf.sparse.to_dense(W_mapped)
    dense_out = np.cumsum(w.toarray(), axis=1) * (w.toarray() > 0)
    
    
    assert np.all(np.abs(tf_out - dense_out) < 1E-3)

if __name__ == '__main__':
    test_map_values()
    
