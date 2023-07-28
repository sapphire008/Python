#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:26:24 2023

@author: edwardcui
"""

import numpy as np
import scipy.sparse

def set_operation_counts(x, y, operation="intersection", pad=None):
    """Batch-wise count after set operations"""
    # Input shapes
    n_d, m_x = x.shape
    n_d, m_y = y.shape

    # Use np.unique to create convert from data -> indices
    # This can appropriately handle all data types, including strings
    unique, indices = np.unique(np.hstack((x, y)), return_inverse=True)
    n_unique = len(unique)
    if pad is not None:
        pad_index = np.where(unique == pad)[0][0]
    else:
        pad_index = -1

    # From flattened index -> original shape
    indices = indices.reshape(n_d, -1)
    indices_x = indices[:, :m_x]
    indices_y = indices[:, m_x:]
    
    # Use coo format to create to create binary indicator matrices
    # e.g. index = [1, 3], n_unique = 5 -> [0, 1, 0, 1, 0]
    def _create_coo_matrix_from_dense_idx(idx, m):
        r = np.repeat(np.arange(n_d), m) # row index
        c = idx.ravel() # flatten
        data = np.ones_like(r, dtype=int)
        data[c==pad_index] = 0 # filter out pad index
        return scipy.sparse.coo_matrix((data, (r, c)), shape=(n_d, n_unique), dtype=int)
    
    def _create_coo_matrix_from_sparse_idx(idx, m):
        idx = idx.tocoo() # convert to coo format from other formats
        r = idx.row
        c = idx.data
        data = np.ones_like(r, dtype=int)
        data[c==pad_index] = 0 # filter out pad index
        return scipy.sparse.coo_mamtrix((data, (r, c)), shape=(n_d, n_unique), dtype=int)
    
    x_hat = _create_coo_matrix_from_dense_idx(indices_x, m_x)
    y_hat = _create_coo_matrix_from_dense_idx(indices_y, m_y)
    
    # set operations in binary
    if operation == "intersection":
        res = x_hat.multiply(y_hat)
    elif operation == "union":
        res = x_hat + y_hat
        res.data = np.minimum(res.data, 1)
    elif operation == "difference":
        res = x_hat - y_hat
        res.data = np.maximum(res.data, 0)

    return res.sum(axis=1).A.ravel()


x = np.array([[1,2,3,4,5], [2,3,4,5,6], [3, 5, 1, 0, 0]]*1024)
y = np.array([[5,6,7,8,9], [2,3,5,7,8], [3, 1, 0, 0, 0]]*1024)
    
%timeit res_sparse = set_operation_counts(x, y, operation="intersection", pad=0)

# %% normal for loop
import numpy as np

def batch_wise_intersection(x, y, pad=None):
    batch_size = x.shape[0]
    res = [[]] * batch_size
    for ii in range(batch_size):
        res[ii] = set(x[ii, :]).intersection(set(y[ii, :]))
        if pad is not None:
            res[ii] = res[ii].difference(set([pad]))
        
    return res

x = np.array([[1,2,3,4,5], [2,3,4,5,6], [3, 5, 1, 0, 0]]*1024)
y = np.array([[5,6,7,8,9], [2,3,5,7,8], [3, 1, 0, 0, 0]]*1024)

%timeit res = batch_wise_intersection(x, y, pad=0)


# %% tensorflow
import tensorflow as tf

%timeit res_tf = tf.sets.size(tf.sets.intersection(x, y))

#%% indicator

x = np.array([["a", "b", "c", "d"], ["b","c", "d", "e"]])
y = np.array([["b", "a", "c", "e"], ["b", "a", "", ""]])
n_x, n_d = x.shape
n_y, n_d = y.shape
values, ix = np.unique(np.vstack((x,y)), return_inverse=True)
n_unique = len(values)

# reshape the inverse array. ix_x_hat will be an array the same size
# as x, where ix_x_hat[i,j] gives the index of x[i,j] in values. That
# is to say, values[ix_x_hat[i,j]] == x[i,j]
ix_hat = ix.reshape(-1, x.shape[1])
ix_x_hat = ix_hat[:n_x]
ix_y_hat = ix_hat[n_x:]



