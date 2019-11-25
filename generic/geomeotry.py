# -*- coding: utf-8 -*-
"""
Created on Sun Aug 30 01:41:55 2015

functions for geometry

@author: Edward
"""

import numpy as np

def spm_matrix(P, order='T*R*Z*S'):
    """Python adaptaion of spm_matrix
     returns an affine transformation matrix
     FORMAT [A, T, R, Z, S] = spm_matrix(P, order)
     P(1)  - x translation
     P(2)  - y translation
     P(3)  - z translation
     P(4)  - x rotation about - {pitch} (radians)
     P(5)  - y rotation about - {roll}  (radians)
     P(6)  - z rotation about - {yaw}   (radians)
     P(7)  - x scaling
     P(8)  - y scaling
     P(9)  - z scaling
     P(10) - x affine
     P(11) - y affine
     P(12) - z affine

     order (optional) application order of transformations.

     A     - affine transformation matrix
    ___________________________________________________________________________

     spm_matrix returns a matrix defining an orthogonal linear (translation,
     rotation, scaling or affine) transformation given a vector of
     parameters (P).  By default, the transformations are applied in the
     following order (i.e., the opposite to which they are specified):

     1) shear
     2) scale (zoom)
     3) rotation - yaw, roll & pitch
     4) translation

     This order can be changed by calling spm_matrix with a string as a
     second argument. This string may contain any valid MATLAB expression
     that returns a 4x4 matrix after evaluation. The special characters 'S',
     'Z', 'R', 'T' can be used to reference the transformations 1)-4)
     above. The default order is 'T*R*Z*S', as described above.

     SPM uses a PRE-multiplication format i.e. Y = A*X where X and Y are 4 x n
     matrices of n coordinates.

    __________________________________________________________________________
     Copyright (C) 2008 Wellcome Trust Centre for Neuroimaging

     Karl Friston
     $Id: spm_matrix.m 1149 2008-02-14 14:29:04Z volkmar $
     """
    # pad P with 'null' parameters
    #---------------------------------------------------------------------------
    q  = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    P  = np.concatenate((P,q[(np.size(P) + 1):12]))

    # Transformation matrices
    T = np.array([[1,0,0,P[0]],  # translation
                  [0,1,0,P[1]],
                  [0,0,1,P[2]],
                  [0,0,0,1]])

    R1 = np.array([[1,              0,              0,              0],
                   [0,              np.cos(P[3]),   np.sin(P[3]),   0],
                   [0,             -np.sin(P[3]),   np.cos(P[3]),   0],
                   [0,              0,              0,              1]])

    R2 = np.array([[np.cos(P[4]),   0,              np.sin(P[4]),   0],
                   [0,              1,              0,              0],
                   [-np.sin(P[4]),  0,              np.cos(P[4]),   0],
                   [0,              0,              0,              1]])

    R3 = np.array([[np.cos(P[5]),   np.sin(P[5]),   0,              0],
                   [-np.sin(P[5]),  np.cos(P[5]),   0,              0],
                   [0,              0,              1,              0],
                   [0,              0,              0,              1]])

    R   = R1.dot(R2).dot(R3) # rotation

    Z = np.array([[P[6],    0,      0,      0], # scale
                  [0,       P[7],   0,      0],
                  [0,       0,      P[8],   0,],
                  [0,       0,      0,      1]])

    S = np.array([[1,   P[9],   P[10],  0], # shear
                  [0,   1,      P[11],  0],
                  [0,   0,      1,      0],
                  [0,   0,      0,      1]])

    # order = '('+order.replace('*', ').dot(')+')' # convert to np dot product
    # A = eval(order)
    v = {'T':T, 'R':R, 'Z':Z, 'S':S}
    l = order.split('*')
    A = v[l[0]].dot(v[l[1]]).dot(v[l[2]]).dot(v[l[3]])

def ndim_rotation_matrix(x, y):
    """
    Implemented based on MATLAB code from
    https://math.stackexchange.com/questions/598750/finding-the-rotation-matrix-in-n-dimensions

    x, y are n-dimensional column vectors

    u = x / |x|
    v = y - (u'*y).*u
    v = v / |v|

    cos(theta) = x' * y / (|x| |y|)
    sin(theta) = sqrt(1-cos(theta)^2)

    R = I - u*u' - v*v' + [u, v] R_theta [u, v]'
    """
    u = x / np.linalg.norm(x, ord=2)
    v = y - u.T.dot(y)*(u)
    v = v / np.linalg.norm(v, ord=2)

    cost = float(x.T.dot(y) / (np.linalg.norm(x, ord=2) * np.linalg.norm(y, ord=2)))
    sint = float(np.sqrt(1-cost**2))

    R = np.eye(x.shape[0]) - u.dot(u.T) - v.dot(v.T) + \
        np.c_[u, v].dot(np.array([[cost, -sint], [sint, cost]])).dot(np.c_[u, v].T)

    return R
