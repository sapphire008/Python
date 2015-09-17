# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 21:07:31 2015

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for plotting 3D
from spm_matrix import spm_matrix

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

    def is_numeric(obj): # to check if a numpy object is numeric
        attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
        return all(hasattr(obj, attr) for attr in attrs)

    if not is_numeric(A) or np.ndim(A)!=2 or any([s!=4 for s in np.shape(A)]):
        raise(IOError(\
        "Order expression '%s' did not return a valid 4x4 matrix."%(order)))

    return(A, T, R, Z, S)

def Ellipsoid(center, radii, rvec=np.eye(3), numgrid=100):
    """Matrix description of ellipsoid
    center: center [x0,y0,z0]
    radii: radii of the ellipsoid [rx, ry, rz]
    rvec: vector of the radii that indicates orientation. Default identity
    numgrid: number of points to estimate the ellipsoid. The higher the
        number, the smoother the plot. Defualt 100.
    return: x, y, z coordinates
    """
    # Spherical coordinate
    u = np.linspace(0.0, 2.0*np.pi, numgrid) # 100 grid resolution
    v = np.linspace(0, np.pi, numgrid-10) #100 grid resolution
    # Convert to Cartesian
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    X = np.rollaxis(np.array([x,y,z]), 0, 3)
    X = X.dot(rvec.T) + center.reshape((1, 1, -1)) # rotation and translation
    return(X[:,:,0], X[:,:,1], X[:,:,2])

def SimulateEllipsoid(P=None):
    """Generate a cloud of points within an ellipsoid given parameter P"""
    if P is None:
        P = np.array([1,2,3,np.pi/3, np.pi/4, np.pi/6,4,5,6,0,0,0])
    A, _, R, _, _ = spm_matrix(P)
    # make up data
    X = np.concatenate((np.random.randn(1000,3), np.ones((1000,1))),axis=1)
    # Transform data
    X = X.dot(A.T)
    # homogenize data
    X = X[:,:-1] / X[:,-1][...,np.newaxis]
    # Reverse to extract parameters
    # mean
    centroid = np.mean(X, axis=0)
    # subtract mean
    Y = X - centroid
    # principle component analysis
    U, S, V = np.linalg.svd(1.0/np.sqrt(Y.shape[0])*Y)
    # retrieve radii
    radii = S # variance = S**2
    # retrieve rotation
    rvec = V
    return(X, centroid, radii, rvec)

def P2M(P):
    """polynomail to matrix form"""
    nd = (np.sqrt(np.size(P)*8+9)-3.0)/2.0
    M = np.eye(1.0 + nd)
    M = M*np.diag(np.concatenate((P[0:nd],np.array([-1.0]))))*0.5
    count = nd
    for ind in xrange(int(nd)):
        M[(ind+1):-1, ind] = P[count:(count+nd-ind-1)]
        count +=nd-ind-1
    M[-1,:-1] = P[-nd:]
    M = M + M.T
    return(M)

def M2P(M):
    """Matrix to polynomial form"""
    P = np.diag(M)[:-1]
    for ind in xrange(np.shape(M)[0]):
        P = np.concatenate((P, M[(ind+1):-1, ind]), axis=1)
    P = np.concatenate((P,M[-1,:-1]), axis=1)
    return(P)

if __name__ == '__main__':
    # Start figure
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    # simulate data
    P = np.array([1,2,3,0, np.pi/4, 0,5,10,15,0,0,0])
    X, _, _, _ = SimulateEllipsoid(P)
    x, y, z = X[:,0], X[:,1], X[:,2]
    ax.scatter(x, y, z, color='k', alpha=0.1)

    # calcualte ellipsoid fit
    _, center, rvec, radii, _ = spm_matrix(P)
    rvec = rvec[:3,:3]
    center = center[:3,-1]
    radii = np.diag(radii)[:3]
    x,y,z = Ellipsoid(center, radii, rvec)
    ax.plot_surface(x,y,z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.7)

    # show principla axes
    for n, XYZ in enumerate(rvec.T):
        x, y, z = zip(center, XYZ * radii[n] + center)
        ax.plot(x, y, z)

    #ax.dist=2
    ax.view_init(elev=0, azim=0) # set perspective
    #ax.dist = 5

    # label axes
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
