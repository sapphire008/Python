# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 21:07:31 2015

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.append("C:/Users/Edward/Documents/Assignments/Scripts/Python/")
from generic.spm_matrix import spm_matrix

import time


def rotation(theta):
    tx,ty,tz = theta
    Rx = np.array([[1,0,0], [0, np.cos(tx), -np.sin(tx)], [0, np.sin(tx), np.cos(tx)]])
    Ry = np.array([[np.cos(ty), 0, -np.sin(ty)], [0, 1, 0], [np.sin(ty), 0, np.cos(ty)]])
    Rz = np.array([[np.cos(tz), -np.sin(tz), 0], [np.sin(tz), np.cos(tz), 0], [0,0,1]])
    return np.dot(Rx, np.dot(Ry, Rz))
    

def Ellipsoid(center, radii, rvec=np.eye(3), A=None, numgrid=100):
    """Matrix description of ellipsoid
    center: center [x0,y0,z0]
    radii: radii of the ellipsoid [rx, ry, rz]
    rvec: vector of the radii that indicates orientation. Default identity
    A: orientation matrix 3x3, where each row is a vector of radii, can be
        supplied directly instead of radii and rvec
    numgrid: number of points to estimate the ellipsoid. The higher the 
        number, the smoother the plot. Defualt 100.
    return: x, y, z coordinates
    """
    # Ellipsoid (x-c)'A(x-c) = 1
    # find the rotation matrix and radii of the axes
    # A = eigvec * diag(eigvalue) * inv(eigvec)
    if A is None:
        A = rvec.dot(np.diag(1.0/radii**2)).dot(np.linalg.inv(rvec))
    # Note that this step is necessary to sort out x, y, z
    U, S, V = np.linalg.svd(A) # A = USV', V is returned as V'
    radii = 1.0/np.sqrt(S)
    # Spherical coordinate
    u = np.linspace(0.0, 2.0*np.pi, numgrid) # 100 grid resolution
    v = np.linspace(0, np.pi, numgrid-10) #100 grid resolution
    # Convert to Cartesian
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    X = np.rollaxis(np.array([x,y,z]), 0, 3)
    X = X.dot(V.T) + center.reshape((1, 1, -1)) # rotation and translation
    return(X[:,:,0], X[:,:,1], X[:,:,2])

def SimulateEllipsoid(P=None):        
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
    X, center, radii, rvec = SimulateEllipsoid()
    x, y, z = X[:,0], X[:,1], X[:,2]
    ax.scatter(x, y, z, color='k', alpha=0.1)
    
    # calcualte ellipsoid fit
    x,y,z = Ellipsoid(center, radii, rvec)
    ax.plot_surface(x,y,z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.7)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
