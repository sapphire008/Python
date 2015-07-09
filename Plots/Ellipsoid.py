# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 21:07:31 2015

@author: Edward
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    X = X.dot(V) + center.reshape((1, 1, -1))
    return(X[:,:,0], X[:,:,1], X[:,:,2])

# Ellipsoid (x-v)'A(x-v) = 1
#A = np.array([[1,0,0], [0,2,0], [0,0,2]])
#c = np.array([0,0,0])
#
## find the rotation matrix and radii of the axes
#U, S, V = np.linalg.svd(A) # USV'
#radii = 1.0/np.sqrt(S)
#
## Polar coordinate
#u = np.linspace(0.0, 2.0*np.pi, 100)
#v = np.linspace(0, np.pi, 100)
## Convert to Cartesian
#x = radii[0] * np.outer(np.cos(u), np.sin(v))
#y = radii[1] * np.outer(np.sin(u), np.sin(v))
#z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
#
#for i in range(len(x)):
#    for j in range(len(x)):
#        x[i,j], y[i,j], z[i,j] = np.dot([x[i,j], y[i,j], z[i,j]], V) + c

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
#        
    X = np.random.randn(1000,3)  * np.array([1,2,3]) + np.array([[6,7,8]])
    X = np.dot(X, rotation((0,0,np.pi/4)).T)
    x, y, z = X[:,0][...,np.newaxis], X[:,1][...,np.newaxis], X[:,2][...,np.newaxis]
    # a,b,c = Y[:,0][...,np.newaxis], Y[:,1][...,np.newaxis], Y[:,2][...,np.newaxis]

    #ax.scatter(x,y,z,s=20, c=u'k', depthshade=True)
#    ax = fig.add_subplot(212,projection='3d')
#    ax.scatter(a,b,c,s=20, c=u'k', depthshade=True)

    # fit with ellipse
    #c, radii, evecs, v, M = FitEllipsoid(X, flag=0)
    #A = np.array([[1,0,0],[0,0.25,0],[0,0,0.25]]) # shape matrix
    radii = np.array([1,4,9])
    center = np.array([6,7,8])
    rvec = np.array([[2,-2,1],[1,2,2],[2,1,-2]])
    
    x,y,z = Ellipsoid(center, radii, rvec)
    
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.plot_surface(x,y,z, rstride=4, cstride=4, color='b', linewidth=0, alpha=0.2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
