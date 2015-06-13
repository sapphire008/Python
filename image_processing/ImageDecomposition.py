# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 02:06:08 2014

@author: Edward
"""
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def pca_image(H):
    # Principal Component Analysis of image
    # input: H, list of image path
    # return: projection matrix (with important dimensions first),
    # variance and mean
    # adapted from 
    # http://www.janeriksolem.net/2009/01/pca-for-images-using-python.html

    # read images as ndarray, then faltten it
    X = np.array([np.array(mpimg.imread(m)).flatten() for m in enumerate(H)],'f')
    #get dimensions
    num_data,dim = X.shape
    #center data, remove mean
    mean_X = X.mean(axis=0)
    for i in range(num_data):
        X[i] -= mean_X
    # pca algorithm
    if dim>100:
        print 'PCA - compact trick used'
        M = np.dot(X,X.T) #covariance matrix
        ED,EV = np.linalg.eigh(M) #eigenvalues and eigenvectors
        tmp = np.dot(X.T,EV).T #this is the compact trick
        V = tmp[::-1] #reverse since last eigenvectors are the ones we want
        S = np.sqrt(ED)[::-1] #reverse since eigenvalues are in increasing order
    else:
        print 'PCA - SVD used'
        U,S,V = np.linalg.svd(X)
        V = V[:num_data] #only makes sense to return the first num_data
    
    #return the projection matrix, the variance and the mean
    return V,S,mean_X

