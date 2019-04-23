# -*- coding: utf-8 -*-
"""
Created on Sun Jun 22 00:22:31 2014

@author: Edward
"""

# Segmenting the picture in regions
import numpy as np
import pylab as pl
from scipy.ndimage.interpolation import zoom
from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering


def SpectralClusterImage(input_image, beta=5, eps=1e-6, n_regions=11, assign_labels='discretize',downsample_factor=np.NaN, order=3):
    """ Spectral Cluster an image
        Inputs:
            input_image: ndarray of image
            beta: Take a decreasing function of the gradient: an exponential
                The smaller beta is, the more independent the segmentation is of 
                the acutal image. For beta=1, the segmentation is close to a 
                voronoi. Default is 5.
            eps: error term. Default is 1E-6
            n_regions: number of regions to decompose into. Default is 11.
            assign_labels: ways of decomposition. Selecting from 'discretize' and 
                'kmeans'. Default is 'discretize'.
            downsample_factor: downsampling before spectral decomposition. Default
                is to keep the original sampling. Enter a single number to apply
                the kernel for both dimensions of the image, or enter as a sequence
                to apply different kernel for each dimension
            order: downsampling method, order of B-spline interpolation
    """
    # Downsample the image
    if not np.isnan(downsample_factor):
        zoom(input_image, zoom=downsample_factor, order=order)
    # Convert the image into a graph with the value of the gradient on the edges
    graph = image.img_to_graph(input_image)
    # Take a decreasing function of the gradient: an exponential
    # The smaller beta is, the more independent the segmentation is of the
    # acutal image. For beta=1, the segmentation is close to a voronoi
    graph.data = np.exp(-beta * graph.data / input_image.std()) + eps 
    # Apply spectral clustering  (this step goes much faster if yuo have pyamg 
    # installed) 
    labels = spectral_clustering(graph, n_clusters=n_regions,
                                 assign_labels='discretize')
    labels = labels.reshape(input_image.shape)
    # Visualizing the resulting regions
    pl.figure(figsize=(5,5))
    pl.imshow(input_image, cmap=pl.cm.gray)
    for lb in range(n_regions):
        pl.contour(labels == lb, contour=1,
                   color=[pl.cm.spectral(lb / float(n_regions)), ])
    # Get rid of x, y tick marks
    pl.xticks(())
    pl.yticks(())



                            