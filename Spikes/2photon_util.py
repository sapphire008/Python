# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 19:24:54 2015

@author: Edward
"""
import numpy as np
from MATLAB import *

class Neuron(object):
    """Contains the descriptive geometry of a neuron"""
    def __init__(self, centroid=None, axes=None, index=0):
        self.centroid = centroid
        self.axes = axes
        self.index = index

def spiral(r, z, ts=0.01):
    """Calculate the trajectory of a spiral
    Inputs:
        r: either a scaler or [x, y] to specify the radius of the spiral
        z: elevation of the spiral.
        ts: step size, default 0.01. Smaller value, higher resolution.

    Outputs:
        X, Y, Z: Cartesian coordinates of the spiral
    """



def smartTrjectory(neuron):
    """Given a population of neurons, calculte a smart trajectory coordinates to
    scan these neurons.
    Returned as [t, X, Y, Z]

    usage: t, X, Y, Z = smartTrajectory(neurons)
    where neurons is a list of description of the neurons in 3D.
    """

    # Sort the neurons by decreasing depths zi from the top of the acquisition
    # and compute their radius
