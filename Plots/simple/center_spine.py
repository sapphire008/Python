# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 02:49:09 2016

@author: Edward
"""
import matplotlib.pyplot as plt
import numpy as np



def center_spines(ax=None, centerx=0, centery=0, endarrow=False):
    """Centers the axis spines at <centerx, centery> on the axis "ax", and
    places arrows at the end of the axis spines."""
    if ax is None:
        ax = plt.gca()

    # Set the axis's spines to be centered at the given point
    # (Setting all 4 spines so that the tick marks go in both directions)
    ax.spines['left'].set_position(('data', centerx))
    ax.spines['bottom'].set_position(('data', centery))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.spines['left'].set_capstyle('butt')
    ax.spines['bottom'].set_capstyle('butt')



if __name__ == '__main__':
    x = np.arange(-5, 5)
    y = x
    
    line, = plt.plot(x, y)
    center_spines()
    plt.axis('equal')
    plt.show()