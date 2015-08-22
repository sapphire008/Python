"""
Matplotlib adapation of R's beeswarm pacakge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

def beeswarm(values, positions=None, method="swarm",
             ax=None, s=20, col="black", xlim=None, ylim=None,
             labels=None, labelrotation="vertical", **kwargs):
    """
    beeswarm(values, positions=None, method="swarm",
         ax=None, s=20, col="black", xlim=None, ylim=None,
         labels=None, labelrotation="vertical", **kwargs)
     Inputs:
         * values: an array of a sequence of vectors
         * positions: sets the horizontal positions of the swarms.
            Ticks and labels are set to match the positions.
            If none, set positions to range(len(values))
            Default: None
         * method: how to jitter the x,y  coordinates. Choose from
            "swarm", "hex", "center", "square"
            Default: swarm
         * ax: use this axis for plotting. If none supplied, make a new one
            Default: None
         * s: size of points in points^2 (assuming 72 points/inch).
            Defautt: 20
         * col: color of points. Can be:
            - a single string: color all points that color
            - a vector of strings length len(values): gives color for each group
            - a vector of strings length sum([len(values[i]) for i in range(len(values))])
                 gives color for each point
            - a vector of strings any other length: cycle through the list of colors.
                 (really pretty if not useful)
            Default: "black"
         * xlim: tuple giving (xmin, xmax). If not specified, either get
             from the supplied ax or recalculate
         * ylim: tuple giving (ymin, ymax). If not specified, eiterh get
             from the supplied as or recalculate
         * labels: list of labels for each group.
             Default: range(len(values))
         * labelrotation: rotation of x label.
             Default: "vertical"
     Returns:
         * bs: pandas.DataFrame with columns: xorig, yorig, xnew, ynew, color
         * ax: the axis used for plotting
    """
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    if positions is None:
        positions = np.arange(len(values))

def _calculateSwarm(x, dsize, gsize, side=0L, priority="ascending"):
    """do swarm layout"""
    # parse some inputs
    if x is None:
        return
    if side not in [-1,0,1]:
        raise(Exception('"side" must be -1,0,or 1'))
        
    # make sure things are operating in float
    x, dsize, gsize = np.float32(x), float(dsize), float(gsize)
    # take out missing values
    x = x[~np.isnan(x)]

    # initialize data frame
    out = pd.DataFrame({'x':x/dsize, 'y':0}, index=np.arange(len(x)))

    # Determine the order in which points will be placed
    out = {
    "ascending": out.sort('x', ascending=True, axis=0),
    "descending": out.sort('x', ascending=False, axis=0),
    "none": out, # do not reorder
    "random": out.reindex(np.random.permutation(out.index)), # randomly gitter
    "density": out.reindex(np.argsort(-KernelDensity(kernel='gaussian', bandwidth=0.2).fit(out['x']).score_samples(out['x']))),  # arrange outward from densest areas
    }.get(priority)
    
    # place the points: we will place once point at a time
    if len(out.index)>1:
        for ii in range(1, len(out.index)):
            xi = out.loc[ii,'x']
            # identify previously-placed points with potential to overlap the current point
            isPotOverlap = (abs(xi - out['x'])<1) & (np.arange(len(out.index))<ii)
            if any(isPotOverlap):
                global pre_x, pre_y, poty_off
                pre_x = out.loc[isPotOverlap,'x']
                pre_y = out.loc[isPotOverlap,'y']
                poty_off = np.sqrt(1-((xi - pre_x)**2)) # potential y offsets
                poty = {
                -1: np.concatenate(([0.], pre_y - poty_off)),
                0: np.concatenate(([0.], pre_y + poty_off, pre_y - poty_off)),
                1: np.concatenate(([0.], pre_y + poty_off))
                }.get(side)

                def checkoverlap(y): # check for overlaps afterward
                    return(any(((xi - pre_x) ** 2 + (y - pre_y) ** 2) < 0.999))
                poty_bad = np.array([checkoverlap(y) for y in poty])
                poty[poty_bad] = np.inf
                out.loc[ii,'y'] = poty[np.argmin(abs(poty))]
            else:
                out.loc[ii,'y'] = 0
    out.loc[np.isnan(out['x']),'y'] = np.nan  # missing x values should have missing y values
    return(out.sort_index()['y'] * gsize)



def swarmx(x,y, xsize=0.08, ysize=0.08, log=None, side=0L, priority="ascending"):
    """jitter points horizontally
    * priority: select from "ascending" (default) ,"descending","density","random","none"

    """
    return


def swarmy(x,y,xsize=0.08,ysize=0.08, priority="ascending"):
    """jitter points vertically"""
    return
