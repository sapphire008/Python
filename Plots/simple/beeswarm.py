"""
Matplotlib adapation of R's beeswarm pacakge
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity

from pdb import set_trace

def beeswarm(df, values, group=None, cluster=None, positions=None,
            method='swarm', corral='none', corralWidth=None,
            side=int(0), priority='ascending', ax=None, orientation='vertical',
            xlim=None, ylim=None, xlab=None, ylab=None, legendon=True,
            legend=None, legendtitle=None, labels=None, labelson=True,
            ticklabelrotation='horizontal', log=False, s=33., dpi=72.,
            figsize=(10.,5.), color=('k','r'), colortheme='cluster',
            reset_index=False, 
            **kwargs):
    """
     Helper functions:
         * connect_paired_dots
         * add_average_bar
         
     Inputs:
         * df: data frame
         * values: column name of the data to be plotted
         * group: column name of the category vector
         * cluster: column name of the vector that further divides each group
         * positions: sets the horizontal positions of the swarms.
            Ticks and labels are set to match the positions.
            If none, set positions to range(len(values))
            Default: None
         * method: how to jitter the x,y  coordinates. Choose from
            "swarm", "hex", "center", "square"
            Default: swarm
         * corral: Method to adjust points that would be placed outside their own group region.
            Choose from 'none', 'gutter', 'wrap','random', 'omit'
         * corralWidth: Width of the "corral" in user coordinates. If missing, a sensible value will be chosen.
         * side: Direction to perform jittering
            0: both directions; 1: to the right or upwards; -1: to the left or downwards.
         * priority: Order used to perform point layout when method is "swarm"; ignored otherwise.
         * ax: use this axis for plotting. If none supplied, make a new one
            Default: None.
         * orientation: direction of swarm plot. Defualt 'vertical'
         * xlim, ylim: specify x,y axes limits; specify each as a tuple
         * xlab, ylab: x, y axes labels
         * legendon: turn on legend if cluster is not None. Default True.
         * legend: a list of names for legend
         * legendtitle: title of legend
         * labels: tick label of categorical axis
         * labelson: turn on or off xlabel (Default True)
         * labelrotation: rotation of x label.
             Default: 'horizontal'
         * log: specify a function to apply log transformation of data, e.g. np.log10
             Default: False.
         * s: size of points in points^2 (assuming 72 points/inch).
            Default: 33 (corresponding roughly to 0.08 inch)
         * dpi: dots per point. Default 72.
         * figsize: figure size (width, height). Default (10.0,5.0). 
             Need to specify before plotting for proper scatter spacing.
         * color: color of points. Can be:
            - a single string: color all points that color
            - a list of colors which will be cycled through
            - a list of colors with length identical to number of rows of df
            Default: ('black','red')
         * colortheme: mode of color theme
            - 'group': different colors for different group, but same for different clusters within the group
            - 'cluster': (Default) different colors for different clusters, but the same set of colors for each group
            - 'multi': different colors for different group x clusters
            - 'floral': as in floral pattern (e.g. wallpaper, dresses), where each point within
                    group x cluster are cycled using color. Make sure set 'cluster=None'.
         * reset_index: reset the index of the df before making the plot. Default False
         * kwargs: whichever arguments that are relevent for plt.scatter

     Returns:
         * ax: the axis used for plotting
         * bs: pandas.DataFrame with columns: xorig, yorig, xnew, ynew, color

     Caveats:
         The beeswarm algorithm depends heavily on dpi, dot size and figure
         size. It is necessary to fine tune these three parameters so that the
         plot looks nicely. By default. We set dpi to be 72, dot size to be 33
         (coresponding roughly to 0.08 inch, use by R's beeswarm package), and
         figure size to be (10,5)
    """
    # Reset df index
    if reset_index:
        df = df.reset_index(drop=True)
    
    # Create axis handle if not specified in the argument
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1)
    # The algorithm heavily depends on dpi and figure size
    ax.get_figure().set_dpi(dpi)
    ax.get_figure().set_size_inches(figsize)

    # Create positions vector if not speicifed in the argument
    # set_trace()
    ngroup = len(np.unique(df[group])) if group in list(df.columns.values) else 1
    if positions is None:
        positions = np.arange(ngroup)
    elif len(positions) != ngroup:
        raise(ValueError('"positions" must have length equal to %d, the number of groups'%(ngroup)))
    print(positions)

    # Set the extent of axis
    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        xx = max(positions) - min(positions) + 1
        xmin = min(positions)-0.1*xx
        xmax = max(positions)+0.1*xx
        ax.set_xlim(left=xmin, right=xmax)
        xlim = ax.get_xlim() if xlim is None else xlim
    if ylim is not None:
        ax.set_ylim(bottom=ylim[0], top=ylim[1])
    else:
        yy = max(df[values]) - min(df[values])
        ymin = min(df[values])-.05*yy
        ymax = max(df[values])+0.05*yy
        ax.set_ylim(bottom=ymin, top=ymax)
        ylim = ax.get_ylim() if ylim is None else ylim

    # Get dot size
    xsize, ysize = xydotsize(ax, s=s, dpi=dpi)

    # Create legends if not specified in the argument
    ncluster = len(np.unique(df[cluster])) if cluster in list(df.columns.values) else 1
    if ncluster==1:
        legend = None
    else:
        if legend is None:
            pass
        elif isinstance(legend, (list, tuple, np.ndarray)) and len(legend) != ncluster: # sanity check
            raise(ValueError('"legend" must have length equal to %d, the number of clusters'%(ncluster)))
    
    # Create group labels if not specified in the argument
    if labels is None:
        labels = stable_unique(df[group].values)

    # Get the color vector
    if isinstance(color, str): color = (color)
    color_spec = {
    'group': colorvect([group], df, color=color),
    'cluster': colorvect([cluster], df, color=color) if cluster is not None else color[0],
    'multi': colorvect([group] if cluster is None else [group, cluster], df, color=color),
    'floral': [color[d%len(color)] for d in df.index]
    }.get(colortheme)
    if color_spec is None:
        raise(ValueError('Unrecognized color theme: %s')%(colortheme))
        
    
    # Create a new dataframe
    bs = pd.DataFrame({'xorig':0, 'yorig':df[values], 'xnew':0, 'ynew':df[values], 'color':color_spec})
    
    #set_trace()

    # Adjust data along the grouping dimension: for now, plot vertically
    g_offset, g_pos, d_pos = [], [], []
    for n, g in enumerate(stable_unique(df[group].values)):
        y = df.loc[df[group]==g, values]
        x =  np.repeat(positions[n], len(y))
        # jitter data
        if method == 'swarm':
            if orientation == 'vertical':
                g_pos.append(swarmx(x, y, xsize=xsize, ysize=ysize, side=side, priority=priority, ylog=log))
                g_offset.append(g_pos[n] - x)
            else: # horizontal
                g_pos = swarmy(x, y, xsize=xsize, ysize=ysize, side=side, priority=priority, xlog=log)
                g_offset = g_pos - y
        else: # other methods
            if orientation == 'vertical':
                g_pos, d_pos = gridx(x, y, xsize=xsize, ysize=ysize, dlim=ylim, method=method, side=side, log=log)
                g_offset = g_pos - x
            else: # horizontal
                g_pos, d_pos = gridy(x, y, xsize=xsize, ysize=ysize, dlim=xlim, method=method, side=side, log=log)
                g_offset = g_pos - y

    # check corral
    g_offset = _corral(positions, g_offset, size_g=xsize, ax=ax, corral=corral, corralWidth=corralWidth)

    # parse data frame
    for n, g in enumerate(stable_unique(df[group].values)):
        if orientation == 'vertical':
            bs.loc[df[group]==g, 'xorig'] = positions[n] # original position
            bs.loc[df[group]==g, 'xnew'] = np.array(g_pos[n]) # group offset
            if method != 'swarm': # data offset for non-swarm
                bs.loc[df[group]==g, 'ynew'] = np.array(d_pos[n])
        else:
            bs.loc[df[group]==g, 'yorig'] = positions[n] # original position
            bs.loc[df[group]==g, 'ynew'] = np.array(g_pos[n]) # group offset
            if method != 'swarm':  # data offset for non-swarm
                bs.loc[df[group]==g, 'xnew'] = np.array(d_pos[n])

    # Readjust the axis again in case any dots are being cutt off
    if orientation == 'vertical': # adjust x axis
        xx = max(bs['xnew']) - min(bs['xnew'])
        xmin = min(bs['xnew'])-0.1*xx
        xmax = max(bs['xnew'])+0.1*xx
        ax.set_xlim(left=xmin, right=xmax)
    else: # horizontal, adjust y axis
        yy = max(bs['ynew']) - min(bs['ynew'])
        ymin = min(bs['ynew'])-.05*yy
        ymax = max(bs['ynew'])+0.05*yy
        ax.set_ylim(bottom=ymin, top=ymax)
    

    # Do the plot
    if cluster is None or ncluster==1:
        ax.scatter(bs['xnew'], bs['ynew'], s=s, c=bs['color'], **kwargs)
    else: # iterate over clusters
        for m, cl in enumerate(stable_unique(df[cluster].values)):
            ind = df[cluster]==cl
            ax.scatter(bs.loc[ind,'xnew'], bs.loc[ind, 'ynew'], s=s, \
                    c=bs.loc[ind, 'color'],
                    label=cl if legend is None else legend[m], **kwargs)
        if legendon: # turn on legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.,\
                        title=legendtitle)
            
    # set axis tick / group label
    ax.set_xticks(positions)
    if labels is not None and labelson:
        ax.set_xticklabels(labels, rotation=ticklabelrotation)
    elif not labelson:
        ax.set_xticklabels([""]*len(np.unique(df[group])))

    # set x, y label
    if xlab is not None: ax.set_xlabel(xlab)
    if ylab is not None: ax.set_ylabel(ylab)
    
    if xlim is not None: ax.set_xlim(*xlim)
    if ylim is not None: ax.set_ylim(*ylim)
    # Return items
    return(ax, bs)

def xydotsize(ax, s=None, dpi=None, scale=(1.25,1.25)):
    """ Determine dot size in data axis.
    scale: helps further increasing space between dots
    s: font size in points
    """
    figw, figh = ax.get_figure().get_size_inches() # figure width, height in inch
    dpi = float(ax.get_figure().get_dpi()) if dpi is None else float(dpi)
    w = (ax.get_position().xmax-ax.get_position().xmin)*figw # axis width in inch
    h = (ax.get_position().ymax-ax.get_position().ymin)*figh # axis height in inch
    xran = ax.get_xlim()[1]-ax.get_xlim()[0] # axis width in data
    yran = ax.get_ylim()[1]-ax.get_ylim()[0] # axis height in data
    if s is None:
        xsize=0.08*xran/w*scale[0] # xscale * proportion of xwidth in data
        ysize=0.08*yran/h*scale[1] # yscale * proportion of yheight in data
    else:
        xsize=np.sqrt(s)/dpi*xran/w*scale[0] # xscale * proportion of xwidth in data
        ysize=np.sqrt(s)/dpi*yran/h*scale[1] # yscale * proportion of yheight in data

    return(xsize, ysize)

def _calculateSwarm(x, dsize, gsize, side=int(0), priority='ascending'):
    """Implement swarm layout algorithm
    gsize: group dimension size
    dsize: data dimension size
    """
    # parse some inputs
    if x is None:
        return
    if side not in [-1,0,1]:
        raise(ValueError('"side" must be -1,0,or 1'))

    # make sure things are operating in float
    x, dsize, gsize = np.float32(x), np.float32(dsize), np.float32(gsize)
    # take out missing values
    x = x[~np.isnan(x)]
    #global out

    # initialize data frame
    out = pd.DataFrame({'x':x/dsize, 'y':0}, index=np.arange(len(x))) #*0.7987600831790864

    # Determine the order in which points will be placed
    out = {
    'ascending': out.sort_values('x', ascending=True, axis=0),
    'descending': out.sort_values('x', ascending=False, axis=0),
    'none': out, # do not reorder
    'random': out.reindex(np.random.permutation(out.index)), # randomly gitter
    'density': out.reindex(np.argsort(-KernelDensity(kernel='gaussian',\
                bandwidth=0.2).fit(out['x'][:,np.newaxis]).score_samples(\
                out['x'][:,np.newaxis]))),  # arrange outward from densest areas, result is simlar but not the same as R
    }.get(priority)

    # place the points: we will place once point at a time
    if len(out.index)>1:
        for ii in range(1, len(out.index)):
            xi = out.loc[out.index[ii], 'x']# get ii of sorted
            # identify previously-placed points with potential to overlap the current point
            isPotOverlap = (np.abs(xi - np.array(out['x']))<1.0) & (np.arange(len(out.index))<ii)
            #print(xi, np.where(isPotOverlap))
            if any(isPotOverlap):
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
                out.loc[out.index[ii], 'y'] = poty[np.argmin(abs(poty))]
            else:
                out.loc[out.index[ii], 'y'] = 0

    out.loc[np.isnan(out['x']),'y'] = np.nan  # missing x values should have missing y values
    return(out.sort_index()['y'] * gsize)

def swarmx(x, y, xsize, ysize, side=int(0), priority='ascending', xlog=False, ylog=False):
    """jitter points horizontally
    xlog, ylog: must be a function, e.g. np.log10; Default False
    """
    #print(y)
    if xlog:
        x = xlog(x)
    if ylog:
        y = ylog(y)
    g_pos = x + _calculateSwarm(y, dsize=ysize, gsize=xsize, side=side, priority=priority)
    if xlog:
        # get base: will not work with log1p
        b = np.exp(np.log(5.0)/xlog(5.0)) # 5.0, or any constants to reverse calculate base
        g_pos = b**g_pos
    return(g_pos)
    #return(pd.DataFrame({x=x_new, y=y}))

def swarmy(x, y, xsize, ysize, side=int(0), priority='ascending', xlog=False, ylog=False):
    """ jitter points vertically
    xlog, ylog: must be a function, e.g. np.log10; Default False
    """
    if xlog:
        x = xlog(x)
    if ylog:
        y = ylog(y)
    g_pos = y + _calculateSwarm(x, dsize=xsize, gsize=ysize, side=side, priority=priority)
    if ylog:
        # get base: will not work with log1p
        b = np.exp(np.log(5.0)/ylog(5.0)) # 5.0, or any constants to reverse calculate base
        g_pos = b**g_pos
    return(g_pos)
    #return(pd.DataFrame({x=x, y=y_new}))


def _calculateGrid(x, dsize, gsize, dlim, method='hex', side=int(0), log=False):
    """
    Implement the non-swarm arrangement methods
    dlim: data dimension limit
    gsize: group dimension size
    dsize: data dimension size
    log: must be a function, e.g. np.log10, otherwise/default False
    """
    global d_index, breaks, mids, xx
    xx = x
    if method == "hex": dsize = dsize*np.sqrt(3.0)/2.0
    if log:
        # get base: will not work with log1p
        b = np.exp(np.log(5.0)/log(5.0)) # 5.0, or any constants to reverse calculate base
        breaks = b**np.arange(log(dlim[0]), log(dlim[1])+dsize, dsize)
        mids = pd.Series(b**(log(breaks[:-1]) + log(breaks[1:]))/2.0)
    else: # if data axis is NOT on a log scale
        breaks = np.arange(dlim[0], dlim[1]+dsize, dsize)
        mids = pd.Series((breaks[:-1] + breaks[1:]) / 2.0)
    if len(breaks) == 1 and np.isnan(breaks[0]):
        d_index, d_pos = x, x
    else:
        d_index = pd.Series(pd.cut(pd.Series(x), bins=breaks, labels=False))
        d_pos = d_index.apply(lambda x: mids[x])
    #print(d_index)
    # now determine positions along the group axis
    v_s = {}
    for item in stable_unique(d_index):
        vals = np.arange(list(d_index).count(item))
        v_s[item] = {
        'center': {-1: vals - np.max(vals),
                    0: vals - np.mean(vals),
                    1: vals - 1.0
                   }.get(side),
        'square': {-1: vals - np.max(vals),
                    0: vals - np.floor(np.mean(vals)),
                    1: vals -1.0
                    }.get(side),
        'hex': {-1: vals - np.max(vals) - (0. if (item%2) == 1 else 0.5),
                 0: vals - (np.floor(np.mean(vals))+0.25 if (item%2)==1 else np.ceil(np.mean(vals))-0.25),
                 1: vals - (1.0 if item%2==1 else 0.5)
                }.get(side)
        }.get(method, ValueError('Unrecognized method: %s' %(method)))
        # raise if is an exception
        if isinstance(v_s[item], Exception):
            raise(v_s[item])
    x_index = unsplit(v_s, d_index)
    return(x_index.apply(lambda x: x*gsize), d_pos)

def gridx(x, y, xsize, ysize, dlim, method='hex', side=int(0),log=False):
    """ jitter points horizontally"""
    g_offset, d_pos = _calculateGrid(y, dsize=ysize, gsize=xsize, dlim=dlim, method=method, side=side, log=log)
    return(g_offset+x, d_pos) # new_x, new_y

def gridy(x, y, xsize, ysize, dlim,  method='hex', side=int(0), log=False):
    """ jitter points vertically"""
    g_offset, d_pos = _calculateGrid(x, dsize=xsize, gsize=ysize, dlim=dlim, method=method, side=side, log=log)
    return(g_offset+y, d_pos) # new_y, new_x

def unsplit(x,f):
    """
    same as R's unsplit function
    Read of the values specified in f from x to a vector

    Inputs:
      x: dictionary of value->[items]
      f: vector specifying values to be read off to the vector
    """
    y = pd.DataFrame({'y':[None]*len(f)})
    for item in set(f):
        y.ix[np.array(f==item),'y'] = x[item]
    return(y['y'])

def is_numeric(obj):
    """ check if an object is numeric"""
    attrs = ['__add__', '__sub__', '__mul__', '__div__', '__pow__']
    return all(hasattr(obj, attr) for attr in attrs)

def _corral(positions, g_offset, size_g, ax, corral='none', corralWidth=None):
    """Implement corral method to check for runaway points"""
    if corral == 'none':
        return(g_offset)
    if corralWidth is None:
        if len(positions)>1:
            corralWidth = np.min(positions[-1] - positions[-len(positions)]) - (2*size_g)
        else:
            corralWidth = 2 * (np.min(np.diff([ax.get_xlim[0]] + positions + [ax.get_ylim[1]])) - size_g)
    else:
        if not is_numeric(corralWidth):
            raise(ValueError('"corralWidth" must be a number'))
        if corralWidth <=0:
            raise(ValueError('"corralWidth" must be greater than 0'))
    halfCorralWidth = corralWidth / 2.0
    # calculate g_offset based on corral method
    g_offset = {
    'gutter': [np.minimum(halfCorralWidth, np.maximum(-halfCorralWidth,zz)) for zz in g_offset],
    'wrap': [(zz + halfCorralWidth) % (halfCorralWidth * 2) - halfCorralWidth for zz in g_offset],
    'random': [np.random.uniform(-halfCorralWidth, halfCorralWidth, len(zz)) if (zz > halfCorralWidth).all() or (zz < halfCorralWidth).all() else zz for zz in g_offset],
    'omit': [np.nan if (zz>halfCorralWidth).all() or (zz<-halfCorralWidth).all() else zz for zz in g_offset]
    }.get(corral, ValueError('Unrecognized corral method: %s' %(corral)))
    if isinstance(g_offset,Exception):
        raise(g_offset)
    else:
        g_offset = np.array(g_offset)

    return(g_offset)

def colorvect(factors, df, color=('k','r')):
    """Parse the color vector.
    Cycle through the list of colors provided
    """
    # create group by object
    groupby = df.groupby(list(factors), sort=False)
    # get indices of unique group
    return([color[c] for c in groupby.grouper.group_info[0] % len(color)])
    
def stable_unique(a):
    indices = np.unique(a, return_index=True)[1]
    return [a[index] for index in sorted(indices)]


def add_average_bar(ax, bs, pos='left', cap_marker_edge_width=1, label_values=None, label_values_offset=0, fmt="o", color="k", *args, **kwargs):
    """
    ax: axis of the beeswarm
    bs: data frame returned by beeswarm
    pos: position of the average bars of the beeswarm.
        - "left": left of the swarm
        - "right": right of the swarm
        - a list of custom positions. Needs to be the same length as the number of swarms
    cap_marker_edge_width: default 1. This deals with seaborn.
    *args, **kwargs: additional arguments for ax.errorbar
    """
    gp = bs.groupby(by='xorig', sort=False)
    mean0 = gp.mean()
    serr0 = gp.agg(lambda x: np.std(x) / np.sqrt(np.shape(x)[0]))
    if pos == 'left':
        pos0 = gp.min()['xnew'].values-1.5*gp.std()['xnew']
    elif pos == 'right':
        pos0 = gp.max()['xnew'].values-1.5*gp.std()['xnew']
    elif isinstance(pos, (list, tuple, np.ndarray)): # assume a list of positions
        pos0 = np.asarray(pos)
    else:
        raise(TypeError('Unknown type of pos'))
    
    _, caps, _ = ax.errorbar(pos0, mean0['ynew'].values, serr0['ynew'].values, fmt=fmt, color=color, *args, **kwargs)
    
    if cap_marker_edge_width is not None:
        for cap in caps:
            cap.set_markeredgewidth(cap_marker_edge_width)
            
    if label_values:
        if not isinstance(label_values_offset, (list, tuple, np.ndarray)):
            label_values_offset = [label_values_offset]*len(pos0)
        for p, yo, m, s in zip(pos0, label_values_offset, mean0['ynew'].values, serr0['ynew'].values):
            if isinstance(label_values, str) :
                if label_values == 'vertical':
                    ax.text(p, m+s*1.1+yo, "{:.1f}\n$\pm$\n{:.1f}".format(m, s), va='bottom', ha='center')
                elif label_values == 'mean':
                    ax.text(p, m+s*1.1+yo, "{:.1f}".format(m), va='bottom', ha='center')
            
            else:
                ax.text(p, m+s*1.1+yo, "{:.1f}$\pm${:.1f}".format(m, s), va='bottom', ha='center')
    
    return ax


def connect_paired_dots(ax, bs, zorder=0, pairs=None, *args, **kwargs):
    """
    Connect paired beeswarms
    ax: axis of the beeswarm
    bs: data frame returned by beeswram, assuming the first half is group1 and second half is group 2
    pairs: positions of the dots. Default is the unique value of the xorig
    *args, **kwargs, additional arguements for ax.plot
    """

    nrows = bs.shape[0] # should be a even number of rows
    if nrows%2 != 0:
        raise(ValueError('Number of rows of bs must be even!'))
        
    num_lines = int(nrows/2)
    
    if pairs is None:
        pairs = np.sort(bs['xorig'].unique())
    
    # Sepeating the left and right dots
    bs0 = bs.loc[bs['xorig']==pairs[0],:]
    bs1 = bs.loc[bs['xorig']==pairs[1],:]
    
    for k in range(num_lines):
        ax.plot(np.asarray([bs0.iloc[k]['xnew'], bs1.iloc[k]['xnew']]), 
                np.asarray([bs0.iloc[k]['ynew'], bs1.iloc[k]['ynew']]), zorder=0, *args, **kwargs)
        
    return ax
    
    
        

if __name__=='asdf':#'__main__':
    from ImportData import FigureData
    df = FigureData(dataFile='D:/Edward/Documents/Assignments/Scripts/Python/Plots/example/beeswarm.csv')
    df = df.table
    values = 'time_survival'
    group = 'ER'
    cluster = 'event_survival'
    ax, bs = beeswarm(df, values, group=group, cluster=cluster, figsize=(6,5),
                        method='swarm', legend=('yes','no'), legendtitle='Survival',corral='gutter')
