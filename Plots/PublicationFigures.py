# -*- coding: utf-8 -*-
"""
Created on Sat Jun 06 13:35:08 2015

@author: Edward
"""
DEBUG = True

import os
import numpy as np
from ImportData import FigureData
#import matplotlib
#matplotlib.use('Agg') # use 'Agg' backend
import matplotlib.pyplot as plt

plotType = 'neuro'
style = 'Vstack'
exampleFolder = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/'

# global variables
# fontname = os.path.abspath(os.path.join(os.path.dirname(__file__), 'resource/Helvetica.ttf')) # font .ttf file path
fontname = 'Arial'
fontsize = {'title':16, 'xlab':12, 'ylab':12, 'xtick':10,'ytick':10, 'texts':8,
            'legend': 12, 'legendtitle':12} # font size
color = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd154','#17becf'] # tableau10, or odd of tableau20
marker = ['o', 's', 'd', '^', '*', 'p']# scatter plot line marker cycle
hatch = ['/','\\','-', '+', 'x', 'o', 'O', '.', '*'] # fill patterns potentially used for filled objects such as bars
canvas_size = (6,5)

class PublicationFigures(object):
    """Generate publicatino quantlity figures
        Data: FigureData, or data file path
        PlotType: currently supported plot types include:
            ~ LinePlot: for categorical data, with error bar
                style:
                    'Twin' -- Same plot, 2 y-axis (left and right of plot)
                    'Vstacked' (default) -- vertically stacked subplots
            ~ Beeswarm: beeswarm plot; boxplot with scatter points
                style: 'hex','swarm' (default),'center','square'
    """
    def __init__(self, dataFile=None, savePath=None, *args, **kwargs):
        """Initialize class
        """
        if isinstance(dataFile, (str,list,tuple,np.ndarray)):
            self.LoadData(dataFile, *args, **kwargs) # load data
        elif isinstance(dataFile, FigureData):
            self.data = dataFile
        self.savePath = savePath
        self.cache=0 # for progressive draw of objects

    def LoadData(self, dataFile, *args, **kwargs):
        """To be called after object creation"""
        self.data = FigureData(dataFile, *args, **kwargs)
        # Set some variables to help with indexing
        g = globals()
        for item in ['x','y','z','by']:
            g['_'+item] = self.data.meta[item] \
                    if item in self.data.meta else None

    def AdjustFigure(canvas_size=canvas_size, tight_layout=True):
        """Used as a decotrator to set the figure properties"""
        def wrap(func):
            def wrapper(self, *args, **kwargs):
                res = func(self, *args, **kwargs)#execute the function as usual
                self.SetFont() # adjust font
                if canvas_size is not None:
                    self.fig.set_size_inches(canvas_size) # set figure size
                if tight_layout:
                    self.fig.tight_layout() # tight layout
                return(res)
            return(wrapper)
        return(wrap)

    def AdjustAxs(otypes=[np.ndarray], excluded=None):
        """Used as a decorator to set the axis properties"""
        def wrap(func):
            # vectorize the func so that it can be applied to single axis or
            # multiple axes
            func_vec = np.vectorize(func, otypes=otypes, excluded=excluded)
            def wrapper(self, ax=None, *args, **kwargs):
                if ax is None: # if not specified, use default axis
                    res = func_vec(self.axs, *args, **kwargs)
                else:
                    res = func_vec(ax, *args, **kwargs)
                return(res)
            return(wrapper)
        return(wrap)

    def Save(self,savePath=None, dpi=300):
        """
        savePath: full path to save the image. Image type determined by file
            extention
        dpi: DPI of the saved image. Default 300.
        """
        if savePath is not None: # overwrite with new savePath
            self.savePath = savePath
        if self.savePath is None: # save to current working directory
            self.savePath = os.path.join(os.getcwd(),'Figure.eps')
        self.fig.savefig(self.savePath, bbox_inches='tight', dpi=dpi)

    """ ####################### Plot utilities ####################### """
    @AdjustFigure(canvas_size=(50,5), tight_layout=False)
    def Traces(self, outline='vertical', scaleref='scalebar', scalepos='last',
               annotation=None, annstyle='last', color=['#000000', '#ff0000',
               '#0000ff','#ffa500', '#007f00','#00bfbf', '#bf00bf']):
        """Plot time series / voltage and current traces
        outline: connfigurations of the plot. Default 'vertical'
            - 'vertical': arrange the traces vertically, each as a subplot.
            - 'horizontal': arrange the traces horizontally, concatenating the
                            traces along time
            - 'overlap': Plot all the traces in the same axis, cycling through
                     the color list.
            - np.ndarray specify the position of the index spanning a grid
                e.g. [[0,1],[2,3],[4,5]] would specify data[0] at first row,
                first column; data[1] at first row, second column; data[2] at
                second row, first column, ... and so on
        scaleref: style of scale reference
            - 'scalebar': use scale bar (Default)
            - 'axis': use axis
        scalepos: where to set scale reference.
            - 'last': (Default) set scale reference only in the last row and
                last column of the subplots. Note that if scaleref is 'axis',
                vertical scales will be shown in every subplot, while
                horitontal scale will not be set until the last subplots
            - 'each': set a scale reference at each subplot.  Note that if
                scaleref is 'axis', both vertical and horizontal scales will
                be shown in every subplots.
                horitontal scale will not be set until the last subplots
        annotation: a string or list of strings to put into the annotation.
            The annotation text are also parsed from data.meta['annotation'].
        annstyle: style of annotation.
            - 'last': (Default) annotation all together after plotting the
                  traces. Print each item in the list of annotations as a line
            - 'each': each subplot of traces gets an annotation next to it
        color: default MATLAB's color scheme
        """
        import matplotlib.gridspec as gridspec
        # Define the layout of a single canvas
        gs = gridspec.GridSpec(2,3, width_ratios = [1, 50, 3], height_ratios=[7,1])
        # Set up all the axes
        ax = dict()
        fig = plt.figure()
        ax['trace'] = fig.add_subplot(gs[1])
        ax['initial'] = fig.add_subplot(gs[0], sharex=ax['trace'])
        ax['scalebar'] = fig.add_subplot(gs[2], sharex=ax['trace'])
        ax['annotation'] = fig.add_subplot(gs[3:], sharex=ax['trace'])
        #ax = list(ax.values())

        # text annotation
        ax['annotation'].text(0., 0.5, self.data.meta['notes'][0])
        # scalebar
        self.AddTraceScaleBar(xunit='ms', yunit='mV', color='k', ax=ax['scalebar'])
        # initial value
        ax['initial'].text(0., 0.5, '%0.2fmV'%(self.data.table[0]['VoltA'][0]))

        plt.gcf().set_size_inches(6,3)
        # parse subplot configuration
        if outline is None:
            tmp = np.array(_y)
            if tmp.ndim<2: tmp = tmp[np.newaxis,:]
            outline =np.arange(0,np.array(_y).size).reshape(tmp.shape)
        elif outline == 'vertical':
            outline = np.arange(0, np.array(_y).size)[:,np.newaxis]
        elif outline == 'horizontal':
            outline = np.arange(0,np.array(_y).size)[np.newaxis,:]
        elif outline == 'overlap':
            outline = np.array([[0]])
        elif isinstance(outline, (np.ndarray,list,tuple)):
            outline = np.asarray(outline)
            if outline.ndim<2: outline = outline[np.newaxis,:] # horizontal
        else:
            raise(TypeError('Unrecognized plot configuration'))

        nrows, ncols = outline.shape
        # Create figure and axis
        # use gridspec instead, then use additional grid to add scalebar
        self.fig, self.axs = plt.subplots(nrows=nrows,ncols=ncols,sharex=False, sharey=False)

        for n in np.nditer(outline):
            if n is None: continue # if no data, continue to next loop
            n = int(n)
            r,c = np.unravel_index(n, _y.shape, order='C')
            ax = self.axs[n] if self.axs.ndim<2 else self.axs[r, c]
            x, y = _x[r], _y[r,c]
            # for lazy error handling
            xlabel=self.get_field(self.data.meta,'xlabel',r)
            ylabel=self.get_field(self.data.meta,'ylabel',r,c)
            xunit=self.get_field(self.data.meta,'xunit',r)
            yunit=self.get_field(self.data.meta,'yunit',r,c)
            annotation=self.get_field(self.data.meta,'annotation',r)
            # parse scale reference
            if scalepos == 'each':
                ref = scaleref
            elif scalepos == 'last':
                if (ncols==1 and ax.is_last_row()) or \
                (nrows==1 and ax.is_last_col()) or \
                (ncols>1 and nrows>1 and \
                    (ax.is_last_row() or ax.is_last_col())):
                    ref = scaleref
                else: # not the last line  yet
                    ref = 'yaxis'  if scaleref == 'axis' else None

            # do the plot
            self.TimeSeries(self.data.table[r][x], self.data.table[r][y],
                            ax=ax,xunit=xunit,yunit=yunit,
                            xlabel=xlabel,ylabel=ylabel,
                            scaleref=ref, annotation=annotation \
                                            if annstyle=='each' else None)
        # set aspect ratio
        #self.SetAspectRatio(r=2, adjustable='box-forced')
        plt.subplots_adjust(wspace=0.01)
        self.TurnOffAxis()

        if annstyle == 'last': # ['last','each']
            self.TextAnnotation(text=annotation) # description of the trace

    def TimeSeries(self, X, Y, ax=None, xunit='ms', yunit='mV',
                   xlabel=None, ylabel=None, scaleref='scalebar',
                   annotation=None, color='k'):
        """Make a single time series plot"""
        if ax is None: ax = self.axs
        hline = ax.plot(X, Y, color=color)[0]
        if scaleref == 'scalebar': # Use scale bar instead of axis
            self.AddTraceScaleBar(xunit=xunit, yunit=yunit, color=hline, ax=ax)
        elif scaleref == 'yaxis': # use only y axis
            self.ShowOnlyYAxis(ax)
            ax.set_ylabel(ylabel)
        elif scaleref == 'axis': # Use axis
            self.SetDefaultAxis(ax) # use default axis
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
        else: # do not draw any reference on this axis
            self.TurnOffAxis(ax)
    
    @AdjustFigure(tight_layout=False)
    def SingleEpisodeTraces(self, table, notes="", color='k', channels=['A'], 
                            streams=['Volt','Cur']):
        """Helper function to export traces from a single episode.
           Arrange all plots vertcially"""
        self.fig, self.axs = plt.subplots(nrows=len(channels)*len(streams), 
                                          ncols=1, sharex=True)
        pcount = 0
        yunit_dict = {'Volt':'mV','Cur':'pA','Stim':'pA'}
    
        for c in channels: # iterate over channels
            for s in streams: # iterate over streams
                self.axs[pcount].plot(table['time'],table[s+c], 
                                      label=pcount, c='k')
                self.AddTraceScaleBar(xunit='ms', yunit=yunit_dict[s],
                                      ax=self.axs[pcount])
                position = [0, table[s+c][0]]
                text = '%.0f'%(position[1]) + yunit_dict[s]
                self.TextAnnotation(text=text, position=position, 
                                    ax=self.axs[pcount], color=color,
                                    xoffset='-', yoffset=None, fontsize=None,
                                    ha='right',va='center')
                pcount += 1
    
        # Finally, annotate the episode information at the bottom
        pad = np.array(self.axs[-1].get_position().bounds[:2]) *\
                        np.array([1.0, 0.8])
        self.fig.text(pad[0], pad[1], notes, ha='left',va='bottom')
        

    @AdjustFigure()
    def Scatter(self, color=color, marker=marker, alpha=0.5, legend_on=True):
        """2D Scatter plot
        color = blue, magenta, purple, orange, green
        marker = circle, pentagon, pentagram star,star, + sign
        """
        global _x, _y, _by
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1)
        # Get number of groups
        group = np.unique(self.data.table[_by]) if _by is not None else [1]
        for n,gp in enumerate(group):
            label = self.get_field(self.data.meta,'legend',n)
            # select subset of data rows
            series = self.data.table[self.data.table[_by]==gp] \
                                if _by is not None else self.data.table
            # plot
            plt.scatter(series[_x], series[_y], alpha=alpha, s=50,
                        marker=marker[n%len(marker)],
                        color=color[n%len(color)],
                        label=label[n] if label is not None else None)
        self.SetDefaultAxis()
        self.axs.set_xlabel(self.data.meta['xlabel'])
        self.axs.set_ylabel(self.data.meta['ylabel'])
        if legend_on and label is not None :
            self.axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    @AdjustFigure()
    def Scatter3D(self, color='k', marker=['.', '+', 'x', (5, 2), '4']):
        from mpl_toolkits.mplot3d import Axes3D # for 3D plots
        self.fig = plt.figure()
        self.axs = self.fig.add_subplot(111, projection='3d')
        color=list(color)
        global _x, _y, _z, _by
        # Get number of groups
        group = np.unique(self.data.table[_by]) if _by is not None else [1]
        for n,gp in enumerate(group):
            label = self.get_field(self.data.meta,'legend',n)
            # select subset of data rows
            series = self.data.table[self.data.table[_by]==gp] \
                                if _by is not None else self.data.table
            # plot
            self.axs.scatter(series[_x], series[_y], series[_z],
                             label=label, zdir=u'z', s=144, #sqrt(s) point font
                             c = color[n%len(color)],
                             marker=marker[n%len(marker)],
                             depthshade=True)
        self.axs.set_xlabel(self.data.meta['xlabel'])
        self.axs.set_ylabel(self.data.meta['ylabel'])
        self.axs.set_zlabel(self.data.meta['zlabel'])
        # Add annotations
        #self.AddRegions()
        self.SetDefaultAxis3D() # default axis, view, and distance

    @AdjustFigure()
    def BarPlot(self, style='Vertical', width=0.27, gap=0, space=0.25,
                color=color, hatch=None, alpha=0.4, linewidth=0):
        """Plot bar graph
        style: style of bar graph, can choose 'Vertical' and 'Horizontal'
        width: width of bar. Default 0.27
        gap: space between bars. Default 0.
        space: distances between categories. Deafult 0.25
        """
        global _x, _y, _z, _by
        # initialize plot
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        # Get number of groups
        group = np.unique(self.data.table[_by]) if _by is not None else [1]
        # Center of each category
        ns = len(group) # number of series
        inc = space+(ns-1)*gap+ns*width
        self.x = np.arange(0,len(np.unique(self.data.table[_x]))*inc,inc)
        # leftmost position of bars
        pos = self.x-ns/2*width - (ns-1)/2*gap

        for n,gp in enumerate(group):
            label = self.get_field(self.data.meta,'legend',n)
            # select subset of data rows
            series = self.data.table[self.data.table[_by]==gp] \
                                if _by is not None else self.data.table
            err = self.data.parse_errorbar(series) # get errorbar
            pos = pos if n==0 else pos+width+gap
            if style=='Vertical':
                bars = self.axs.bar(pos[:series.shape[0]], series[_y],
                                    width,  yerr=err, alpha=alpha,
                                    align='center', label=label)
            else:
                bars = self.axs.barh(pos[:series.shape[0]], series[_y],
                                     width,  yerr=err, alpha=alpha,
                                     align='center', label=label)
            # Set color
            self.SetColor(bars, color, n, linewidth)
            # Set hatch if available
            self.SetHatch(bars, hatch, n)
        self.SetDefaultAxis()
        if style=='Vertical':
            self.SetCategoricalXAxis()
            self.AdjustBarPlotXAxis()
        else: # horizontal
            self.AdjustBarPlotYAxis()
            self.SetCategoricalYAxis()
        # Set labels
        self.axs.set_xlabel(self.data.meta['xlabel'])
        self.axs.set_ylabel(self.data.meta['ylabel'])

        if n>0: # for multiple series, add legend
            self.axs.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    @AdjustFigure()
    def Boxplot(self, color=color):
        """boxplot"""
        # initialize plot
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        self.x = [0,1]
        self.axs.boxplot(np.array(self.data.table[_y]).T)
        self.SetDefaultAxis()

    @AdjustFigure(canvas_size=None)
    def Beeswarm(self, style= "swarm",color=color, theme='cluster', **kwargs):
        """Beeswarm style boxplot
        * style: beeswarm dot style,['swarm' (default),'hex','center','square']
        * theme: ['cluster' (Default), 'group', 'multi', 'floral'].
                Details see beeswarm doc string
        """
        from simple.beeswarm import beeswarm
        global _x, _y, _by
        # initialize plot
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        # get some label parameters
        group = self.get_field(self.data.meta, 'group')
        legend = self.get_field(self.data.meta, 'legend')
        legendtitle = self.get_field(self.data.meta, 'legendtitle')
        # Do the plot
        self.axs, _ = beeswarm(_y, df=self.data.table, group=_x, cluster=_by,\
                                method=style,ax=self.axs, color=color,\
                                colortheme=theme, figsize=canvas_size,\
                                legend=legend, legendtitle=legendtitle, \
                                labels=group, **kwargs)

        # Format style
        # make sure axis tickmark points out
        self.axs.tick_params(axis='both',direction='out')
        self.axs.spines['right'].set_visible(False)
        self.axs.spines['top'].set_visible(False)
        self.axs.xaxis.set_ticks_position('bottom')
        self.axs.yaxis.set_ticks_position('left')
        # Set Y label, if exsit
        try:
            self.axs.set_ylabel(self.data.meta['ylabel'])
        except:
            pass

    @AdjustFigure()
    def Violinplot(self, color=color):
        """violin plot / boxplot"""

    @AdjustFigure()
    def Histogram(self, style='Hstack'):
        """Plot histogram"""
        n, bins, patches = P.hist(x, 50, normed=1, histtype='stepfilled')
        P.setp(patches, 'facecolor', 'g', 'alpha', 0.75)
        return

    def HistogramHstack(self):
        return

    @AdjustFigure()
    def LinePlot(self,style='Vstack',xtime='categorical',margins=(0.,0.25)):
        """Line plots with errorbars
        style: ['Vstack' (default), 'Twin'] style of multiple subplots.
            - 'Vstack': vertically stacked subplots
            - 'Twin': can handle only up to 2 subplots
        xtime: used to plot time series with errorbars. Specify an array of
                time points.
        """
        # set categorical x
        self.x = list(self.data.table.index) if xtime=='categorical' else xtime
        global _x, _y
        _y = [_y] if isinstance(_y, str) else _y
        if style=='Twin' and len(_y) == 2:
            self.LinePlotTwin()
        else:
            self.LinePlotVstack()
        # Must set margins before setting aspect ratio
        self.SetMargins(x=margins[0], y=margins[1])
        # Set aspect ratio
        self.SetAspectRatio(r=2, adjustable='box-forced', margins=margins)
        if xtime == 'categorical':
            self.SetCategoricalXAxis() # make some space for each category

    def LinePlotVstack(self):
        """ Line plots stacked vertically"""
        self.fig, self.axs = plt.subplots(nrows=len(_y), ncols=1, sharex=True)
        self.axs = np.array([self.axs]) if len(_y)<2 else self.axs
        err = self.data.parse_errorbar(simplify=False) # get errorbar
        for n, ax in enumerate(self.axs):
            # Plot error bar
            ax.errorbar(self.x,self.data.table[_y[n]], color='k',yerr = err[n])
        self.axs = self.axs[0] if len(_y)<2 else self.axs
        self.SetVstackAxis() # set vertical stacked subplot axes

    def LinePlotTwin(self, color=('k','r')):
        """ Line plots with 2 y-axis"""
        self.fig, self.axs = plt.subplots()
        # Must set label on the first axis in order to show up in the plot
        self.axs.set_xlabel(self.data.meta['xlabel'])
        # Construct another axis sharing xaxis with current axis
        self.axs = np.array([self.axs, self.axs.twinx()])
        err = self.data.parse_errorbar(simplify=False) # get error bar
        self.SetTwinPlotAxis(color=color) # set twin plot subplot axes
        for n, ax in enumerate(self.axs):
             # Plot error bar
            ax.errorbar(np.array(self.x), np.array(self.data.table[_y[n]]),
                        color=color[n], yerr=err[n])

    """ ####################### Axis schemas ####################### """
    @AdjustAxs()
    def SetDefaultAxis(ax):
        """Set default axis appearance"""
        ax.tick_params(axis='both',direction='out')
        ax.spines['left'].set_visible(True)
        ax.spines['left'].set_capstyle('butt')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_capstyle('butt')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

    @AdjustAxs()
    def SetDefaultAxis3D(ax, elev=45, azim=60, dist=12):
        ax.tick_params(axis='both', direction='out')
        ax.view_init(elev=elev, azim=azim) # set perspective
        ax.dist = dist # use default axis distance 10
        if ax.azim > 0: # z axis will be on the left
            ax.zaxis.set_rotate_label(False) # prevent auto rotation
            a = ax.zaxis.label.get_rotation()
            ax.zaxis.label.set_rotation(90+a) # set custom rotation
            ax.invert_xaxis() # make sure (0,0) in front
            ax.invert_yaxis() # make sure (0,0) in front
        else:
            ax.invert_xaxis() # make sure (0,0) in front
        #ax.zaxis.label.set_color('red')
        #ax.yaxis._axinfo['label']['space_factor'] = 2.8

    @AdjustAxs()
    def TurnOffAxis(ax):
        """Turn off all axis"""
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    @AdjustAxs()
    def ShowOnlyXAxis(ax):
        """Turn off all axis but only X axis"""
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_visible(True)
        ax.yaxis.set_visible(False)

    @AdjustAxs()
    def ShowOnlyYAxis(ax):
        """Turn off all axis but only Y axis"""
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(True)

    def AdjustBarPlotXAxis(self):
        """Adjust bar plot's x axis for categorical axis"""
        # get y axis extent
        ymin, ymax = self.axs.get_ybound()
        if ymax <= 0.0: # only negative data present
            # flip label to top
            self.axs.spines['bottom'].set_position('zero') # zero the x axis
            self.axs.tick_params(labelbottom=False, labeltop=True)
        elif ymin >= 0.0: # only positive data present. Default
            self.axs.spines['bottom'].set_position('zero') # zero the x axis
        else: # mix of positive an negative data : set all label to bottoms
            self.axs.spines['bottom'].set_visible(False)
            self.axs.spines['top'].set_visible(True)
            self.axs.spines['top'].set_position('zero')
        self.axs.xaxis.set_ticks_position('none')

    def AdjustBarPlotYAxis(self):
        """Adjust bar plot's y axis for categorical axis"""
        #set all label to left
        self.axs.spines['left'].set_visible(False)
        self.axs.spines['right'].set_visible(True)
        self.axs.spines['right'].set_position('zero')
        self.axs.yaxis.set_ticks_position('none')

    def SetTwinPlotAxis(self, color=('k', 'r')):
        """Axis style  of 2 plots sharing y axis"""
        spineName = ('left','right')
        for n, ax in enumerate(self.axs):             # For twin Plot
            ax.tick_params(axis='both',direction='out') # tick mark out
            ax.spines['top'].set_visible(False) # remove top boundary
            ax.xaxis.set_ticks_position('bottom') # keep only bottom ticks
            ax.set_ylabel(self.data.meta['ylabel'][n]) # set y label
            ax.yaxis.label.set_color(color[n]) # set y label color
            ax.tick_params(axis='y',color=color[n]) # set y tick color
            [tl.set_color(color[n]) for tl in ax.get_yticklabels()]
            ax.spines[spineName[n]].set_color(color[n]) # set y spine color
        self.axs[0].spines['right'].set_visible(False) # leave only left spine
        self.axs[1].spines['left'].set_visible(False) # only only right spine

    @AdjustAxs()
    def PadY(ax):
        """Set extra padding if data points / lines are cut off"""
        arr = np.array([l.get_ydata() for l in ax.lines])
        MAX, MIN = np.max(arr), np.min(arr)
        ytick_arr = ax.get_yticks()
        inc = np.mean(np.diff(ytick_arr)) # extra padding
        if np.min(ytick_arr)>=MIN:
            ax.set_ylim(MIN-inc, ax.get_ylim()[-1])
        if np.max(ytick_arr)<=MAX:
            ax.set_ylim(ax.get_ylim()[0], MAX+inc)

    def SetVstackAxis(self):
        """Axis style of vertically stacked subplots"""
        def SVsA(ax, n):
            ax.tick_params(axis='both', direction='out') #tick mark out
            ax.spines['top'].set_visible(False) # remove top boundary
            ax.spines['right'].set_visible(False) # remove right spine
            ax.yaxis.set_ticks_position('left') # keep only left ticks
            ax.set_ylabel(self.data.meta['ylabel'][n]) # set different y labels
            if ax.is_last_row():     #keep only bottom ticks
                ax.xaxis.set_ticks_position('bottom')
                ax.set_xlabel(self.data.meta['xlabel']) # x label
            else:
                ax.xaxis.set_visible(False)
                ax.spines['bottom'].set_visible(False)
        SVsA_vec = np.frompyfunc(SVsA,2,1)
        num_axs = len(self.axs) if isinstance(self.axs, np.ndarray) else 1
        SVsA_vec(self.axs, range(num_axs))
        self.fig.tight_layout(h_pad=0.01) # pad height

    def SetHstackAxis(self):
        """Axis style of horizontally stacked / concatenated subplots"""
        def SHsA(ax, n):
            ax.tick_params(axis='both', direction='out') # tick mark out
            ax.spine['top'].set_visible(False) #remove top boundary
            ax.spine['right'].set_visible(False) # remove right spine
            ax.yaxis.set_ticks_position('left') # keep only left ticks
            ax.set_xlabel(self.data.meta['xlabel'][n]) # set different x labels
            if ax.is_first_col(): # keep only first ticks
                ax.yaxis.set_ticks_position('left')
                ax.set_ylabel(self.data.meta['ylabel'][0]) # y label
            else:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
        SHsA_vec = np.frompyfunc(SHsA,2,1)
        num_axs = len(self.axs) if isinstance(self.axs, np.ndarray) else 1
        SHsA_vec(self.axs, range(num_axs))
        self.fig.tight_layout(w_pad=0.01) # pad width

    def SetCategoricalXAxis(self, ax=None):
        """Additional settings for plots with categorical data"""
        # change the x lim on the last, most buttom subplot
        if ax is None: # last axis, or self.axs is a single axis
            ax = self.axs[-1] if isinstance(self.axs, np.ndarray) else self.axs
        if ax.get_xlim()[0] >= self.x[0]:
            ax.set_xlim(ax.get_xticks()[0]-1,ax.get_xlim()[-1])
        if ax.get_xlim()[-1] <= self.x[-1]:
            ax.set_xlim(ax.get_xlim()[0], ax.get_xticks()[-1]+1)
        plt.xticks(self.x, np.unique(self.data.table[_x]))

    def SetCategoricalYAxis(self, ax=None):
        """Additional settings for plots with categorical data"""
        if ax is None: # last axis, or self.axs is a single axis
            ax = self.axs[-1] if isinstance(self.axs, np.ndarray) else self.axs
        if ax.get_ylim()[0] >= self.x[0]:
            ax.set_ylim(ax.get_yticks()[0]-0.5,ax.get_ylim()[-1])
        if ax.get_ylim()[-1] <= self.x[-1]:
            ax.set_ylim(ax.get_ylim()[0], ax.get_yticks()[-1]+0.5)
        plt.yticks(self.x, np.unique(self.data.table[_x]))

    def SetDiscontinousAxis(self, x=None, y=None):
        """Plot with discontious axis. Allows one discontinuity for each axis.
        Assume there is only 1 plot in the figure
        x: x break point
        y: y break point
        """
        raise(NotImplementedError('This method is yet to be implemeneted'))
        if x is not None and y is not None:
            f,axs = plt.subplots(2,2,sharex=True,sharey=True)
        elif x is not None and y is None:
            f,axs = plt.subplots(2,1,sharey=True)
        elif x is None and y is not None:
            f,axs = plt.subplots(2,1,sharex=True)
        line = self.axs.get_lines()
        # plot the same data in all the subplots
        [ax.add_line(line) for ax in axs]
        # set axis
        self.SetVstackAxis() # set vertical stacked subplot axes
        for ax in self.axs:
            if not ax.is_first_col:
                ax.yaxis.set_visible(False)
                ax.spines['left'].set_visible(False)
        # add slashes between two plots

    @AdjustAxs(excluded=['margins'])
    def SetAspectRatio(ax, r=2, adjustable='box-forced',margins=(0,0)):
        """Set aspect ratio of the plots, across all axes.
        Must set margins before calling this function to set aspect
        ratios.
        r: ratio in data domains
        adjustable: see 'adjustable' argument for axes.set_aspect
        margins: account extra margins when setting aspect ratio.
        Default is (0,0)
        """
        if not isinstance(r, str):
            dX = np.diff(ax.get_xlim())/(1+2*margins[0])
            dY = np.diff(ax.get_ylim())/(1+2*margins[1])
            aspect = dX/dY/r
        else:
            aspect = r
        ax.set_aspect(aspect=aspect, adjustable=adjustable)

    @AdjustAxs()
    def SetMargins(ax, x=0.25, y=0.25):
        """Wrapper for setting margins"""
        ax.margins(x,y)

    def SetColor(self, plotobj, color, n, linewidth=0):
        """Set colors. Would allow optionally turn off color"""
        if color is not None:
            for p in plotobj:
                p.set_color(color[n%len(color)])
                p.set_linewidth(linewidth)
        else:
            for p in plotobj:
                p.set_color('w')
                p.set_edgecolor('k')
                p.set_linewidth(1)

    def SetHatch(self, plotobj, hatch, n):
        """Set hatch for patchs"""
        if hatch is not None:
            for p in plotobj:
                p.set_hatch(hatch[n%len(hatch)])

    """ #################### Text Annotations ####################### """
    def AddTraceScaleBar(self,xunit,yunit,color='k',linewidth=None,\
                         fontsize=None,ax=None):
        """Add scale bar on trace. Specifically designed for voltage /
        current / stimulus vs. time traces."""
        if ax is None: ax=self.axs
        def scalebarlabel(x, unitstr):
            if unitstr.lower()[0] == 'm':
                return(str(x)+unitstr if x<1000 else str(x/1000)+
                    unitstr.replace('m',''))
            elif unitstr.lower()[0] == 'p':
                return(str(x)+unitstr if x<1000 else str(x/1000)+
                    unitstr.replace('p','n'))

        self.TurnOffAxis(ax) # turn off axis
        X, Y = np.ptp(ax.get_xticks()), np.ptp(ax.get_yticks())
        # calculate scale bar unit length
        X, Y = self.roundto125(X/5), self.roundto125(Y/5)
        # Parse scale bar labels
        xlab, ylab = scalebarlabel(X, xunit), scalebarlabel(Y, yunit)
        # Get color of the scalebar
        if color is None:
            color = ax.get_lines()[0]
        if 'matplotlib.lines.Line2D' in str(type(color)):
            color = color.get_color()
        if linewidth is None:
            try:
                linewidth = ax.get_lines()[0]
            except:
                raise(AttributeError('Did not find any line in this axis. Please explicitly specify the linewidth'))
        if 'matplotlib.lines.Line2D' in str(type(linewidth)):
            linewidth = linewidth.get_linewidth()
        if fontsize is None:
            fontsize = ax.yaxis.get_major_ticks()[2].label.get_fontsize()
        # Calculate position of the scale bar
        xi = np.max(ax.get_xticks()) + X/2.0
        yi = np.mean(ax.get_yticks())
        # calculate position of text
        xtext1, ytext1 = xi+X/2.0, yi-Y/10.0 # horizontal
        xtext2, ytext2 = xi+X+X/10.0, yi+Y/2.0 # vertical
        # Draw text
        txt1 = ax.text(xtext1, ytext1, xlab, ha='center',va='top',
                             color=color, size=fontsize)
        self.AdjustText(txt1, ax=ax) # adjust texts just added
        txt2 = ax.text(xtext2, ytext2, ylab, ha='left',va='center',
                             color=color, size=fontsize)
        self.AdjustText(txt2,ax=ax) # adjust texts just added
        # Draw Scale bar
        ax.annotate("", xy=(xi,yi), xycoords='data',  # horizontal
                          xytext=(xi+X,yi), textcoords = 'data',
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0,
                          color=color, linewidth=linewidth))
        ax.annotate("", xy=(xi+X,yi), xycoords='data',  # vertical
                          xytext=(xi+X,yi+Y), textcoords = 'data',
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0,
                          color=color,linewidth=linewidth))
        #txtbb = txt1.get_bbox_patch().get_window_extent()
        #print(txtbb.bounds)
        #print(self.axs.transData.inverted().transform(txtbb).ravel())

    def TextAnnotation(self, text="", position='south', ax=None, xoffset=None,
                       yoffset=None, color='k', fontsize=None, **kwargs):
        """Annotating with text
        color: color of the text and traces. Default 'k'. If None, use the same
               color of the trace
        xoffset: the amount of space in horizontal direction, e.g. text around
                 traces. 
                 ~ If None, no offsets. 
                 ~ If "+", add a space of a single character to x position.
                   The number of "+"s in the argument indicates the number of 
                   times that the single character space will be added.
                 ~ If "-", subtract a space of single character to x position.
                   Rule for multiple "-"s is the same for "+"
                 ~ If a number, add this number to x position
        yoffset: the amount of space in vertical direction, e.g. between lines
                 of text. The yoffset is applied the same as xoffset, only to
                 y position
        fontsize: size of the font. Default None, use the same font size as 
                the x tick labels
        **kwargs: additional argument for ax.text
        """
        ax = self.axs if ax is None else ax
      
        if isinstance(position, str):
            # get axis parameter
            X, Y = np.ptp(ax.get_xticks()), np.ptp(ax.get_yticks())
            xytext = {
            'north': (np.mean(ax.get_xticks()),np.max(ax.get_yticks())+Y/10.0),
            'south': (np.mean(ax.get_xticks()),np.min(ax.get_yticks())-Y/10.0),
            'east': (np.max(ax.get_xticks())+X/10.0,np.mean(ax.get_yticks())),
            'west': (np.min(ax.get_xticks())-X/10.0,np.mean(ax.get_yticks())),
            'northeast':(np.max(ax.get_xticks())+X/10.0,
                         np.max(ax.get_yticks()) + Y/10.0),
            'northwest':(np.min(ax.get_xticks())-X/10.0,
                         np.max(ax.get_yticks()) + Y/10.0),
            'southeast':(np.max(ax.get_xticks())+X/10.0,
                         np.min(ax.get_yticks()) - Y/10.0),
            'southwest':(np.max(ax.get_xticks())-X/10.0,
                         np.min(ax.get_yticks()) - Y/10.0)
            }.get(position,ValueError('Unrecognized position %s'%position))
            if isinstance(xytext, Exception):
                raise(xytext)
        else: # assume numeric
            xytext = position
        
        if fontsize is None:
            fontsize = ax.yaxis.get_major_ticks()[2].label.get_fontsize()
            
        def calloffset(offset, ind): # xy offset modification 
            if offset is not None:
                if '+' in offset:
                    xytext[ind]+=offset.count('+')*self.xydotsize(ax,
                                                s=fontsize,scale=(1.,1.))[ind]           
                elif '-' in offset:
                    xytext[ind]-=offset.count('-')*self.xydotsize(ax, 
                                                s=fontsize,scale=(1.,1.))[ind]
                else:
                    try:
                        xytext[ind] += float(offset)
                    except:
                        pass
            return(xytext)
                
        xytext = calloffset(xoffset, 0)
        xytext = calloffset(yoffset, 1)

        if color is None:
            color = ax.get_lines()[0]
        if 'matplotlib.lines.Line2D' in str(type(color)):
            color = color.get_color()
               
        txt = ax.text(xytext[0], xytext[1], text,color=color, size=fontsize,
                      **kwargs)
        self.AdjustText(txt, ax=ax)

    def AnnotateOnGroup(self, m, text='*', vpos=None):
        """Help annotate statistical significance over each group in Beeswarm /
        bar graph. Annotate several groups at a time.
        m: indices of the group. This is required.
        text: text annotation above the group. Default is an asterisk '*'
        vpos: y value of the text annotation, where text is positioned. Default
              is calculated by this method. For aesthetic reason, vpos will be
              applied to all groups specified in m
        """
        # Calculate default value of vpos
        if vpos is None:
            I = self.axs.get_yticks()
            yinc = I[1]-I[0] # y tick increment
            Y = [max(x) for x in self.data.series['y']]
            vpos = max([Y[k] for k in m])+yinc/5.0
        X = self.axs.get_xticks()
        for k in m:
            txt = self.axs.text(X[k], vpos, text, ha='center',va='top')
            # adjust text so that it is not overlapping with data or title
            self.AdjustText(txt)

    def AnnotateBetweenGroups(self, m=0, n=1, text='*', hgap=0):
        """Help annotate statistical significance between two groups in the
        Beeswarm / bar plot. Annotate one pair at a time.
            m, n indicates the index of the loci to annotate between.
            By default, m=0 (first category), and n=1 (second category)
        text: annotation text above the bracket between the two loci.
            Default is an asterisk "*" to indicate significance.
        hgap: horizontal gap between neighboring annotations. Default is 0.
            No gap will be added at m=0 or n=1
        All annotations are stored in self.axs.texts, a list of text objects.
        """
        # Calculate Locus Position
        X = self.axs.get_xticks()
        I = self.axs.get_yticks()
        Y = [max(x) for x in self.data.table[_y]]
        yinc = I[1]-I[0] # y tick increment
        ytop = max(I) + yinc/10.0 # top of the annotation
        yoffset = (ytop-max(Y[m],Y[n]))/2.0
        # position of annotation bracket
        xa, xb = X[m]+hgap*int(m!=0), X[n]-hgap*int(n!=max(X))
        ya, yb = yoffset + Y[m], yoffset + Y[n]
        # position of annotation text
        xtext, ytext = (X[m]+X[n])/2.0, ytop+yinc/10.0
        # Draw text
        txt = self.axs.text(xtext,ytext, text, ha='center',va='bottom')
        # adjust text so that it is not overlapping with data or title
        self.AdjustText(txt)
        # Draw Bracket
        self.axs.annotate("", xy=(xa,ya), xycoords='data',
                          xytext=(xa,ytop), textcoords = 'data',
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0))
        self.axs.annotate("", xy=(xa,ytop), xycoords='data',
                          xytext=(xb,ytop), textcoords = 'data',
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0))
        self.axs.annotate("", xy=(xb,ytop), xycoords='data',
                          xytext=(xb,yb), textcoords = 'data',
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0))

    def AdjustText(self, txt, ax=None):
        """Adjust text so that it is not being cutoff"""
        #renderer = self.axs.get_renderer_cache()
        if ax is None: ax = self.axs
        txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=1'))
        plt.draw() # update the text draw
        txtbb = txt.get_bbox_patch().get_window_extent() # can specify render
        xmin, ymin, xmax, ymax = tuple(ax.transData.inverted().
                                        transform(txtbb).ravel())
        xbnd, ybnd = ax.get_xbound(), ax.get_ybound()
        if xmax > xbnd[-1]:
            ax.set_xbound(xbnd[0], xmax)
        if xmin < xbnd[0]:
            ax.set_xbound(xmin, xbnd[-1])
        if ymax > ybnd[-1]:
            ax.set_ybound(ybnd[0], ymax)
        if ymin < ybnd[0]:
            ax.set_ybound(ymin, ybnd[-1])

    def RemoveAnnotation(self):
        """Remove all annotation and start over"""
        self.axs.texts = []


    """ ################# Geometry Annotations ####################### """
    def DrawEllipsoid(self, center, radii, rvec=np.eye(3), \
                      numgrid=100, ax=None, color=color, alpha=0.6):
        """Draw an ellipsoid given its parameters
        center: center [x0,y0,z0]
        radii: radii of the ellipsoid [rx, ry, rz]
        rvec: vector of the radii that indicates orientation. Default identity
        numgrid: number of points to estimate the ellipsoid. The higher the
            number, the smoother the plot. Defualt 100.
        """
        # Caculate ellipsoid coordinates
        from simple.ellipsoid import Ellipsoid as Ellipsoid
        x,y,z = Ellipsoid(center, radii, rvec, numgrid)
        if ax is None:
            if not (isinstance(self.axs, np.ndarray) or \
                    isinstance(self.axs, list)):
                ax = self.axs # only 1 axis
            else:
                return
        ax.plot_surface(x,y,z,rstride=4,cstride=4,linewidth=0,\
                              alpha=alpha,color=color[self.cache%len(color)])
        self.cache += 1 # increase color cache index by 1

    def DrawRect(self, x,y,w,h, ax=None, color=color, alpha=0.6):
        """Draw a rectangular bar
        x,y,w,h: xcenter, ycenter, width, height
        """
        from matplotlib.patches import Rectangle
        if ax is None:
            if not (isinstance(self.axs, np.ndarray) or \
                    isinstance(self.axs, list)):
                ax = self.axs # only 1 axis
            else:
                return
        ax.add_patch(Rectangle((x-w/2.0, y-h/2.0), w, h, angle=0.0, \
                            facecolor=color[self.cache%len(color)]))
        self.caceh += 1 # increase color cache index by 1
        # Send the patch to the background, but right above the previous patch
        self.SetZOrder(style='overlay')

    def SetZOrder(self, plotobj=None, style=None, order=None):
        """Organizing layers of plot. This is helpful when exporting to .eps"""
        raise(NotImplementedError('This mehtod is not implemented'))
        if style == 'back': # send the layer all the way back
            None
        elif style == 'front': # send the layer all the way front
            None
        elif style == 'overlay':
            # First identify the same type of objects starting from the bottom,
            # then put the current object on top of the top-most found object
            patch_order = 0
            self.SetZOrder(plotobj, style=None, order=patch_order)
        if order>0: # send the plot object 1 layer forward
            None
        elif order<0: # send the plot object 1 layer backward
            None

    """ ####################### Misc ####################### """
    @staticmethod
    def roundto125(x, r=np.array([1,2,5,10])): # helper static function
        """5ms, 10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s, etc.
        5mV, 10mV, 20mV, etc.
        5pA, 10pA, 20pA, 50pA, etc."""
        p = int(np.floor(np.log10(x))) # power of 10
        y = r[(np.abs(r-x/(10**p))).argmin()] # find closest value
        return(y*(10**p))

    @staticmethod
    def get_field(struct, *args): # layered /serial indexing
        try:
            for m in args:
                struct = struct[m]
            return(struct)
        except:
            return(None)

    @staticmethod
    def ind2sub(ind, size, order='C'):
        """MATLAB's ind2usb
        order: in 'C' order by default"""
        return(np.unravel_index(ind, size,order=order))

    @staticmethod
    def sub2ind(sub, size, order='C'):
        """MATLAB's sub2ind
        order: in 'C' order by default"""
        return(np.ravel_multi_index(sub, dims=size, order=order))
        
    @staticmethod
    def xydotsize(ax, s=None, dpi=None, scale=(1.25,1.25)):
        """ Determine dot size in data axis.
        scale: helps further increasing space between dots
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


    def SetFont(self, fontsize=fontsize,fontname=fontname,items=None,ax=None,
                fig=None):
        """Change font properties of all axes
        fontsize: size of the font, specified in the global variable
        fontname: fullpath of the font, specified in the global variable
        items: select a list of items to change font. ['title', 'xlab','ylab',
               'xtick','ytick', 'texts','legend','legendtitle']
        ax: which axis or axes to change the font. Default all axis in current
            instance. To skip axis, input as [].
        fig: figure handle to change the font (text in figure, not in axis).
        Default is any text items in current instance. To skip, input as [].
        """
        if (fontname is None) and (fontsize is None):
            return
        import matplotlib.font_manager as fm
        
        if ax is None:
            ax = self.axs
            
        if fig is None:
            fig = self.fig

        def get_ax_items(ax):
            """Parse axis items"""
            itemDict={'title':[ax.title], 'xlab':[ax.xaxis.label],
                    'ylab':[ax.yaxis.label], 'xtick':ax.get_xticklabels(),
                    'ytick':ax.get_yticklabels(),
                    'texts':ax.texts if isinstance(ax.texts,(np.ndarray,list))
                                         else [ax.texts],
                    'legend': [] if ax.legend_ is None
                                        else ax.legend_.get_texts(),
                    'legendtitle':[] if ax.legend_ is None
                                        else [ax.legend_.get_title()]}
            itemList, keyList = [], []
            if items is None: # get all items
                for k, v in iter(itemDict.items()):
                    itemList += v
                    keyList += [k]*len(v)
            else: # get only specified item
                for k in items:
                    itemList += itemDict[k] # add only specified in items
                    keyList += [k]*len(itemDict[k])
            
            return(itemList, keyList)
            
        def get_fig_items(fig):
            """Parse figure text items"""
            itemList = fig.texts if isinstance(fig.texts,(np.ndarray,list)) \
                                    else [fig.texts]
            keyList = ['texts'] * len(itemList)
            
            return(itemList, keyList)
                 
        def CF(itemList, keyList):
            """Change font given item"""
            # initialize fontprop object
            fontprop = fm.FontProperties(style='normal', weight='normal',
                                         stretch = 'normal')
            if os.path.isfile(fontname): # check if font is a file
                fontprop.set_file(fontname)
            else:# check if the name of font is available in the system
                if not any([fontname.lower() in a.lower() for a in
                        fm.findSystemFonts(fontpaths=None, fontext='ttf')]):
                     print('Cannot find specified font: %s' %(fontname))
                fontprop.set_family(fontname) # set font name
            # set font for each object
            for n, item in enumerate(itemList):
                if isinstance(fontsize, dict):
                    fontprop.set_size(fontsize[keyList[n]])
                elif n <1: # set the properties only once
                    fontprop.set_size(fontsize)
                item.set_fontproperties(fontprop) # change font for all items
            
        def CF_ax(ax): # combine CF and get_ax_items
            if not ax: # true when empty or None
                return # skip axis font change
            itemList, keyList = get_ax_items(ax)
            CF(itemList, keyList)
            
        def CF_fig(fig): # combine CF and get_fig_items
            if not fig: # true when empty or None
                return # skip figure font change
            itemsList, keyList = get_fig_items(fig)
            CF(itemsList, keyList)
        
        # vecotirze the closure
        CF_ax_vec = np.frompyfunc(CF_ax, 1,1)
        CF_fig_vec = np.frompyfunc(CF_fig, 1,1)
        
        # Do the actual font change
        CF_ax_vec(ax)
        CF_fig_vec(fig)

if __name__ == "__main__":
    dataFile = os.path.join(exampleFolder, '%s.csv' %plotType)
    # Load data
    if plotType != 'neuro':
        K = PublicationFigures(dataFile=dataFile, savePath=os.path.join(exampleFolder,'%s.png'%plotType))
    if plotType == 'lineplot':
        # Line plot example
        K.LinePlot(style=style)
        #K.axs[0].set_ylim([0.5,1.5])
        #K.axs[1].set_ylim([0.05, 0.25])
    elif plotType == 'boxplot':
        # boxplot example
        K.Boxplot()
    elif plotType == 'beeswarm':
        # Beeswarm example
        K.Beeswarm()
        #K.AnnotateOnGroup(m=[0,1])
        #K.AnnotateBetweenGroups(text='p=0.01234')
    elif plotType == 'trace':
        # Time series example
        K.Traces()
    elif plotType == 'barplot':
        K.BarPlot(style='Vertical')
    elif plotType == 'scatter':
        K.Scatter()
    elif plotType == 'scatter3d':
        K.Scatter3D()
    elif plotType == 'boxplot':
        K.Boxplot()
    elif plotType == 'neuro':
        #base_dir = 'D:/Data/2015/08.August/Data 25 Aug 2015/Neocortex B.25Aug15.S1.E%d.dat'
        #eps = [14, 22, 29, 39] # AHP
        #data = [base_dir%(epi) for epi in eps]
        #K = PublicationFigures(dataFile=data, savePath=os.path.join(exampleFolder,'multiple_traces.png'), old=True, channels=['A'], streams=['Volt'])
        #K.Traces(outline='overlap')
    
        data = 'D:/Data/2015/07.July/Data 10 Jul 2015/Neocortex K.10Jul15.S1.E38.dat'
        K = PublicationFigures(dataFile=data, savePath=os.path.join(exampleFolder,'single_episode_traces.png'), old=True, channels=['A'], streams=['Volt', 'Cur'])
        K.SingleEpisodeTraces(K.data.table, notes=K.data.meta['notes'][0], channels=['A'], streams=['Volt','Cur'])

    # Final clean up
    #K.fig.show()
    K.Save()