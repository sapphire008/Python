# -*- coding: utf-8 -*-
"""
Created on Sat Jun 06 13:35:08 2015

@author: Edward
"""
DEBUG = True



import os
import numpy as np
from beeswarm import beeswarm
from ImportData import NeuroData

import matplotlib.pyplot as plt

plot_type = 'lineplot'
exampleFolder = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/example/'


class PublicationFigures(object):
    """Generate publicatino quantlity figures
        Data: FigureData, or data file path
        PlotType: currently supported plot types include:
            ~ LinePlot: for categorical data, with error bar
                Style:  
                    'Twin' -- Same plot, 2 y-axis (left and right of plot)
                    'Vstacked' (default) -- vertically stacked subplots
            ~ Beeswarm: beeswarm plot; boxplot with scatter points
                Style: 'hex','swarm' (default),'center','square'
    """
    def __init__(self, dataFile=None, SavePath=None):
        """Initialize class        
        """
        if isinstance(dataFile, str):
            self.LoadData(dataFile) # load data
        elif isinstance(dataFile, NeuroData):
            self.data = dataFile
        self.SavePath = SavePath
        # Set basic plot properties
        
    def LoadData(self, dataFile):
        """To be called after object creation"""
        self.data = NeuroData(dataFile)
        
    def Save(self, SavePath=None):
        if SavePath is not None: # overwrite with new savepath
            self.SavePath = SavePath
        if self.SavePath is None: # save to current working directory
            self.SavePath = os.path.join(os.getcwd(),'Figure.eps')
        self.fig.savefig(self.SavePath)
        
    """ ####################### Plot utilities ####################### """      
    def Traces(self, groupings=None, scalebar=True, annotation=None, 
               color=('#000000','#FF9966','#3399FF','#FF0000','#00FF99',
               '#FF33CC','#CC99FF')):
        """Plot time series / voltage and current traces
        groupings: grouping of y data. E.g [[1,2],[3]] will result two
        subplots, where y1 and y2 are in the same subplot above, and y3 below.
        color: default (black, orange, blue, red, green, magenta, purple)
                
        """
        m = 0 # row, indexing y axis data
        n = 0 # column, indexing x axis or time data
        c = 0 # indexing color cycle or traces in a subplot
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        hline = plt.plot(self.data.series['x'][m], self.data.series['y'][n], color=color[c%len(color)])
        # set aspect ratio
        self.SetAspectRatio(2,1)
        if scalebar: # Use scale bar instead of axis
            self.AddTraceScaleBar(hline[0], xunit=self.data.names['x'][m], 
                                  yunit=self.data.names['y'][m]) 
        else:
            self.SetDefaultAxis() # use default axis
        if annotation:
            self.TextAnnotation(text=annotation) # description of the trace
        
       
    def Histogram(self):
        """Plot histogram"""
        return
        
    def Beeswarm(self, Style='swarm'):
        """Beeswarm style boxplot"""
        colors = ['red','cyan','green','magenta','blue','black']
        # boardcasting color cycle
        num = self.data.num['x']
        colors = num/len(colors)*colors+colors[0:num%len(colors)]
        self.bs, self.axs = beeswarm(self.data.series['x'], method=Style, 
                                     labels=self.data.names['x'],col=colors)
        # Format style
        # make sure axis tickmark points out
        self.axs.tick_params(axis='both',direction='out')
        self.axs.spines['right'].set_visible(False)
        self.axs.spines['top'].set_visible(False)
        self.axs.xaxis.set_ticks_position('bottom')
        self.axs.yaxis.set_ticks_position('left')
        # Set Y label, if exsit
        try:
            self.axs.set_ylabel(self.data.names['y'][0])
        except:
            pass
        # save current figure handle
        self.fig = plt.gcf()
     
    def LinePlot(self, Style='Vstack'):
        if Style == "Vstack":
            self.LinePlotVstack()
        elif Style == "Twin":
            self.LinePlotTwin()
        self.AdjustCategoricalXAxis() # make some space for each category
  
    def LinePlotTwin(self, colors=('k','r')):
        """ Line plots with 2 y-axis"""
        self.x = range(1,len(self.data.series['x'][0])+1)
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True, figsize=(10,5))
        self.axs = [self.axs, self.axs.twinx()]
        for n, ax in enumerate(self.axs):
             # Plot error bar
            ax.errorbar(self.x,self.data.series['y'][n],
                yerr = [self.data.stats['y']['ebp'][n],
                self.data.stats['y']['ebn'][n]], color=colors[n])
        self.SetTwinPlotAxis(colors = colors) # set twin plot subplot axes

    def LinePlotVstack(self):
        """ Line plots stacked vertically"""
        self.x = range(1,len(self.data.series['x'][0])+1)
        self.fig, self.axs = plt.subplots(nrows=self.data.num['y'],ncols=1,
                                          sharex=True, figsize=(10, 5))
        for n, ax in enumerate(self.axs):
            # Plot error bar
            ax.errorbar(self.x,self.data.series['y'][n],
                yerr = [self.data.stats['y']['ebp'][n],
                self.data.stats['y']['ebn'][n]], color='k')
        self.SetVstackAxis() # set vertical stacked subplot axes
        
    """ ####################### Axis schemas ####################### """
    def SetDefaultAxis(self):
        """Set default axis appearance"""
        def SDA(ax): # short for set default axis
            ax.tick_params(axis='both',direction='out')
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(True)
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.set_ticks_position('left')
        SDA_vec = np.vectorize(SDA) # vectorize the closure
        SDA_vec(self.axs)
        
    def TurnOffAxis(self):
        """Turn off all axis"""
        def TOA(ax): # short for turn off axis
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
        TOA_vec = np.vectorize(TOA) # vectorize the closure
        TOA_vec(self.axs)
        
    def SetTwinPlotAxis(self, colors=('k', 'r')):
        spineName = ('left','right')
        for n, ax in enumerate(self.axs):             # For twin Plot
            ax.tick_params(axis='both',direction='out') # tick mark out
            ax.spines['top'].set_visible(False) # remove top boundary
            ax.xaxis.set_ticks_position('bottom') # keep only bottom ticks
            ax.set_ylabel(self.data.names['y'][n]) # set y label
            ax.yaxis.label.set_color(colors[n]) # set y label color
            ax.tick_params(axis='y',colors=colors[n]) # set y tick color
            ax.spines[spineName[n]].set_color(colors[n]) # set y spine color
        self.axs[0].set_xlabel(self.data.names['x'][0]) # x label
        
    def SetVstackAxis(self):
        for n, ax in enumerate(self.axs):             # For Vstack
            ax.tick_params(axis='both', direction='out') #tick mark out
            ax.spines['top'].set_visible(False) # remove top boundary
            ax.spines['right'].set_visible(False) # remove right spine
            ax.yaxis.set_ticks_position('left') # keep only left ticks
            ax.set_ylabel(self.data.names['y'][n]) # set y label
            if ax.is_last_row():     #keep only bottom ticks       
                ax.xaxis.set_ticks_position('bottom') 
                ax.set_xlabel(self.data.names['x'][0]) # x label
            else:
                ax.xaxis.set_visible(False)
                ax.spines['bottom'].set_visible(False)                   
                
    def AdjustCategoricalXAxis(self): # additional for plots with categorical data
        # change the x lim on the last, most buttom subplot
        self.axs[-1].set_xlim([0,len(self.data.series['x'][0])+1])
        plt.xticks(self.x, self.data.series['x'][0])
        # Add some margins to the plot so that it is not touching the axes
        plt.margins(0.025,0.025)
        self.fig.tight_layout() # enforce tight layout
        
    def SetDiscontinousAxis(self, x=None, y=None):
        """Plot with discontious axis. Allows one discontinuity for each axis.
        Assume there is only 1 plot in the figure
        x: x break point
        y: y break point
        """
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
        
    def SetAspectRatio(self, r=2, adjustable='box-forced'):
        def SAR(ax):
            X, Y = np.ptp(ax.get_xticks()), np.ptp(ax.get_yticks()) 
            ax.set_aspect(X/Y/r, adjustable=adjustable)
        SAR_vec = np.vectorize(SAR) # vectorize the closure
        SAR_vec(self.axs)
        
        
    """ ####################### Annotations ####################### """
    def AddTraceScaleBar(self, hline, xunit, yunit, color=None):
        def roundto125(x): # helper static function
            """5ms, 10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s, etc.
            5mV, 10mV, 20mV, etc.
            5pA, 10pA, 20pA, 50pA, etc."""
            r = np.array([1,2,5,10])
            x = int(x)/5
            p = int(np.log10(x)) # power of 10
            y = r[(np.abs(r-x/(10**p))).argmin()] # find closest value
            return(y*(10**p))
        def scalebarlabel(x, unitstr):
            if unitstr.lower()[0] == 'm':
                return(str(x)+unitstr if x<1000 else str(x/1000)+
                    unitstr.replace('m',''))
            elif unitstr.lower()[0] == 'p':
                return(str(x)+unitstr if x<1000 else str(x/1000)+
                    unitstr.replace('p','n'))

        self.TurnOffAxis() # turn off axis
        X, Y = np.ptp(self.axs.get_xticks()), np.ptp(self.axs.get_yticks())
        # calculate scale bar unit length
        X, Y = roundto125(X), roundto125(Y)
        # Parse scale bar labels
        xlab, ylab = scalebarlabel(X, xunit), scalebarlabel(Y, yunit)
        # Get color of the scalebar
        color = plt.getp(hline,'color')
        # Calculate position of the scale bar
        xi = np.max(self.axs.get_xticks()) + X/10.0
        yi = np.mean(self.axs.get_yticks())
        # calculate position of text
        xtext1, ytext1 = xi+X/2.0, yi-Y/10.0 # horizontal
        xtext2, ytext2 = xi+X+X/10.0, yi+Y/2.0 # vertical
        # Draw text
        txt1 = self.axs.text(xtext1, ytext1, xlab, ha='center',va='top', color=color)
        self.AdjustText(txt1)
        txt2 = self.axs.text(xtext2, ytext2, ylab, ha='left',va='center', color=color)
        self.AdjustText(txt2)
        # Draw Scale bar
        self.axs.annotate("", xy=(xi,yi), xycoords='data',  # horizontal
                          xytext=(xi+X,yi), textcoords = 'data', 
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0, color=color))
        self.axs.annotate("", xy=(xi+X,yi), xycoords='data',  # vertical
                          xytext=(xi+X,yi+Y), textcoords = 'data', 
                          annotation_clip=False,arrowprops=dict(arrowstyle="-",
                          connectionstyle="arc3", shrinkA=0, shrinkB=0, color=color))
        
    def TextAnnotation(self, text="", position='south'):
        return
    
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
            Y = [max(x) for x in self.data.series['x']]
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
        Y = [max(x) for x in self.data.series['x']]
        yinc = I[1]-I[0] # y tick increment
        ytop = max(I) + yinc/10.0 # top of the annotation
        yoffset = (ytop-max(Y[m],Y[n]))/2.0
        # position of annotation bracket
        xa, xb = X[m]+hgap*int(m!=0), X[n]-hgap*int(n!=max(X)) 
        ya, yb = yoffset + Y[m], yoffset + Y[n] 
        # position of annotation text
        xtext, ytext = (X[m]+X[n])/2.0, ytop+yinc/10.0
        # Draw text
        txt = self.axs.text(xtext,ytext, text, ha='center',va='top')
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
    def AdjustText(self,txt):
        """Adjust text so that it is not overlapping with data or title"""
        txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=1'))
        txtbb = txt.get_bbox_patch().get_window_extent()
        ymax = self.axs.transData.inverted().transform(txtbb).ravel()[-1]
        ybnd = self.axs.get_ybound()
        if ymax > ybnd[-1]:
            self.axs.set_ybound(ybnd[0], ymax)
            
    def RemoveAnnotation(self):
        """Remove all annotation and start over"""
        self.axs.texts = []
    
    def CleanUpFont(self):
        # clean up font 
        return
        

if __name__ == "__main__":
    dataFile = os.path.join(exampleFolder, '%s.txt' %plotType)
    # Load data
    K = PublicationFigures(dataFile=dataFile, SavePath=os.path.join(exampleFolder,'%s.png'%plotType))

    # Line plot example
    K.LinePlot(Style='Twin')
    #K.SetAspectRatio(2)
    K.axs[0].set_ylim([0.5,1.5])
    K.axs[1].set_ylim([0.05, 0.25])
    
    # Beeswarm example
    #K.Beeswarm()
    #K.AnnotateOnGroup(m=[0,1])
    #K.AnnotateBetweenGroups()
    
    # Time series example
    #K.Traces()
