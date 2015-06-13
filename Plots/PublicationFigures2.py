# -*- coding: utf-8 -*-
"""
Created on Sat Jun 06 13:35:08 2015

@author: Edward
"""
import os
import numpy as np
#import matplotlib
#matplotlib.use('Agg') # use 'Agg' backend
import matplotlib.pyplot as plt
from beeswarm import *

#sys.path.append('C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots')

#dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/beeswarm.txt'
dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/lineplot.txt'

class FigureData(object):
    """Parse input text file data for figures
    """
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = {'x':[],'y':[],'z':[]} 
        self.stats = {'x':{},'y':{},'z':{}}
        self.names = {'x':[],'y':[],'z':[]}
        self.num = {'x':[],'y':[],'z':[]} # count number of data sets
        if dataFile is not None and isinstance(dataFile, str):
            self.LoadData(dataFile)

    def LoadData(self, dataFile):
        """Load data in text file"""
        with open(dataFile, 'rb') as fid:
            for line in fid: # iterate each line
                if line[0] == "#":
                    continue  # skip comments
                # split comma delimited string
                # series code, series name,@datatype, data1, data2, data3, ...
                lst = [s.strip() for s in line.split(',')]
                # Parse variable
                v = lst[0][0] # variable name
                stats = lst[0][1:-1]
                # Read the data
                seriesData = self.ReadData(lst[1][1:], lst[3:])
                # Organize the data to structure
                if stats != "": #stats, not empty
                    if stats in self.stats[v].keys(): # key exists already
                        self.stats[v][stats].append(seriesData)
                    else: # add new key / create new list
                        self.stats[v][stats] = [seriesData]
                else: # series data
                    self.series[v].append(seriesData)
                    self.names[v].append(lst[2][1:-1])

            fid.close()
            # Parse number of data set
            for v in self.series.keys():
                self.num[v] = len(self.series[v])

    @staticmethod
    def ReadData(valueType, seriesList):
        if valueType == 'str':
            return(np.array(seriesList))
        elif valueType == 'float':
            return(np.array(seriesList).astype(np.float))
        elif valueType == 'int':
            return(np.array(seriesList).astype(np.int))
        else: # unrecognized type
            BaseException('Unrecognized data type')    

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
        elif isinstance(dataFile, FigureData):
            self.data = dataFile
        self.SavePath = SavePath
        
    def LoadData(self, dataFile):
        """To be called after object creation"""
        self.data = FigureData(dataFile)
        
    def Save(self, SavePath=None):
        if SavePath is not None: # overwrite with new savepath
            self.SavePath = SavePath
        if self.SavePath is None: # save to current working directory
            self.SavePath = os.path.join(os.getcwd(),'Figure.eps')
        self.fig.savefig(self.SavePath)
        
    def TimeSeries(self, Style='Vstack'):
        """Plot time series / voltage and current traces"""
        return
       
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
        self.SetLinePlotStyle()
  
    def LinePlotTwin(self):
        """ Line plots with 2 y-axis"""
        self.x = range(1,len(self.data.series['x'][0])+1)
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        self.axs = [self.axs, self.axs.twinx()]
        colors = ('k','r')
        spineName = ('left','right')
        for n, ax in enumerate(self.axs):
             # Plot error bar
            ax.errorbar(self.x,self.data.series['y'][n],
                yerr = [self.data.stats['y']['ebp'][n],
                self.data.stats['y']['ebn'][n]], color=colors[n])
            # For twin Plot
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.label.set_color(colors[n])
            ax.tick_params(axis='y',colors=colors[n])
            ax.spines[spineName[n]].set_color(colors[n])
        self.axs[0].set_xlabel(self.data.names['x'][0]) # x label

    def LinePlotVstack(self):
        """ Line plots stacked vertically"""
        self.x = range(1,len(self.data.series['x'][0])+1)
        self.fig, self.axs = plt.subplots(nrows=self.data.num['y'],ncols=1,
                                          sharex=True)
        for n, ax in enumerate(self.axs):
            # Plot error bar
            ax.errorbar(self.x,self.data.series['y'][n],
                yerr = [self.data.stats['y']['ebp'][n],
                self.data.stats['y']['ebn'][n]], color='k')
            # For Vstack
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_visible(False)
            if n < (self.data.num['y']-1): # first several plots
                ax.xaxis.set_ticks_position('none')
                ax.spines['bottom'].set_visible(False)
            else: # last plot
                ax.xaxis.set_ticks_position('bottom')
        self.axs[-1].set_xlabel(self.data.names['x'][0]) # x label
        
    """ ####################### Axis schemas ####################### """
    def SetLinePlotStyle(self):
        for n, ax in enumerate(self.axs):
            # Set ylabel
            ax.set_ylabel(self.data.names['y'][n])
            # make sure axis tickmark points out
            ax.tick_params(axis='both',direction='out')
            # Set axis visibility
            ax.spines['top'].set_visible(False)
        # change the x lim on the last, most buttom subplot
        ax.set_xlim([0,len(self.data.series['x'][0])+1])
        plt.xticks(self.x, self.data.series['x'][0])
        # Add some margins to the plot so that it is not touching the axes
        plt.margins(0.02,0.02)
        self.fig.tight_layout() # enforce tight layout
        
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
            ax.xaxis.set_ticks_position('none')
            ax.yaxis.set_ticks_position('none')
        TOA_vec = np.vectorize(TOA) # vectorize the closure
        TOA_vec(self.axs)
    
    """ ####################### Annotations ####################### """
    def AnnotateOnGroup(self, m, text='*', vpos=None):
        """Help annotate statistical significance over each group in Beeswarm /
        bar graph. Annotate several groups at a time.
        m: indices of the group. This is required.
        text: text annotation above the group. Default is an asterisk '*'
        vpos: y value of the text annotation, where text is positioned. Default
              is calculated by this method. For aesthetic reason, vpos will be
              applied to all groups specified in i
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
            
if __name__ == "__main__":
    # Load data
    K = PublicationFigures(dataFile=dataFile, SavePath='C:/QQDownload/asdf_beeswarm.png')
    # Beeswarm example
    #K.Beeswarm()
    #K.AnnotateOnGroup(m=[0,1])
    #K.AnnotateBetweenGroups()
    #K.axs.set_ylim([-3,7])
    
    # Line plot example
    K.LinePlot()
    #K.axs[0].set_ylim([0,2.0])
    #K.axs[1].set_ylim([0.05, 0.25])

