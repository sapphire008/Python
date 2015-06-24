# -*- coding: utf-8 -*-
"""
Created on Sat Jun 06 13:35:08 2015

@author: Edward
"""
import re
import numpy as np
#import matplotlib
#matplotlib.use('Agg') # use 'Agg' backend
import matplotlib.pyplot as plt
from beeswarm import *

dataFile = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/beeswarm.txt'

class FigureData(object):
    """Parse input text file data for figures
    """
    def __init__(self, dataFile=None):
        """Initialize class"""
        self.series = dict()
        self.names = dict()
        self.num = []# count number of data sets
        if dataFile is not None and isinstance(dataFile, str):
            self.loadData(dataFile)

    def loadData(self, dataFile):
        """Load data in text file"""
        with open(dataFile, 'rb') as fid:
            for line in fid: # iterate each line
                if line[0] == "#":
                    continue  # skip comments
                # split comma delimited string
                # series code, series name,@datatype, data1, data2, data3, ...
                lst = line.strip().split(',')
                # Parse data series name
                self.names[lst[1]] = lst[2][1:-1]
                # Parse data
                if lst[0][1:] == 'str':
                    self.series[lst[1]] = np.array(lst[3:])
                elif lst[0][1:] == 'float':
                    self.series[lst[1]] = np.array(lst[3:]).astype(np.float)
                elif lst[0][1:] == 'int':
                    self.series[lst[1]] = np.array(lst[3:]).astype(np.int)
                else: # unrecognized type
                    BaseException('Unrecognized data type')
            fid.close()
            # Parse number of data set
            for x in self.series.keys():
                self.num.extend(map(int, re.findall(r'\d+',x)))
            self.num = max(self.num)


class PublicationFigures(FigureData):
    """Generate publicatino quantlity figures
        Data: FigureData, or data file path
        PlotType: currently supported plot types include:
            ~ LinePlot: for categorical data, with error bar
                Style:  
                    'Twin' -- Same plot, 2 y-axis (left and right of plot)
                    'Vstacked' (default) -- vertically stacked subplots
            ~ Beeswarm: beeswarm plot; boxplot with scatter points
                Style: 'hex','swarm' (default),'center','square'
                    '
    """
    def __init__(self, Data=None, PlotType=None, Style=None, SavePath=None):
        """Initialize class        
        """
        if Data is None:
            return
        elif isinstance(Data, str):
            self.data = FigureData(Data) # load data
        elif isinstance(Data, FigureData):
            self.data = Data
        self.PlotType = PlotType
        self.Style = Style
        self.SavePath = SavePath
        # Do Plots
        self.DoPlots()
        
    def DoPlots(self):
        # Switch between plots
        if self.PlotType is None:
            return
        if self.PlotType == 'LinePlot':
            if self.Style is None:
                self.Style = 'Vstack'
            if self.Style == 'Twin':
                self.LinePlotTwin()
            elif self.Style == 'Vstack':
                self.LinePlotVstack()
            # Fix style
            self.SetLinePlotStyle()
        elif self.PlotType == 'Beeswarm':
            if self.Style is None:
                self.Style = 'swarm'
            self.Beeswarm()
        else: # unrecognized plot type
            BaseException('Unsupported Plot Type')
        #self.fig.show()
        if self.SavePath is not None:
            self.fig.savefig(self.SavePath)
       
    def Beeswarm(self):
        colors = ['red','cyan','green','magenta','blue','black']
        # boardcasting color cycle
        colors = self.data.num/len(colors)*colors+colors[0:self.data.num%len(colors)]
        # Data
        datavect = []
        [datavect.append(self.data.series['y'+str(n+1)]) for n in range(self.data.num)]
        print(datavect)
        # names
        datanames = []
        [datanames.append(self.data.names['y'+str(n+1)]) for n in range(self.data.num)]
        self.bs, ax = beeswarm(datavect, method=self.Style, 
                                     labels=datanames, 
                                     col=colors)
        # Format style
        # make sure axis tickmark points out
        ax.tick_params(axis='both',direction='out')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        # save current figure handle
        self.fig = plt.gcf()
        self.axs = ax
        # Do annotation: compare significance
        X = ax.get_xticks()
        Y = [max(x) for x in datavect]
        self.label_diff(0,1,'p=0.0370',X,Y)
        self.label_diff(1,2,'p<0.0001',X,Y)

    def label_diff(self, i,j,text,X,Y):
        # Custom function to draw the diff bars
        x = (X[i]+X[j])/2
        y = 1.1*max(Y[i], Y[j])
        dx = abs(X[i]-X[j])

        props = {'connectionstyle':'bar','arrowstyle':'-',\
                 'shrinkA':20,'shrinkB':20,'lw':2}
        self.axs.annotate(text, xy=(X[i],y+7), zorder=10)
        self.axs.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)


        
    def LinePlotTwin(self):
        self.x = range(1,len(self.data.series['x1'])+1)
        self.fig, self.axs = plt.subplots(nrows=1,ncols=1, sharex=True)
        self.axs = [self.axs, self.axs.twinx()]
        colors = ('k','r')
        spineName = ('left','right')
        for n, ax in enumerate(self.axs):
            # Plot error bar
            ax.errorbar(self.x,self.data.series['y'+str(n+1)],
                yerr = [self.data.series['yebp'+str(n+1)],
                self.data.series['yebn'+str(n+1)]], color=colors[n])
            # For twin Plot
            ax.xaxis.set_ticks_position('bottom')
            ax.yaxis.label.set_color(colors[n])
            ax.tick_params(axis='y',colors=colors[n])
            ax.spines[spineName[n]].set_color(colors[n])
        self.axs[0].set_xlabel(self.data.names['x1'])

    def LinePlotVstack(self):
        self.x = range(1,len(self.data.series['x1'])+1)
        self.fig, self.axs = plt.subplots(nrows=self.data.num, ncols=1,sharex=True)
        for n, ax in enumerate(self.axs):
            # Plot error bar
            ax.errorbar(self.x,self.data.series['y'+str(n+1)],
                yerr = [self.data.series['yebp'+str(n+1)],
                self.data.series['yebn'+str(n+1)]], color='k')
            # For Vstack
            ax.yaxis.set_ticks_position('left')
            ax.spines['right'].set_visible(False)
            if n < (self.data.num-1): # first several plots
                ax.xaxis.set_ticks_position('none')
                ax.spines['bottom'].set_visible(False)
            else: # last plot
                ax.xaxis.set_ticks_position('bottom')
        self.axs[-1].set_xlabel(self.data.names['x1'])

    def SetLinePlotStyle(self):
        for n, ax in enumerate(self.axs):
            # Set ylabel
            ax.set_ylabel(self.data.names['y'+str(n+1)])
            # make sure axis tickmark points out
            ax.tick_params(axis='both',direction='out')
            # Set axis visibility
            ax.spines['top'].set_visible(False)
        # change the x lim on the last, most buttom subplot
        ax.set_xlim([0,len(self.data.series['x1'])+1])
        plt.xticks(self.x, self.data.series['x1'])
        # Add some margins to the plot so that it is not touching the axes
        plt.margins(0.02,0.02)
        self.fig.tight_layout() # enforce tight layout
#            
#    def TimeSeries(self):
#        """Plot traces
#        Style:
#        'Classic': 
#        'Vstack': plots stacked vertically
#        """
      
if __name__ == "__main__":
    #K = PublicationFigures(Data=dataFile,PlotType='LinePlot',Style='Twin',SavePath='C:/QQDownload/asdf_twin.eps')
    # Allows further tuning
    # to set the ylim
    #K.axs[0].set_ylim([0,2.0])
    #K.axs[1].set_ylim([0.05, 0.25])
    K = PublicationFigures(Data=dataFile,PlotType='Beeswarm', Style='swarm', SavePath='C:/QQDownload/asdf_beeswarm.png')
    K.axs.set_ylim([-3,7])

