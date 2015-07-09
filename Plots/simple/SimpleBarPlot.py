# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 16:56:32 2015
Simple Bar plot function
@author: Edward
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# global variables
# fontname = 'C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/resource/Helvetica.ttf' # font .ttf file path
# platform specific fonts
#import sys
#fontname = {'darwin': 'Helvetica', # Mac
#            'win32':'Arial', # Windows
#            'linux': 'FreeSans', # Linux
#            'cygwin': 'Arial' # use Windows
#            }.get(sys.platform)
fontname = 'Helvetica'
fontsize = {'title':16, 'xlab':12, 'ylab':12, 'xtick':10,'ytick':10,'texts':10, 
            'legend': 10, 'legendtitle':10} # font size

def SimpleBarPlot(groups, values, errors, savepath=None, width = 0.27, 
                  size=(3, 3), color=['#1f77b4']):
    """Takes 3 inputs and generate a simple bar plot
    e.g. groups = ['dog','cat','hippo']
         values = [15, 10, 3]
         errors = [0.5, 0.3, 0.8]
         savepath: path to save figure. Format will be parsed by the extension
             of save name
        width: distance between each bars
        size: figure size, in inches. Input as a tuple. Default (3,3)
        color: default tableau10's steelblue color (hex: #1f77b4)
    """
    # Get bar plot function according to style
    ngroups = len(groups) # group labels    
    # leftmost position of bars
    pos = np.arange(ngroups)
    # initialize the plot
    fig, axs = plt.subplots(nrows=1, ncols = 1, sharex=True)
    # plot the series
    axs.bar(pos, values, width, yerr=errors, color=color, align='center')
    # set axis
    axs.tick_params(axis='both',direction='out')
    axs.spines['left'].set_visible(True)
    axs.spines['right'].set_visible(False)
    axs.spines['top'].set_visible(False)
    axs.spines['bottom'].set_visible(True)
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')
    ymin, ymax = axs.get_ybound()
    if ymax <= 0.0: # only negative data present
        # flip label to top
        axs.spines['bottom'].set_position('zero') # zero the x axis
        axs.tick_params(labelbottom=False, labeltop=True)
    elif ymin >= 0.0: # only positive data present. Default
        axs.spines['bottom'].set_position('zero') # zero the x axis
    else: # mix of positive an negative data : set all label to bottoms
        axs.spines['bottom'].set_visible(False) 
        axs.spines['top'].set_visible(True)
        axs.spines['top'].set_position('zero')
    axs.xaxis.set_ticks_position('none')
    # Set x categorical label
    x = range(0,len(groups))
    if axs.get_xlim()[0] >= x[0]:
        axs.set_xlim(axs.get_xticks()[0]-1,axs.get_xlim()[-1])
    if axs.get_xlim()[-1] <= x[-1]:
        axs.set_xlim(axs.get_xlim()[0], axs.get_xticks()[-1]+1)
    plt.xticks(x, groups)
    # Set font
    itemDict = {'title':[axs.title], 'xlab':[axs.xaxis.label], 
                'ylab':[axs.yaxis.label], 'xtick':axs.get_xticklabels(),
                'ytick':axs.get_yticklabels(), 
                'texts':axs.texts if isinstance(axs.texts, np.ndarray) 
                        or isinstance(axs.texts, list) else [axs.texts], 
                'legend': [] if axs.legend_ is None 
                                else axs.legend_.get_texts(), 
                'legendtitle':[] if axs.legend_ is None 
                                    else [axs.legend_.get_title()]}
    itemList, keyList = [], []
    for k, v in iter(itemDict.items()):
        itemList += v
        keyList += [k]*len(v)   
    # initialize fontprop object
    fontprop = fm.FontProperties(style='normal', weight='normal',
                                 stretch = 'normal')
    if os.path.isfile(fontname): # check if font is a file
        fontprop.set_file(fontname)
    else:# check if the name of font is available in the system
        if not any([fontname.lower() in a.lower() for a in 
                fm.findSystemFonts(fontpaths=None, fontext='ttf')]):
            print('%s font not found. Use system default.' %(fontname))
        fontprop.set_family(fontname) # set font name
    # set font for each object
    for n, item in enumerate(itemList):
        if isinstance(fontsize, dict):
            fontprop.set_size(fontsize[keyList[n]])                                        
        elif n <1: # set the properties only once
            fontprop.set_size(fontsize)
        item.set_fontproperties(fontprop) # change font for all items
    # Set figure size
    fig.set_size_inches(size[0],size[1])
    # Save the figure
    if savepath is not None:
        fig.savefig(savepath, bbox_inches='tight', rasterized=True, dpi=300)
    return(fig, axs)
    
    
if __name__=='__main__':
    groups = ['dog','cat','hippo']
    values = [-15, 10, 3]
    errors = [0.5, 0.3, 0.8]
    SimpleBarPlot(groups, values, errors, savepath='C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots/barplot.eps')
    