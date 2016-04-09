# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 02:21:18 2016

General utilities for plotting

@author: Edward
"""
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def SetFont(ax, fig, fontsize=12,fontname='Arial',items=None,):
        """Change font properties of all axes
        ax: which axis or axes to change the font. Default all axis in current
            instance. To skip axis, input as [].
        fig: figure handle to change the font (text in figure, not in axis).
        Default is any text items in current instance. To skip, input as [].
        fontsize: size of the font, specified in the global variable
        fontname: fullpath of the font, specified in the global variable
        items: select a list of items to change font. ['title', 'xlab','ylab',
               'xtick','ytick', 'texts','legend','legendtitle']
       
        """
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

        
def AdjustAxs(otypes=[np.ndarray], excluded=None):
    """Used as a decorator to set the axis properties"""
    def wrap(func):
        # vectorize the func so that it can be applied to single axis or
        # multiple axes
        func_vec = np.vectorize(func, otypes=otypes, excluded=excluded)
        def wrapper(ax, *args, **kwargs):
            res = func_vec(ax, *args, **kwargs)
            return(res)
        return(wrapper)
    return(wrap)
    
def SetAxisOrigin(ax, xcenter='origin', ycenter='origin', xspine='bottom', yspine='left'):
    """Set the origin of the axis"""
    if xcenter == 'origin':
        xtick = ax.get_xticks()
        if max(xtick)<0:
            xcenter = max(xtick)
        elif min(xtick)>0:
            xcenter = min(xtick)
        else:
            xcenter = 0
            
    if ycenter == 'origin':
        ytick = ax.get_yticks()
        if max(ytick)<0:
            ycenter = max(ytick)
        elif min(ytick)>0:
            ycenter = min(ycenter)
        else:
            ycenter = 0
            
    xoffspine = 'top' if xspine == 'bottom' else 'bottom'     
    yoffspine = 'right' if yspine=='left' else 'left'
    
           
    ax.spines[xspine].set_position(('data', ycenter))
    ax.spines[yspine].set_position(('data', xcenter))
    ax.spines[xoffspine].set_visible(False)
    ax.spines[yoffspine].set_visible(False)
    ax.xaxis.set_ticks_position(xspine)
    ax.yaxis.set_ticks_position(yspine)
    ax.spines[xspine].set_capstyle('butt')
    ax.spines[yspine].set_capstyle('butt')