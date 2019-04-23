# -*- coding: utf-8 -*-
"""
Created on Tue Jul 07 16:56:32 2015
Simple Bar plot function
@author: Edward
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import copy
# from plots import *

from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker, AuxTransformBox


from pdb import set_trace


# global variables
fontname = 'D:/Edward/Documents/Assignments/Scripts/Python/Plots/resource/Helvetica.ttf' # font .ttf file path
# platform specific fonts
#import sys
#fontname = {'darwin': 'Helvetica', # Mac
#            'win32':'Arial', # Windows
#            'linux': 'FreeSans', # Linux
#            'cygwin': 'Arial' # use Windows
#            }.get(sys.platform)
# fontname = 'Helvetica'
fontsize = {'title':10, 'xlab':8, 'ylab':8, 'xtick':6,'ytick':6,'texts':6,
            'legend': 6, 'legendtitle':6} # font size

# unit in points. This corresponds to 0.25mm (1pt  = 1/72 inch)
bar_line_property = {'border': 0.70866144, 'h_err_bar': 0.70866144, 'v_err_bar': 0.70866144,
                     'xaxis_tick': 0.70866144, 'yaxis_tick': 0.70866144, 'xaxis_spine': 0.70866144,
                     'yaxis_spine': 0.70866144} # in mm.

def barplot(groups, values, errors=None, Ns=None, pos=None,
                width=0.5, space=0.1, size=(2,2),
                color=['#BFBFBF'], xlabpos='hug', ylab="", set_axis=True,
                showvalue=True, showerror=False, bardir='+', border=[0.75, 0.5],
                capsize=4, ax=None, iteration=0, numdigit="{:.1f}", xpad=5,
                enforce_ylim=False, Ns_color='k', values_color='k',ylim=None,
                outsidevalue_thresh_px=20, xticklabdir='horizontal',
                DEBUG=False, **kwargs):
    """Takes 3 inputs and generate a simple bar plot
    e.g. groups = ['dog','cat','hippo']
         values = [-15, 10, 3]
         errors = [3, 2, 1]
         Ns = [5, 6, 5]. To be labeled at the base, inside the bar.
         pos: position of the bar groups. By defualt, np.arange(ngroups).
              Or specify as a list that is the same length as ngroups
        width: bar width
        space: space between bars within the group
        size: figure size, in inches. Input as a tuple. Default (3,3). Better

        color: default grey #BFBFBF
        xlabpos: xlabel position.
            'hug': always label at the base of the bar. Positive bars label
                    underneath, negative bars label above;
            'away': label outside of the graph area
        ylab: ylabel string
        showvalue: show value of the bar right under the error bar
        bardir: direction of the bar to show.
            "+" only show outward bars
            "-" only show inward bars.
            Otherwise, show both direction
        border: additional border to add to the left and right of the bars
                [0.75, 0.5]
        capsize: errorbar capsize (Default 4 point font)
        numdigit: format of the value if showvalue. Default {:.2f}
        iteration: number of times / groups to draw the bars.
        xpad: padding of xtick labels. Deafult 5
        enforce_ylim: enforce y-limit, given by the argument ylim.
                      Choose [True, False, 0]. Effect of this option depends on
                      multiple factors. Play to determine the best option.
        Ns_color: Ns text color. Default 'k'
        values_color: values text color. Deafult 'k'
        ylim: specify y-axis limit.
        outsidevalue_thresh_px: If bar size is shorter than this number of
            pixels, write the value outside the bar. Default 20 pixels.
        xticklabdir: set to 'vertical' to rotate the xticklabels

    Use add_comparison to add comparison
    """
    values = np.asarray(values)
    errors = np.asarray(errors)
    # Get bar plot function according to style
    ngroups = len(groups) # group labels
    # leftmost position of bars
    if pos is None:
        pos = np.arange(ngroups)
    elif len(pos) != ngroups:
        raise(ValueError('Length of argument "pos" must be the same as the number of groups of bars'))

    # Adjust spaciing
    pos = np.asarray(pos)+0.1+iteration*(width+space)
    # initialize the plot
    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols = 1, sharex=True, figsize=size)
        if ylim is not None:
            ax.set_ylim(ylim)
    else:
        fig = ax.get_figure()
        fig.set_size_inches(size)
    # errorbar property
    errors = np.array(errors)
    if errors.ndim==1:
        err = np.zeros((2, errors.shape[0]))
        for m, v in enumerate(values):
            if np.isnan(v):
                err[:,m] = 0
                continue
            if bardir == '+':
                d = 1 if v>=0 else 0
            elif bardir == '-':
                d = 0 if v>=0 else 1

            err[d,m] = errors[m]
    else:
        err = errors

    # plot the series
    rec = ax.bar(pos, values, width, yerr=err, color=color, align='center', capsize=capsize, ecolor='k', edgecolor='k', linewidth=bar_line_property['border'],capstyle='projecting', joinstyle='miter',
                 error_kw={'elinewidth':bar_line_property['v_err_bar'], 'capthick':bar_line_property['h_err_bar'], 'solid_capstyle':'projecting', 'solid_join_style':'miter'},**kwargs)
    rec.pos = pos
    # rec.err_height = values+np.sign(values)*(err if np.ndim(err)<2 else np.max(err,axis=0)) # height with errorbar, + or -

    if bardir == "+" and (enforce_ylim==True or enforce_ylim is 0):
        if all(np.array(values)>0) and max(ax.get_ylim())>=0: # all positive values
            ax.set_ylim([0,max(ax.get_ylim())])
        elif all(np.array(values)<0) and max(ax.get_ylim())<=0: # all negative values
            ax.set_ylim([min(ax.get_ylim()), 0])
        else: # has a mix of positive and negative values
            pass
    elif isinstance(enforce_ylim, (tuple, list)) and len(enforce_ylim)==2:
        ax.set_ylim(enforce_ylim) # specified ylim

    if DEBUG:
        print(bardir)

    # set axis
    if set_axis:
        ax.spines['left'].set_linewidth(bar_line_property['xaxis_spine'])
        ax.spines['right'].set_linewidth(bar_line_property['xaxis_spine'])
        ax.spines['bottom'].set_linewidth(bar_line_property['yaxis_spine'])
        ax.spines['top'].set_linewidth(bar_line_property['yaxis_spine'])
        ax.xaxis.set_tick_params(width=bar_line_property['xaxis_tick'])
        ax.yaxis.set_tick_params(width=bar_line_property['yaxis_tick'])
        ax.tick_params(axis='both',direction='out')
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if ylim is None:
            ymin, ymax = ax.get_ybound()
        else:
            ymin, ymax = ylim


        if ymax <= 0.0: # only negative data present
            if DEBUG:
                print('All negative data')
            # flip label to top
            if ylim is not None:
                ax.set_ylim(ylim)
            elif enforce_ylim is 0:
                ax.spines['bottom'].set_position('zero') # zero the x axis
            else:
                ax.spines['bottom'].set_position(('data',ymax))
            ax.tick_params(labelbottom=False, labeltop=True)
        elif ymin >= 0.0: # only positive data present. Default
            if DEBUG:
                print('All positive data')
            if ylim is not None:
                ax.set_ylim(ylim)
            elif enforce_ylim is 0:
                ax.spines['bottom'].set_position('zero') # zero the x axis
            else:
                ax.spines['bottom'].set_position(('data',ymin))
        else: # mix of positive an negative data : set all label to bottoms
            if DEBUG:
                print('Mix of positive and negative data')
            ax.spines['bottom'].set_visible(False)
            ax.spines['top'].set_visible(True)
            ax.spines['top'].set_position('zero')
            if ylim is not None:
                ax.set_ylim(ylim)
            elif enforce_ylim==True or enforce_ylim is 0: #really strong enforcement
                if np.all(np.array(values)>=0):
                    ax.set_ylim([0, ymax])
                else:
                    ax.set_ylim([ymin, 0])

    # Set x categorical label
    ax.xaxis.set_ticks_position('none')
    ax.set_xticks(pos)
    ax.set_xticklabels(groups, va='center', ha='center', rotation=xticklabdir)
    ax.tick_params(axis='x', which='major', pad=xpad)

    yrange = ax.get_ylim()[1]-ax.get_ylim()[0] # axis height in data
    yori = (0 - ax.get_ylim()[0])/yrange
    def hugxticks(values, ax, yrange, yroi):
        if all(np.array(values)>0) or all(np.array(values)<0):
            return

        original_y = [a.get_position()[1] for a in ax.get_xticklabels()]
        [a.set_y(yori) for a in ax.get_xticklabels()] # move below origin first
        plt.draw()
        for v, a in zip(values, ax.get_xticklabels()):
            txtbb = a.get_window_extent()
            _, ymin, _, ymax = tuple(ax.transData.inverted().transform(txtbb).ravel())
            ypos = abs((ymin+ymax)/2.0)
            if v>=0:
                a.set_y(yori + ypos/yrange*1/2)
            else:
                a.set_y(yori + ypos/yrange*3/2)

        return original_y
    # Set x label vertical position
    if xlabpos == 'hug':
        hugxticks(values, ax, yrange, yori)

    # Set Ns
    if Ns is not None:
        plt.draw()
        #if xlabpos != 'hug':
        #    original_y = hugxticks(values, ax, yrange, yori)
        for i, (v, a, n) in enumerate(zip(values, ax.get_xticklabels(), Ns)):
            if np.isnan(v):
                continue
            txtbb = a.get_window_extent()
            xmin, ymin, xmax, ymax = tuple(ax.transData.inverted().transform(txtbb).ravel())
            xtext = (xmin+xmax)/2.0
            yoffset = ax.transData.inverted().transform((0, 2))[1] - ax.transData.inverted().transform((0,0))[1]

            if  enforce_ylim==True or enforce_ylim is 0 or (not(all(values>0)) and not(all(values<0))):
                ybase = 0
            else:
                ybase = ax.get_ylim()[np.argmin(np.abs(ax.get_ylim()))]


            if abs(ax.transData.transform((0,v))[1] - ax.transData.transform((0,0))[1]) <outsidevalue_thresh_px: # small bar, less than 10 pixels
                ke = max(np.abs(err[:,i]))
                n_text_color = Ns_color # its on the outside anyway
                if v>0:
                    ytext = ybase + ke + yoffset
                    va = 'bottom'
                elif v==0:
                    if np.all(np.array(values)>=0):
                        ytext = ybase + ke + yoffset
                        va = 'bottom'
                    else:
                        ytext = ybase - ke - yoffset
                        va = 'bottom'
                else: # v<0
                    ytext = ybase - ke - yoffset
                    va = 'top'
            else:
                if v>=0:
                    ytext = ybase + yoffset
                    va = 'bottom'
                else: # v<0
                    ytext = ybase - yoffset
                    va = 'top'
                if all(np.array(rec[i].get_facecolor()) == np.array([0.,0.,0.,1.])) and Ns_color in ['k', '#000000', [0,0,0], (0,0,0), [0,0,0,1], (0,0,0,1), np.array([0,0,0]), np.array([0,0,0,1])]:
                    n_text_color = 'w'
                else:
                    n_text_color = Ns_color
            # ytext = -(ymin+ymax)/2.0
            ax.text(xtext,ytext, "("+str(int(n))+")", ha='center',va=va, color=n_text_color)

    if showvalue:
        plt.draw()
        for (i, v), a in zip(enumerate(values), ax.get_xticklabels()):
            if np.isnan(v):
                continue
            txtbb = a.get_window_extent()
            xmin, ymin, xmax, ymax = tuple(ax.transData.inverted().transform(txtbb).ravel())
            xtext = (xmin+xmax)/2.0
            yoffset = ax.transData.inverted().transform((0, 2))[1] - ax.transData.inverted().transform((0,0))[1]
            bar_size = abs(ax.transData.transform((0,v))[1] - ax.transData.transform((0,0))[1])


            if  bar_size < 2*outsidevalue_thresh_px: # small bar, less than 10 pixels
                ke = max(np.abs(err[:,i])) # put value outside of errorbar
                v_text_color = values_color # its on the outside anyway
                if v>0:
                    ytext = v + ke + yoffset*3 + 2*(bar_size<outsidevalue_thresh_px and Ns is not None) * yoffset
                    va = 'bottom'
                elif v==0:
                    if np.all(np.array(values)>=0):
                        ytext = v + ke + yoffset*5
                        va = 'bottom'
                    else:
                        ytext = v - ke - yoffset*5
                        va = 'bottom'
                else: # v<0
                    ytext = v - ke - yoffset*3 - 2*(bar_size<outsidevalue_thresh_px and Ns is not None) * yoffset
                    va = 'top'
            else:
                if v>=0:
                    ytext = v - yoffset
                    va = 'top'
                else: # v<0
                    ytext = v + yoffset
                    va = 'bottom'
                if all(np.array(rec[i].get_facecolor()) == np.array([0.,0.,0.,1.])) and values_color in Ns_color in ['k', '#000000', [0,0,0], (0,0,0), [0,0,0,1], (0,0,0,1), np.array([0,0,0]), np.array([0,0,0,1])]:
                    v_text_color = 'w'
                else:
                    v_text_color = values_color
            if not showerror:
                ax.text(xtext, ytext, numdigit.format(v), ha='center', va=va, color=v_text_color)
            else:
                ax.text(xtext, ytext, (numdigit+"\n±\n"+numdigit).format(v, float(np.nanmean(err[:,i]))), ha='center', va=va, color=v_text_color)

    ax.set_xticks(pos-iteration*(width+space)/2)
    # Set ylabel
    ax.set_ylabel(ylab)
    # Set xaxis limit
    ax.set_xlim([min(pos) - border[0], max(pos) + border[1]])

    # Line drawing
    setBarplotErrorbarStyle(rec)
    equalAxLineWidth(ax)
    setAxisLineStyle(ax)

    # Save the figure
#    if savepath is not None:
#        fig.savefig(savepath, bbox_inches='tight', rasterized=True, dpi=300)
    return(fig, ax, rec)


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

def setBarplotErrorbarStyle(rec):
    if 'ErrorbarContainer' in str(type(rec)):
        children = rec.get_children()
    else:
        children = rec.errorbar.get_children()
    for c in children:
        if c is None:
            continue
        elif isinstance(c, matplotlib.lines.Line2D):
            c.set_dash_capstyle = 'projecting'
            c.set_dash_joinstyle = 'miter'
            c.set_solid_capstyle = 'projecting'
            c.set_solid_joinstyle = 'miter'
        elif isinstance(c, matplotlib.collections.LineCollection):
            try: # for future
                c.set_capstyle = 'projecting'
            except:
                pass
            try:
                c.set_joinstyle = 'miter'
            except:
                pass

@AdjustAxs()
def equalAxLineWidth(ax, lineproperty ={'xaxis_tick': 0.70866144,
                                    'yaxis_tick': 0.70866144,
                                    'xaxis_spine': 0.70866144,
                                    'yaxis_spine': 0.70866144}):
    ax.spines['left'].set_linewidth(bar_line_property['xaxis_spine'])
    ax.spines['right'].set_linewidth(bar_line_property['xaxis_spine'])
    ax.spines['bottom'].set_linewidth(bar_line_property['yaxis_spine'])
    ax.spines['top'].set_linewidth(bar_line_property['yaxis_spine'])
    ax.xaxis.set_tick_params(width=bar_line_property['xaxis_tick'])
    ax.yaxis.set_tick_params(width=bar_line_property['yaxis_tick'])

@AdjustAxs()
def setAxisLineStyle(ax, lineproperty={'xaxis_tick_capstyle':'projecting',
                                       'xaxis_tick_joinstyle':'miter',
                                       'yaxis_tick_capstyle':'projecting',
                                       'yaxis_tick_joinstyle':'miter',
                                       'xaxis_spine_capstyle':'projecting',
                                       'xaxis_spine_joinstyle':'miter',
                                       'yaxis_spine_capstyle':'projecting',
                                       'yaxis_spine_joinstyle':'miter',
                                       }):
    # Ticks
    for i in ax.xaxis.get_ticklines():
        i._marker._capstyle = lineproperty['xaxis_tick_capstyle']
        i._marker._joinstyle = lineproperty['xaxis_tick_joinstyle']

    for i in ax.yaxis.get_ticklines():
        i._marker._capstyle = lineproperty['yaxis_tick_capstyle']
        i._marker._joinstyle = lineproperty['yaxis_tick_joinstyle']

    # Spines
    ax.spines['left']._capstyle = lineproperty['yaxis_spine_capstyle']
    ax.spines['left']._joinstyle = lineproperty['yaxis_spine_joinstyle']
    ax.spines['right']._capstyle = lineproperty['yaxis_spine_capstyle']
    ax.spines['right']._joinstyle = lineproperty['yaxis_spine_joinstyle']
    ax.spines['top']._capstyle = lineproperty['xaxis_spine_capstyle']
    ax.spines['top']._joinstyle = lineproperty['xaxis_spine_joinstyle']
    ax.spines['bottom']._capstyle = lineproperty['xaxis_spine_capstyle']
    ax.spines['bottom']._joinstyle = lineproperty['xaxis_spine_joinstyle']


def add_comparison(x_bar=[0.1, 1.1], y_bar=[0.5, 0.5], x_text=0.6, y_text=0.5, text='*', ax=None, va='bottom', ha='center', *args, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.plot(x_bar, y_bar, color='k', lw=bar_line_property['h_err_bar'])
    ax.text(x_text, y_text, text, va=va, ha=ha,*args, **kwargs)

    return ax



if __name__=='__main__':
    savepath='D:/Edward/Documents/Assignments/Scripts/Python/Plots/example/barplot2.eps'
    # fig, ax = plt.subplots(1,1)
    groups = ['dog','cat','hippo']
    values = [-15, 5, 9]
    errors = [3, 2, 1]
    Ns = [5, 13, 4]
    # ax.bar(np.arange(3), values, width=0.2, color='b', align='center')
    fig, ax, rec1 = barplot(groups, values, errors, Ns, width=0.3, space=0.05, numdigit="{:d}", ylab='weight gained (kg)', iteration=0)

    groups = ['dog', 'cat', 'hippo']
    values = [-13, 6, 8]
    errors = [2, 3, 1]
    Ns = [8, 10, 7]
    # ax.bar(np.arange(3)+0.2, values, width=0.2, color='r', align='center')
    # Draw another group bars next to the previous group
    fig, ax, rec2 = barplot(groups, values, errors, Ns, width=0.3, space=0.1, numdigit="{:d}", ylab='weight gained (kg)', iteration=1, ax=ax, size=(5,5))

    #SetFont(ax, fig, fontsize=fontsize, fontname='Helvetica')
    # fig.savefig(savepath, bbox_inches='tight', rasterized=True, dpi=300)
