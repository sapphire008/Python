# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:55:23 2016

Data export utility of Synapse

@author: Edward
"""

import sys
import os

import numpy as np
import matplotlib
matplotlib.use("PS")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker, AuxTransformBox
import matplotlib.ticker as tic

from pdb import set_trace
sys.path.append('D:/Edward/Documents/Assignments/Scripts/Python/PySynapse')
from util.spk_util import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# %%
# Helper functions
def SetFont(ax, fig, fontsize=12,fontname='Arial',items=None):
        """Change font properties of all axes
        ax: which axis or axes to change the font. Default all axis in current
            instance. To skip axis, input as [].
        fig: figure handle to change the font (text in figure, not in axis).
        Default is any text items in current instance. To skip, input as [].
        fontsize: size of the font, specified in the global variable
        fontname: fullpath of the font, specified in the global variable
        items: select a list of items to change font. ['title', 'xlab','ylab',
               'xtick','ytick', 'texts','legend','legendtitle','textartist']
        """        
        def unpack_anchor_offsetbox(box):
            """Getting only text area items from the anchor offset box"""
            itemList = []
            counter = 0
            maxiter=100 # terminate at this iteration
            def unpacker(box):
                return box.get_children()

            # vectorize
            unpacker = np.frompyfunc(unpacker, 1,1)
            # Get the children
            while counter<maxiter and box:
                # recursively unpack the anchoroffsetbox or v/hpacker
                box = np.hstack(unpacker(box)).tolist()
                for nn, b in enumerate(box):
                    if 'matplotlib.text.Text' in str(type(b)):
                        itemList.append(b)
                        box[nn] = None
                # remove recorded
                box = [b for b in box if b is not None]
                counter += 1

            return itemList

        def get_ax_items(ax):
            """Parse axis items"""
            itemDict={'title':[ax.title], 'xlab':[ax.xaxis.label],
                    'ylab':[ax.yaxis.label], 'xtick':ax.get_xticklabels(),
                    'ytick':ax.get_yticklabels(),
                    'texts':ax.texts if isinstance(ax.texts,(np.ndarray,list))
                                         else [ax.texts],
                    'legend': [] if not ax.legend_
                                        else ax.legend_.get_texts(),
                    'legendtitle':[] if not ax.legend_
                                        else [ax.legend_.get_title()],
                    'textartist':[] if not ax.artists
                                        else unpack_anchor_offsetbox(ax.artists)}
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

def roundto125(x, r=np.array([1,2,5,10])): # helper static function
        """5ms, 10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s, etc.
        5mV, 10mV, 20mV, etc.
        5pA, 10pA, 20pA, 50pA, etc."""
        p = int(np.floor(np.log10(x))) # power of 10
        y = r[(np.abs(r-x/(10**p))).argmin()] # find closest value
        return(y*(10**p))

def AdjustText(txt, ax=None):
        """Adjust text so that it is not being cutoff"""
        #renderer = self.axs.get_renderer_cache()
        if ax is None: ax = plt.gca()
        txt.set_bbox(dict(facecolor='w', alpha=0, boxstyle='round, pad=1'))
        # plt.draw() # update the text draw
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

def AddTraceScaleBar(xunit, yunit, color='k',linewidth=None,\
                         fontsize=None, ax=None, xscale=None, yscale=None, loc=5):
        """Add scale bar on trace. Specifically designed for voltage /
        current / stimulus vs. time traces.
        xscale, yscale: add the trace bar to the specified window of x and y.
        """
        if ax is None: ax=plt.gca()
        def scalebarlabel(x, unitstr):
            x = int(x)
            if unitstr.lower()[0] == 'm':
                return(str(x)+unitstr if x<1000 else str(int(x/1000))+
                    unitstr.replace('m',''))
            elif unitstr.lower()[0] == 'p':
                return(str(x)+unitstr if x<1000 else str(int(x/1000))+
                    unitstr.replace('p','n'))

        ax.set_axis_off() # turn off axis
        X = np.ptp(ax.get_xlim()) if xscale is None else xscale
        Y = np.ptp(ax.get_ylim()) if yscale is None else yscale
        # calculate scale bar unit length
        X, Y = roundto125(X/5), roundto125(Y/5)
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
        # xi = np.max(ax.get_xlim()) + X/2.0
        # yi = np.mean(ax.get_ylim())
        # calculate position of text
        # xtext1, ytext1 = xi+X/2.0, yi-Y/10.0 # horizontal
        # xtext2, ytext2 = xi+X+X/10.0, yi+Y/2.0 # vertical
        # Draw the scalebar
        box1 = AuxTransformBox(ax.transData)
        box1.add_artist(plt.Rectangle((0,0),X, 0, fc="none"))
        box2 = TextArea(xlab, minimumdescent=False, textprops=dict(color=color))
        boxh = VPacker(children=[box1,box2], align="center", pad=0, sep=2)
        box3 = AuxTransformBox(ax.transData)
        box3.add_artist(plt.Rectangle((0,0),0,Y, fc="none"))
        box4 = TextArea(ylab, minimumdescent=False, textprops=dict(color=color))
        box5 = VPacker(children=[box3, boxh], align="right", pad=0, sep=0)
        box = HPacker(children=[box5,box4], align="center", pad=0, sep=2)
        anchored_box = AnchoredOffsetbox(loc=loc, pad=-9, child=box, frameon=False)
        ax.add_artist(anchored_box)
        return(anchored_box)

@AdjustAxs()
def TurnOffAxis(ax):
    """Turn off all axis"""
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def writeEpisodeNote(zData, viewRange, channels, initFunc=None):
    if initFunc is None:
        initFunc = lambda x: x[0]

    ts = zData.Protocol.msPerPoint
    notes = []
    # Make notes for each channel  
    for ch in channels:
        V = initFunc(spk_window(getattr(zData, 'Voltage')[ch], ts, viewRange))
        I = initFunc(spk_window(getattr(zData, 'Current')[ch], ts, viewRange))
        notes.append("Channel %s %.1f mV %d pA"%(ch, V , I ))
        # notes.append("Channel %s %.1f mV %d pA"%(ch, min(getattr(zData, 'Voltage')[ch]), min(getattr(zData, 'Current')[ch])  ))
    final_notes = zData.Protocol.readDataFrom + ' ' + ' '.join(notes) + ' WCTime: ' + zData.Protocol.WCtimeStr + ' min'
    return final_notes
# %%
def PlotTraces(df, index, viewRange, saveDir, colorfy=False, dpi=300, setFont='default', fig_size=None, nullRange=None):
    # np.savez('R:/tmp.npz', df=df, index=index, viewRange=[viewRange], saveDir=saveDir, colorfy=colorfy)
    # return
    # set_trace()
    # Start the figure
    nchannels = len(viewRange.keys())
    if not fig_size: # if not specified size, set to (4*nchannels, 4)
        fig_size = (4, 4*nchannels)

    if not colorfy:
        colorfy=['k']
    fig, _= plt.subplots(nrows=nchannels, ncols=1, sharex=True)
    ax = fig.get_axes()
    # text annotation area
    textbox = []
    for n, i in enumerate(index):
        zData = df['Data'][i]
        ts = zData.Protocol.msPerPoint
        channels = []
        
        for c, m in enumerate(viewRange.keys()):
            # Draw plots
            X = spk_window(zData.Time, ts, viewRange[m][0])
            Y = spk_window(getattr(zData, m[0])[m[1]], ts, viewRange[m][0])
            if nullRange is not None:
                if isinstance(nullRange, list):
                    Y -= np.mean(spk_window(Y, ts, nullRange))
                else: # a single number
                    Y -=  Y[time2ind(nullRange, ts)]
            ax[c].plot(X, Y, color=colorfy[n%len(colorfy)])
            # Draw initial value
            # initY = min(Y)
            initY = Y[0]
            InitVal = "{0:0.0f}".format(initY)      
            if m[0] == 'Voltage':
                InitVal += 'mV'
            elif m[0] == 'Current':
                InitVal += 'pA'
            else: # Stimulus
                InitVal = ''
                
            if m[1] not in channels:
                channels.append(m[1])
            
            # set_trace()
            ax[c].text(X[0]-50,  Y[0]-1, InitVal, ha='right', va='center', color=colorfy[n%len(colorfy)])
        
        final_notes = writeEpisodeNote(zData, viewRange[m][0], channels=channels)
        # Draw more annotations
        textbox.append(TextArea(final_notes, minimumdescent=False, textprops=dict(color=colorfy[n%len(colorfy)])))

    box = VPacker(children=textbox, align="left",pad=0, sep=2)
    annotationbox = AnchoredOffsetbox(loc=3, child=box, frameon=False, bbox_to_anchor=[1, 1.1])
    ax[-1].add_artist(annotationbox)

    # set axis
    scalebar = [annotationbox]
    for c, vr in enumerate(viewRange.items()):
        l, r = vr
        ax[c].set_xlim(r[0])
        ax[c].set_ylim(r[1])
        # Add scalebar
        scalebar.append(AddTraceScaleBar(xunit='ms', yunit='mV' if l[0]=='Voltage' else 'pA', ax=ax[c]))
        plt.subplots_adjust(hspace = .001)
        # temp = 510 + c
        temp = tic.MaxNLocator(3)
        ax[c].yaxis.set_major_locator(temp)

    if (isinstance(setFont, str) and setFont.lower() == 'default') or \
                    (isinstance(setFont, bool) and setFont):
        SetFont(ax, fig, fontsize=10, fontname=os.path.join(__location__,'../resources/Helvetica.ttf'))

    # save the figure
    fig.set_size_inches(fig_size)

    # plt.subplots_adjust(hspace=-0.8)
    fig.savefig(saveDir, bbox_inches='tight', bbox_extra_artists=tuple(scalebar), dpi=dpi)
    # Close the figure after save
    plt.close(fig)

    return(ax)


def PlotTracesVertically(df, index, viewRange, saveDir, colorfy=False):
    return

def PlotTracesHorizontally(df, index, viewRange, saveDir, colorfy=False):
    return

def data2csv(data):
    return

def embedMetaData(ax):
    """embedding meta data to a figure"""
    return



if __name__ == '__main__':
    sys.path.append("D:/Edward/Documents/Assignments/Scripts/Python/PySynapse")

    data = np.load('R:/tmp.npz')
    df, index, viewRange, saveDir, colorfy = data['df'].tolist(), data['index'].tolist(), data['viewRange'][0],\
                                                data['saveDir'].tolist(), data['colorfy'].tolist()
    # plot the figure
    ax= PlotTraces(df=df, index=index, viewRange=viewRange, saveDir='R:/tmp.eps', colorfy=colorfy, setFont=True)
