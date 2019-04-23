# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 09:55:23 2016

Data export utility of Synapse

@author: Edward
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use("PS")
matplotlib.rcParams['svg.fonttype'] = 'none'
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker, AuxTransformBox
import matplotlib.ticker as tic

# matplotlib.rcParams['pdf.fonttype'] = 42
# matplotlib.rcParams['ps.fonttype'] = 42

from pdb import set_trace

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
sys.path.append(os.path.join(__location__, '..')) # for debug only
from util.spk_util import *
from util.svg2eps import *

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
                    'xminortick': ax.get_xminorticklabels(),
                    'yminortick': ax.get_yminorticklabels(),
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
                if not fontname.lower() in [f.name.lower() for f in fm.fontManager.ttflist] and \
                   not fontname.lower() in [f.name.lower() for f in fm.fontManager.afmlist]:
                        #any([fontname.lower() in a.lower() for a in
                        #fm.findSystemFonts(fontpaths=None, fontext='ttf')]):
                     print('Cannot find specified font: %s' %(fontname))
                fontprop.set_family(fontname) # set font name
            # set font for each object
            for n, item in enumerate(itemList):
                if isinstance(fontsize, dict):
                    if keyList[n] in fontsize.keys():
                        fontprop.set_size(fontsize[keyList[n]])
                elif n < 1: # set the properties only once
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
            
def roundto125(x, r=np.array([1,2,5,10])): # helper static function
        """5ms, 10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s, etc.
        5mV, 10mV, 20mV, etc.
        5pA, 10pA, 20pA, 50pA, etc.
        """
        p = int(np.floor(np.log10(x))) # power of 10
        y = r[(np.abs(r-x/(10**p))).argmin()] # find closest value
        return(y*(10**p))


def butterFilter(y, Order, Wn, Btype="low"):
    b, a = butter(Order, Wn, btype=Btype, analog=False, output='ba')
    y_filt = filtfilt(b, a, y)
    return y_filt

def AddTraceScaleBar(xunit, yunit, color='k',linewidth=None,\
                         fontsize=None, ax=None, xscale=None, yscale=None,
                         loc=5, bbox_to_anchor=None):
        """Add scale bar on trace. Specifically designed for voltage /
        current / stimulus vs. time traces.
        xscale, yscale: add the trace bar to the specified window of x and y.
        """
        if ax is None: ax=plt.gca()
        def scalebarlabel(x, unitstr):
            x = int(x)
            if unitstr.lower()[0] == 'm':
                return(str(x)+" " + unitstr if x<1000 else str(int(x/1000))+ " " +
                        unitstr.replace('m',''))
            elif unitstr.lower()[0] == 'p':
                return(str(x)+" "+ unitstr if x<1000 else str(int(x/1000))+ " " +
                        unitstr.replace('p','n'))
            else: # no prefix
                return(str(x)+" " + unitstr)

        ax.set_axis_off() # turn off axis
        X = np.ptp(ax.get_xlim()) if xscale is None else xscale
        Y = np.ptp(ax.get_ylim()) if yscale is None else yscale
        # calculate scale bar unit length
        X, Y = roundto125(X/5), roundto125(Y/(5 if Y<1200 else 10))
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
                linewidth=0.70
                #raise(AttributeError('Did not find any line in this axis. Please explicitly specify the linewidth'))
        if 'matplotlib.lines.Line2D' in str(type(linewidth)):
            linewidth = linewidth.get_linewidth()
        # print(linewidth)
        if fontsize is None:
            fontsize = ax.yaxis.get_major_ticks()[2].label.get_fontsize()
        scalebarBox = AuxTransformBox(ax.transData)
        scalebarBox.add_artist(matplotlib.patches.Rectangle((0, 0), X, 0, fc="none", edgecolor='k', linewidth=linewidth, joinstyle='miter', capstyle='projecting')) #TODO capstyle
        scalebarBox.add_artist(matplotlib.patches.Rectangle((X, 0), 0, Y, fc="none", edgecolor='k', linewidth=linewidth, joinstyle='miter', capstyle='projecting'))
        scalebarBox.add_artist(matplotlib.text.Text(X/2, -Y/20, xlab, va='top', ha='center', color='k'))
        scalebarBox.add_artist(matplotlib.text.Text(X+X/20, Y/2, ylab, va='center', ha='left', color='k'))
        anchored_box = AnchoredOffsetbox(loc=loc, pad=-9, child=scalebarBox, frameon=False, bbox_to_anchor=bbox_to_anchor)
        ax.add_artist(anchored_box)
        return(anchored_box)

def DrawAnnotationArtists(artist_dict, axs):
    """Draw the same annotation objects displayed on the graphics window 
    when exporting to matplotlib figures
        * ann_dict: dictionaries of each artist
    """
    # TODO
    for key, artist in artist_dict.items():
        # Find out which axis to draw on
        ax = axs[artist['layout'][2]]
        if isinstance(ax, list):
            ax = ax[artist['layout'][3]]
        if artist['type'] == 'box':
            mpl_artist = matplotlib.patches.Rectangle((artist['x0'], artist['y0']), artist['width'], artist['height'],
                                                      ec=artist['linecolor'] if artist['line'] else 'none',
                                                      linewidth=artist['linewidth'] if artist['line'] else None, linestyle=artist['linestyle'],
                                                      fc=artist['fillcolor'],fill=artist['fill'],
                                                      joinstyle='miter',capstyle='projecting')
            #set_trace()
            #mpl_artist = matplotlib.patches.Rectangle((100, 0), 500, 10)

            ax.add_patch(mpl_artist)
        elif artist['type'] == 'line':
            mpl_artist = matplotlib.lines.Line2D([artist['x0'], artist['x1']], [artist['y0'], artist['y1']],
                                           color=artist['linecolor'], linewidth=artist['linewidth'],
                                           linestyle=artist['linestyle'],
                                           solid_joinstyle='miter', solid_capstyle='projecting',
                                           dash_joinstyle='miter', dash_capstyle='projecting')
            ax.add_artist(mpl_artist)
        elif artist['type'] == 'circle':
            pass
        elif artist['type'] == 'arrow':
            pass
        elif artist['type'] == 'symbol':
            pass
        elif artist['type'] == 'curve':
            mpl_artist = matplotlib.lines.Line2D(artist['x'], artist['y'], color=artist['linecolor'],
                                                 linewidth=0.5669291338582677, solid_joinstyle='bevel',
                                                 solid_capstyle='butt')
            ax.add_artist(mpl_artist)
        elif artist['type'] == 'event':
            ax.text(float(artist['eventTime'][0] - 0.1 * np.diff(ax.get_xlim())),
                    float(np.mean(artist['y'])),
                    '{:d} APs'.format(len(artist['eventTime'])),
                    color=artist['linecolor'], va='center', ha='left')

            for et in artist['eventTime']:
                mpl_artist = matplotlib.lines.Line2D([et, et], artist['y'], color=artist['linecolor'],
                                                     linewidth=0.5669291338582677, solid_joinstyle='bevel',
                                                     solid_capstyle='butt')
                ax.add_artist(mpl_artist)

        elif artist['type'] == 'ttl':
            pass
        else:
            pass

        # Add the artist to the plot
        #ax.add_artist(mpl_artist)


@AdjustAxs()
def TurnOffAxis(ax):
    """Turn off all axis"""
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)

def writeEpisodeNote(zData, viewRange, channels, initFunc=None, mode='Simple'):
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
    if mode.lower() == 'simple' and zData.Protocol.acquireComment != 'PySynapse Arithmetic Data':
        final_notes = os.path.basename(os.path.splitext(zData.Protocol.readDataFrom)[0]) + ' ' + ' '.join(notes) + ' WCTime: ' + zData.Protocol.WCtimeStr + ' min'
    elif mode.lower() == 'label only':
        final_notes = os.path.basename(os.path.splitext(zData.Protocol.readDataFrom)[0])
    else: # Full
        final_notes = zData.Protocol.readDataFrom + ' ' + ' '.join(notes) + ' WCTime: ' + zData.Protocol.WCtimeStr + ' min'
    return final_notes
# %%
def PlotTraces(df, index, viewRange, saveDir, colorfy=False, artists=None, dpi=300, fig_size=None,
               adjustFigH=True, adjustFigW=True, nullRange=None, annotation='Simple',  showInitVal=True,
               setFont='default', fontSize=10, linewidth=1.0, monoStim=False, stimReflectCurrent=True,
               plotStimOnce=False, filterDict=None, **kwargs):
    """Export multiple traces overlapping each other"""    
    # np.savez('R:/tmp.npz', df=df, index=index, viewRange=[viewRange], saveDir=saveDir, colorfy=colorfy)
    # return
    # set_trace()
    # Start the figure
    # viewRange= {(channel, stream):[[xmin,xmax],[ymin),ymax]]}
    nchannels = len(viewRange.keys())
    if not fig_size: # if not specified size, set to (4, 4*nchannels)
        fig_size = (4, 4*nchannels)

    if not colorfy:
        colorfy=['k']
    fig, _ = plt.subplots(nrows=nchannels, ncols=1, sharex=True)
    ax = fig.get_axes()
    # text annotation area
    textbox = []
    for n, i in enumerate(index):
        zData = df['Data'][i]
        ts = zData.Protocol.msPerPoint
        channels = []
        
        for c, m in enumerate(viewRange.keys()):
            # Draw plots
            X = zData.Time
            Y = getattr(zData, m[0])[m[1]]
            # null the trace, but ignore arithmetic data since their data were already nulled
            if nullRange is not None and zData.Protocol.acquireComment != 'PySynapse Arithmetic Data':
                if isinstance(nullRange, list):
                    Y = Y - np.mean(spk_window(Y, ts, nullRange))
                else: # a single number
                    Y = Y - Y[time2ind(nullRange, ts)]
                    
            # window the plot
            X = spk_window(X, ts, viewRange[m][0])
            Y = spk_window(Y, ts, viewRange[m][0])

            # Apply filter if toggled filtering, but do not filter stimulus
            if isinstance(filterDict, dict) and m[0]!='Stimulus':
                Y = butterFilter(Y, filterDict['order'], filterDict['wn'], filterDict['btype'])

            # Stim channel reflects current channel
            if stimReflectCurrent and m[0]=='Stimulus':
                CurBase = spk_window(zData.Current[m[1]], ts, viewRange[m][0]) # use view range of stimulus on current
                CurBase = np.mean(spk_window(CurBase, ts, [0,50]))
                Y = Y + CurBase
                
            # do the plot
            if m[0] in ['Voltage', 'Current'] or not monoStim:
                ax[c].plot(X, Y, color=colorfy[n%len(colorfy)], lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            else: # Stimulus
                if plotStimOnce and n > 0:
                    pass
                else:
                    ax[c].plot(X, Y, color='k', lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            # Draw initial value
            if showInitVal:
                InitVal = "{0:0.0f}".format(Y[0])
                if m[0] == 'Voltage':
                    InitVal += ' mV'
                elif m[0] == 'Current':
                    InitVal += 'pA'
                elif m[0] == 'Stimulus':
                    if stimReflectCurrent:
                        InitVal += ' pA'
                    else:
                        InitVal = ''

                ax[c].text(X[0]-0.03*(viewRange[m][0][1]-viewRange[m][0][0]),  Y[0]-1, InitVal, ha='right', va='center', color=colorfy[n%len(colorfy)])

        if m[1] not in channels:
            channels.append(m[1])

        if annotation.lower() != 'none':
            final_notes = writeEpisodeNote(zData, viewRange[m][0], channels=channels, mode=annotation)
            # Draw more annotations
            textbox.append(TextArea(final_notes, minimumdescent=False, textprops=dict(color=colorfy[n%len(colorfy)])))
    
    # Group all the episode annotation text
    if annotation.lower() != 'none':
        box = VPacker(children=textbox, align="left", pad=0, sep=2)
        annotationbox = AnchoredOffsetbox(loc=3, child=box, frameon=False, bbox_to_anchor=[1, 1.1])
        ax[-1].add_artist(annotationbox)
        scalebar = [annotationbox]
    else:
        scalebar = []

    # Draw annotation artists
    DrawAnnotationArtists(artists, axs=ax)

    # set axis
    for c, vr in enumerate(viewRange.items()):
        l, r = vr
        ax[c].set_xlim(r[0])
        ax[c].set_ylim(r[1])
        # Add scalebar
        scalebar.append(AddTraceScaleBar(xunit='ms', yunit='mV' if l[0]=='Voltage' else 'pA', ax=ax[c]))
        plt.subplots_adjust(hspace=.001)
        # temp = 510 + c
        temp = tic.MaxNLocator(3)
        ax[c].yaxis.set_major_locator(temp)

    if (isinstance(setFont, str) and setFont.lower() in ['default', 'arial', 'helvetica']) or \
                    (isinstance(setFont, bool) and setFont):
        SetFont(ax, fig, fontsize=fontSize, fontname=os.path.join(__location__,'../resources/Helvetica.ttf'))
    else:
        SetFont(ax, fig, fontsize=fontSize, fontname=setFont)

    # save the figure
    if adjustFigH:
        fig_size = (fig_size[0], fig_size[1]*nchannels)

    fig.set_size_inches(fig_size)

    # plt.subplots_adjust(hspace=-0.8)
    fig.savefig(saveDir, bbox_inches='tight', bbox_extra_artists=tuple(scalebar), dpi=dpi, transparent=True)
    # Close the figure after save
    plt.close(fig)
    # Convert from svg to eps
    if '.svg' in saveDir:
        svg2eps_ai(source_file=saveDir, target_file=saveDir.replace('.svg', '.eps'))


    return(ax)

def PlotTracesConcatenated(df, index, viewRange, saveDir, colorfy=False, artists=None, dpi=300,
                           fig_size=None, nullRange=None, hSpaceType='Fixed', hFixedSpace=0.10,
                           adjustFigW=True, adjustFigH=True, trimH=(None,None),
                           annotation='Simple', showInitVal=True, setFont='default', fontSize=10,
                           linewidth=1.0, monoStim=False, stimReflectCurrent=True, **kwargs):
    """Export traces arranged horizontally.
    Good for an experiments acquired over multiple episodes.
    trimH: (t1, t2) trim off the beginning of first episode by t1 seconds, and the
        the end of the last episode by t2 seconds
    """
    nchannels = len(viewRange.keys())
    if not colorfy:
        colorfy=['k']
    fig, _= plt.subplots(nrows=nchannels, ncols=1, sharex=True)
    ax = fig.get_axes()
    # text annotation area
    textbox = []
    nullBase = dict()
    currentTime = 0
    maxWindow = max(df['Duration'])
    for n, i in enumerate(index): #iterate over episodes
        zData = df['Data'][i]
        ts = zData.Protocol.msPerPoint
        channels = []
        for c, m in enumerate(viewRange.keys()): # iterate over channels/streams
            # Draw plots
            X = zData.Time + currentTime
            Y = getattr(zData, m[0])[m[1]]
            # null the trace
            if nullRange is not None:
                if n == 0: # calculate nullBase
                    if isinstance(nullRange, list):
                        nullBase[(m[0],m[1])] = np.mean(spk_window(Y, ts, nullRange))
                    else:
                        nullBase[(m[0],m[1])] = Y[time2ind(nullRange, ts)]
                Y = Y - nullBase[(m[0],m[1])]
            if n == 0 and trimH[0] is not None:
                X = spk_window(X, ts, (trimH[0], None))
                Y = spk_window(X, ts, (trimH[0], None))
            elif n + 1 == len(index) and trimH[1] is not None:
                X = spk_window(X, ts, (None, trimH[1]))
                Y = spk_window(X, ts, (None, trimH[1]))

            # Stim channel reflects current channel
            if stimReflectCurrent and m[0]=='Stimulus':
                CurBase = spk_window(zData.Current[m[1]], ts, viewRange[m][0]) # use view range of stimulus on current
                CurBase = np.mean(spk_window(CurBase, ts, [0,50]))
                Y = Y + CurBase
            # do the plot
            if m[0] in ['Voltage', 'Current'] or not monoStim: # temporary workaround
                ax[c].plot(X, Y, color=colorfy[n%len(colorfy)], lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            else:
                ax[c].plot(X, Y, color='k', lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            # Draw the initial value, only for the first plot
            if n == 0 and showInitVal:
                InitVal = "{0:0.0f}".format(Y[0])
                if m[0] == 'Voltage':
                    InitVal += ' mV'
                elif m[0] == 'Current':
                    InitVal += ' pA'
                elif m[0] == 'Stimulus':
                    if stimReflectCurrent:
                        InitVal += ' pA'
                    else:
                        InitVal = ''

                ax[c].text(X[0]-0.03*(viewRange[m][0][1]-viewRange[m][0][0]), Y[0]-1, InitVal, ha='right', va='center', color=colorfy[n%len(colorfy)])

            if m[1] not in channels:
                channels.append(m[1])

        if annotation.lower() != 'none':
            final_notes = writeEpisodeNote(zData, viewRange[m][0], channels=channels, mode=annotation)
            # Draw some annotations
            textbox.append(TextArea(final_notes, minimumdescent=False, textprops=dict(color=colorfy[n%len(colorfy)])))

        # Set some spacing for the next episode
        if n+1 < len(index):
            if hSpaceType.lower() == 'fixed':
                currentTime = currentTime + (len(Y)-1)*ts + maxWindow * hFixedSpace / 100.0
            elif hSpaceType.lower() in ['real time', 'realtime', 'rt']:
                currentTime = currentTime + (df['Data'][index[n+1]].Protocol.WCtime - zData.Protocol.WCtime)*1000

    # Group all the episodes annotation text
    if annotation.lower() != 'none':
        box = VPacker(children=textbox, align="left",pad=0, sep=2)
        annotationbox = AnchoredOffsetbox(loc=3, child=box, frameon=False, bbox_to_anchor=[1, 1.1])
        ax[-1].add_artist(annotationbox)
        scalebar = [annotationbox]
    else:
        scalebar = []
    
    # set axis
    for c, vr in enumerate(viewRange.items()):
        ax[c].set_ylim(vr[1][1])
        # Add scalebar
        scalebar.append(AddTraceScaleBar(xunit='ms', yunit='mV' if vr[0][0]=='Voltage' else 'pA', ax=ax[c]))
        plt.subplots_adjust(hspace = .001)
        temp = tic.MaxNLocator(3)
        ax[c].yaxis.set_major_locator(temp)

    # Set font
    if (isinstance(setFont, str) and setFont.lower() in ['default', 'arial', 'helvetica']) or \
                (isinstance(setFont, bool) and setFont):
        SetFont(ax, fig, fontsize=fontSize, fontname=os.path.join(__location__,'../resources/Helvetica.ttf'))
    else:
        SetFont(ax, fig, fontsize=fontSize, fontname=setFont)
            
    # figure out and set the figure size
    if adjustFigW:
        fig_size = (np.ptp(ax[0].get_xlim()) / maxWindow * fig_size[0], fig_size[1])
    
    if adjustFigH:
        fig_size = (fig_size[0], fig_size[1]*nchannels)
        
    fig.set_size_inches(fig_size)
    
    fig.savefig(saveDir, bbox_inches='tight', bbox_extra_artists=tuple(scalebar), dpi=dpi)
    # Close the figure after save
    plt.close(fig)
    # Convert svg file to eps
    if '.svg' in saveDir:
        svg2eps_ai(source_file=saveDir, target_file=saveDir.replace('.svg', '.eps'))
    
    return(ax)
    
    
def PlotTracesAsGrids(df, index, viewRange, saveDir=None, colorfy=False, artists=None, dpi=300,
                      fig_size=None, adjustFigH=True, adjustFigW=True, nullRange=None, 
                      annotation='Simple', setFont='default',gridSpec='Vertical', showInitVal=True,
                      scalebarAt='all', fontSize=10, linewidth=1.0, monoStim=False,
                      stimReflectCurrent=True, plotStimOnce=False, **kwargs):
    "Export Multiple episodes arranged in a grid; default vertically""" 
    if not colorfy:
        colorfy = ['k']
        
    nchannels = len(viewRange.keys())
    nepisodes = len(index)
    if isinstance(gridSpec, str):
        nrows, ncols = {
        'ver': (nchannels*nepisodes, 1),
        'hor': (1, nchannels*nepisodes),
        'cha': (nchannels, nepisodes),
        'epi': (nepisodes, nchannels)
        }.get(gridSpec[:3].lower(), (None, None))
    
        if nrows is None:
            raise(ValueError('Unrecognized gridSpec: {}'.format(gridSpec)))
    else:
        raise(TypeError('Unrecognized type of argument: "gridSpec"'))
        
    fig, _ = plt.subplots(nrows=nrows, ncols=ncols, sharex=True)
    ax = fig.get_axes()
            
    # text annotation area
    textbox = []
    viewRange_dict = {}
    row, col = 0,0 # keep track of axis used
    first_last_mat = [[],[]]
    for n, i in enumerate(index):
        zData = df['Data'][i]
        ts = zData.Protocol.msPerPoint
        channels = []
        
        for c, m in enumerate(viewRange.keys()):
            # Draw plots
            X = zData.Time
            Y = getattr(zData, m[0])[m[1]]
            # null the trace
            if nullRange is not None:
                if isinstance(nullRange, list):
                    Y = Y - np.mean(spk_window(Y, ts, nullRange))
                else: # a single number
                    Y = Y - Y[time2ind(nullRange, ts)]
            # window the plot
            X = spk_window(X, ts, viewRange[m][0])
            Y = spk_window(Y, ts, viewRange[m][0])
            # Stim channel reflects current channel
            if stimReflectCurrent and m[0]=='Stimulus':
                CurBase = spk_window(zData.Current[m[1]], ts, viewRange[m][0]) # use view range of stimulus on current
                CurBase = np.mean(spk_window(CurBase, ts, [0,50]))
                Y = Y + CurBase
            # do the plot
            ind = np.ravel_multi_index((row,col), (nrows, ncols), order='C')
            if n == 0:
                first_last_mat[0].append(ind)
            elif n == len(index)-1:
                first_last_mat[-1].append(ind)
            
            if m[0] in ['Voltage', 'Current'] or not monoStim:
                ax[ind].plot(X, Y, color=colorfy[n%len(colorfy)], lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            else: # Stimulus
                if plotStimOnce and n > 0:
                    pass
                else:
                    ax[ind].plot(X, Y, color='k', lw=linewidth, solid_joinstyle='bevel', solid_capstyle='butt')
            # View range
            viewRange_dict[(row,col)] = list(m)+list(viewRange[m])
            # Draw initial value
            if showInitVal:
                InitVal = "{0:0.0f}".format(Y[0])
                if m[0] == 'Voltage':
                    InitVal += ' mV'
                elif m[0] == 'Current':
                    InitVal += ' pA'
                elif m[0] == 'Stimulus':
                    if stimReflectCurrent:
                        InitVal += ' pA'
                    else:
                        InitVal = ''

                ax[ind].text(X[0]-0.03*(viewRange[m][0][1]-viewRange[m][0][0]),  Y[0]-1, InitVal, ha='right', va='center', color=colorfy[n%len(colorfy)])

            if m[1] not in channels:
                channels.append(m[1])
            
            # update axis
            row, col = {
                'ver': (row+1, col),
                'hor': (row, col+1),
                'cha': (row+1 if c<nrows-1 else 0, col+1 if c==nrows-1 else col),
                'epi': (row+1 if c==ncols-1 else row, col+1 if c<ncols-1 else 0)
                }.get(gridSpec[:3].lower())
            
        if annotation.lower() != 'none':
            final_notes = writeEpisodeNote(zData, viewRange[m][0], channels=channels, mode=annotation)
            # Draw more annotations
            textbox.append(TextArea(final_notes, minimumdescent=False, textprops=dict(color=colorfy[n%len(colorfy)])))
            
        # Group all the episode annotation text
    if annotation.lower() != 'none':
        box = VPacker(children=textbox, align="left",pad=0, sep=2)
        annotationbox = AnchoredOffsetbox(loc=3, child=box, frameon=False, bbox_to_anchor=[1, 1.1])
        ax[-1].add_artist(annotationbox)
        scalebar = [annotationbox]
    else:
        scalebar = []

    # set axis
    for c, vr in enumerate(viewRange_dict.items()):
        l, r = vr
        ind = np.ravel_multi_index(l, (nrows, ncols), order='C')
        ax[ind].set_xlim(r[2])
        ax[ind].set_ylim(r[3])
        # Add scalebar
        if scalebarAt.lower()=='all' or (scalebarAt.lower()=='first' and ind in first_last_mat[0]) or (scalebarAt.lower()=='last' and ind in first_last_mat[-1]):
            scalebar.append(AddTraceScaleBar(xunit='ms', yunit='mV' if r[0]=='Voltage' else 'pA', ax=ax[ind]))
        else: # including 'none'
            TurnOffAxis(ax=ax[ind])
            
        plt.subplots_adjust(hspace = .001)
        # temp = 510 + c
        temp = tic.MaxNLocator(3)
        ax[ind].yaxis.set_major_locator(temp)

        # Draw annotation artist for each export
        DrawAnnotationArtists(artists, axs=[ax[ind]])
                        
    if (isinstance(setFont, str) and setFont.lower() in ['default', 'arial', 'helvetica']) or \
                    (isinstance(setFont, bool) and setFont):
        SetFont(ax, fig, fontsize=fontSize, fontname=os.path.join(__location__,'../resources/Helvetica.ttf'))
    else:
        SetFont(ax, fig, fontsize=fontSize, fontname=setFont)

    # save the figure
    if adjustFigW:
        fig_size = (fig_size[0]*ncols, fig_size[1])
    if adjustFigH:
        fig_size = (fig_size[0], fig_size[1]*nrows)
    fig.set_size_inches(fig_size)
    
    # plt.subplots_adjust(hspace=-0.8)
    fig.savefig(saveDir, bbox_inches='tight', bbox_extra_artists=tuple(scalebar), dpi=dpi, transparent=True)
    # Close the figure after save
    plt.close(fig)
    if '.svg' in saveDir:
        svg2eps_ai(source_file=saveDir, target_file=saveDir.replace('.svg', '.eps'))

    return(ax)
    

def data2csv(data):
    return

def embedMetaData(ax):
    """embedding meta data to a figure"""
    return



if __name__ == '__main__':
    sys.path.append("D:/Edward/Documents/Assignments/Scripts/Python/PySynapse")
#
#    data = np.load('R:/tmp.npz')
#    df, index, viewRange, saveDir, colorfy = data['df'].tolist(), data['index'].tolist(), data['viewRange'][0],\
#                                                data['saveDir'].tolist(), data['colorfy'].tolist()
#    # plot the figure
#    ax= PlotTraces(df=df, index=index, viewRange=viewRange, saveDir='R:/tmp.eps', colorfy=colorfy, setFont=True)
    nrows, ncols = 5,2
    row, col  = 0, 0
    for n in np.arange(0,2):
        for c in np.arange(0, 5):
            print((row, col))
            row, col = {
            'ver': (row+1, col),
            'hor': (row, col+1),
            'cha': (row+1 if c<nrows-1 else 0, col+1 if c==nrows-1 else col),
            'epi': (row+1 if c==ncols-1 else row, col+1 if c<ncols-1 else 0)
            }.get('cha')
