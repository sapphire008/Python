# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 17:56:32 2015

@author: Edward
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 11:16:23 2015

@author: Edward
"""

import sys
sys.path.append('D:/Edward/Documents/Assignments/Scripts/Python/Plots')
import numpy as np
import matplotlib.pyplot as plt
from PublicationFigures import PublicationFigures as PF

color = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd154','#17becf'] # tableau10, or odd of tableau20

def SingleEpisodeTraces(base_dir, result_dir, eps=None, channels=['A'], 
                 streams=['Volt','Cur']):
    """Helper function to export traces from a single episode"""
    if eps is None:
        return
    # Load data
    data = [base_dir %(epi) for epi in eps]
    K = PF(dataFile=data,savePath=result_dir, old=True, channels=channels, streams=streams)

    
    # Arrange all plots vertcially
    fig, ax = plt.subplots(nrows=len(channels)*len(streams), ncols=1, sharex=True)
    pcount = 0
    yunit_dict = {'Volt':'mV','Cur':'pA','Stim':'pA'}

    for c in channels: # iterate over channels
        for s in streams: # iterate over streams
            ax[pcount].plot(K.data.table['time'], K.data.table[s+c], label=pcount, c='k')
            K.AddTraceScaleBar(xunit='ms', yunit=yunit_dict[s],ax=ax[pcount])
            dataposition = [K.data.table['time'][0], K.data.table[s+c][0]]
            datatext = '%.0f'%(dataposition[1]) + yunit_dict[s]
            K.TextAnnotation(text=datatext, position=dataposition, ax=ax[pcount], color='k',
                       xoffset='-', yoffset=None, fontsize=None,ha='right',va='center')
            pcount += 1

    # Finally annotate the episode information at the bottom
    # fig.suptitle(K.data.meta['notes'][0])
    pad = np.array(ax[-1].get_position().bounds[:2]) * np.array([1.0, 0.8])
    fig.text(pad[0], pad[1], K.data.meta['notes'][0], ha='left',va='bottom')
    K.SetFont(ax=ax, fig=fig)
    
    fig.set_size_inches(10,5)
    #K.Save()
    return(K, ax, fig)
    
    
def MultipleTraces(base_dir, result_dir, eps=None, channel='A', stream='Volt', color=color, window=None):
    """Helper function to draw multiple traces in a single axis"""
    if eps is None:
        return
    # load data
    data = [base_dir %(epi) for epi in eps]
    nep = len(eps)
    K = PF(dataFile=data, savePath=result_dir, old=True, channels=[channel], streams=[stream])
    
    # Initialize axis
    fig, ax = plt.subplots(nrows=1, ncols=1)
    yunit_dict = {'Volt':'mV','Cur':'pA','Stim':'pA'}
    # Draw plots
    for n in range(nep):
        x, y = K.data.table[n]['time'], K.data.table[n][stream+channel]
        ts = x[1] - x[0]
        if window is not None:
            x, y = x[int(window[0]/ts) : int(window[1]/ts)], y[int(window[0]/ts) : int(window[1]/ts)]
            
        ax.plot(x, y, label=n, c=color[n%nep])
        if n == (nep-1): # add trace bar for the last episode
            K.AddTraceScaleBar(xunit='ms', yunit=yunit_dict[stream], ax=ax)
        if n == 0: # annotate the first episode
            dataposition = [np.array(x)[0], np.array(y)[0]]
            datatext = '%.0f'%(dataposition[1]) + yunit_dict[stream]
            K.TextAnnotation(text=datatext, position=dataposition, ax=ax, color='k',
                       xoffset='-', yoffset=None, fontsize=None,ha='right',va='center')
    
    # update the graph               
    fig.canvas.draw()    
    # Finally, annotate the episode information at the bottom
    pad = np.array(ax.get_position().bounds[:2]) * np.array([1.0, 0.8])
    #fontsize = ax.yaxis.get_major_ticks()[2].label.get_fontsize()
    inc = 0.025    
    #inc = K.xydotsize(ax, s=fontsize,scale=(1.,1.))[1]
    #print(inc)
    #inc = inc/(ax.get_ybound()[1] - ax.get_ybound()[0])*(ax.get_position().bounds[3]-ax.get_position().bounds[1])
    #print(inc)

    for n, _ in enumerate(eps):
        # print(pad[0], pad[1]+inc*n)
        fig.text(pad[0], pad[1]-inc*n, K.data.meta['notes'][n], ha='left',va='bottom', color=color[n%len(color)])
    
    K.SetFont(ax=ax, fig=fig)
    fig.set_size_inches(6,4)
    return(K, ax, fig)
    
def ConcatenatedTraces(base_dir, result_dir, eps=None, channel='A', stream='Volt', gap=0.05, color='k'):
    """Heper function to export horizontally concatenated traces
    gap: gap between consecutive plots. gap * duration of plot. Default is 0.05,
         or 5% of the duration of the plot.
    """
    if eps is None:
        return
    # load data
    data = [base_dir %(epi) for epi in eps]
    nep = len(eps)
    K = PF(dataFile=data, savePath=result_dir, old=True, channels=[channel], streams=[stream])
    
    # Initialize axis
    fig, ax = plt.subplots(nrows=1, ncols=1)
    yunit_dict = {'Volt':'mV','Cur':'pA','Stim':'pA'}
    
    gap *= max([x['time'].iloc[-1] - x['time'].iloc[0] 
                    for x in K.data.table])

    # initialize the time
    x0 = 0.0    
    # Draw plots
    for n in range(nep):
        x, y = K.data.table[n]['time'], K.data.table[n][stream+channel]
        # update time shift
        x = x + x0
        x0 = x.iloc[-1] + gap
            
        ax.plot(x, y, label=n, c=color)
        if n == (nep-1): # add trace bar for the last episode
            K.AddTraceScaleBar(xunit='ms', yunit=yunit_dict[stream], ax=ax, 
                               xscale=x.iloc[-1]-x.iloc[0])
        if n == 0: # annotate the first episode
            dataposition = [np.array(x)[0], np.array(y)[0]]
            datatext = '%.0f'%(dataposition[1]) + yunit_dict[stream]
            K.TextAnnotation(text=datatext, position=dataposition, ax=ax, color='k',
                       xoffset='-', yoffset=None, fontsize=None,ha='right',va='center')
                       
    # update the graph                       
    fig.canvas.draw()    
    # Finally, annotate the episode information at the bottom
    pad = np.array(ax.get_position().bounds[:2]) * np.array([1.0, 0.8])
    #fontsize = ax.yaxis.get_major_ticks()[2].label.get_fontsize()
    inc = 0.025    
    #inc = K.xydotsize(ax, s=fontsize,scale=(1.,1.))[1]
    #print(inc)
    #inc = inc/(ax.get_ybound()[1] - ax.get_ybound()[0])*(ax.get_position().bounds[3]-ax.get_position().bounds[1])
    #print(inc)

    for n, _ in enumerate(eps):
        # print(pad[0], pad[1]+inc*n)
        fig.text(pad[0], pad[1]-inc*n, K.data.meta['notes'][n], ha='left',va='bottom', color=color)
    
    K.SetFont(ax=ax, fig=fig)
    fig.set_size_inches(50,4)
    return(K, ax, fig)
    
        
if __name__ == '__main__':
    #base_dir = 'D:/Data/2015/07.July/Data 10 Jul 2015/Neocortex K.10Jul15.S1.E%d.dat'
    #result_dir = 'C:/Users/Edward/Documents/Assignments/Case Western Reserve/StrowbridgeLab/Projects/TeA Persistence Cui and Strowbridge 2015/analysis/ADP under Pirenzepine - 07152015/example.svg'
    #eps = [38]
    #channels=['A']
    #streams=['Volt','Cur']
    #K, ax, fig = SingleEpisodeTraces(base_dir, result_dir, eps=eps)
    #######################################    
    #base_dir = 'D:/Data/Traces/2015/10.October/Data 21 Oct 2015/Neocortex C.21Oct15.S1.E%d.dat'
    #result_dir = 'C:/Users/Edward/Documents/Assignments/Case Western Reserve/StrowbridgeLab/Projects/TeA Persistence Cui and Strowbridge 2015/analysis/Self termination with stimulation - 10222015/example.eps'
    #result_dir = 'C:/Users/Edward/Desktop/asdf.svg'
    #eps = range(53, 58, 1)
    #channel = 'A'
    #stream = 'Volt'
    #K, ax, fig = MultipleTraces(base_dir, result_dir, eps=eps, channel=channel, stream=stream, window=[2000, 4000])
    #fig.savefig(result_dir, bbox_inches='tight', dpi=300)
    ########################################
    base_dir = 'D:/Data/Traces/2015/06.June/Data 17 Jun 2015/Neocortex H.17Jun15.S1.E%d.dat'
    eps = np.arange(150, 160, 1)
    result_dir = 'C:/Users/Edward/Desktop/concatplot.svg'
    channel = 'A'
    stream = 'Volt'
    K, ax, fig = ConcatenatedTraces(base_dir, result_dir, eps=eps, channel=channel, stream=stream, gap=0.05)
    fig.savefig(result_dir, bbox_inches='tight', dpi=300)
    
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
