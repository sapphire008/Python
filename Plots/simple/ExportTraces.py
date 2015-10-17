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
sys.path.append('C:/Users/Edward/Documents/Assignments/Scripts/Python/Plots')
import numpy as np
import matplotlib.pyplot as plt
from PublicationFigures import PublicationFigures as PF

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
            dataposition = [0, K.data.table[s+c][0]]
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
    
    
def ConcatenatedTraces():
    """Heper function to export horizontally concatenated traces"""
            
if __name__ == '__main__':
    base_dir = 'D:/Data/2015/07.July/Data 10 Jul 2015/Neocortex K.10Jul15.S1.E%d.dat'
    result_dir = 'C:/Users/Edward/Documents/Assignments/Case Western Reserve/StrowbridgeLab/Projects/TeA Persistence Cui and Strowbridge 2015/analysis/ADP under Pirenzepine - 07152015/example.svg'
    eps = [38]
    channels=['A']
    streams=['Volt','Cur']
    K, ax, fig = SingleEpisodeTraces(base_dir, result_dir, eps=eps)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
