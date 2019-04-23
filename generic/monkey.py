# -*- coding: utf-8 -*-
"""
Created on Tue May  8 15:08:22 2018

@author: Edward
"""

from ImportData import load_trace
from spk_util import *

def get_SFA_bins(Cell="NeocortexChRNBM K.16Oct17", Episode="S1.E1", num_bins=40):
    zData = load_trace([Cell, Episode])
    ts = zData.Protocol.msPerPoint
    stim = spk_get_stim(zData.Stimulus['A'], ts)
    _, spike_times, _ = spk_count(spk_window(zData.Voltage['A'], ts, stim), ts)
    
    time_bins = np.linspace(0, stim[1]-stim[0], num_bins+1)
    time_bins = np.c_[time_bins[:-1], time_bins[1:]]
    time_bins = np.c_[time_bins.mean(axis=1), time_bins]
    
    spike_bins = np.zeros(time_bins.shape[0])
    
    for n, t in enumerate(time_bins):
        spike_bins[n] = np.logical_and(spike_times >= time_bins[n, 1], spike_times < time_bins[n, 2]).sum(dtype=int)
    
    spike_bins = np.cumsum(spike_bins, dtype=int)
    
    return spike_bins



spike_bins_1 = get_SFA_bins(Episode="S1.E1")
spike_bins_2 = get_SFA_bins(Episode="S1.E6")
spike_bins_3 = get_SFA_bins(Episode="S1.E10")
sb = np.c_[spike_bins_1, spike_bins_2, spike_bins_3]
mean_sb = sb.mean(axis=1)
serr_sb = serr(sb, axis=1)
