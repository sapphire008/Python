# -*- coding: utf-8 -*-
"""
Created on Sun Dec 06 09:02:29 2015

GA spike detection

Zarifia et al., 2015
A new evolutionary approach for neural spike detection based on genetic algorithm

@author: Edward
"""

import sys
sys.path.append("D:/Edward/Documents/Assignments/Scripts/Python/Plots")
from ImportData import NeuroData
sys.path.append("D:/Edward/Documents/Assignments/Scripts/Python/Spikes")
from spk_util import *
sys.path.append("D:/Edward/Documents/Assignments/Scripts/Python/generic")
from MATLAB import *
from matplotlib import pyplot as plt

import scipy.signal as sg
import scipy.optimize as op

def GA_spk_detect(Vs, ts):
    Vs = np.concatenate(([Vs[0]], Vs, [Vs[-1]]))
    psi = Vs[1:-1]**2 - Vs[2:] * Vs[:-2]
    Vs = Vs[1:-1] # recover original time series
    
    def find_C(psi, C):
        thresh = C * np.mean(psi)
        ind, pks = findpeaks(psi, mph=thresh, mpd=20)
        # Calculate SNR
        # Debug plot
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.plot(Vs)
        ax.plot(ind, 150*np.ones(len(ind)), 'o')
        
        
        
        
    
if __name__ == '__main__':
    datadir = 'D:/Data/Traces/2015/11.November/Data 20 Nov 2015/Slice C.20Nov15.S1.E10.dat'
    # Load data
    zData = NeuroData(datadir, old=True)
    ts = zData.Protocol.msPerPoint
    Vs = zData.Current['A']
    #Vs = spk_filter(Vs, ts, Wn=[300., 3000.], btype='bandpass')
    Vs = spk_window(Vs, ts, [0,5000])