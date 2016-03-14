# -*- coding: utf-8 -*-
"""
CWT unsupervised spike detector
Created on Mon Dec 21 23:47:39 2015
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

from wavelet import *


def bior_wavelet():
    """
    """
    return

def spk_detect_cwt(Vs,ts):
    """Unsupervised spike detection using continuous wavelet transform"""
    
    
    
if __name__ == '__main__':
    datadir = 'D:/Data/Traces/2015/11.November/Data 20 Nov 2015/Slice C.20Nov15.S1.E10.dat'
    # Load data
    zData = NeuroData(datadir, old=True)
    ts = zData.Protocol.msPerPoint
    Vs = zData.Current['A']
    #Vs = spk_filter(Vs, ts, Wn=[300., 3000.], btype='bandpass')
    Vs = spk_window(Vs, ts, [0,5000])