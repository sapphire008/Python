# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 15:51:22 2019

@author: Edward
"""
import os
import time
from spk_util import *
from MATLAB import *
from importData import *


def monitor_access_resistance_vclamp(ep_file, printResults=True, scalefactor=1.0, window=[995, 1015], fid=None):
    zData = NeuroData(dataFile=ep_file, old=True, infoOnly=False)
    ts = zData.Protocol.msPerPoint
    Is = spk_window(zData.Current['A'], ts, window)
    Vs = spk_window(zData.Stimulus['A'], ts, window)
    
    # Estimate capacitance
    R_series, tau, rsquare = spk_vclamp_series_resistance(Is, Vs, ts, window=window, scalefactor=scalefactor, direction='up')
    
    if printResults:
        if fid is not None:
            fid.write('{}\n'.format(ep_file))
            fid.write('R_series = {:.4f}\n'.format(R_series))
            fid.write('tau = {:.4f}\n'.format(tau))
            fid.write('rsquare = {:.4f}\n'.format(rsquare))
            fid.write('scale factor = {:.4f}\n'.format(scalefactor))
        
        print('{}'.format(ep_file))
        print('R_series = {:.4f}'.format(R_series))
        print('tau = {:.4f}'.format(tau))
        print('rsquare = {:.4f}'.format(rsquare))
        print('scale factor = {:.4f}'.format(scalefactor))
        
        
if __name__ == '__main__':
    Path = 'D:/Data/Traces/test'
    current_files, _ = SearchFiles(Path, '*.dat', 'D') # get current list of files
    pause_timer = 5 # 5 seconds
    scalefactor = 1. # current scale factor
    if not isempty(current_files):
        monitor_access_resistance_vclamp(current_files[-1], scalefactor=scalefactor)
    # Start the log
    fid = open(os.path.join(Path, 'R_series_monitor.log'), 'a')
    while True:
        new_files, _ = SearchFiles(Path, '*.dat', 'D')
        new_0 = np.setdiff1d(new_files, current_files)
        if not isempty(new_0):
            current_files = new_files
            # call function
            monitor_access_resistance_vclamp(current_files[-1], scalefactor=scalefactor, fid=fid)
        time.sleep(pause_timer)
        
    fid.close()
        