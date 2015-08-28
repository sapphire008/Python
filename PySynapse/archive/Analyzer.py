# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 22:46:12 2015

@author: Edward
"""

import numpy as np
from scipy.signal import savgol_filter, butter
import Data


def movingAverageFilter(data, window=10):
    longSide
    flatData = np.ones(1, window)
    cheatShift = int(windowSize / 2);
    outData = filter(flatData./(windowSize),1,cat(longSide, flatData.*inData(1), inData, flatData.*inData(end)));
    outData = outData(windowSize + cheatShift:length(inData) + windowSize + cheatShift - 1);
    
def butterworthFilter(data, FilterOrder, NormalizedFilterFrequency):
    [B,A] = butter(FilterOrder, NormalizedFilterFrequency)
    data_mean = mean(data);
    dataFilt = [data-data_mean, zeros(1,2^nextpow2(numel(data))-numel(data))];# pad zeros to data
    dataFilt = filtfilt(B,A,dataFilt);#filter data
    dataFilt = dataFilt(1:numel(data))+data_mean; #recover data
    return(dataFilt)

def filterData(data, FilterType, FilterLength, FilterOrder):
    if FilterType=='sgolay':
        return(savgol_filter(data, FilterLength, FilterOrder))
    elif FilterType == 'movingaverage':
        data
        asdf
    elif FilterType = 'medfilt':
        asdf
    elif FilterType = 'butter':
        return(butterworthFilter(data, FilterOrder, FilterLength))
    else:
        raise ValueError("Unrecognized filter type: %s\n" %(FilterType))

def DetectEvent(zData, cumulativeDerThresh, event_type = 'EPSP', FilterType='movingaverage',FilterLength=5, FilterOrder=2, DerivFilterType='sgolay',DerivFilterLength=7, DerivFilterOrder=2, Window=None,SIUArtifact=None):
    event_type = event_type.upper()
    FilterType = FilterType.lower()
    if event_type == 'EPSP' or event_type == 'IPSC':
        PSPDOWN = False # post synaptic event is not down
    elif event_type == 'IPSP' or event_type == 'EPSC':
        PSPDOWN = True # post synaptic event is down
        # switch sign of threshold if downward event
        cumulativeDerThresh = -1 * cumulativeDerThresh;
    else: # fall through
        raise(ValueError("Unrecognized event type: %s\n" % (event_type)))
        
    
    
        
                
    