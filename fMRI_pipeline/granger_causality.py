# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:14:35 2014

@author: dcui
"""

# Import some modules
PYTHONPKGPATH = '/hsgs/projects/jhyoon1/pkg64/pythonpackages/'

#from __future__ import print_function  # Python 2/3 compatibility
import sys,os
sys.path.append(os.path.join(PYTHONPKGPATH,'nibabel-1.30'))
#import nibabel# required for nipy
sys.path.append(os.path.join(PYTHONPKGPATH,'nitime'))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import csv2rec

#import nitime
import nitime.analysis as nta
import nitime.timeseries as ts
import nitime.utils as tsu
from nitime.viz import drawmatrix_channels

# Define some parameters
DataPath = '/hsgs/projects/jhyoon1/midbrain_pilots/mid/analysis/EffectiveConnectivity/'
result_dir = '/hsgs/projects/jhyoon1/midbrain_pilots/mid/analysis/EffectiveConnectivity/'
subjects = ['MP020_050613','MP021_051713','MP022_051713','MP023_052013','MP024_052913',
            'MP025_061013','MP026_062613','MP027_062713','MP028_062813','MP029_070213',
            'MP030_070313','MP031_071813','MP032_071013','MP033_071213','MP034_072213',
            'MP035_072613','MP036_072913','MP037_080613']
target_data = '%s_timeseries_data.csv'
col_offset = 8# starting which column are the ROIs data?
TR = 2 # TR
f_ub = np.inf #upper bound of frequency of interest: low pass
f_lb = 1/128 # lower bound of frequency of interest: high pass
granger_order = 1 #predict the current behavior of the time-series based on how many time points back?
conditions = ['lose5','lose1','lose0','gain0','gain1','gain5']

# Initialize stack objects to record data
stack_coh = np.zeros(len(subjects),dtype = {'names': conditions,'formats': ['f4']*len(conditions) })
stack_gl = np.zeros_like(stack_coh)
stack_R = np.zeros_like(stack_coh)

def granger_causality_analysis(time_series, f_lb, f_ub, granger_order, roi_names,result_dir, s='', c=''):
    # initialize GrangerAnalyzer object
    G = nta.GrangerAnalyzer(time_series, order = granger_order)
    # initialize CoherenceAnalyzer 
    C = nta.CoherenceAnalyzer(time_series)
    # initialize CorrelationAnalyzer
    R = nta.CorrelationAnalyzer(time_series)
    # get the index of the frequency band of interest for different analyzer
    freq_idx_G = np.where((G.frequencies > f_lb) * (G.frequencies < f_ub))[0]
    freq_idx_C = np.where((C.frequencies> f_lb) * (C.frequencies < f_ub)) [0]
    # average the last dimension
    coh = np.mean(C.coherence[:, :, freq_idx_C], -1) 
    gl = np.mean(G.causality_xy[:, :, freq_idx_G], -1)
    # Difference in HRF between ROI may result misattriution of causality
    # examine diference between x-->y and y-->x
    g2 = np.mean(G.causality_xy[:,:,freq_idx_G] - G.causality_yx[:,:,freq_idx_G],-1)
    # Figure organization:
    # causality_xy: x-->y, or roi_names[0]-->roi_names[1], roi_names[0]-->roi_names[2], etc.
    # this makes: given a column, transverse through rows. Directionality is
    # from current column label to each row label
    # plot coherence from x to y, 
    drawmatrix_channels(coh, roi_names, size=[10., 10.], color_anchor=0)
    plt.title(('%s %s pair-wise Coherence' % (s, c)).replace('  ',' '))
    plt.savefig(os.path.join(result_dir,s,('%s_%s_pairwise_coherence.png' % (s, c)).replace('__','_')))
    # plot correlation from x to y
    #drawmatrix_channels(R.corrcoef, roi_names, size=[10., 10.], color_anchor=0)
    #plt.title(('%s %s pair-wise Correlation' % (s, c)).replace('  ',' '))
    #plt.savefig(os.path.join(result_dir,s,('%s_%s_pairwise_correlation.png' % (s, c)).replace('__','_')))
    # plot granger causality from x to y
    drawmatrix_channels(gl, roi_names, size=[10., 10.], color_anchor=0)
    plt.title(('%s %s pair-wise Granger Causality' % (s, c)).replace('  ',' '))
    plt.savefig(os.path.join(result_dir,s,('%s_%s_pairwise_granger_causality.png' % (s, c)).replace('__','_')))
    # plot granger causliaty forward-backward difference
    drawmatrix_channels(g2, roi_names, size=[10., 10.], color_anchor = 0)
    plt.title(('%s %s pair-wise Granger Causality Forward-Backward Difference' % (s, c)).replace('  ',' '))
    plt.savefig(os.path.join(result_dir,s,('%s_%s_granger_causality_forward_backward_diff.png' % (s, c)).replace('__','_')))
    # close all the figures
    plt.close("all")
    return(coh, gl, g2, G, C, R)


# Pair-wise Granger Causality
for n, s in enumerate(subjects): # transverse through subjects
    # construct current data path
    current_data = os.path.join(DataPath,s,target_data % (s))
    # read csv file
    data_rec = csv2rec(current_data)
    roi_names = np.array(data_rec.dtype.names)[(0+col_offset):]
    n_seq = len(roi_names) # number of rois        
    n_samples = data_rec.shape[0] # number of time points
    data = np.zeros((n_seq, n_samples)) # initialize output data
    # import the data to numpy
    for n_idx, roi in enumerate(roi_names):
        data[n_idx] = data_rec[roi]
        
    # normalize the data of each ROI to be in units of percent change
    pdata = tsu.percent_change(data)
     # get the index of rows/observations to include
    phase_idx = np.logical_or(data_rec['phases'] == 'Cue', data_rec['phases'] == 'Delay')
    pdata = np.delete(pdata,np.where(np.logical_not(phase_idx)),axis=1)
    cond_names = np.delete(data_rec['conditions'], np.where(np.logical_not(phase_idx)), axis=0)
    #pdata[:,np.where(np.logical_not(phase_idx))] = 0
    # initialize TimeSeries object
    #time_series = ts.TimeSeries(pdata,sampling_interval=TR)
    # Do granger causality analysis for each condition
    for m, c in enumerate(conditions):
        # get time points of current condition
        time_series = ts.TimeSeries(pdata[:,np.where(cond_names==c)], sampling_interval=TR)
        # do granger causality analysis
        coh, gl, g2, G, C, R = granger_causality_analysis(time_series, f_lb, f_ub, granger_order, roi_names, result_dir, s, c)
        # store current values into a structure array
        stack_coh[n] = stack_coh[n] + (coh,)
        stack_gl[n][m] = stack_gl[n] + (gl,)
        stack_R[n][m] = stack_R[n] + (R.corrcoef,)
        # clear current time series
        time_series_c = None
        
    # stack the value to sum_'s
    stack_coh = np.dstack((stack_coh, coh))
    stack_gl = np.dstack((stack_gl, gl))
    stack_R = np.dstack((stack_R, R.corrcoef))
    # save extracted data
    np.savez(os.path.join(result_dir,s,'%s_data.npz' % (s)),G,C,R,coh,gl,g2)
    # clear the variables
    coh, gl, g2, G, C, R = (None, None, None, None, None, None)
    

# calcualte mean
mean_coh = np.mean(stack_coh, axis = 2)
mean_gl = np.mean(stack_gl, axis = 2)
mean_R = np.mean(stack_R, axis = 2)
# plot group averaged
figGroup_coh = drawmatrix_channels(mean_coh, roi_names,size=[10.,10.], color_anchor=0)
plt.title('Control pair-wise Coherence')
plt.savefig(os.path.join(result_dir,'Control_pair_wise_coherence.png'))
figGroup_gl = drawmatrix_channels(mean_gl, roi_names,size=[10.,10.], color_anchor=0)
plt.title('Control pair-wise Granger Causality')
plt.savefig(os.path.join(result_dir,'Control_pair_wise_granger_causality.png'))
figGroup_R = drawmatrix_channels(mean_R, roi_names,size=[10.,10.], color_anchor=0)
plt.title('Control pair-wise Correlation')
plt.savefig(os.path.join(result_dir,'Control_pair_wise_correlation.png'))

np.savez(os.path.join(result_dir,'Control_Group_average.npz'), stack_coh, 
         stack_gl, stack_R, mean_coh, mean_gl, mean_R)
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    













