# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:06:23 2015

Spike sorting EC-PC

Not working at all. Specifically, the distribution of Hilbert power spectrum
is quite different from what is being noted in the paper. Therefore, the model
/ curve fitting procedure fails. 

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

def spk_snr(Vs, method='rms'):
    if method == 'rms':
        return( (np.max(Vs) - np.min(Vs)) / rms(Vs) )
    else:
        raise(NotImplementedError("'%s' method is not implemented"%(method)))

# Least square solver function
def lstsq(x, y):
    """Solve m, c for linear equation
    y = mx + c given x, y
    """
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y)[0]
    return(m, c)
    
    
def hilbert_Z(Vs):
    """Calculate power of Hilbert transform"""
     # Data variance
    sigma_squared = np.var(Vs)
    # Hilbert transform of neurla recording Vs
    Vs, l = padzeros(Vs) # pad zeros for faster fft performance
    Vst = sg.hilbert(Vs)
    # Instantaneous energy in Hilbert space, normalized to data variance
    Z = np.abs(Vst)**2 / sigma_squared
    Z = Z[0:l]
    # Truncate when Z0 < var(Vs)
    #Z = Z[Z<sigma_squared]
    #Z = Z[Z<40]
    return(Z)

def spk_detect_EC_PC(Vs, ts, window=5.0, p_thresh=0.95, fz_thresh=1e-6,
                     z_thresh=3.):
    """Detect extracellular spikes with EC-PC algorithm.
    Inputs:
        Vs: time series of recording
        ts: sampling rate, in [ms]
        window: sliding window to detect spike. Consider this parameter
        as the width of the spike. Default 2.0 ms.
        p_thresh: probability threshold for spike detection. Default 0.95.
        fz_thresh: minimum power to use to calcualte EC-PC. The power needs to
                   be greater than 0. Default 1e-6.
        z_thresh: threshold for power. Use this to filter out low power noise.
                  Default 5.0.
        
    Returns:
        ind: indices of detected spikes
        
    The idea is that neural recordings have two components, the exponentially 
    distributed noise (EC), and power distributed spikes / signal (PC). 
    This method is claimed to beat other popular threshold based spike
    detection algorithms, including RMS, median (Qurioga et al., 2004), 
    nonlinear energy operator (NEO), and continuous wavelet transform (CWT) 
    (Nenadic and Burdick, 2005), shown in:
    Wing-kin Tam, Rosa So, Cuntai Guan, Zhi Yang. EC-PC spike detection for
    high performance Brain-Computer Interface. 2015. IEEE.
    
    The script is based on the most updated and detailed version of a series
    of papers lead by Zhi Yang:
    Yin Zhou, Tong Wu, Amir Rastegarnia, Cuntai Guan, Edward Keefer, Zhi Yang.
    On the robustness of EC–PC spike detection method for online neural 
    recording. 2014. Journal of Neuroscience Methods. 235: 316-330.
    """
    # Hilbert transform power Z
    Z0 = hilbert_Z(Vs)
    
    # Use distribution fitting to find a, b, c, lambda1, lambda2
    f_Z, Z0 = np.histogram(Z0, bins=500, density=True)
    # Center the bin
    Z0 = Z0[:-1] + (Z0[1] - Z0[0])/2.
    # Take f_Z > 0 only
    Z0 = Z0[f_Z>fz_thresh]
    f_Z = f_Z[f_Z>fz_thresh]
    # Take Z0 > z_thresh only
    f_Z = f_Z[Z0>z_thresh]
    Z0 = Z0[Z0>z_thresh]
    
    def f(Z, lambda1, lambda2, a, b, c):
        f_n_Z = a*np.exp(-lambda1 * Z)
        f_d_Z = b / (Z**((3+2*lambda2)/(2*lambda2)) + c)
        # f_d_Z = b / (Z**lambda2 + c)
        return(f_n_Z + f_d_Z)
    
    popt, _ = op.curve_fit(f, Z0, f_Z)
    lambda1, lambda2, a, b, c = popt
    f_n_Z = a*np.exp(-lambda1 * Z0)
    f_d_Z = b / (Z0**((3+2*lambda2)/(2*lambda2)) + c)
    # f_d_Z = b / (Z0**lambda2 + c)
    p_Z = f_d_Z / (f_d_Z + f_n_Z)
       
    # Debug plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(Z0, f_Z, label='Data')
    ax.plot(Z0, f_n_Z, label='EC')
    ax.plot(Z0, f_d_Z, label='PC')
    ax.plot(Z0, f_d_Z + f_d_Z, label='Sum')
    ax.legend()
    ax.set_yscale('log')
    fig.set_size_inches(20, 6)
        
    # Slide the window across the time series and calculate the probability
    m = int(np.ceil(window / ts / 2.0))
    m = np.arange(-m, m+1, 1)
    # calculate expected iterations
    t_ind = np.arange(-min(m), len(Vs), len(m))
    nbin = len(t_ind)
    P_vect = np.zeros(nbin)
    nbin -= 1
    for n, t in enumerate(t_ind):
        i = m + t
        if n == nbin:
            i = i[i<len(Vs)]
        # calcualte power of the current window
        Z_i = hilbert_Z(Vs[i])
        Z_i = max(Z_i) # winner takes all
        # Find the closest match of Z in p_Z
        P_vect[n] = np.interp(Z_i, Z0, p_Z)
    
    # Threshold filter the probability vector
    ind = np.where(P_vect<p_thresh)[0]
    ind = t_ind[ind]

    # Debug plot
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(Vs)
    ax.plot(ind,140*np.ones(len(ind)), 'o')
    fig.set_size_inches(30,10)
    
    
    
if __name__ == '__main__':
    datadir = 'D:/Data/Traces/2015/11.November/Data 20 Nov 2015/Slice C.20Nov15.S1.E10.dat'
    # Load data
    zData = NeuroData(datadir, old=True)
    ts = zData.Protocol.msPerPoint
    Vs = zData.Current['A']
    #Vs = spk_filter(Vs, ts, Wn=[300., 3000.], btype='bandpass')
    Vs = spk_window(Vs, ts, [0,5000])
    spk_detect_EC_PC(Vs, ts, window=5.0, p_thresh=0.95, fz_thresh=1e-6,
                     z_thresh=3.0)
#    window=2.0
#    p_thresh=0.95
#    fz_thresh=1e-6
#    z_thresh=5.