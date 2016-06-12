# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:19:49 2015

Some routines for electrophysiology data analyses

@author: Edward
"""
import sys
import numpy as np
import scipy.signal as sg
import scipy.stats as st
# from pdb import set_trace
try:
    from MATLAB import *
except:
    try:
        from util.MATLAB import *
    except:
        sys.path.append('D:/Edward/Documents/Assignments/Scripts/Python/Spikes/')
        from MATLAB import *
    
        

def time2ind(t, ts, t0=0):
    """Convert a time point to index of vector
    ind = time2ind(t, ts, t0)
    Inputs:
        t: current time in ms
        ts: sampling rate in ms
        t0: (optional) time in ms the first index corresponds
            to. Defualt is 0.

    Note that as long as t, ts, and t0 has the same unit of time,
    be that second or millisecond, the program will work.

    Output:
        ind: index
    """
    if np.isscalar(t):
        t = [t]

    return(np.array([int(a) for a in (np.array(t) - t0) / ts]))

def ind2time(ind, ts, t0=0):
    """Convert an index of vector to temporal time point
     t = ind2time(ind, ts, t0=0)
     Inputs:
         ind: current index of the vector
         ts: sampling rate in ms
         t0: (optional) what time in ms does the first index
             correspond to? Defualt is 0

     Note that as long as t, ts, and t0 has the same unit of time,
     be that second or millisecond, the program will work.

     Output:
         t: current time in ms
    """
    if np.isscalar(ind):
        ind = [ind]
    return(np.array(ind) * ts + t0)

def spk_window(Vs, ts, Window, t0=0):
    """Window the time series
    Vs = spk_window(Vs, ts, Window, t0=0)
    Inputs:
      Vs: voltage time series
      ts: sampling rate [ms]
      Window: temporal window, in the format of [min_sec, max_sec]
      t0: (optional) what time in seconds does the first index
          correspond to? Defualt is 0.

    Note that as long as ts, Window, and t0 has the same unit,
    be that second or millisecond, the program will work.

    Output:
        Vs: windows Vs
    """
    dur = len(Vs)
    # Start / end indices
    def parse_window(x, none_allowed, min_allowed, max_allowed, func):
        if x is None:
            x = none_allowed
        else:
            x = func(x) # apply the transformation
            if x < min_allowed:
                x = min_allowed
            elif x > max_allowed:
                x = max_allowed

        return x

    func = lambda y: time2ind(y, ts, t0)[0]
    start_ind = parse_window(Window[0], 0, 0, dur, func=func)
    end_ind = parse_window(Window[1], dur, 0, dur, func=func)

    return Vs[start_ind:end_ind]


def spk_average(Vs, ts=None, Window=None, axis=0, t0=0):
    """Find the average of a series of traces
    Vs = spk_averagetrace(Vs, ts, Window, axis=0, t0=0)
    Inputs:
      Vs: time series
      ts: sampling rate [ms]. Necessary when specified Window.
      Window: temporal window, in the format of [min_ms, max_ms]
      dim: dimension to averge the trace.
           * None: average everything.
           * 0 over time. (Default)
           * 1 over trials.
      t0: (optional) what time in seconds does the first index
           correspond to? Defualt is 0.
    """
    # Window the time series
    if Window is not None:
        if ts is None:
            raise(ValueError('Please specify sampling rate ts.'))
        Vs = spk_window(Vs, ts, Window, t0);

    # Take the average
    if axis is not None:
        Vs = np.mean(Vs, axis=axis)
    else:
        Vs = np.mean(Vs) # average everything

    return(Vs)


def spk_count(Vs, ts, msh=-10.0, msd=1.0, **kwargs):
    """ Count the number of action potentials given a time series, using simple
        threshold based peak detection algorithm
    num_spikes, spike_time, spike_heights = spk_count(Vs, ts, **kwargs)
    Inputs:
        Vs: voltage time series [mV].
        ts: sampling rate [ms]
        msh: minimum height of the spike. Default -10.0 [mV].
        msd: minimum distances between detected spikes. Default 1.0 [ms].
        **kwargs: optional inputs for "findpeaks"

    Note that min_spike_height needs to be in the same unit as Vs;
              min_spike_dist needs to be in the same unit as ts.

    Outputs:
       num_spikes: number of spikes for each trial
       spike_time: indices of the spike, returned as one cell array of time
                   vectors per trial
       spike_heights: voltage of the spike [mV], returned as one cell array
                   of spike heights per trial
    """

    # find spikes
    if msd is not None:
        msd = float(msd) / float(ts)
    ind, spike_heights = findpeaks(Vs, mph=msh, mpd=msd, **kwargs)
    # Count number of spikes
    num_spikes = len(ind)
    # Convert spike timing
    spike_time = ind2time(ind, ts)

    return(num_spikes, spike_time, spike_heights)

def spk_filter(Vs, ts, Wn, N=6, btype='bandpass'):
    """Filter time series
    Vs: time series
    ts: sampling rate in [ms]. IMPORTANT: 'ts' must be millisecond
    Wn: cutoff frequency of the filter
    N: order of filter. Default 6.
    """
    Nq = 1.0/ts*1000/2
    #wp, ws = np.array(wp)/Nq, np.array(ws)/Nq
    #N, Wn = sg.buttord(wp, ws, gpass=gp, gstop=gs, analog=False)
    Wn = np.array(Wn) / Nq
    b, a = sg.butter(N, Wn, btype=btype, analog=False, output='ba')
    Vs_mean = np.mean(Vs)
    Vs -= Vs_mean
    l = len(Vs)
    pad = 2**nextpow2(l)
    if (pad - l) < (0.1*l):
        pad = 2**(nextpow2(l)+1)
    pad = pad - l # legnth of padding
    Vs = np.concatenate((Vs, np.zeros(1,pad)))
    Vs = sg.filtfilt(b, a, Vs, axis=-1, padtype=None, padlen=None)
    Vs = Vs[0:l] + Vs_mean
    return(Vs)


def spk_dirac(ts=1., dur=1, phi=0., h=1., collapse=True):
    """Make (summed) Dirac Delta function
        delta = spk_dirac(ts=1., dur=1., phi=0., h=1., collapse=True)
    Inputs:
        ts: sampling rate in seconds. Default is 1 second.
        dur: duration of the time series in seconds.
            ~Input a single value so that the time window is twice of this
             input, centered at 0. e.g. dur = 5 --> [-5, 5] time window
            ~Alternatively, input as a time window in seconds, e.g.
             dur = [-2, 5].
            ~If no input, default [-1, 1] window.
        phi: phase shift [seconds] of the Delta function. Default is 0 as in
             classic Dirac delta function. Input as a vector to return one
             Delta function for each phi.
        h: Height of the singularity (origin) where non-zero value occur.
           Deafult heigth is 1.
        collapse: [true|false] collaspe Dirac function with different phase
                  shifts by adding across phi (columns). Default is true.

    Output:
       delta: if not collpased, returns a matrix with row corresponding to
              time and columns corresponding to different phi; if collapsed,
              only a column vector is returned.

    Example usage: eph_dirac(1, [-5,5],[-2,-3,1,3],1,true).
    Returns a vector [0;0;1;1;0;0;1;0;1;0;0];
    """
    if np.isscalar(dur):
        dur = [-dur, dur]

    phi_ind = time2ind(phi, ts, dur[0])
    if collapse:
        # initialize the vector
        delta = np.zeros(int(1+np.diff(time2ind(dur, ts))))
        delta[phi_ind] = h
    else:
        # initialize a matrix
        delta = np.zeros((int(1+np.diff(time2ind(dur, ts))), np.size(phi)))
        for p, n in enumerate(phi_ind):
            delta[p, n] = h

    return(delta)

def spk_firing_rate(Vs, ts, method='gaussian', debug=False, sigma=300., n=5,
                    window=500.):
    """Estimate a continuous, time varying firing rate
    R = spk_firing_rate(Vs, ts, method, ...)
    Inputs:
        Vs: voltage time series, N x M matrix with N time points and M trials
            in units of [mV]
        ts: sampling rate [ms]
        method: method of calculating firing rate of a single trial. Default is
                'gaussian'. Available options are the following:
        1). 'gaussian': specify a Gaussian moving kernel to calculate firing
                        rate. (..., 'gaussian', sigma=0.3., n=5),
                        where sigma is the standard deviation (default is 0.1s)
                        and n is the number of standard deviations of Gaussian
                        kernel to use to convolve the data (default 5).
        2). 'rect': sepcify a rectangular moving kernel to calculate firing
                    rate. The default setting is (..., 'rect', window=0.5),
                    which specifies a moving kernel of 500ms.
        debug: turn on debug to print spike detection message.
    Outputs:
        R: time series of the same dimension as Vs, containing calculated
           firing rate in units of [Hz]

    A note for the unit: if ts, sigma, and window are in ms, the returned
    firing rate will be in MHz, instead of Hz.

    Implementation based on review by:
    Cunningham, J.P., Gilja, V., Ryu, S.I., Shenoy, K.V. Methods for
    Estimating Neural Firing Rates, and Their Application to Brain-Machine
    Interfaces. Neural Network. 22(9): 1235-1246 (2009).
    """
    # Detect spikes first
    _, t_spk, _ = spk_count(Vs, ts)
    assert t_spk is not None and len(t_spk)>0, "No spikes detected"
    if debug is True:
        print("Detected %d spikes\n"%(len(t_spk)))
    t_window = ind2time([0, len(Vs)-1], ts)

    # Make dirac function based on spike times
    R = spk_dirac(ts, t_window, t_spk, 1., True)

    # Switch between selection of convolution functions
    if method == 'gaussian':
        w = stationary_gaussian_kernel(ts, sigma=sigma, n=n)
        mtype = 'ks' # kernel smoothing (stationay)
    elif method == 'rect':
        w = stationary_rect_kernel(ts, window=window)
        mtype = 'ks' # kernel smoothing (stationary)
    else:
        raise(NotImplementedError('%s kernel method is not implemented'\
                %(method)))

    if mtype == 'ks': # kernel smoothing (stationary)
        # Convolve to get the firing rate
        R = np.convolve(R, w, mode='same')

    # convert from MHz to Hz
    R *= 1000.

    return(R)


def stationary_gaussian_kernel(ts, sigma=300., n=5):
    """Make gaussian kernel
    ts: sampling rate [ms]
    n: use n standard deviations below and above 0 (mean).
    sigma: standard deviation (width of Gaussian kernel) [ms].
           During Up state, sigma = 10ms according to:
           Neske, G.T., Patrick, S.L., Connor, B.W. Contributions of
           Diverse Excitatory and Inhibitory Neurons to Recurrent Network
           Activity in Cerebral Cortex. The Journal of Nueroscience.
           35(3): 1089-1105 (2015). But this sd size may be too small for
           other processes. So default is set to 300ms for a smoother
           firing rate curve.
    """
    t = np.arange(-n*sigma, n*sigma+ts, ts)
    w = 1./(np.sqrt(2.*np.pi)*sigma)*np.exp(-t**2/(2.*sigma**2))
    return(w)


def stionary_rect_kernel(ts, window=500.):
    """Make rectangular kernel
    ts: sampling rate [ms]
    window:window length in [ms]
    """
    # boxcar function
    t = time2ind(window, ts)
    w = np.concatenate((np.zeros(10), np.ones(t), np.zeros(10)))
    return(w)
    

def detectPSP_template_matching(Vs, ts, event, w=200, tau_RISE=1, tau_DECAY=4, mph=0.5, mpd=1, step=1, criterion='se', thresh=3):
    """Intracellular post synaptic potential event detection based on
    Jonas et al, 1993: Quantal components of unitary EPSCs at the mossy fibre synapse on CA3 pyramidal cells of rat hippocampus.
    Clements and Bekkers, 1997: Detection of spontaneous synaptic events with an optimally scaled template.
    Cited by Guzman et al., 2014: Stimfit: quantifying electrophysiological data with Python
    Inputs:    
        * Vs: voltage or current time series  
        * ts: sampling rate [ms]
        * event: event type
        * w: window of the template [ms]
        * tau_RISE: rise time of the template gamma function [ms]. Default 1ms
        * tau_DECAY: decay time of the template gamma function [ms]. Default 4ms
        * mph: minimum event size [mV]. Default 0.5mV
        * mpd: minimum event distance [ms]. Default 1ms
        * step: step size to match template. Default is 1ms
        * criterion: ['se'|'corr']
            'se': standard error [Default]
            'corr': correlation
        * thresh: threshold applied on detection criterion to detect the event.
            This value depends on the criterion selected.
            'se': which is the default setting. Recommend 3 [Default]
            'corr': Recommend 0.95 (significance level of the correlation)
            
    """
    step = step/ts
    t_vect = np.arange(0,w+ts,ts)
    def p_t(t, tau_RISE, tau_DECAY): # Template function. Upward PSP
        g = (1.0 - np.exp(-t/tau_RISE)) * np.exp(-t/tau_DECAY)
        g = g / np.max(g) # normalize the peak
        return g
        
    p = p_t(t_vect, tau_RISE, tau_DECAY)
    
    # Do some preprocessing first
    r_mean = np.mean(Vs)
    r = Vs - r_mean
    r = np.concatenate((r, np.zeros_like(p)))
    # length of trasnversion
    h = len(p)
    # Append some zeros
    chi_sq = np.zeros((np.arange(0, len(Vs), step).size,4)) # store fitting results
    A = np.vstack([p, np.ones(h)]).T
    for n, k in enumerate(np.arange(0, len(Vs), step, dtype=int)): # has total l steps
        chi_sq[n,0] = int(k) # record index        
        r_t = r[int(k):int(k+h)]
        q = np.linalg.lstsq(A, r_t)
        m, c = q[0] # m=scale, c=offset
        chi_sq[n, 1:3] = q[0]
        if criterion=='se':
            chi_sq[n,3] = float(q[1]) # sum squared residual
        elif criterion == 'corr':
            chi_sq[n,3] = np.corrcoef(r_t, m*p+c)[0,1]
    
    if criterion=='se':
        DetectionCriterion = chi_sq[:,1] / (np.sqrt(chi_sq[:,3]/(h-1)))
        if event in ['IPSP', 'EPSC']:
            DetectionCriterion = -1.0 * DetectionCriterion
    elif criterion=='corr':
        DetectionCriterion = chi_sq[:,3]
        DetectionCriterion = DetectionCriterion / np.sqrt((1-DetectionCriterion**2) /(h-2)) # t value
        DetectionCriterion = 1-st.t.sf(np.abs(DetectionCriterion), h-1)
    
    # Run through general peak detection on the detection criterion trace
    ind, _ = findpeaks(DetectionCriterion, mph=thresh, mpd=mpd/ts)
          
    pks = chi_sq[ind,1]
    ind = chi_sq[ind,0]
    # Filter out the detected events that is less than the minimum peak height requirement
    if event in ['EPSP', 'IPSC']:    
        valid_ind = pks>abs(mph)
    else:
        valid_ind = pks<-1*abs(mph)
    
    pks = pks[valid_ind]
    ind = ind[valid_ind]
    event_time = ind2time(ind, ts)
    
        
    return event_time, pks, DetectionCriterion, chi_sq
            

def detectPSP_deconvolution():
    return


if __name__ == '__main__':
    # test smoothed firing rate
#    from ImportData import *
#    zData = 'D:/Data/Traces/2015/09.September/Data 24 Sep 2015/Neocortex J.24Sep15.S1.E16.dat'
#    zData = NeuroData(zData, old=True)
#    Vs = zData.Voltage['A']
#    ts = zData.Protocol.msPerPoint
#    R = spk_firing_rate(Vs, ts, sigma=300.)
#    from matplotlib import pyplot as plt
#    plt.plot(R)
    from ImportData import *
    zData = NeuroData('D:/Data/Traces/2015/08.August/Data 10 Aug 2015/Neocortex A.10Aug15.S1.E35.dat', old=True)
    Vs = zData.Voltage['A']
    ts = zData.Protocol.msPerPoint

    
    ind, pks, DetectionCriterion, chi_sq = detectPSP_template_matching(Vs, ts, 'EPSP', step=10, criterion='se', thresh=3)
    
    print(pks[0])
    print(pks[1])
    print(pks[2])
    print(pks[3])
    
    from matplotlib import pyplot as plt
    plt.plot(Vs)
    plt.plot(ind, -65*np.ones_like(ind), 'ro')
    plt.figure()
    plt.plot(DetectionCriterion)
