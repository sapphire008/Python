# -*- coding: utf-8 -*-
"""
Created on Thu Nov 19 17:19:49 2015

Some routines for electrophysiology data analyses

@author: Edward
"""
import sys
import os
import numpy as np
import scipy.signal as sg
import scipy.stats as st
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

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
        if x is None or np.isnan(x):
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
                        where sigma is the standard deviation (default is 0.3s)
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

def detrending(Vs, ts, mode='linear'):
    """Detrend the data. Useful for calculating mean independent noise.
    mode:
        'mean': simply remove mean
        'linear' (Deafult),'nearest', 'zero', 'slinear', 'quadratic', 'cubic': using interp1d
        'polyN': fit a polynomial for Nth degree. e.g. 'poly3' fits a cubic curve
    Do not mistake 'linear' mode as removing a global linear trend. For removing global linear trend,
    use 'poly1'

    Note that after detrending the mean would be zero. To keep the mean of the
    trace, remove mean before detrending, then add mean back.
    """
    if mode=='mean':
        return Vs - np.mean(Vs)
    else:
        x = np.arange(0, len(Vs)*ts, ts)
        if mode in ['linear','nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
            p = interp1d(x, Vs, kind=mode)
        elif mode[:4]=='poly':
            deg = str2num(mode[4:])
            p = np.poly1d(np.polyfit(x,  Vs, deg))

        y_hat = p(x)

        return Vs - y_hat

def detectSpikes_cell_attached(Is, ts, msh=30, msd=10, basefilt=20, maxsh=300,
                               removebase=False, **kwargs):
    """Detect cell attached extracellular spikes
    Is: current time series
    ts: sampling rate (ms)
    msh: min spike height (Default 30pA)
    msd: min spike distance (Default 10ms)
    basefilt: baseline medfilt filter order in ms (Default 20)
    maxsh: maximum spike height. Helpful to remove stimulation artifacts.
            (Default 300)
    removebase: remove baseline when returning height. This will result
            absolute height of spike relative to the baseline. If set to false,
            returning the value of the spike, before filtering. (Default False)
    **kwargs: additional arguments for "findpeaks"
    """
    # Make sure medfilt kernel size is odd
    # Median filter out the spikes to get baseline
    Base = medfilt1(Is, int(basefilt/ts/2)*2+1) # Use custom medfilt1
    msd = msd / ts
    Is = Is - Base
    # Invert Is because of voltage clamp mode, resulting inward current being
    # negative.
    [LOCS, PKS] = findpeaks(-Is, mph=msh, mpd=msd, **kwargs)
    num_spikes = len(PKS)
    spike_time = ind2time(LOCS, ts)
    # Remove peaks exceeding max height
    ind = np.where(PKS<maxsh)
    LOCS = LOCS[ind]
    PKS = PKS[ind]
    if removebase:
        spike_heights = PKS
    else:
        spike_heights = -PKS + Base[LOCS]
    return num_spikes, spike_time, spike_heights


def spk_vclamp_series_resistance(Is, Vs, ts, window=[995,1015], scalefactor=1.0, direction='up'):
    """Calculate the series resistance based on capacitance artifact
    * Inputs:
        - Is: current time series (pA)
        - Vs: voltage step time series (mV)
        - ts: sampling rate (ms)
        - window: a window that contains the capcitance artifact, [baseline, end_of_artifact]
        - scalefactor: scale factor of the current time series
        - direction ["up"(default)|"down"]: is the artifact upswing or downswing
    * Outputs:
        - R_series: series resistance [MOhm]
        - tau: time constant of the exponential fit on the artifact [ms]
        - rsquare: adjusted R square of exponential fit on the artifact
    """
    if window is not None:
        Is = spk_window(Is, ts, window)
        Vs = spk_window(Vs, ts, window)
        
    if direction != 'up':
        Is = -Is
        Vs = -Vs

    index = np.argmax(Is)
    Is_fit = Is[index:]
    Is_fit = Is_fit - np.mean(Is_fit[-5:])
    Ts_fit = np.arange(0, len(Is_fit)*ts, ts)
    
    # plt.plot(Ts_fit, Is_fit)
    # Fitting the best possible
    f0 = lambda x, a, b: a*np.exp(-b*x)
    popt1, pcov1 = curve_fit(f0, Ts_fit, Is_fit, [np.max(Is_fit), 0.5])
    gof1 = goodness_of_fit(Ts_fit, Is_fit, popt1, pcov1, f0)
    #print(gof1['adjrsquare'])
    #return Ts_fit, Is_fit, popt1

    if gof1['adjrsquare'] > 0.85:
        tau = 1.0 / np.abs(popt1[1])    
        rsquare = gof1['adjrsquare']
    else:
        f0 = lambda x, a, b, c: a*np.exp(-b*x)+c
        popt2, pcov2 = curve_fit(f0, Ts_fit, Is_fit,  [np.max(Is_fit), 0.5, np.min(Is_fit)])
        gof2 = goodness_of_fit(Ts_fit, Is_fit, popt2, pcov2, f0)
        if gof2['adjrsquare'] > 0.85:
            tau = 1.0 / np.abs(popt2[1])
            rsquare = gof2['adjrsquare']
        else:    
            f0 = lambda x, a, b, c, d: a*np.exp(-b*x) + c*np.exp(d*x)
            popt3, pcov3 = curve_fit(f0, Ts_fit, Is_fit,  [np.max(Is_fit), 0.5, np.min(Is_fit), 0.5])
            gof3 = goodness_of_fit(Ts_fit, Is_fit, popt3, pcov3, f0)
            tau = np.max(1.0/np.array([popt3[1], popt3[3]]))
            rsquare = gof3['adjrsquare']

    # Integrate the current over the window to get total charge
    Is = Is - np.mean(Is[-5:])
    Q = np.sum(Is[Is>0]) * ts / scalefactor
    
    C_m = Q / np.abs(Vs[-1] - Vs[0]) # [pF]
    R_series = tau / C_m * 1000 # [MOhm]

    return R_series, tau, rsquare


def spk_get_stim(Ss, ts, longest_row=True, decimals=0):
    """Serve as an example on how to extract the strongest 
    and longest stimulus given the stimulus trace
    
    Inputs:
        Ss: time series of stimulus trace
        ts: sampling rate [seconds]
    Returns [start, end, intensity]
    """
    stim_amp = np.max(Ss)
    stim = np.where(Ss == stim_amp)[0]
    consec_index = getconsecutiveindex(stim)
    # Get the longest stimulus
    if longest_row:
        longest_row = np.argmax(np.diff(consec_index, axis=1))
        stim = stim[consec_index[longest_row, :]]
        stim = np.round(ind2time(stim, ts), decimals=decimals)
        stim = np.concatenate((stim, np.asarray([stim_amp])), axis=0)
    else: # can have multiple stims
        tmp_stim = np.empty((consec_index.shape[0], consec_index.shape[1]))
        for r in range(consec_index.shape[0]):
            tmp_stim[r, :] = np.round(ind2time(stim[consec_index[r,:]], ts), decimals=decimals)
            
        stim = np.c_[tmp_stim, stim_amp*np.ones((consec_index.shape[0], 1))]
        
                    
    return stim


def spk_get_rin(Vs, ts, neg=[], Ss=None, base_win=[-100, 0], rin_win=[-100,0], base_func=np.mean, rin_func=np.mean, relative_rin_win_end=True):
    """
    Vs: voltage [mV]
    ts: sampling interval [ms]
    neg: a window of the Rin negative step [start, end (,intensity)], either size 2 or 3. If size 2, Ss argument must be specified
    Ss: time series of the same length as Vs. Needed when len(neg)==2
    base_win: baseline window
    rin_win: Rin window
    base_func: function applied to the base_win to extract the number. Default np.mean
    rin_func: function applied to the rin_win. Default np.mean
    relative_rin_win_end: If True: base_win is relative to neg[0], and rin_win to neg[1]
    """
    if len(neg) == 3:
        Rin =(rin_func(spk_window(Vs, ts, rin_win + neg[1])) - base_func(spk_window(Vs, ts, base_win + neg[0]))) / neg[2] * 1000
    elif len(neg) == 2: 
        if Ss is None:
            raise(Exception("Stimulus intensity is not specified"))
        
        Rin = (rin_func(spk_window(Vs, ts, rin_win + neg[1])) - base_func(spk_window(Vs, ts, base_win + neg[0]))) /  \
               (np.mean(spk_window(Ss, ts, rin_win + neg[1])) - np.mean(spk_window(Ss, ts, base_win + neg[0]))) * 1000
    else:
        raise(Exception("Length of neg must be at least 2"))
        
    return Rin
        

# %%
def spk_time_distance(spike_time, method="victor&purpura", *args, **kwargs):
    if method == "victor&purpura":
        spkd_victor_and_purpura(tli, tlj, cost)
    return

def spkd_victor_and_purpura(tli, tlj, cost=0):
    """Calculate the "spike time" distance (Victor & Purpura 1996) for a single
    cost between a pair of spike trains
    
    tli: vector of spike times for first spike train
    tlj: vector of spike times for second spike train
    cost: cost per unit time to move a spike
     
    Translated from origial MATLAB script by Daniel Reich
    http://www-users.med.cornell.edu/~jdvicto/spkdm.html
    
    Original MATALB script license:
    Copyright (c) 1999 by Daniel Reich and Jonathan Victor.
    Translated to Matlab by Daniel Reich from FORTRAN code by Jonathan Victor.
    """
    nspi = len(tli)
    nspj = len(tlj)
    if cost == 0:
        return abs(nspi-nspj)
    elif cost == np.inf:
        return nspi+nspj
    
    scr = np.zeros((nspi+1, nspj+1))
    # Initialize margins with cost of adding a spike
    scr[:, 0] = np.arange(0, nspi+1, 1)
    scr[0, :] = np.arange(0, nspj+1, 1)
    if nspi and nspj: # if neither is zero
        for i in range(1, nspj+1):
            for j in range(1, nspj+1):
                # Finding the minimum of adding a spike, removing a spike, or moving a spike
                scr[i,j]=np.min([scr[i-1,j]+1, scr[i,j-1]+1, scr[i-1,j-1]+cost*np.abs(tli[i-1]-tlj[j-1])])

    return scr[nspi, nspj]

def spkd_victor_purpura_interval(tli, tlj, cost=0, tsamp=2000):
    """Calculates distance between two spike trains in the
    spike interval metric by a continuum modification of the 
    sellers algorithm
    
    Inputs:
        tli: vector of spike times for first spike train
        tlj: vector of spike times for second spike train
        cost: cost per unit time to move a spike
        tsamp: the length of the entire interval
        
    """
    
    nspi = len(tli) # number of spike times in train 1
    nspj = len(tlj) # number of spike times in train 2
    
    ni = nspi + 1 # number of intervals in train 1
    nj = nspj + 1 # number fo intervals in train 2
    scr = np.zeros((ni+1, nj+1))
    
    # Define calculation for a cost of zero
    if cost == 0:
        d = np.abs(ni-nj)
        return d, scr
    
    # Initialize margins with cost of adding a spike
    scr[:, 0] = np.arange(0, ni+1, 1)
    scr[0, :] = np.arange(0, nj+1, 1)
    
    tli_diff = np.diff(tli)
    tlj_diff = np.diff(tlj)
    
    for i in range(0, ni):
        if i>0 and i<ni-1: # in the middle
            di = tli_diff[i-1]
        elif i==0 and i==ni-1: # ni == 1 at the beginning
            di = tsamp
        elif i==0 and i<ni-1: # ni > 1 at the beginning
            di = tli[i]
        else: # otherwise
            di = tsamp - tli[i-1]
        
        iend = i==0 or i==ni-1
        
        # Unrolled loop for j = 1
        # -----------------------
        if nj == 1:
            dj = tsamp
        else: # j < nj
            dj = tlj[0]
        
        if iend:
            dist = 0
        else: # jend
            dist = np.max([0, dj-di])
            
        scr[i+1, 1] = np.min([scr[i,1]+1, scr[i+1, 0]+1, scr[i,0]+cost*dist])
        
        # Main code
        # -----------------------
        for j in range(1, nj-1):
            dj = tlj_diff[j-1]
            
            if iend:
                dist = np.max([0, di-dj])
            else:
                dist = np.abs(di-dj)
            
            scr[i+1, j+1] = np.min([scr[i, j+1]+1, scr[i+1, j]+1, scr[i,j]+cost*dist])
            
        # Unrolled loop for j = nj
        if nj == 0:
            dj = tsamp
        else:
            dj = tsamp - tlj[nj-2]
            
        if iend:
            dist = 0
        else:
            dist = np.max([0, dj-di])
        
        scr[i+1, nj] = np.min([scr[i, nj]+1,  scr[i+1, nj-1]+1,  scr[i, nj-1]+cost*dist])
    
    return scr[ni, nj]


# %% Simultaions
def spk_make_epsp_train(event_time, duration=10000, ts=0.1, 
                        alpha_dict={'duration':1000, 'amplitude':150, 'tau1':50, 'tau2':100}):    
    alpha_dict['ts'] = ts
    
    T = np.arange(0, duration+ts, ts)
    R = spk_dirac(ts=ts, dur=[0, duration], phi=event_time, h=1., collapse=True)
    epsp = alpha(**alpha_dict)
    epsp_train = sg.convolve(R, epsp, mode="full")[:len(T)] # faster
    return epsp_train

# %%
if __name__ == '__main__':
    from ImportData import *
    from matplotlib import pyplot as plt
    
    Base = 'Neocortex N.30May16.S1.E67'
    Similar = 'Neocortex N.30May16.S1.E71'
    Similar2 = 'Neocortex N.30May16.S1.E58'
    Different = 'Neocortex N.30May16.S1.E64'
    Different2 = 'Neocortex N.30May16.S1.E68'
    
    def get_spike_num(path):
        zData = load_trace(path)
        ts = zData.Protocol.msPerPoint
        stim = spk_get_stim(zData.Stimulus['A'], ts)
        Vs = spk_window(zData.Voltage['A'], ts, np.array([0, 2000]) + stim[0])
        _, spk_time, _ = spk_count(Vs, ts)
        return spk_time, Vs
        
    
    Base_spk, _ = get_spike_num(Base)
    Similar_spk,_ = get_spike_num(Similar)
    Similar2_spk,Vs = get_spike_num(Similar2)
    Different_spk,_ = get_spike_num(Different)
    Different2_spk,_ = get_spike_num(Different)
    
    d = spkd_victor_and_purpura(Base_spk, Similar_spk, cost=0.1)
        
    d = spkd_victor_purpura_interval(Base_spk, Similar_spk, cost=0.1, tsamp=2000)
    

