"""Filtering routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from kwiklib.utils.six.moves import range


# -----------------------------------------------------------------------------
# Signal processing functions
# -----------------------------------------------------------------------------
def bandpass_filter(**prm):
    """Bandpass filtering."""
    rate = prm['sample_rate']
    order = prm['filter_butter_order']
    low = prm['filter_low']
    high = prm['filter_high']
    return signal.butter(order,
                         (low/(rate/2.), high/(rate/2.)),
                         'pass')

def apply_filter(x, filter=None):
    if x.shape[0] == 0:
        return x
    b, a = filter
    try:
        out_arr = signal.filtfilt(b, a, x, axis=0)
    except TypeError:
        out_arr = np.zeros_like(x)
        for i_ch in range(x.shape[1]):
            out_arr[:, i_ch] = signal.filtfilt(b, a, x[:, i_ch])
    return out_arr

def decimate(x):
    q = 16
    n = 50
    axis = 0

    b = signal.firwin(n + 1, 1. / q, window='hamming')
    a = 1.

    y = signal.lfilter(b, a, x, axis=axis)

    sl = [slice(None)] * y.ndim
    sl[axis] = slice(n // 2, None, q)

    return y[sl]


# -----------------------------------------------------------------------------
# Whitening
# -----------------------------------------------------------------------------
"""
  * Get the first chunk of data
  * Detect spikes the usual way
  * Compute mean on each channel on non-spike data
  * For every pair of channels:
      * estimate the covariance on non-spike data
  * Get the covariance matrix
  * Get its square root C' (sqrtm)
  * Get u*C' + (1-u)*s*Id, where u is a parameter, s the std of non-spike data
    across all channels
  * Option to save or not whitened data in FIL
  * All spike detection is done on whitened data

"""
def get_whitening_matrix(x):
    C = np.cov(x, rowvar=0)
    # TODO

def whiten(x, matrix=None):
    if matrix is None:
        matrix = get_whitening_matrix(x)
    # TODO




