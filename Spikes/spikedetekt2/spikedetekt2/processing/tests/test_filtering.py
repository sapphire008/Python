"""Filtering tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy import signal
from spikedetekt2.processing import (apply_filter, bandpass_filter,
    get_whitening_matrix, whiten, decimate)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_apply_filter():
    """Test bandpass filtering on a combination of two sinusoids."""
    rate = 10000.
    low, high = 100., 200.
    # Create a signal with small and high frequencies.
    t = np.linspace(0., 1., rate)
    x = np.sin(2*np.pi*low/2*t) + np.cos(2*np.pi*high*2*t)
    # Filter the signal.
    filter = bandpass_filter(filter_low=low,
        filter_high=high, filter_butter_order=4, sample_rate=rate)
    x_filtered = apply_filter(x, filter=filter)
    # Check that the bandpass-filtered signal is weak.
    assert np.abs(x[int(2./low*rate):-int(2./low*rate)]).max() >= .9
    assert np.abs(x_filtered[int(2./low*rate):-int(2./low*rate)]).max() <= .1
    
def test_decimate():
    x = np.random.randn(16000, 3)
    y = decimate(x)
    
def test_whitening():
    x = np.random.randn(10000, 2)
    x[:, 1] += x[:,0]
    M = get_whitening_matrix(x)
    # TODO
    
    
    