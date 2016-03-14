"""Thresholding routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
from collections import namedtuple

import numpy as np
from scipy import signal

from spikedetekt2.processing import apply_filter

DoubleThreshold = namedtuple('DoubleThreshold', ['strong', 'weak'])


# -----------------------------------------------------------------------------
# Thresholding
# -----------------------------------------------------------------------------
def get_threshold(raw_data, filter=None, channels=slice(None), **prm):
    """Compute the threshold from the standard deviation of the filtered signal
    across many uniformly scattered excerpts of data.
    
    threshold_std_factor can be a tuple, in which case multiple thresholds
    are returned.
    
    """
    nexcerpts = prm.get('nexcerpts', None)
    excerpt_size = prm.get('excerpt_size', None)
    use_single_threshold = prm.get('use_single_threshold', True)
    threshold_strong_std_factor = prm.get('threshold_strong_std_factor', None)
    threshold_weak_std_factor = prm.get('threshold_weak_std_factor', None)
    threshold_std_factor = prm.get('threshold_std_factor', 
        (threshold_strong_std_factor, threshold_weak_std_factor))
    
    if isinstance(threshold_std_factor, tuple):
        # Fix bug with use_single_threshold=False: ensure that 
        # threshold_std_factor has 2 dimensions (threshold_weak_strong, channel)
        threshold_std_factor = np.array(threshold_std_factor)[:,None]
    
    # We compute the standard deviation of the signal across the excerpts.
    # WARNING: this may use a lot of RAM.
    excerpts = np.vstack(
        # Filter each excerpt.
        apply_filter(excerpt.data[:,:], filter=filter)
            for excerpt in raw_data.excerpts(nexcerpts=nexcerpts, 
                                             excerpt_size=excerpt_size))
    
    # Get the median of all samples in all excerpts,
    # on all channels...
    if use_single_threshold:
        median = np.median(np.abs(excerpts))
    # ...or independently for each channel.
    else:
        median = np.median(np.abs(excerpts), axis=0)
    
    # Compute the threshold from the median.
    std = median / .6745
    threshold = threshold_std_factor * std
    
    if isinstance(threshold, np.ndarray):
        return DoubleThreshold(strong=threshold[0], weak=threshold[1])
    else:
        return threshold

