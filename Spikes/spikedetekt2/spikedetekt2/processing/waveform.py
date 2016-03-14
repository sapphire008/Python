"""Alignment routines."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np
from scipy.interpolate import interp1d

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
class InterpolationError(Exception):
    pass

def get_padded(Arr, Start, End):
    '''
    Returns Arr[Start:End] filling in with zeros outside array bounds
    
    Assumes that EITHER Start<0 OR End>len(Arr) but not both (raises error).
    '''
    if Start < 0 and End >= Arr.shape[0]:
        raise IndexError("Can have Start<0 OR End>len(Arr) but not both.\n \
                           This error has probably occured because your Thresholds \n \
                             are aritificially low due to early artifacts\n \
                             Increase the parameter CHUNKS_FOR_THRESH ")
    if Start < 0:
        StartZeros = np.zeros((-Start, Arr.shape[1]), dtype=Arr.dtype)
        return np.vstack((StartZeros, Arr[:End]))
    elif End > Arr.shape[0]:
        EndZeros = np.zeros((End-Arr.shape[0], Arr.shape[1]), dtype=Arr.dtype)
        return np.vstack((Arr[Start:], EndZeros))
    else:
        return Arr[Start:End]
        
        
# -----------------------------------------------------------------------------
# Waveform class
# -----------------------------------------------------------------------------
class Waveform(object):
    def __init__(self, fil=None, raw=None, masks=None, 
                 s_min=None,  # relative to the start of the chunk
                 s_fracpeak=None,   # relative to the start of the chunk
                 s_start=None,  # start of the chunk, absolute,
                 recording=0,
                 channel_group=None):
        self.fil = fil
        self.raw = raw
        self.masks = masks
        self.s_min = s_min  # start of the waveform, relative to the start
                            # of the chunk
        self.s_start = s_start  # start of the chunk, absolute (wrt the exp)
        # peak fractional time of the waveform, absolute (relative to the exp)
        self.sf_offset = s_fracpeak + s_start
        self.s_offset = int(self.sf_offset)
        self.s_frac_part = self.sf_offset - self.s_offset
        self.channel_group = channel_group
        self.recording = recording
        
    def __cmp__(self, other):
        return self.sf_offset - other.sf_offset
        
    def __repr__(self):
        return '<Waveform on channel group {chgrp} at sample {smp}>'.format(
            chgrp=self.channel_group,
            smp=self.sf_offset
        )


# -----------------------------------------------------------------------------
# Waveform extraction
# -----------------------------------------------------------------------------
def extract_waveform(component, chunk_fil=None, chunk_raw=None,
                     chunk_extract=None,  # =chunk_fil or its abs()
                     threshold_strong=None, threshold_weak=None, 
                     probe=None, **prm):
    s_start = component.s_start  # Absolute start of the chunk.
    keep_start = component.keep_start  # Absolute start of the kept chunk.
    keep_end = component.keep_end  # Absolute end of the kept chunk.
    recording = component.recording  # Recording index of the current raw data 
                                     # section
    
    s_before = prm['extract_s_before']
    s_after = prm['extract_s_after']
    
    
    component_items = component.items
    assert len(component_items) > 0
    
    # Get samples and channels in the component.
    if not isinstance(component_items, np.ndarray):
        component_items = np.array(component_items)
    
    # The samples here are relative to the start of the chunk.
    comp_s = component_items[:,0]  # shape: (component_size,)
    comp_ch = component_items[:,1]  # shape: (component_size,)
    
    # Find the channel_group of the spike.
    # Make sure the channel is in the probe, otherwise pass the waveform.
    if component_items[0][1] not in probe.channel_to_group:
        return None
    channel_group = probe.channel_to_group[component_items[0][1]]
    # List of channels in the current channel group.
    channels = probe.channel_groups[channel_group].channels

    # Total number of channels across all channel groups.
    # chunk_extract = chunk_extract[:,probe.channels]
    nsamples, nchannels = chunk_extract.shape
    # nchannels = len(channels)
    # assert nchannels == probe.nchannels
    
    # Get binary mask.
    masks_bin = np.zeros(nchannels, dtype=np.bool)  # shape: (nchannels,)
    masks_bin[sorted(set(comp_ch))] = 1
        
    # Get the temporal window around the waveform.
    # These values are relative to the start of the chunk.
    s_min, s_max = np.amin(comp_s) - 3, np.amax(comp_s) + 4  
    s_min = max(s_min, 0)
    s_max = min(s_max, nsamples)
    s_offset = s_start + s_min  # absolute offset of the waveform (wrt the exp)
    
    # Extract the waveform values from the data.
    # comp shape: (some_length, nchannels)
    # contains the filtered chunk on weak threshold crossings only
    # small temporal window around the waveform
    comp = np.zeros((s_max - s_min, nchannels), dtype=chunk_extract.dtype)
    comp[comp_s - s_min, comp_ch] = chunk_extract[comp_s, comp_ch]
    # the sample where the peak is reached, on each channel, relative to
    # the beginning
    
    # Find the peaks (relative to the start of the chunk).
    peaks = np.argmax(comp, axis=0) + s_min  # shape: (nchannels,)
    # peak values on each channel
    # shape: (nchannels,)
    peaks_values = chunk_extract[peaks, np.arange(0, nchannels)] * masks_bin
    
    # Compute the float masks.
    masks_float = np.clip(  # shape: (nchannels,)
        (peaks_values - threshold_weak) / (threshold_strong - threshold_weak), 
        0, 1)
    masks_float = masks_float[channels]  # keep shank channels
    
    # Compute the fractional peak.
    power = prm.get('weight_power', 1.)
    comp_normalized = np.clip(
        (comp - threshold_weak) / (threshold_strong - threshold_weak),
        0, 1)
    comp_power = np.power(comp_normalized, power)
    u = np.arange(s_max - s_min)[:,np.newaxis]
    # Spike frac time relative to the start of the chunk.
    s_fracpeak = np.sum(comp_power * u) / np.sum(comp_power) + s_min
    
    # Realign spike with respect to s_fracpeak.
    s_peak = int(s_fracpeak)
    # Get block of given size around peaksample.
    wave = get_padded(chunk_fil,
                      s_peak - s_before - 1,
                      s_peak + s_after + 2)
    wave = wave[:,channels] # keep shank channels
    
    # Perform interpolation around the fractional peak.
    old_s = np.arange(s_peak - s_before - 1, s_peak + s_after + 2)
    new_s = np.arange(s_peak - s_before, s_peak + s_after) + (s_fracpeak - s_peak)
    try:
        f = interp1d(old_s, wave, bounds_error=True, kind='cubic', axis=0)
    except ValueError: 
        raise InterpolationError("Interpolation error at time {0:d}".format(
                                 s_offset))
    wave_aligned = f(new_s)
    
    # Get unfiltered spike.
    wave_raw = get_padded(chunk_raw,
                          s_peak - s_before,
                          s_peak + s_after)
    wave_raw = wave_raw[:,channels] # keep shank channels
                          
    # Create the Waveform instance.
    waveform = Waveform(fil=wave_aligned, raw=wave_raw, masks=masks_float, 
                    s_min=s_min, s_start=s_start, s_fracpeak=s_fracpeak, 
                    channel_group=channel_group,
                    recording=recording)
                    
    # Only keep the waveforms that are within the chunk window.
    if keep_start <= waveform.sf_offset < keep_end:
        return waveform
    else:
        return None
        
        
