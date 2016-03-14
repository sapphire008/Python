"""Alignment tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import numpy as np

from spikedetekt2.processing import extract_waveform, Component
from kwiklib.utils import Probe


# -----------------------------------------------------------------------------
# Test probe
# -----------------------------------------------------------------------------
n = 5
probe_adjacency_list = [(i, i+1) for i in range(n-1)]
PROBE = Probe({0: {'channels': range(n),
                   'graph': probe_adjacency_list}})


# -----------------------------------------------------------------------------
# Test component
# -----------------------------------------------------------------------------
CHUNK_WEAK = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1],
        ])
CHUNK_STRONG = np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])
COMPONENT = np.array([(1, 2), (2, 2), (2, 3), 
                      (3, 3), (4, 3), (4, 4)])
THRESHOLD_STRONG = .8
THRESHOLD_WEAK = .5
CHUNK_EXTRACT = THRESHOLD_WEAK * np.random.rand(5, 5)
CHUNK_EXTRACT[COMPONENT[:,0], COMPONENT[:,1]] = THRESHOLD_WEAK + \
                        (1-THRESHOLD_WEAK) * np.random.rand(COMPONENT.shape[0])
CHUNK_EXTRACT[2, 2] = (1 + THRESHOLD_STRONG) / 2.


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------
def test_extract_waveform_1():
    waveform = extract_waveform(Component(COMPONENT),
                                chunk_extract=CHUNK_EXTRACT,
                                chunk_fil=CHUNK_EXTRACT,
                                chunk_raw=CHUNK_EXTRACT,
                                extract_s_before=1,
                                extract_s_after=2,
                                threshold_strong=THRESHOLD_STRONG,
                                threshold_weak=THRESHOLD_WEAK, 
                                probe=PROBE)
    assert waveform.raw.shape == (3, 5)
    assert waveform.fil.shape == (3, 5)
    assert waveform.masks.shape == (5,)
    
    