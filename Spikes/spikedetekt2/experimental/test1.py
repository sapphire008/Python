"""Test script illustrating how to run spikedetekt on a .dat file."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from spikedetekt2 import *

DIRPATH = 'data'
filename = 'dat1s'

sample_rate = 20000
duration = 1.
nchannels = 32
chunk_size = 20000
nsamples = int(sample_rate * duration)
raw_data = 'data/dat1s.dat'

prm = get_params(**{
    'nchannels': nchannels,
    'sample_rate': sample_rate,
    'chunk_size': chunk_size,
    'detect_spikes': 'negative',
})
prb = {'channel_groups': [
    {
        'channels': range(nchannels),
        'graph': [
                    [0, 1], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4],
                    [3, 4], [3, 5], [4, 5], [4, 6], [5, 6], [5, 7],
                    [6, 7], [6, 8], [7, 8], [7, 9], [8, 9], [8, 10],
                    [9, 10], [9, 11], [10, 11], [10, 12], [11, 12], [11, 13],
                    [12, 13], [12, 14], [13, 14], [13, 15], [14, 15], [14, 16],
                    [15, 16], [15, 17], [16, 17], [16, 18], [17, 18], [17, 19],
                    [18, 19], [18, 20], [19, 20], [19, 21], [20, 21], [20, 22],
                    [21, 22], [21, 23], [22, 23], [22, 24], [23, 24], [23, 25],
                    [24, 25], [24, 26], [25, 26], [25, 27], [26, 27], [26, 28],
                    [27, 28], [27, 29], [28, 29], [28, 30], [29, 30], [29, 31],
                    [30, 31]
                ],
    }
]}

# Delete the files if the script is called with "reset" option.
if 'reset' in sys.argv:
    if files_exist(filename, dir=DIRPATH):
        delete_files(filename, dir=DIRPATH)

# Create empty files if they do not exist yet.
if not files_exist(filename, dir=DIRPATH):
    create_files(filename, dir=DIRPATH, prm=prm, prb=prb)
    
    # Open the files.
    files = open_files(filename, dir=DIRPATH, mode='a')
    
    # Add data.
    add_recording(files, 
                  sample_rate=sample_rate,
                  nchannels=nchannels)
    add_cluster_group(files, channel_group_id='0', id='noise', name='Noise')
    add_cluster(files, channel_group_id='0',)
    
    # Close the files
    close_files(files)
    
# Open the files in writing mode and run SpikeDetekt
with Experiment(filename, dir=DIRPATH, mode='a') as exp:
    run(raw_data, experiment=exp, prm=prm, probe=Probe(prb))

# Open the file in read-only mode to look into it.
with Experiment(filename, dir=DIRPATH) as exp:
    print "spikes:", len(exp.channel_groups[0].spikes)
    fm = exp.channel_groups[0].spikes.features_masks
    wf = exp.channel_groups[0].spikes.waveforms_filtered
    
    print "Features & masks:", fm.shape
    print "Waveforms:", wf.shape
    
