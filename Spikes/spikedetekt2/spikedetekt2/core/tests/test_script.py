"""Main module tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path as op
import numpy as np
import tempfile

from kwiklib import (excerpts, get_params, pydict_to_python, get_filenames,
    itervalues, create_trace, Experiment)
from spikedetekt2.core.script import run_spikedetekt


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = None
prm_filename = 'myexperiment.prm'
prb_filename = 'myprobe.prb'
dat_filename = 'myexperiment.dat'
name = 'myexperiment'

sample_rate = 20000.
duration = 1.
nchannels = 8
nsamples = int(sample_rate * duration)

    
def setup():
    global DIRPATH
    DIRPATH = tempfile.mkdtemp()
    
    # Create DAT file.
    raw_data = create_trace(nsamples, nchannels)
    for start, end in excerpts(nsamples, nexcerpts=10, excerpt_size=10):
        raw_data[start:end] += np.random.randint(low=-10000, high=10000, 
                                                 size=(10, nchannels))
    raw_data.tofile(op.join(DIRPATH, dat_filename))

    # Create PRM file.
    prm = get_params(**{
        'raw_data_files': dat_filename,
        'experiment_name': name,
        'nchannels': nchannels,
        'sample_rate': sample_rate,
        'detect_spikes': 'positive',
        'prb_file': prb_filename,
    })
    prm_contents = pydict_to_python(prm)
    with open(op.join(DIRPATH, prm_filename), 'w') as f:
        f.write(prm_contents)
    
    # Create PRB file.
    prb_contents = """
    nchannels = %NCHANNELS%
    channel_groups = {0:
        {
            'channels': list(range(nchannels)),
            'graph': [(i, i + 1) for i in range(nchannels - 1)],
        }
    }""".replace('%NCHANNELS%', str(nchannels)).replace('    ', '')
    with open(op.join(DIRPATH, prb_filename), 'w') as f:
        f.write(prb_contents)

def teardown():
    os.remove(op.join(DIRPATH, prm_filename))
    os.remove(op.join(DIRPATH, prb_filename))
    
    files = get_filenames(name, dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Main tests
# -----------------------------------------------------------------------------
def test_main_1():
    run_spikedetekt(op.join(DIRPATH, prm_filename))
    
    # Open the data files.
    with Experiment(name, dir=DIRPATH) as exp:
        nspikes = len(exp.channel_groups[0].spikes)
        assert exp.channel_groups[0].spikes.clusters.main.shape[0] == nspikes
        assert exp.channel_groups[0].spikes.features_masks.shape[0] == nspikes
        assert exp.channel_groups[0].spikes.waveforms_filtered.shape[0] == nspikes
        
        fm = exp.channel_groups[0].spikes.features_masks
        assert fm[:,:,0].min() < fm[:,:,0].max()
        
        # Make sure the masks are not all null.
        assert fm[:,:,1].max() > 0
        