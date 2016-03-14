"""Main module tests."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import os
import os.path as op
import tempfile
import shutil

import numpy as np
import tables as tb
from nose import with_setup

from kwiklib.dataio import (BaseRawDataReader, read_raw, create_files,
    open_files, close_files, add_recording, add_cluster_group, add_cluster,
    get_filenames, Experiment, excerpts)
from spikedetekt2.core import run
from kwiklib.utils import itervalues, get_params, Probe, create_trace


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
DIRPATH = tempfile.mkdtemp()

sample_rate = 2000.
duration = 1.
nchannels = 8
nsamples = int(sample_rate * duration)
raw_data = .1 * np.random.randn(nsamples, nchannels)
# Add "spikes".
for start, end in excerpts(nsamples, nexcerpts=100, excerpt_size=10):
    raw_data[start:end] *= 5

prm = get_params(**{
    'nchannels': nchannels,
    'sample_rate': sample_rate,
    'detect_spikes': 'positive',
    'save_high': True,
    'save_raw': True,
})
prb = {0:
    {
        'channels': list(range(nchannels)),
        'graph': [(i, i + 1) for i in range(nchannels - 1)],
    }
}

def setup():
    create_files('myexperiment', dir=DIRPATH, prm=prm, prb=prb)

    # Open the files.
    files = open_files('myexperiment', dir=DIRPATH, mode='a')

    # Add data.
    add_recording(files,
                  sample_rate=sample_rate,
                  nchannels=nchannels)
    add_cluster_group(files, channel_group_id='0', id='0', name='Noise')
    add_cluster(files, channel_group_id='0',)

    # Close the files
    close_files(files)

def teardown():
    files = get_filenames('myexperiment', dir=DIRPATH)
    [os.remove(path) for path in itervalues(files)]


# -----------------------------------------------------------------------------
# Processing tests
# -----------------------------------------------------------------------------
def test_run_nospikes():
    """Read from NumPy array file."""
    # Run the algorithm.
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(np.zeros((nsamples, nchannels)),
            experiment=exp, prm=prm, probe=Probe(prb))

    # Open the data files.
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        assert len(exp.channel_groups[0].spikes) == 0

@with_setup(setup,)
def test_run_1():
    """Read from NumPy array file."""
    # Run the algorithm.
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(raw_data, experiment=exp, prm=prm, probe=Probe(prb),)

    # Open the data files.
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        nspikes = len(exp.channel_groups[0].spikes)
        assert exp.channel_groups[0].spikes.clusters.main.shape[0] == nspikes
        assert exp.channel_groups[0].spikes.features_masks.shape[0] == nspikes
        assert exp.channel_groups[0].spikes.waveforms_filtered.shape[0] == nspikes

        assert isinstance(exp.channel_groups[0]._node.pca_waveforms,
            tb.Array)

        # Assert the log file exists.
        logfile = exp.gen_filename('log')
        assert os.path.exists(logfile)

        assert exp.recordings[0].raw.shape == (nsamples, nchannels)
        assert exp.recordings[0].high.shape == (nsamples, nchannels)
        assert exp.recordings[0].low.shape[0] in range(nsamples // 16 - 2,
                                                       nsamples // 16 + 3)
        assert exp.recordings[0].low.shape[1] == nchannels

@with_setup(setup,)
def test_run_2():
    """Read from .dat file."""
    path = os.path.join(DIRPATH, 'mydatfile.dat')
    (raw_data * 1e4).astype(np.int16).tofile(path)

    # Run the algorithm.
    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(path, experiment=exp, prm=prm, probe=Probe(prb))

    # Open the data files.
    with Experiment('myexperiment', dir=DIRPATH) as exp:
        assert len(exp.channel_groups[0].spikes)

@with_setup(setup,)
def test_run_canonical_pcs():
    prm_canonical = prm.copy()

    canonical_pcs = np.ones((prm['nfeatures_per_channel'],
                             prm['waveforms_nsamples'],
                             prm['nchannels']))
    prm_canonical['canonical_pcs'] = canonical_pcs

    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(raw_data, experiment=exp, prm=prm_canonical, probe=Probe(prb),)

@with_setup(setup,)
def test_diagnostics():

    dir = tempfile.mkdtemp()

    path = op.join(dir, 'diagnostics.py')
    with open(path, 'w') as f:
        f.write(
        'def diagnostics(prm=None, **kwargs):\n'
        '   print(prm)\n'
        '\n')

    prm['diagnostics_path'] = path

    with Experiment('myexperiment', dir=DIRPATH, mode='a') as exp:
        run(np.zeros((nsamples, nchannels)),
            experiment=exp, prm=prm, probe=Probe(prb))

    shutil.rmtree(dir)
