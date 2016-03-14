"""This module provides functions used to write HDF5 files in the new file
format."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import json
import os
import warnings
from collections import OrderedDict, Iterable

import numpy as np
import tables as tb

from utils import convert_dtype, ensure_vector
from spikedetekt2.utils.six import itervalues, iteritems, string_types
from spikedetekt2.utils import warn, debug, COLORS_COUNT

# Disable PyTables' NaturalNameWarning due to nodes which have names starting 
# with an integer.
warnings.simplefilter('ignore', tb.NaturalNameWarning)


# -----------------------------------------------------------------------------
# File names
# -----------------------------------------------------------------------------
RAW_TYPES = ('raw.kwd', 'high.kwd', 'low.kwd')
FILE_TYPES = ('kwik', 'kwx') + RAW_TYPES

def get_filenames(name, dir=None):
    """Generate a list of filenames for the different files in a given 
    experiment, which name is given."""
    # Default directory: working directory.
    if dir is None:
        dir = os.getcwd()
    name = os.path.splitext(name)[0]
    return {type: os.path.join(dir, name + '.' + type) for type in FILE_TYPES}
    
def get_basename(path):
    bn = os.path.basename(path)
    bn = os.path.splitext(bn)[0]
    if bn.split('.')[-1] in ('raw', 'high', 'low'):
        return os.path.splitext(bn)[0]
    else:
        return bn


# -----------------------------------------------------------------------------
# Opening/closing functions
# -----------------------------------------------------------------------------
def open_file(path, mode=None):
    if mode is None:
        mode = 'r'
    try:
        f = tb.openFile(path, mode)
        return f
    except IOError as e:
        warn("IOError: " + str(e.message))
        return

def open_files(name, dir=None, mode=None):
    filenames = get_filenames(name, dir=dir)
    return {type: open_file(filename, mode=mode) 
            for type, filename in iteritems(filenames)}

def close_files(name, dir=None):
    if isinstance(name, string_types):
        filenames = get_filenames(name, dir=dir)
        files = [open_file(filename) for filename in itervalues(filenames)]
    else:
        files = itervalues(name)
    [file.close() for file in files if file is not None]
    
def files_exist(name, dir=None):
    files = get_filenames(name, dir=dir)
    return os.path.exists(files['kwik'])
    
def delete_files(name, dir=None):
    files = get_filenames(name, dir=dir)
    [os.remove(path) for path in itervalues(files) if os.path.exists(path)]
      
def get_row_shape(arr, nrows=1):
    """Return the shape of a row of an array."""
    return (nrows,) + arr.shape[1:]
      
def empty_row(arr, dtype=None, nrows=1):
    """Create an empty row for a given array."""
    return np.zeros(get_row_shape(arr, nrows=nrows), dtype=arr.dtype)
        

# -----------------------------------------------------------------------------
# HDF5 file creation
# -----------------------------------------------------------------------------
def create_kwik(path, experiment_name=None, prm=None, prb=None):
    """Create a KWIK file.
    
    Arguments:
      * path: path to the .kwik file.
      * experiment_name
      * prm: a dictionary representing the contents of the PRM file (used for
        SpikeDetekt)
      * prb: a dictionary with the contents of the PRB file
    
    """
    if experiment_name is None:
        experiment_name = ''
    if prm is None:
        prm = {}
    if prb is None:
        prb = {}
    
    file = tb.openFile(path, mode='w')
    
    file.root._f_setAttr('kwik_version', 2)
    file.root._f_setAttr('name', experiment_name)

    file.createGroup('/', 'application_data')
    
    # Set the SpikeDetekt parameters
    file.createGroup('/application_data', 'spikedetekt')
    for prm_name, prm_value in iteritems(prm):
        file.root.application_data.spikedetekt._f_setAttr(prm_name, prm_value)
    
    file.createGroup('/', 'user_data')
    
    # Create channel groups.
    file.createGroup('/', 'channel_groups')
    for igroup, group_info in enumerate(prb.get('channel_groups', [])):
        group = file.createGroup('/channel_groups', str(igroup))
        # group_info: channel, graph, geometry
        group._f_setAttr('name', 'channel_group_{0:d}'.format(igroup))
        group._f_setAttr('adjacency_graph', group_info.get('graph', np.zeros((0, 2))))
        file.createGroup(group, 'application_data')
        file.createGroup(group, 'user_data')
        
        # Create channels.
        file.createGroup(group, 'channels')
        channels = group_info.get('channels', [])
        for channel_idx in channels:
            # channel is the absolute channel index.
            channel = file.createGroup(group.channels, str(channel_idx))
            channel._f_setAttr('name', 'channel_{0:d}'.format(channel_idx))
            
            ############### TODO
            channel._f_setAttr('kwd_index', 0)
            channel._f_setAttr('ignored', False)
            channel._f_setAttr('position', group_info.get('geometry', {}). \
                get(channel_idx, None))
            channel._f_setAttr('voltage_gain', 0.)
            channel._f_setAttr('display_threshold', 0.)
            file.createGroup(channel, 'application_data')
            file.createGroup(channel.application_data, 'spikedetekt')
            file.createGroup(channel.application_data, 'klustaviewa')
            file.createGroup(channel, 'user_data')
            
        # Create spikes.
        spikes = file.createGroup(group, 'spikes')
        file.createEArray(spikes, 'time_samples', tb.UInt64Atom(), (0,),
                          expectedrows=1000000)
        file.createEArray(spikes, 'time_fractional', tb.UInt8Atom(), (0,),
                          expectedrows=1000000)
        file.createEArray(spikes, 'recording', tb.UInt16Atom(), (0,),
                          expectedrows=1000000)
        clusters = file.createGroup(spikes, 'clusters')
        file.createEArray(clusters, 'main', tb.UInt32Atom(), (0,),
                          expectedrows=1000000)
        file.createEArray(clusters, 'original', tb.UInt32Atom(), (0,),
                          expectedrows=1000000)
        
        fm = file.createGroup(spikes, 'features_masks')
        fm._f_setAttr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/features_masks'. \
            format(igroup))
        wr = file.createGroup(spikes, 'waveforms_raw')
        wr._f_setAttr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/waveforms_raw'. \
            format(igroup))
        wf = file.createGroup(spikes, 'waveforms_filtered')
        wf._f_setAttr('hdf5_path', '{{kwx}}/channel_groups/{0:d}/waveforms_filtered'. \
            format(igroup))
        
        # TODO: add clusters 0, 1, 2, 3 by default
        
        # Create clusters.
        clusters = file.createGroup(group, 'clusters')
        file.createGroup(clusters, 'main')
        file.createGroup(clusters, 'original')
        
        # Create cluster groups.
        cluster_groups = file.createGroup(group, 'cluster_groups')
        file.createGroup(cluster_groups, 'main')
        file.createGroup(cluster_groups, 'original')
        
    # Create recordings.
    file.createGroup('/', 'recordings')
    
    # Create event types.
    file.createGroup('/', 'event_types')
            
    file.close()

def create_kwx(path, prb=None, prm=None, has_masks=True):
    """Create an empty KWX file.
    
    Arguments:
      * prb: the PRB dictionary
      * waveforms_nsamples (common to all channel groups if set)
      * nfeatures (total number of features per spike, common to all channel groups if set)
      * nchannels (number of channels per channel group, common to all channel groups if set)
    
    """
    
    if prb is None:
        prb = {}
    if prm is None:
        prm = {}
    
    nchannels = prm.get('nchannels', None)
    nfeatures_per_channel = prm.get('nfeatures_per_channel', None)
    nfeatures = prm.get('nfeatures', None)
    waveforms_nsamples = prm.get('waveforms_nsamples', None)
        
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'channel_groups')
    
    for ichannel_group, chgrp_info in enumerate(prb.get('channel_groups', [])):
        nchannels_ = len(chgrp_info.get('channels', [])) or nchannels or 0
        waveforms_nsamples_ = chgrp_info.get('waveforms_nsamples', waveforms_nsamples) or 0
        nfeatures_per_channel_ = chgrp_info.get('nfeatures_per_channel', nfeatures_per_channel) or 0
        nfeatures_ = chgrp_info.get('nfeatures', nfeatures) or nfeatures_per_channel_ * nchannels_
        
        assert nchannels_ > 0
        assert nfeatures_ > 0
        assert waveforms_nsamples_ > 0
        
        channel_group_path = '/channel_groups/{0:d}'.format(ichannel_group)
        
        # Create the HDF5 group for each channel group.
        file.createGroup('/channel_groups', 
                         '{0:d}'.format(ichannel_group))
                         
        
        # Determine a sensible chunk shape.
        chunkrows = 10485760 // (nfeatures_ * 4)
                         
        # Create the arrays.
        if has_masks:
            # Features + masks.
            file.createEArray(channel_group_path, 'features_masks',
                              tb.Float32Atom(), (0, nfeatures_, 2),
                              chunkshape=(chunkrows, nfeatures_, 2))
        else:
            file.createEArray(channel_group_path, 'features_masks',
                              tb.Float32Atom(), (0, nfeatures_),
                              chunkshape=(chunkrows, nfeatures_))
        
        
        # Determine a sensible chunk shape.
        chunkrows = 10485760 // (waveforms_nsamples_ * nchannels_ * 2)
        
        file.createEArray(channel_group_path, 'waveforms_raw',
                          tb.Int16Atom(), (0, waveforms_nsamples_, nchannels_),
                          chunkshape=(chunkrows, waveforms_nsamples_, nchannels_))
        file.createEArray(channel_group_path, 'waveforms_filtered',
                          tb.Int16Atom(), (0, waveforms_nsamples_, nchannels_),
                          chunkshape=(chunkrows, waveforms_nsamples_, nchannels_))
                                                   
    file.close()
            
def create_kwd(path, type='raw', prm=None,):#recordings=None,):
    """Create an empty KWD file.
    
    Arguments:
      * type: 'raw', 'high', or 'low'
    
    """
        
    if prm is None:
        prm = {}
        
    file = tb.openFile(path, mode='w')
    file.createGroup('/', 'recordings')
    
    file.close()

def create_files(name, dir=None, prm=None, prb=None):
    
    filenames = get_filenames(name, dir=dir)
    
    create_kwik(filenames['kwik'], prm=prm, prb=prb)
    create_kwx(filenames['kwx'], prb=prb, prm=prm)
    
    create_kwd(filenames['raw.kwd'], 'raw', prm=prm)
    create_kwd(filenames['high.kwd'], 'high', prm=prm)
    create_kwd(filenames['low.kwd'], 'low', prm=prm)
    
    return filenames

    
# -----------------------------------------------------------------------------
# Adding items in the files
# -----------------------------------------------------------------------------
def add_recording(fd, id=None, name=None, sample_rate=None, start_time=None, 
                  start_sample=None, bit_depth=None, band_high=None,
                  band_low=None, downsample_factor=1., nchannels=None,
                  nsamples=None, data=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    
    if data is not None:
        nsamples, nchannels = data.shape
    
    assert nchannels
    
    # The KWIK needs to be there.
    assert kwik is not None
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        recordings = sorted([n._v_name 
                             for n in kwik.listNodes('/recordings')])
        if recordings:
            id = str(max([int(r) for r in recordings if r.isdigit()]) + 1)
        else:
            id = '0'
    # Default name: recording_X if X is an integer, or the id.
    if name is None:
        if id.isdigit():
            name = 'recording_{0:s}'.format(id)
        else:
            name = id
    recording = kwik.createGroup('/recordings', id)
    recording._f_setAttr('name', name)
    recording._f_setAttr('start_time', start_time)
    recording._f_setAttr('start_sample', start_sample)
    recording._f_setAttr('sample_rate', sample_rate)
    recording._f_setAttr('bit_depth', bit_depth)
    recording._f_setAttr('band_high', band_high)
    recording._f_setAttr('band_low', band_low)
    
    kwik_raw = kwik.createGroup('/recordings/' + id, 'raw')
    kwik_high = kwik.createGroup('/recordings/' + id, 'high')
    kwik_low = kwik.createGroup('/recordings/' + id, 'low')
    
    kwik_raw._f_setAttr('hdf5_path', '{raw.kwd}/recordings/' + id)
    kwik_high._f_setAttr('hdf5_path', '{high.kwd}/recordings/' + id)
    kwik_low._f_setAttr('hdf5_path', '{low.kwd}/recordings/' + id)
    
    kwik.createGroup('/recordings/' + id, 'user_data')
        
    for type in RAW_TYPES:
        kwd = fd.get(type, None)
        if kwd:
            add_recording_in_kwd(kwd, recording_id=id,
                                 downsample_factor=downsample_factor,
                                 nchannels=nchannels, 
                                 nsamples=nsamples, 
                                 data=data)
    
def add_recording_in_kwd(kwd, recording_id=0,
                         downsample_factor=None, nchannels=None, 
                         nsamples=None, data=None):
    if isinstance(kwd, string_types):
        kwd = open_file(kwd, 'a')
        to_close = True
    else:
        to_close = False
    
    if data is not None:
        nsamples, nchannels = data.shape
    
    recording = kwd.createGroup('/recordings', str(recording_id))
    recording._f_setAttr('downsample_factor', downsample_factor)
    
    dataset = kwd.createEArray(recording, 'data', 
                               tb.Int16Atom(), 
                               (0, nchannels), expectedrows=nsamples)
    
    # Add raw data.
    if data is not None:
        assert data.shape[1] == nchannels
        data_int16 = convert_dtype(data, np.int16)
        dataset.append(data_int16)
            
    kwd.createGroup(recording, 'filter')
    # TODO: filter
    if to_close:
        kwd.close()
    
    return kwd
    
def add_event_type(fd, id=None, evt=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        event_types = sorted([n._v_name 
                             for n in kwik.listNodes('/event_types')])
        if event_types:
            id = str(max([int(r) for r in event_types if r.isdigit()]) + 1)
        else:
            id = '0'
    event_type = kwik.createGroup('/event_types', id)
    
    kwik.createGroup(event_type, 'user_data')
    
    app = kwik.createGroup(event_type, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setAttr('color', None)
    
    events = kwik.createGroup(event_type, 'events')
    kwik.createEArray(events, 'time_samples', tb.UInt64Atom(), (0,))
    kwik.createEArray(events, 'recording', tb.UInt16Atom(), (0,))
    kwik.createGroup(events, 'user_data')
    
def add_cluster(fd, channel_group_id=None, id=None, clustering='main',
                cluster_group=None, color=None,
                mean_waveform_raw=None,
                mean_waveform_filtered=None,
                ):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    if channel_group_id is None:
        channel_group_id = '0'
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    # The channel group id containing the new cluster group must be specified.
    assert channel_group_id is not None
    clusters_path = '/channel_groups/{0:s}/clusters/{1:s}'.format(
        channel_group_id, clustering)
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        clusters = sorted([n._v_name 
                             for n in kwik.listNodes(clusters_path)])
        if clusters:
            id = str(max([int(r) for r in clusters if r.isdigit()]) + 1)
        else:
            id = '0'
    cluster = kwik.createGroup(clusters_path, id)
    
    cluster._f_setAttr('cluster_group', cluster_group)
    cluster._f_setAttr('mean_waveform_raw', mean_waveform_raw)
    cluster._f_setAttr('mean_waveform_filtered', mean_waveform_filtered)
    
    # TODO
    quality = kwik.createGroup(cluster, 'quality_measures')
    quality._f_setAttr('isolation_distance', None)
    quality._f_setAttr('matrix_isolation', None)
    quality._f_setAttr('refractory_violation', None)
    quality._f_setAttr('amplitude', None)
    
    kwik.createGroup(cluster, 'user_data')
    
    app = kwik.createGroup(cluster, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setAttr('color', color or ((int(id) % (COLORS_COUNT - 1)) + 1))
    
def add_cluster_group(fd, channel_group_id=None, id=None, clustering='main',
                      name=None, color=None):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    if channel_group_id is None:
        channel_group_id = '0'
    kwik = fd.get('kwik', None)
    # The KWIK needs to be there.
    assert kwik is not None
    # The channel group id containing the new cluster group must be specified.
    assert channel_group_id is not None
    cluster_groups_path = '/channel_groups/{0:s}/cluster_groups/{1:s}'.format(
        channel_group_id, clustering)
    if id is None:
        # If id is None, take the maximum integer index among the existing
        # recording names, + 1.
        cluster_groups = sorted([n._v_name 
                             for n in kwik.listNodes(cluster_groups_path)])
        if cluster_groups:
            id = str(max([int(r) for r in cluster_groups if r.isdigit()]) + 1)
        else:
            id = '0'
    # Default name: cluster_group_X if X is an integer, or the id.
    if name is None:
        if id.isdigit():
            name = 'cluster_group_{0:s}'.format(id)
        else:
            name = id
    cluster_group = kwik.createGroup(cluster_groups_path, id)
    cluster_group._f_setAttr('name', name)
    
    kwik.createGroup(cluster_group, 'user_data')
    
    app = kwik.createGroup(cluster_group, 'application_data')
    kv = kwik.createGroup(app, 'klustaviewa')
    kv._f_setAttr('color', color or ((int(id) % (COLORS_COUNT - 1)) + 1))
    
def _normalize_inplace(x):
    if x is None:
        return
    if x.dtype in (np.float32, np.float64):
        m, M = x.min(), x.max()
        c = max(np.abs(m), np.abs(M))
        x /= float(c)

def add_spikes(fd, channel_group_id=None, clustering='main',
                time_samples=None, time_fractional=0,
                recording=0, cluster=0, cluster_original=0,
                features_masks=None, features=None, masks=None,
                waveforms_raw=None, waveforms_filtered=None,):
    """fd is returned by `open_files`: it is a dict {type: tb_file_handle}."""
    if channel_group_id is None:
        channel_group_id = '0'
    kwik = fd.get('kwik', None)
    kwx = fd.get('kwx', None)
    # The KWIK needs to be there.
    assert kwik is not None
    # The channel group id containing the new cluster group must be specified.
    assert channel_group_id is not None

    spikes = kwik.root.channel_groups.__getattr__(channel_group_id).spikes
        
    time_samples = ensure_vector(time_samples)
    nspikes = len(time_samples)
    
    ds_features_masks = kwx.root.channel_groups.__getattr__(channel_group_id).features_masks
    ds_waveforms_raw = kwx.root.channel_groups.__getattr__(channel_group_id).waveforms_raw
    ds_waveforms_filtered = kwx.root.channel_groups.__getattr__(channel_group_id).waveforms_filtered
        
    nfeatures = ds_features_masks.shape[1]
    
    if features_masks is None:
        # Default features and masks
        if features is None:
            features = np.zeros((nspikes, nfeatures), dtype=np.float32)
        if masks is None:
            masks = np.zeros((features.shape[0], nfeatures), dtype=np.float32)
        
        # Ensure features and masks have the right number of dimensions.
        # features.shape is (1, nfeatures)
        # masks.shape is however  (nchannels,)
        if features.ndim == 1:
            features = np.expand_dims(features, axis=0)
        if masks.ndim == 1:
            masks = np.expand_dims(masks, axis=0)
        
        # masks.shape is now    (1,nchannels,)
        # Tile the masks if needed: same mask value on each channel.
        if masks.shape[1] < features.shape[1]:
            nfeatures_per_channel = features.shape[1] // masks.shape[1]
            masks = np.repeat(masks, nfeatures_per_channel, axis = 1)
        # masks.shape is (1, nfeatures) - what we want
        # Concatenate features and masks
        features_masks = np.dstack((features, masks))
        
    
    time_fractional = ensure_vector(time_fractional, size=nspikes)
    recording = ensure_vector(recording, size=nspikes)
    cluster = ensure_vector(cluster, size=nspikes)
    cluster_original = ensure_vector(cluster_original, size=nspikes)
    
    if waveforms_raw is None:
        waveforms_raw = empty_row(ds_waveforms_raw, nrows=nspikes)
    if waveforms_raw.ndim < 3:
        waveforms_raw = np.expand_dims(waveforms_raw, axis=0)
        
    if waveforms_filtered is None:
        waveforms_filtered = empty_row(ds_waveforms_filtered, nrows=nspikes)
    if waveforms_filtered.ndim < 3:
        waveforms_filtered = np.expand_dims(waveforms_filtered, axis=0)
        
    # Make sure we add the correct number of rows to every object.
    assert len(time_samples) == nspikes
    assert len(time_fractional) == nspikes
    assert len(recording) == nspikes
    assert len(cluster) == nspikes
    assert len(cluster_original) == nspikes
    assert features_masks.shape[0] == nspikes
    assert waveforms_raw.shape[0] == nspikes
    assert waveforms_filtered.shape[0] == nspikes
        
    # WARNING: need to normalize the waveforms before converting them to
    # int16. They need to be in [-1,1].
    # _normalize_inplace(waveforms_raw)
    # _normalize_inplace(waveforms_filtered)
    
    spikes.time_samples.append(time_samples)
    spikes.time_fractional.append(time_fractional)
    spikes.recording.append(recording)
    spikes.clusters.main.append(cluster)
    spikes.clusters.original.append(cluster_original)
    ds_features_masks.append(features_masks)
    
    # HACK: disable normalization for the time being, to try to
    # retrieve old output of SD1
    # ds_waveforms_raw.append(convert_dtype(waveforms_raw, np.int16))
    # ds_waveforms_filtered.append(convert_dtype(waveforms_filtered, np.int16))
    ds_waveforms_raw.append(waveforms_raw.astype(np.int16))
    ds_waveforms_filtered.append(waveforms_filtered.astype(np.int16))
