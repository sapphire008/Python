"""Launching script."""

# -----------------------------------------------------------------------------
# Imports
# -----------------------------------------------------------------------------
import logging
import os
import sys
import os.path as op
import tempfile
import argparse
import spikedetekt2

import numpy as np
import tables as tb

from kwiklib import (Experiment, get_params, load_probe, create_files, 
    read_raw, Probe, convert_dtype, read_clusters,
    files_exist, add_clustering, delete_files, exception)
from spikedetekt2.core import run


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def _load_files_info(prm_filename, dir=None):
    dir_, filename = op.split(prm_filename)
    dir = dir or dir_
    basename, ext = op.splitext(filename)
    if ext == '':
        ext = '.prm'
    prm_filename = op.join(dir, basename + ext)
    assert op.exists(prm_filename)
    
    # Load PRM file.
    prm = get_params(prm_filename)
    nchannels = prm.get('nchannels')
    assert nchannels > 0
    
    # Find PRB path in PRM file, and load it
    prb_filename = prm.get('prb_file')
    if not op.exists(prb_filename):
        prb_filename = op.join(dir, prb_filename)
    prb = load_probe(prb_filename)

        
    # Find raw data source.
    data = prm.get('raw_data_files')
    if isinstance(data, basestring):
        if data.endswith('.dat'):
            data = [data]
    if isinstance(data, list):
        for i in range(len(data)):
            if not op.exists(data[i]):
                data[i] = op.join(dir, data[i])
        
    experiment_name = prm.get('experiment_name')
    
    return dict(prm=prm, prb=prb, experiment_name=experiment_name, nchannels=nchannels,
                data=data, dir=dir)    
    
def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

def which(program):
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None
    
def print_path():
    print '\n'.join(os.environ["PATH"].split(os.pathsep))
    
def check_path():
    prog = 'klustakwik'
    if not (which(prog) or which(prog + '.exe')):
        print("Error: '{0:s}' is not in your system PATH".format(prog))
        return False
    return True


# -----------------------------------------------------------------------------
# SpikeDetekt
# -----------------------------------------------------------------------------
def run_spikedetekt(prm_filename, dir=None, debug=False, convert_only=False):
    info = _load_files_info(prm_filename, dir=dir)
    experiment_name = info['experiment_name']
    prm = info['prm']
    prb = info['prb']
    data = info['data']
    dir = dir or info['dir']
    nchannels = info['nchannels']
    
    # Make sure spikedetekt does not run if the .kwik file already exists
    # (i.e. prevent running it twice on the same data)
    assert not files_exist(experiment_name, dir=dir, types=['kwik']), "The .kwik file already exists, please use the --overwrite option."
    
    # Create files.
    create_files(experiment_name, dir=dir, prm=prm, prb=prb, 
                 create_default_info=True, overwrite=False)
    
    # Run SpikeDetekt.
    with Experiment(experiment_name, dir=dir, mode='a') as exp:
        # Avoid reopening the KWD file if it's already opened.
        if isinstance(data, str) and data.endswith('kwd'):
            data = exp._files['raw.kwd']
        run(read_raw(data, nchannels=nchannels), 
            experiment=exp, prm=prm, probe=Probe(prb),
            _debug=debug, convert_only=convert_only)


# -----------------------------------------------------------------------------
# KlustaKwik
# -----------------------------------------------------------------------------
def write_mask(mask, filename, fmt="%f"):
    with open(filename, 'w') as fd:
        fd.write(str(mask.shape[1])+'\n') # number of features
        np.savetxt(fd, mask, fmt=fmt)

def write_fet(fet, filepath):
    with open(filepath, 'w') as fd:
        #header line: number of features
        fd.write('%i\n' % fet.shape[1])
        #next lines: one feature vector per line
        np.savetxt(fd, fet, fmt="%i")

def save_old(exp, shank, dir=None):
    chg = exp.channel_groups[shank]
            
    # Create files in the old format (FET and FMASK)
    fet = chg.spikes.features_masks[...]
    if fet.ndim == 3:
        masks = fet[:,:,1]  # (nsamples, nfet)
        fet = fet[:,:,0]  # (nsamples, nfet)
    else:
        masks = None
    res = chg.spikes.time_samples[:]
    
    times = np.expand_dims(res, axis =1)
    masktimezeros = np.zeros_like(times)
    
    fet = convert_dtype(fet, np.int16)
    fet = np.concatenate((fet, times),axis = 1)
    mainfetfile = os.path.join(dir, exp.name + '.fet.' + str(shank))
    write_fet(fet, mainfetfile)
    
    if masks is not None:
        fmasks = np.concatenate((masks, masktimezeros),axis = 1)
        fmaskfile = os.path.join(dir, exp.name + '.fmask.' + str(shank))
        write_mask(fmasks, fmaskfile, fmt='%f')
    
def run_klustakwik(filename, dir=None, **kwargs):
    # Open the KWIK files in append mode so that we can write the clusters.
    with Experiment(filename, dir=dir, mode='a') as exp:
        name = exp.name
        shanks = exp.channel_groups.keys()
        
        # Set the KlustaKwik parameters.
        params = dict()
        for key, value in kwargs.iteritems():
            if key == 'maskstarts' or key == 'maxpossibleclusters':
                print ("\nERROR: All PRM KlustaKwik parameters must now be prefixed by KK_ or they will be ignored."
                "\nSee https://github.com/klusta-team/example/blob/master/params.prm for an example."
                "\nPlease update or comment out the parameters to use the defaults, then re-run with klusta --cluster-only.")
                return False
                
            if key[:3] == 'kk_':
                params[key[3:]] = value
                
        # Check for conditions which will cause KK to fail.
        if not (params.get('maskstarts', 500) <= params.get('maxpossibleclusters', 1000)):
            print "\nERROR: Condition not met: MaskStarts <= MaxPossibleClusters."
            return False
            
        if (((params.get('maskstarts', 500) == 0) or (params.get('usedistributional', 1) == 0)) and not
            (params.get('minclusters', 100) <= params.get('maxclusters',110) <= params.get('maxpossibleclusters', 1000))):
            print "\nERROR: Condition not met: MinClusters <= MaxClusters <= MaxPossibleClusters."
            return False
            
        # Switch to temporary directory.
        start_dir = os.getcwd()
        tmpdir = os.path.join(start_dir, '_klustakwik')
        if not os.path.exists(tmpdir):
            os.mkdir(tmpdir)
        os.chdir(tmpdir)
        
        for shank in shanks:
            # chg = exp.channel_groups[shank]   
            save_old(exp, shank, dir=tmpdir)
            
            # Generate the command for running klustakwik.
            cmd = 'klustakwik {name} {shank} {params}'.format(
                name=name,
                shank=shank,
                params=' '.join(['-{key} {val}'.format(key=key, val=str(val))
                                    for key, val in params.iteritems()]),
            )
            
            # Save a file with the KlustaKwik run script so user can manually re-run it if it aborts (or edit)
            script_filename = "runklustakwik_" + str(shank) + ".sh"
            scriptfile = open(script_filename, "w")
            scriptfile.write(cmd)
            scriptfile.close()
    
            # Run KlustaKwik.
            os.system(cmd)
            
            # Read back the clusters.
            clu_filename = name + '.clu.' + str(shank)
            
            if not os.path.exists(clu_filename):
                print "\nERROR: Couldn't open the KlustaKwik output file {0}".format(clu_filename)
                print ("This is probably due to KlustaKwik not completing successfully. Please check for messages above.\n"
                "You can re-run KlustaKwik by calling klusta with the --cluster-only option. Please verify the\n"
                "printed parameters carefully, and if necessary re-run with the default KlustaKwik parameters.\n"
                "Common causes include running out of RAM or not prefixing the PRM file KlustaKwik parameters by KK_.")
                return False
            
            clu = read_clusters(clu_filename)
            
            # Put the clusters in the kwik file.
            add_clustering(exp._files, channel_group_id=str(shank), name='original',
                           spike_clusters=clu, overwrite=True)
            add_clustering(exp._files, channel_group_id=str(shank), name='main',
                           spike_clusters=clu, overwrite=True)
        
        # Switch back to original dir.
        os.chdir(start_dir)
        

# -----------------------------------------------------------------------------
# All-in-one script
# -----------------------------------------------------------------------------
def run_all(prm_filename, dir=None, debug=False, overwrite=False, 
            runsd=True, runkk=True, convert_only=False):
            
    if not os.path.exists(prm_filename):
        exception("The PRM file {0:s} doesn't exist.".format(prm_filename))
        return
    
    info = _load_files_info(prm_filename, dir=dir)
    experiment_name = info['experiment_name']
    prm = info['prm']
    prb = info['prb']
    data = info['data']
    nchannels = info['nchannels']
    
    if files_exist(experiment_name, dir=dir) & runsd == True:
        if overwrite:
            delete_files(experiment_name, dir=dir, types=('kwik', 'kwx', 'high.kwd', 'low.kwd'))
        else:
            print(("\nERROR: A .kwik file already exists. To overwrite, call klusta with the --overwrite option,\n"
                   "which will overwrite existing .kwik, .kwx, .high.kwd, and .low.kwd files, or delete them manually first."))
            return False   
    if runsd:
        run_spikedetekt(prm_filename, dir=dir, debug=debug, convert_only=convert_only)
    if runkk:
        run_klustakwik(experiment_name, dir=dir, **prm)
        
def main():
    
    if not check_path():
        return
    
    parser = argparse.ArgumentParser(description='Run SpikeDetekt and/or KlustaKwik.')
    parser.add_argument('prm_file',
                       help='.prm filename')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='run the first few seconds of the data for debug purposes')
    parser.add_argument('--overwrite', action='store_true', default=False,
                       help='overwrite the KWIK files if they already exist')
                       
    parser.add_argument('--detect-only', action='store_true', default=False,
                       help='run only SpikeDetekt')
    parser.add_argument('--cluster-only', action='store_true', default=False,
                       help='run only KlustaKwik (after SpikeDetekt has run)')
    parser.add_argument('--convert-only', action='store_true', default=False,
                       help='only convert raw data to Kwik format, no spike detection')
    parser.add_argument('--version', action='version', version='Klusta-Suite version {0:s}'.format(spikedetekt2.__version__))        


    args = parser.parse_args()

    runsd, runkk, convert_only = True, True, False

    if args.detect_only:
        runkk = False
    if args.cluster_only:
        runsd = False
    if args.convert_only:
        runkk = False
        convert_only = True
    
    run_all(args.prm_file, debug=args.debug, overwrite=args.overwrite,
            runsd=runsd, runkk=runkk, convert_only=convert_only)
        
if __name__ == '__main__':
    main()
