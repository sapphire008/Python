#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:

PYTHONPKGPATH = '/hsgs/projects/jhyoon1/pkg64/pythonpackages/'

#from __future__ import print_function  # Python 2/3 compatibility
import sys,os
sys.path.append(os.path.join(PYTHONPKGPATH,'nibabel-1.30'))
import nibabel# required for nipy
sys.path.append(os.path.join(PYTHONPKGPATH,'nipy-0.3'))
import numpy as np
from nipy.algorithms.registration import SpaceTimeRealign
from nipy.algorithms.slicetiming import timefuncs
from nipy import load_image, save_image

# Input images
def space_time_realign(Images,TR=2,numslices=None,SliceTime='asc_alt_2',RefScan=0,Prefix='ra'):
    '''
    4D simultaneous slice timing and spatial realignment. Adapted from
    Alexis Roche's example script, and extend to be used for multiplex
    imaging sequences
    
    Inputs:
    
        Images: list of images, input as a list of strings/paths to images
        
        numslices: for non-multiplex sequence, default to be the number of
            slices in the image. For multiplex sequence, enter as a tuple,
            such that the first element is the number of planes acquired in
            parallel between each other, and the second element is the number
            of slices of each parallel plane/slab, i.e. (numplanes,numslices)
        
        SliceTime:enter as a string to specify how the slices are ordered.
            Choices are the following
            1).'ascending': sequential ascending acquisition
            2).'descending': sequential descending acquisition
            3).'asc_alt_2': ascending interleaved, starting at first slice
            4).'asc_alt_2_1': ascending interleaved, starting at the second
                slice
            5).'desc_alt_2': descending interleaved, starting at last slice
            6).'asc_alt_siemens': ascending interleaved, starting at the first
                slice if odd number of slices, or second slice if even number
                of slices
            7).'asc_alt_half': ascending interleaved by half the volume
            8).'desc_alt_half': descending interleaved by half the volume
        
        RefScan: reference volume for spatial realignment movement estimation.
            Note that scan 0 is the first scan.
        
        Prefix: prefix of the new corrected images. Default is 'ra'
        
        
    Author: Alexis Roche, 2009.
            Edward Cui, February 2014
    '''
    
    # Load images
    runs = [load_image(run) for run in Images]
    # Parse data info
    if numslices is None:
        numslices = runs[0].shape[2]
        numplanes = 1
    elif isinstance(numslices,tuple):
        (numplanes,numslices) = numslices
    else:
        numplanes = 1
    # Print image info
    if numplanes>1:
        print('Running multiplex: %s' % numplanes)
    print('Number of slices: %s' % numslices)
    # Parse slice timing according to the input
    slice_timing = getattr(timefuncs,SliceTime)(numslices,TR)
    # Repeat the slice timing for multiplex seqquence
    slice_timing = np.tile(slice_timing,numplanes)
    # Print slice timing info
    print('Slice times: %s' % slice_timing)
    # Spatio-temporal realigner
    R = SpaceTimeRealign(runs, tr=TR, slice_times=slice_timing, slice_info=2)
    # Estimate motion within- and between-sessions
    print('Estimating motion ...')
    R.estimate(refscan=RefScan)
    # Resample data on a regular space+time lattice using 4d interpolation
    fname=[None]*len(Images) # output images
    mfname=[None]*len(Images) # output motion parameter files
    print('Saving results ...')
    for n in range(len(Images)):
        # extract motion parameters
        motionparams = np.array([np.concatenate((M.translation,M.rotation),axis=1) for M in R._transforms[n]])
        # set motion parameter file name
        mfname[n] = os.path.join(os.path.split(Images[n])[0], 'rp_a0001.txt')
        # write the motion parameters to file
        np.savetxt(mfname[n],motionparams,fmt='%10.7e',delimiter='\t')
        # resample data
        corr_run = R.resample(n)
        # set image name
        fname[n] = os.path.join(os.path.split(Images[n])[0], Prefix + os.path.split(Images[n])[1])
        # save image
        save_image(corr_run, fname[n])
        print(fname[n])
    return(fname,mfname)

## help with interface from system environment
#import re
#if __name__ == '__main__':
#    try:
#        Images = re.split(sys.argv[0])
#    except:
#        Images = None
#    try:
#        TR = float(sys.argv[1])
#    except:
#        TR = 2
#    try:
#        numslices = tuple(sys.argv[2])
#    except:
#        numslices = None
#    try:
#        SliceTime = str(sys.argv[3])
#    except:
#        SliceTime = 'asc_alt_2'
#    try:
#        RefScan = float(sys.argv[4])
#    except:
#        RefScan = 0
#    try:
#        Prefix = str(sys.argv[5])
#    except:
#        Prefix = 'ra'
#    sys.stdout.write(str(space_time_realign(Images,TR,numslices,SliceTime,RefScan,Prefix)))
    

# whole image
Images = ['/hsgs/projects/jhyoon1/midbrain_Stanford_3T/stop_signal/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/block1/6093_4_1.nii',
          '/hsgs/projects/jhyoon1/midbrain_Stanford_3T/stop_signal/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/block2/6093_5_1.nii',
          '/hsgs/projects/jhyoon1/midbrain_Stanford_3T/mid/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/block1/6093_6_1.nii',
          '/hsgs/projects/jhyoon1/midbrain_Stanford_3T/mid/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/block2/6093_7_1.nii',
          '/hsgs/projects/jhyoon1/midbrain_Stanford_3T/mid/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/block3/6093_8_1.nii',
          '/hsgs/projects/jhyoon1/midbrain_Stanford_3T/RestingState/subjects/funcs/M3020_CNI_011314_adjusted_scale_factor/6093_9_1.nii']
fname = space_time_realign(Images,TR=2,numslices=(3,25),SliceTime='asc_alt_2',RefScan=0,Prefix='ra')

## slab 1
#Images = ['/hsgs/projects/jhyoon1/midbrain_Stanford_3T/RestingState/subjects/funcs/M3020_CNI_011314/0001_1.nii.gz'];
#fname = space_time_realign(Images,TR=2,numslices=None,SliceTime='asc_alt_2',RefScan=0,Prefix='ra')
#
## slab 2
#Images = ['/hsgs/projects/jhyoon1/midbrain_Stanford_3T/RestingState/subjects/funcs/M3020_CNI_011314/0001_2.nii.gz'];
#fname = space_time_realign(Images,TR=2,numslices=None,SliceTime='asc_alt_2',RefScan=0,Prefix='ra')
#
## slab 3
#Images = ['/hsgs/projects/jhyoon1/midbrain_Stanford_3T/RestingState/subjects/funcs/M3020_CNI_011314/0001_3.nii.gz'];
#fname = space_time_realign(Images,TR=2,numslices=None,SliceTime='asc_alt_2',RefScan=0,Prefix='ra')
