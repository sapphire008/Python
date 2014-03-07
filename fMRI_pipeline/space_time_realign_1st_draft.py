#!/usr/bin/env python
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
This script requires the nipy-data package to run. It is an example of
simultaneous motion correction and slice timing correction in
multi-session fMRI data from the FIAC 2005 dataset. Specifically, it
uses the first two sessions of subject 'fiac0'.

Usage:
  python space_time_realign.py

Two images will be created in the working directory for the realigned series::

    rarun1.nii
    rarun2.nii

Author: Alexis Roche, 2009.
"""
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
def space_time_realign(Images,TR=2,numslices=None,SliceTime='asc_alt_2',RefScan=None):
    '''
    4D simultaneous slice timing and spatial realignment. Adapted from
    Alexis Roche's example script, and extend to be used for multiplex
    imaging sequences
    
    Inputs:
    
        Images: list of images, input as a list of strings
        
        numslices: for non-multiplex sequence, default to be the number of
            slices in the image. For multiplex sequence, enter as a tuple,
            such that the first element is the number of planes acquired in
            parallel between each other, and the second element is the number
            of slices of each parallel plane/slab
        
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
        
        RefScan: reference volume for spatial realignment movement estimation
    '''
    
    # load images    
    runs = [load_image(run) for run in Images]
    # parse data info
    if numslices is None:
        numslices = runs[0].shape[2]
        numplanes = 1
    elif isinstance(numslices,tuple):
        numslices = numslices[0]
        numplanes = numplanes[1]
    # parse slice timing according to the input
    slice_timing = getattr(timefuncs,SliceTime)(TR,numslices)
    #repeat the slice timing for multiplex seqquence
    slice_timing = np.tile(slice_timing,numplanes)
    # Spatio-temporal realigner assuming interleaved ascending slice order
    R = SpaceTimeRealign(runs, tr=TR, slice_times=slice_timing, slice_info=2,
                         affine_class='Rigid')
    
    print('Slice times: %s' % slice_timing)
    # Estimate motion within- and between-sessions
    R.estimate(refscan=RefScan)
    # Resample data on a regular space+time lattice using 4d interpolation
    print('Saving results ...')
    for i in range(len(runs)):
        corr_run = R.resample(i)
        fname = os.path.join(os.path.split(Images[i])[0],'ra' + os.path.split(Images[i])[1])
        save_image(corr_run, fname)
        print(fname)
