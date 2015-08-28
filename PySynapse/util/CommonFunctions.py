#!/usr/bin/python
# -*- coding: utf-8 -*-

# last revised 28 July 2015 BWS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy as sp
import os
import sys
import os.path as path
import shutil as shutil

def movingAverage(inArray, windowSize):
    window = np.ones(int(windowSize)) / float(windowSize)
    return np.convolve(inArray, window, 'same')

def sgolayFilter(inArray, windowSize, polyOrder=3):
    return sp.signal.savgol_filter(inArray, windowSize, polyOrder)

def testINIfile(INIfileName):
    if INIfileName[-4:].lower() != ".txt":
        INIfileName += ".txt"
    if path.isfile(INIfileName):
        return INIfileName
    else:
        print("Could not find requested INI file: " + INIfileName)
        return None

def makeBlockName(epiFileName):
    # this routine makes the correct formatted row name for CSV files
    filePath, fileName = path.split(epiFileName)
    fileRoot, fileExt = path.splitext(fileName)
    dots = [i for i, ltr in enumerate(fileRoot) if ltr == "."]
    blockName = fileRoot[0:dots[1]] # to get rid of .S1.E21 stuff
    blockName = blockName.replace("Cell", "")
    blockName = blockName.replace(" ", "")
    return blockName

def getCleanTempFolder():
    if os.name == "posix":
        outputFolder = "/Volumes/RamDrive/FromSynapse"
    else:
        outputFolder = "R:/FromSynapse"
    if not path.isdir(outputFolder):
        os.mkdir(outputFolder)
    else:
        shutil.rmtree(outputFolder) # first remove directory to get rid of old eggs
        os.mkdir(outputFolder)
    return outputFolder

def getTempFolder():
    if os.name == "posix":
        outputFolder = "/Volumes/RamDrive/FromSynapse"
    else:
        outputFolder = "R:/FromSynapse"
    return outputFolder
