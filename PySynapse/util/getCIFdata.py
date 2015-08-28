#!/usr/bin/python
# -*- coding: utf-8 -*-

# revised 20 July 2015 BWS 

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
from numba import jit
import os.path as path
import time as time
import glob as glob
import numpy as np
import csv as csv
import FileIO.ProcessEpisodes as PE 

def version():
    return 0.915

def calcCIFforOneEpisode(fileRoot, eventFolder, testWidthMs):
    dataList, labelList = processInput(fileRoot, eventFolder)
    if len(dataList) == 2:
        combos = [(0,1)]
    else:
        combos = [(0,1), (1,2), (0,2)]
    for oneCombo in combos:
        x1 = oneCombo[0]
        x2 = oneCombo[1]
        retDict = calcCIF(dataList[x1], dataList[x2], testWidthMs)
        print(labelList[x1] + " x " + labelList[x2] + ": ", str(retDict["CIF"]))


def testCIF():
    startTime = time.time()
    a1 = np.array([52., 58, 68, 168, 189.2, 199.2])
    a2 = np.array([35.5, 52.6, 59.1, 70.2, 80.2, 168.2, 170, 192.2])
    widthMs = 1.0
    realCIF, syncCounts = calcCIFcore(a1, a2, widthMs)
    randCIF, pValue = calcCIFbootstrap(a1, a2, realCIF, widthMs, 5000)
    print("Time: " + str(time.time() - startTime), str(realCIF) + " vs " + str(randCIF))
    return realCIF, syncCounts, randCIF, pValue

def calcCIFcore(sourceTrainIn, targetTrainIn, halfWidth):
    # the longer input array is called target while the shorter one is the source
    if len(sourceTrainIn) < 2 | len(targetTrainIn) < 2:
        return 0., 0 # no events on either channel after pruning one away means zero CIF so we are done
    # remove the last time since that entry will always be the same after shuffling intervals
    sourceTrain = np.delete(sourceTrainIn, -1) 
    targetTrain = np.delete(targetTrainIn, -1)
    syncCounts = 0
    for oneEvent in sourceTrain:
        #numHits = np.sum((targetTrain >= (oneEvent - halfWidth)) & (targetTrain < (oneEvent + halfWidth)))
        foundOne = False  
        t1 = oneEvent - halfWidth
        t2 = oneEvent + halfWidth
        for testEvent in targetTrain:
            if testEvent >= t1:
                if testEvent < t2:
                    foundOne = True
                    break
            if testEvent + halfWidth > oneEvent:
                break
        if foundOne:
            syncCounts += 1
    CIF = syncCounts / len(sourceTrain)
    return CIF, syncCounts

def calcCIFbootstrap(sourceTrain, targetTrain, realCIF, halfWidth, bootstrapNum):
    randCIF = 0.
    numGoodRand = 0
    sourceIntervals = np.diff(np.insert(0., 1, sourceTrain))
    targetIntervals = np.diff(np.insert(0., 1, targetTrain))
    for ii in range(bootstrapNum):
        tempRandCIF, randSyncCounts = calcCIFcore(np.cumsum(np.random.permutation(sourceIntervals)), 
                    np.cumsum(np.random.permutation(targetIntervals)), halfWidth)
        randCIF += tempRandCIF
        if tempRandCIF >= realCIF:
            numGoodRand += 1
    return randCIF / bootstrapNum, numGoodRand / bootstrapNum


def calcCIF(array1in, array2in, testWidthMs, startTime, stopTimeIn):
    if stopTimeIn == 0.:
        # do the entire time span covered by the two events files
        stopTime = max(max(array1in), max(array2in))
    else:
        stopTime = stopTimeIn
    array1 = array1in[(array1in >= startTime) & (array1in < stopTime)] - startTime
    array2 = array2in[(array2in >= startTime) & (array2in < stopTime)] - startTime
    if len(array2) >= len(array1):
        source = array1 
        target = array2
    else:
        target = array2
        source = array1
    testHalfWidth = testWidthMs / 2.
    CIF, syncCounts = calcCIFcore(source, target, testHalfWidth)
    retDict = {}
    retDict["CIF"] = CIF
    retDict["SyncCounts"] = syncCounts
    if len(source) < 3 | len(target) < 3:
         # need at least two events per channel after pruning one away during randomization
        retDict["randCIF"] = 0.
        retDict["pValue"] = 1.
        retDict["bootstrapNum"] = 0
    else:
        bootstrapNum = 200
        randCIF, pValue = calcCIFbootstrap(source, target, CIF, testHalfWidth, bootstrapNum)
        if pValue < 0.05:
            # repeat at potentially signficant runs wit more bootstrap runs
            bootstrapNum = 5000
            randCIF, pValue = calcCIFbootstrap(source, target, CIF, testHalfWidth, bootstrapNum)
        retDict["randCIF"] = randCIF
        retDict["pValue"] = pValue
        retDict["bootstrapNum"] = bootstrapNum
    return retDict

def processInput(fileRoot, eventFolder):
    # this is the main routine called with fileRoot not containing a path
    if "," in fileRoot:
        fileRoot = fileRoot.replace(",", "")
    if fileRoot[-4:].lower() == ".dat":
        fileRoot = fileRoot[0:-4] # take off .dat
    testStr = eventFolder + "/" + fileRoot + "*.Events.csv"
    if "," in testStr:
        testStr = testStr.replace(",", "")
    results = glob.glob(testStr)
    if len(results) > 1:
        dataList = []
        labelList = []
        for oneFile in results:
            key = oneFile.split(".Events")[0].split(".")[-1]
            dataList.append(np.array(PE.readEventTimesCore(oneFile)))
            labelList.append(key)
        return dataList, labelList
    else:
        return None

if __name__ == "__main__":
    # first arg should be episode fileRoot without path, second is the events folder with csv files
    #calcCIFforOneEpisode(sys.argv[1], sys.argv[2])
    testCIF()