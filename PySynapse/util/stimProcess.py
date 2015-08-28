#!/usr/bin/python
# -*- coding: utf-8 -*-

# last revised 22 June 2015 BWS

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from array import array
from itertools import repeat

def generateStimWaveform(stimStringIn, msPerPoint, sweepWindowMs, multiplyFactor):
    # takes string like step with some parameters, for example: DC 50 | step 50 250 120 (command units are mV or pA)
    # use multiplyFactor for conversion of real units in string command (pA or mV) into DAC units
    pointsPerMs = round(1. / msPerPoint)
    numPoints = int(sweepWindowMs * pointsPerMs)
    retVector = array("i", repeat(0, numPoints))
    if not "  " in stimStringIn:
        stimString = stimStringIn.lower()
        parts = stimString.split("|")
        for oneCmd in parts:
            subparts = oneCmd.strip().split(" ")
            thisCmd = subparts[0]
            if thisCmd == "dc":
                # synatx is: dc aa*multiplyFactor value
                if len(subparts) == 2:
                    testValue = int(float(subparts[1]) * multiplyFactor)
                    for index in xrange(0, numPoints):
                        retVector[index] += testValue
                else:
                    print("Too many input values for a DC command: " + oneCmd)
                    retVector = None
            elif thisCmd == "step":
                # syntax is: step aa ms to bb ms at cc*multiplyFactor value
                if len(subparts) == 4:
                    startIndex = int(float(subparts[1]) * pointsPerMs)
                    stopIndex = int(float(subparts[2]) * pointsPerMs)
                    testValue = int(float(subparts[3]) * multiplyFactor)
                    if startIndex >= 0 and stopIndex < numPoints:
                        for index in xrange(startIndex, stopIndex):
                            retVector[index] += testValue
                    else:
                        print("Indexing not allowed: " + oneCmd)
                        retVector = None
                else:
                    print("Too many input values for a Step command: " + oneCmd)
                    retVector = None
            elif thisCmd == "train":
                # syntax is: train aa repeats of bb ms On then cc ms Off with dd ms initial delay and ee*multiplyFactor value
                if len(subparts) == 6:
                    numRepeats = int(subparts[1])
                    if numRepeats > 0:
                        onCount = int(float(subparts[2]) * pointsPerMs)
                        offCount = int(float(subparts[3]) * pointsPerMs)
                        initIndex = int(float(subparts[4]) * pointsPerMs)
                        testValue = int(float(subparts[5]) * multiplyFactor)
                        if initIndex >= 0 and initIndex + (numRepeats * onCount) + ((numRepeats - 1) * offCount) < numPoints:
                            for repeatNum in xrange(numRepeats):
                                offset = initIndex + (repeatNum * (onCount + offCount))
                                for index in xrange(onCount):
                                    retVector[offset + index] += testValue
                        else:
                            print("Indexing not allowed: " + oneCmd)
                            retVector = None
                else:
                    print("Too many input values for a Train command: " + oneCmd)
                    retVector = None
            else:
                print("Could not understand requested command: " + thisCmd)
                retVector = None
        return retVector
    else:
        print("input string cannot contain double spaces")
        return None