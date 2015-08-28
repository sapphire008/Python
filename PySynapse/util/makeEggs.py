#!/usr/bin/python
# -*- coding: utf-8 -*-

# revised 5 Aug 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as path
import shutil as shutil
import csv as csv
import FileIO.ProcessEpisodes as PE 

def processInput(inFileRoot, outputFolder, overrideKeyIn="", newValueIn=""):
    # wrapper function for processOneINIfile so that you can batch calls to multiple INI files together
    # in one text file (and comment specific ones out with a leading # character)
    if inFileRoot[-4:].lower() != ".txt":
        inFileRoot += ".txt"
    if not path.isfile(inFileRoot):
        print("Could not find requested INI file (1): " + inFileRoot)
        return None
    with open(inFileRoot, "r") as fINI:
        lines = fINI.readlines()
        if lines[0].strip().lower() != "#ini list":
            # not a batch file so send it on to main routine
            processOneINIfile(inFileRoot, outputFolder, EmptyEggFolder=True, OverrideKey=overrideKeyIn, NewValue=newValueIn)
        else:
            for ii in range(1, len(lines)): # skip first line since that says this is a list of INI files
                oneLine = lines[ii].strip()
                print(str(len(lines)) + " - " + oneLine)
                if oneLine[0] != "#":
                    processOneINIfile(oneLine, outputFolder, EmptyEggFolder=(ii == 1), OverrideKey=overrideKeyIn, NewValue=newValueIn)

def processOneINIfile(INIfileRoot, outputFolder, EmptyEggFolder=True, OverrideKey="", NewValue=""):
    if INIfileRoot[-4:].lower() != ".txt":
        INIfileRoot += ".txt"
    if not path.isfile(INIfileRoot):
        print("Could not find requested INI file (2): " + INIfileRoot)
        return None
    if not path.isdir(outputFolder):
        os.mkdir(outputFolder)
        print("Created new folder for output: " + outputFolder)
    else:
        if EmptyEggFolder:
            shutil.rmtree(outputFolder) # first remove directory to get rid of old eggs
            os.mkdir(outputFolder)
    parserObj = PE.readEpisodeList(INIfileRoot)
    nonexcludedExpts = PE.getListOfNonexcludedExpts(parserObj)
    trialsList = ""
    curTrialList = ""
    curCluster = None
    for oneExptName in nonexcludedExpts:
        newCluster, newEgg = oneExptName.split("_")
        if newCluster != curCluster:
            curCluster = newCluster
            curFolder = outputFolder + "/" + curCluster
            if not path.isdir(curFolder):
                os.mkdir(curFolder)
        epiList = PE.episodeList(parserObj, oneExptName)
        if epiList:
            Parm = getDetectionParms(parserObj, oneExptName)
            if Parm["errorCondition"]:
                return None
            if OverrideKey:
                Parm[OverrideKey] = float(NewValue)
            outputFileName = curFolder + "/" + newEgg + ".csv"
            helperDumpEgg(outputFileName, epiList, Parm, Parm["windowStart"])
            if Parm["includeBasal"]:
                outputFileName = curFolder + "/" + newEgg + "basal.csv"
                helperDumpEgg(outputFileName, epiList, Parm, 0.)
            if Parm["isSequence"]:
                outputFileName = curFolder + "/" + newEgg + "begin.csv"
                helperDumpEgg(outputFileName, epiList, Parm, Parm["seqStartWindow"])
    print("makeEggs.py dumped eggs in: " +  outputFolder)

"""
This routine writes one egg as a csv file (eg A.csv)
"""
def helperDumpEgg(outputFileName, epiList, Parm, windowStartMsIn):
    # revised 17 July 2015 to break out as separate function to allow multiple calls when processing sequences
    if windowStartMsIn < 0.:
        windowStartMs = max(Parm["stimTimes"]) - windowStartMsIn # negative time means go forward from last stim by the abs(timeIn)
    else:
        windowStartMs = windowStartMsIn
    with open(outputFileName, "w") as fCSV:
        writer = csv.writer(fCSV)
        # writer.writerow(chanNames) # header with names of each cells used for egg
        for oneEpiName in epiList: # for rows of egg
            newRow = []
            Factor = 1. / (Parm["windowDur"] / 1000.) # to convert to rate in Hz
            for oneChanName in Parm["chanNames"]: # for columns of egg
                events = PE.readEventTimes(oneEpiName, oneChanName, Parm["eventPath"], stimTimes=Parm["stimTimes"])
                if events:
                    retRate = Factor * sum((ii > windowStartMs) and (ii <= windowStartMs + Parm["windowDur"]) for ii in events)
                    if Parm["baselineStop"] > 0.:
                        newValue = sum((ii > Parm["baselineStart"]) and (ii <= Parm["baselineStop"]) for ii in events)
                        baselineRate = newValue / ((Parm["baselineStop"] - Parm["baselineStart"]) / 1000.)
                        retRate = retRate - baselineRate
                else:
                    retRate = 0.
                newRow.append(str(retRate))
            writer.writerow(newRow)


def getDetectionParms(parserObj, exptName):
    falseStrings = ["none", "off", "false", "no", "0"]
    retDict = {}
    retDict["errorCondition"] = False # added 17 July 2015 BWS
    chanNamesIn = PE.getOneValue(parserObj, exptName, "channels").split(",")
    if not chanNamesIn:
        print("INI file must have channels = xx")
        return None
    retDict["chanNames"] = [oneName.strip() for oneName in chanNamesIn]
    temp = PE.getOneValue(parserObj, exptName, "windowDur")
    if not temp:
        print("INI file must have windowDur = xx in ms")
        return None

    if float(temp) < 100.:
        print("Window start and duration times need to be in milliseconds")
        return None
    retDict["windowDur"] = float(temp)
    temp = PE.getOneValue(parserObj, exptName, "windowStart")
    if not temp:
        print("INI file must have windowStart = xx in ms")
        return None
    retDict["windowStart"] = float(temp)
    baselineStart = PE.getOneValue(parserObj, exptName, "baselineStart")
    if baselineStart:
        retDict["baselineStart"] = float(baselineStart)
    else:
        retDict["baselineStart"]  = 0.
    baselineStop = PE.getOneValue(parserObj, exptName, "baselineStop")
    if baselineStop:
        retDict["baselineStop"] = float(baselineStop)
    else:
        retDict["baselineStop"] = 0.
    eventPath = PE.getOneValue(parserObj, exptName, "eventListPath")
    if not eventPath:
        print("Could not find eventListPath key")
        return None
    retDict["eventPath"] = eventPath
    stimTimesStr = PE.getOneValue(parserObj, exptName, "stimTimes").split(",")
    retDict["stimTimes"] = [float(oneStr) for oneStr in stimTimesStr]
    minAmp = PE.getOneValue(parserObj, "Analysis", "minAmp")
    if minAmp:
        retDict["minAmp"] = float(minAmp)
    else:
        retDict["minAmp"] = 0.
    minInterval = PE.getOneValue(parserObj, exptName, "minInterval")
    if minInterval:
        retDict["minInterval"] = float(minInterval)
    else:
        retDict["minInterval"] = 0.
    includeBasal = PE.getOneValue(parserObj, exptName, "includeBasal")
    if includeBasal:
        retDict["includeBasal"] = includeBasal.lower() not in falseStrings # True if entry is not one of the falseStrings
    else:
        retDict["includeBasal"] = False
    isSequence = PE.getOneValue(parserObj, exptName, "isSequence")
    if isSequence:
        retDict["isSequence"] = isSequence.lower() not in falseStrings # True if entry is not one of the falseStrings
    else:
        retDict["isSequence"] = False
    seqStartWindow = PE.getOneValue(parserObj, exptName, "seqStartWindow")
    if seqStartWindow:
        retDict["seqStartWindow"] = float(seqStartWindow)
    else:
        if retDict["isSequence"]:
            print("The INI file must include a seqStartWindow value (in ms) for generating sequence eggs")
            retDict["errorCondition"] = True
    return retDict


if __name__ == "__main__":
    if len(sys.argv) == 3:
        processInput(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 5:
        processInput(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("incorrect number of parameters sent to makeEggs.py")
