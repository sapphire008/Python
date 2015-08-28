# -*- coding: utf-8 -*-
"""
Routines for handing episode lists and CSV summaries

last revised 5 Aug 2015 BWS

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from numpy import asscalar, fromfile, mean, float64, float32, int16, int32, uint8
import numpy as np
import os.path as path
import os
import statistics as stats
import scipy.stats as sps
import math as math
import pandas as pd
import sys
import os
import csv as csv
if sys.version_info[0] == 3:
    import configparser as CP
else:
    import ConfigParser as CP
from FileIO.CoreDatFileIO import readDatFile

def readEpisodeList(episodeListFileName):
    #"this reads an INI-style description of the expt episode clusters and returns a parser object"
    if not path.isfile(episodeListFileName):
        # first check to see if not path was provided because file is in local directory
        curPath = os.getcwd()
        tempRoot = episodeListFileName
        if tempRoot[-4:].lower() != ".txt":
            tempRoot += ".txt"
        episodeListFileName = curPath + "/" + tempRoot
        if not path.isfile(episodeListFileName):
            print("Could not find requested ini file: " + tempRoot)
            print("  in current folder: " + curPath)
            return None
    # so we know it is a real file at this point
    par = CP.SafeConfigParser()
    retValue = par.read(str(episodeListFileName))
    if len(retValue) > 0:
        return par
    else:
        print("Problem with opening INI file")
        return None

def getListOfNonexcludedExpts(parserObj):
    # will ignore Analysis section
    retList = []
    falseStrings = ["none", "off", "false", "no", "0"]
    for oneExpt in parserObj.sections():
        skipExpt = False
        # first see if INI flag set conditons for selectGroup
        if parserObj.has_option(oneExpt, "selectedExptGroup"):
            selectedGroup = parserObj.get(oneExpt, "selectedExptGroup").lower()
            if parserObj.has_option(oneExpt, "group"):
                curGroup = parserObj.get(oneExpt, "group").lower()
                if curGroup != selectedGroup:
                    skipExpt = True
            else:
                skipExpt = True
        # now test for global enableExpt flag being set True (ie, not one of the falseStrings)
        if not skipExpt:
            globalExcludeEnable = True
            if parserObj.has_option(oneExpt, "enableExcludedExpt"):
                enableExclude = parserObj.get(oneExpt, "enableExcludedExpt")
                if enableExclude.lower() in falseStrings:
                    globalExcludeEnable = False
            if globalExcludeEnable:
                # find key(s) that represent excluded expts if set to True
                excludeKeyList = []
                if parserObj.has_option(oneExpt, "excludeExptKey"):
                    parts = parserObj.get(oneExpt, "excludeExptKey").split(",")
                    for onePart in parts:
                        excludeKeyList.append(onePart.strip())
                else:
                    # default key for excluding entire expts
                    excludeKeyList = ["excludeExpt"]
                # now get answers for each excludedExpt key
                if excludeKeyList:
                    for oneExcludeKey in excludeKeyList:
                        if parserObj.has_option(oneExpt, oneExcludeKey):
                            excludeAnswer = parserObj.get(oneExpt, oneExcludeKey)
                            if excludeAnswer.lower() not in falseStrings:
                                skipExpt = True
        if not skipExpt:
            retList.append(oneExpt)
    if "Analysis" in retList:
        retList.remove("Analysis")
    return retList

def getOneValue(parserObj, sectionName, keyName):
    if parserObj.has_option(sectionName, keyName):
        return parserObj.get(sectionName, keyName)
    else:
        return None

def getAllItems(parserObj, sectionName):
    if parserObj.has_section(sectionName):
        return parserObj.items(sectionName)
    else:
        return None

def getSmartAverage(parserObj, sectionName, chanNameList):
    # returns a Dict of all requested average traces after dealing with requested nullRange and filterCmd
    #  filterCmd is yet to be implemented
    selectedEpiNames = episodeList(parserObj, sectionName)
    if not selectedEpiNames:
        return None
    epiList = [] # to accumulate episode types
    for oneFileName in selectedEpiNames:
        # read in episodes once here so that getTraces does not have to on each call
        epiList.append(readDatFile(str(oneFileName)))
    pointsPerMs = 1. / epiList[0]["msPerPoint"]
    if parserObj.has_option(sectionName, "nullRange"):
        nullRangeStr = parserObj.get(sectionName, "nullRange").split(",")
        nullRange = (float(nullRangeStr[0]), float(nullRangeStr[1]))
    else:
        nullRange = None
    retDict = {}
    retDict["pointsPerMs"] = pointsPerMs
    for oneChanName in chanNameList:
        traces = getTracesFromEpiList(epiList, oneChanName, nullRange, None, None, None)
        retDict[oneChanName] = averageTraces(traces)
    return retDict

def episodeList(parserObj, sectionName):
    #"returns a python list of full names and paths of selected episodes"
    falseStrings = ["none", "off", "false", "no", "0"]
    sectionNames = parserObj.sections()
    if sectionName in sectionNames:
        retValue = []
        outerPath = parserObj.get(sectionName, "outerPath") 
        if os.name == "posix":
            topFolders = next(os.walk("/Volumes"))[1]
            if "BenWork" in outerPath and "BenWork" not in topFolders:
                if "BenHome" in topFolders:
                    outerPath = outerPath.replace("BenWork", "BenHome")
                else:
                    print("Could not find a replacement for BenWork in outer path")
                    return None
        tempPath = outerPath + "/" + parserObj.get(sectionName, "innerPath") + "/" + parserObj.get(sectionName, "fileRoot")
        if "//" in tempPath:
            tempPath.replace("//", "/")
        if parserObj.has_option(sectionName, "selectedEpisodeKey"):
            selectedEpisodeKey = parserObj.get(sectionName, "selectedEpisodeKey")
        else:
            selectedEpisodeKey = "episodes"
        if parserObj.has_option(sectionName, selectedEpisodeKey):
            # this expt has episodes listed so first make exclusion list
            enableExcludedEpisodes = True # default condition unless specified otherwise in the INI file
            if parserObj.has_option(sectionName, "enableExcludedEpisodes"):
                tempStr = parserObj.get(sectionName, "enableExcludedEpisodes").lower()
                if tempStr in falseStrings:
                    enableExcludedEpisodes = False
            epiExclude = [] # will contin a list of excluded epi numbers as integers (not strings)
            if enableExcludedEpisodes and parserObj.has_option(sectionName, "epiExclude"):
                epiExcludeStr = parserObj.get(sectionName, "epiExclude").split(",")
                for i in range(len(epiExcludeStr)):
                    # convert from string to int
                    epiExclude.append(int(epiExcludeStr[i]))
            # now get all selected episodes and check against exclusion list
            if parserObj.has_option(sectionName, "extendEpisodeRange"):
                rangeExtension = int(parserObj.get(sectionName, "extendEpisodeRange"))
            else:
                rangeExtension = 0
            epiList = [] # will contain a list of epi numbers as integers (not strings)
            epiListStr = parserObj.get(sectionName, selectedEpisodeKey).split(",")
            for oneEpiListPart in epiListStr:
                if "-" in oneEpiListPart:
                    # this part of the episode selection string is a range so process ends of it
                    parts = oneEpiListPart.split("-")
                    epiRange = []
                    epiRange.append(int(parts[0]) - rangeExtension)
                    epiRange.append(int(parts[1]) + rangeExtension)
                    for epiNum in range(epiRange[0], 1 + epiRange[1]):
                        if epiNum not in epiExclude and epiNum > 0:
                            epiList.append(epiNum)
                else:
                    # this part is just a single episode number (as a string)
                    epiNum = int(oneEpiListPart)
                    if epiNum not in epiExclude:
                        epiList.append(epiNum)
            if len(epiList) > 0:
                for oneEpi in sorted(set(epiList)): # using set to save only unique epiNums
                    if oneEpi:
                        tempEpi = tempPath + str(oneEpi) + ".dat"
                        retValue.append(tempEpi)
        return retValue
    else:
        print("Could not find section: " + sectionName)
        return None

def getTracesFromEpiList(epiListIn, chanName, nullTimeRange, stimTimes, blankingExtent, filterType):
    #"returns a list of numpy vectors all of the same chan type, eg VoltA, from a python list of episodes"
    if epiListIn:
        if isinstance(epiListIn[0], str):
            # we were given episode fileNames so need to read each episode
            epiList = [] # to accumulate episode types
            for oneFileName in epiListIn:
                epiList.append(readDatFile(str(oneFileName)))
        else:
            epiList = epiListIn
        pointsPerMs = 1. / epiList[0]["msPerPoint"]
        retTraceList = []
        for oneEpi in epiList:
            if chanName not in oneEpi["chanNames"]:
                print("Cannot find requested channel: " + chanName)
                print("  available traces: " + str(oneEpi["chanNames"]))
                return None
            tempTrace = oneEpi["traces"][chanName]
            if nullTimeRange:
                tempTrace = subtractBaseline(tempTrace, pointsPerMs, nullTimeRange, stimTimes)
            if blankingExtent:
                tempTrace = blankTrace(tempTrace, pointsPerMs,stimTimes, blankingExtent)
            if filterType:
                pass
            retTraceList.append(tempTrace)
        return retTraceList
    else:
        return None

def makeExcludeBlockList(procCmd, stimTimesIn=None):
    # for decoding eventProcessingCmd string information used in INI files
    # returns a list of 2-length tuples that specify the start/stop times in ms for each exclusion block
    excludeBlocks = []
    if stimTimesIn:
        if len(stimTimesIn) > 0:
            for oneStimTime in stimTimesIn:
                excludeBlocks.append((oneStimTime - 0.4, oneStimTime + 0.6)) # make exclusion window 
    if procCmd:
        parts = procCmd.split("|")
        for onePart in parts:
            subparts = onePart.strip().split(" ")
            actualCmd = subparts[0].lower()
            if actualCmd == "start":
                if len(subparts) == 2:
                    excludeBlocks.append((0., float(subparts[1]))) # make a list of 2-len tuples
                else:
                    print("Event processing Start command requires one argument")
            elif actualCmd == "exclude":
                if len(subparts) == 3:
                    excludeBlocks.append((float(subparts[1]), float(subparts[2]))) # make a list of 2-len tuples
                else:
                    print("Event processing Exclude command requires two arguments (start stop times in ms")
            elif actualCmd == "siu":
                if len(subparts) > 1:
                    for ii in range(1,len(subparts)): # skip first subpart since that is the command
                        siuTime = float(subparts[ii])
                        excludeBlocks.append((siuTime - 0.4, siuTime + 0.6)) # make exclusion window
            else:
                print("Unknown event processing command: " + actualCmd)
    return excludeBlocks

def readEventTimes(dataFileName, chanName, eventPath, procCmd=None, stimTimes=None):
    # this is a wrapper that checks different folders for the corresponding event csv file  (rev 12 July 2015 BWS)
    tempEventRoot, tempext = path.splitext(path.basename(dataFileName))
    tempEventRoot += "." + chanName + ".Events.csv"
    tempName = eventPath + "/" + tempEventRoot
    if not path.exists(tempName):
        # check subfolders of main event path for csv files if not in top-level events folder
        for subdir in os.listdir(eventPath):
            testFolder = eventPath + "/" + subdir
            tempName = testFolder + "/" + tempEventRoot
            if path.isdir(testFolder) and path.exists(tempName):
                break
    if path.exists(tempName):
        # next line copies input variables to keyword parameters with the same name
        return readEventTimesCore(tempName, procCmd=procCmd, stimTimes=stimTimes)
    else:
        return None

def readEventTimesCore(tempName, procCmd=None, stimTimes=None):
    # returns a list with onsetTimes in ms assuming this information was in first column
    # this function does the actual reading of the event times but requires exact path to event csv file
    if path.exists(tempName):
        eventTimes = []
        excludeBlocks = makeExcludeBlockList(procCmd, stimTimesIn=stimTimes)
        with open(tempName, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            next(reader) # skip header
            for row in reader:
                if row[0][0] != "#":
                    oneTime = float(row[0])
                    skipTime = False
                    if len(excludeBlocks) > 0:
                        for oneBlock in excludeBlocks:
                            if oneTime >= oneBlock[0] and oneTime < oneBlock[1]:
                                skipTime = True
                                break
                    if not skipTime:
                        eventTimes.append(oneTime) # assume first column is onsetTime
        return eventTimes
    else:
        return None

def readEventInfo(tempName, procCmd=None, stimTimes=None):
    # returns a dict with keys to lists for each column
    if path.exists(tempName):
        eventDict = {}
        with open(tempName, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader) # skip header
            for oneKey in headerRow:
                eventDict[oneKey] = []
            for row in reader:
                if row[0][0] != "#":
                    for ii in range(len(headerRow)):
                        eventDict[headerRow[ii]].append(float(row[ii]))
        return eventDict
    else:
        return None

"""
Generates a new CSV output file (xx.Summary.csv) which contains summary stats (mean, SD, etc) for all
columns that contains numbers and ignores columns whose values are strings

This routine generates one set of summary stats for all rows; see createExptSummary if you want per-experiment
summary statistics
"""
def createSummary(inputCSVfile):
    if "/" in inputCSVfile:
        curPath, fileRoot = path.split(inputCSVfile)
        fileRoot = path.splitext(fileRoot)[0]
        outputFile = curPath + "/" + fileRoot + ".Summary.csv"
    else:
        fileRoot = path.splitext(inputCSVfile)[0]
        outputFile = fileRoot + ".Summary.csv"
    fS = "{0:.6f}"
    with open(outputFile, "w") as wCSV:
        writer = csv.writer(wCSV)
        writer.writerow(["Name", "Mean", "SEM", "N", "SD", "CV", "Skew"])
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            firstRealRow = next(reader)
        for ii in range(len(headerRow)):
            if isNumeric(firstRealRow[ii]):
                tempF = []
                with open(inputCSVfile, "r") as fCSV:
                    reader = csv.reader(fCSV) # assume the first line is the header info
                    Temp = next(reader)
                    for row in reader:
                        tempF.append(float(row[ii]))
                    mean = np.nanmean(tempF)
                    sd = np.nanstd(tempF)
                    N = np.count_nonzero(~np.isnan(tempF))
                    sem = sd / math.sqrt(N)
                    CV = sd / mean
                    if np.isnan(tempF).any():
                        skew = 0.
                        print("NaN in " + headerRow[ii])
                    else:
                        skew = sps.skew(tempF)
                    writer.writerow([headerRow[ii], str.format(fS, mean), str.format(fS, sem), str(N), str.format(fS,sd), str.format(fS, CV), str.format(fS, skew)])


def deltaSeqEggSummary(inputCSVfile):
    if "/" in inputCSVfile:
        curPath, fileRoot = path.split(inputCSVfile)
        fileRoot = path.splitext(fileRoot)[0]
        outputFile = curPath + "/" + fileRoot
    else:
        fileRoot = path.splitext(inputCSVfile)[0]
        outputFile = fileRoot
    fS = "{0:.6f}"
    outputFileCur = outputFile + ".DeltaEgg.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            numExpts = 0
            for row in reader:
                if row[1] == "F":
                    numExpts += 1
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            for ii in range(numExpts):
                F = next(reader)
                Fbegin = next(reader)
                R = next(reader)
                Rbegin = next(reader)
                newRow = F
                newRow[1] = "Fdelta"
                for ii in range(2,len(F)):
                    newRow[ii] = str.format(fS, float(F[ii]) - float(Rbegin[ii]))
                writer.writerow(newRow)
                newRow = R
                newRow[1] = "Rdelta"
                for ii in range(3, len(F)):
                    newRow[ii] = str.format(fS, float(R[ii]) - float(Fbegin[ii]))
                writer.writerow(newRow)
    createSummary(outputFileCur)


def seqEggOutputSummary(inputCSVfile):
    if "/" in inputCSVfile:
        curPath, fileRoot = path.split(inputCSVfile)
        fileRoot = path.splitext(fileRoot)[0]
        outputFile = curPath + "/" + fileRoot
    else:
        fileRoot = path.splitext(inputCSVfile)[0]
        outputFile = fileRoot

    outputFileCur = outputFile + ".ADeggSubgroup.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            for row in reader:
                if (row[1] == "Fbegin") or (row[1] == "Rbegin"):
                    writer.writerow(row)
    createSummary(outputFileCur)
    with open(outputFileCur[:-4] + ".Summary.csv", "r") as fSum:
        reader = csv.reader(fSum)
        for row in reader:
            if row[0] == "Magnitude":
                print("AD magnitude = " + row[1])
            if row[0] == "R":
                print("AD R = " + row[1])
            if row[0] == "R2":
                print("AD R2 = " + row[1])

    outputFileCur = outputFile + ".FReggSubgroup.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            for row in reader:
                if (row[1] == "F") or (row[1] == "R"):
                    writer.writerow(row)
    createSummary(outputFileCur)
    with open(outputFileCur[:-4] + ".Summary.csv", "r") as fSum:
        reader = csv.reader(fSum)
        for row in reader:
            if row[0] == "Magnitude":
                print("FR magnitude = " + row[1])
            if row[0] == "R":
                print("FR R = " + row[1])
            if row[0] == "R2":
                print("FR R2 = " + row[1])

def seqPairwiseOutputSummary(inputCSVfile):
    if "/" in inputCSVfile:
        curPath, fileRoot = path.split(inputCSVfile)
        fileRoot = path.splitext(fileRoot)[0]
        outputFile = curPath + "/" + fileRoot
    else:
        fileRoot = path.splitext(inputCSVfile)[0]
        outputFile = fileRoot

    outputFileCur = outputFile + ".ADsubgroup.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            for row in reader:
                if row[1] == "FbeginvsRbegin":
                    writer.writerow(row)
    createSummary(outputFileCur)
    with open(outputFileCur[:-4] + ".Summary.csv", "r") as fSum:
        reader = csv.reader(fSum)
        for row in reader:
            if row[0] == "LDAacc":
                print("AD LDAacc = " + row[1])
            if row[0] == "LDAdeltaAcc":
                print("AD LDAdeltaAcc = " + row[1])
            if row[0] == "LDAtrialAcc":
                print("AD LDAtrialAcc = " + row[1])
            if row[0] == "NCacc":
                print("AD NCacc = " + row[1])
            if row[0] == "ICdist":
                print("AD ICdist = " + row[1])
            if row[0] == "ScaledDist":
                print("AD ScaledDist = " + row[1])

    outputFileCur = outputFile + ".FRsubgroup.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            for row in reader:
                if row[1] == "FvsR":
                    writer.writerow(row)
    createSummary(outputFileCur)
    with open(outputFileCur[:-4] + ".Summary.csv", "r") as fSum:
        reader = csv.reader(fSum)
        for row in reader:
            if row[0] == "LDAacc":
                print("FR LDAacc = " + row[1])
            if row[0] == "LDAdeltaAcc":
                print("FR LDAdeltaAcc = " + row[1])
            if row[0] == "LDAtrialAcc":
                print("FR LDAtrialAcc = " + row[1])
            if row[0] == "NCacc":
                print("FR NCacc = " + row[1])
            if row[0] == "ICdist":
                print("FR ICdist = " + row[1])
            if row[0] == "ScaledDist":
                print("FR ScaledDist = " + row[1])

    outputFileCur = outputFile + ".DFsubgroup.csv"
    with open(outputFileCur, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            writer.writerow(headerRow)
            for row in reader:
                if (row[1] == "FbeginvsR") or (row[1] == "FvsRbegin"):
                    writer.writerow(row)
    createSummary(outputFileCur)


"""
Routine to generate a new CSV summary file (xx.ExptSummary.csv) with summary stats such as Mean, SD, etc for all columns
whose elements are numeric. This function summarizes data for rows within the same experiment--those that have the same column0 
value. See createSummary if you want to summarize all the rows, going across all included experiments
"""
def createExptSummary(inputCSVfile):
    if "/" in inputCSVfile:
        curPath, fileRoot = path.split(inputCSVfile)
        fileRoot = path.splitext(fileRoot)[0]
        outputFile = curPath + "/" + fileRoot + ".ExptSummary.csv"
    else:
        fileRoot = path.splitext(inputCSVfile)[0]
        outputFile = fileRoot + ".ExptSummary.csv"
    fS = "{0:.6f}"
    with open(outputFile, "w") as wCSV:
        writer = csv.writer(wCSV)
        with open(inputCSVfile, "r") as fCSV:
            reader = csv.reader(fCSV) # assume the first line is the header info
            headerRow = next(reader)
            firstRealRow = next(reader) # first row with actual numbers/strings
            # save header row in new CSV file and close reader
            writer.writerow(headerRow)
        with open(inputCSVfile, "r") as fCSV:
            # re-open input CSV file
            reader = csv.reader(fCSV) # assume the first line is the header info
            junk = next(reader) # throw away header row this time
            fS = "{0:.6f}"
            exptRows = None
            curExpt = ""
            for row in reader:
                if row[0] == curExpt:
                    exptRows.append(row)
                else:
                    # ExptName changed
                    if exptRows:
                        # got all the rows for one expt so generate output summary
                        newRow = [curExpt]
                        for ii in range(1, len(headerRow)): # skip first column since that is the exptName
                            if isNumeric(firstRealRow[ii]):
                                tempF = np.nanmean([float(exptRows[jj][ii]) for jj in range(len(exptRows))])
                                newRow.append(str.format(fS, tempF))
                            else:
                                newRow.append("NaN")
                        writer.writerow(newRow)
                    exptRows = [row]
                    curExpt = row[0]
            # now do last expt
            newRow = [curExpt]
            for ii in range(1, len(headerRow)): # skip first column since that is the exptName
                if isNumeric(firstRealRow[ii]):
                    tempF = np.nanmean([float(exptRows[jj][ii]) for jj in range(len(exptRows))])
                    newRow.append(str.format(fS, tempF))
                else:
                    newRow.append("NaN")
            writer.writerow(newRow)

def mergeCSVfiles(keyValue, inFile1, inFile2):
    a = pd.read_csv(inFile1)
    b = pd.read_csv(inFile2)
    merged = a.merge(b, on=keyValue)
    outputFileName = "mergedOutput.csv"
    merged.to_csv(outputFileName, index=False)
    return outputFileName

def isNumeric(inStr):
    try:
        return float(inStr)
    except:
        return None

def averageTraces(traceList):
    # does point by point average of all traces and deals with NaNs
    bufferOut = traceList[0] * 0.
    for oneTrace in traceList:
        bufferOut += oneTrace
    bufferOut = bufferOut / float(len(traceList))
    return bufferOut

def subtractBaseline(inTrace, pointsPerMs, nullTimeRange, stimTimes):
    # inTrace is a 1D vector and nullTimeRange is a 2 value tuple
    if nullTimeRange[0] >= 0:
        startIndex = nullTimeRange[0] * pointsPerMs
        stopIndex = (nullTimeRange[1] * pointsPerMs) - 1
    else:
        # negative times in nullTimeIndex mean relative times from first stim time
        startIndex = (stimTimes[0] + nullTimeRange[0]) * pointsPerMs
        stopIndex = (stimTimes[0] + nullTimeRange[1]) * pointsPerMs
    baselineToSubtract = np.mean(inTrace[startIndex:stopIndex])
    return inTrace - baselineToSubtract

def getEpisodeNumFromFilename(fileName):
    parts = fileName.split(".")
    epiStr = parts[-2]
    return int(epiStr[1:]) # skip E part of string
    
def blankTrace(inTrace, pointsPerMs, stimTimes, blankingExtent, deltaStimMarker = 1.):
    startIndex = int(pointsPerMs * blankingExtent[0]) # typically -0.2 ms
    stopIndex = int(pointsPerMs * blankingExtent[1]) # typically 1.6 ms
    if len(stimTimes) > 0:
        for oneStimTime in stimTimes:
            stimIndex = int(pointsPerMs * oneStimTime)
            replacementValue = inTrace[-1 + startIndex + stimIndex] # one sample before blankingExtent
            for i in range(startIndex + stimIndex, stimIndex):
                inTrace[i] = replacementValue
            inTrace[stimIndex] = replacementValue - deltaStimMarker # to mark actual stim time
            for i in range(stimIndex + 1, stimIndex + stopIndex):
                inTrace[i] = replacementValue
    return inTrace

