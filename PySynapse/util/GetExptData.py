#!/usr/bin/python
# -*- coding: utf-8 -*-

# revised 24 July 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os
import os.path as path
import shutil as shutil
import csv as csv
import numpy as np
import scipy.io as sio
import FileIO.ProcessEpisodes as PE 
from FileIO.CoreDatFileIO import readDatFile
import Util.CommonFunctions as Common

def getExptData(INIfileRoot, exptName):
    falseStrings = ["none", "off", "false", "no", "0"]
    outputFolder = None
    INIfileRoot = Common.testINIfile(INIfileRoot)
    if not INIfileRoot:
        return None
    parserObj = PE.readEpisodeList(INIfileRoot) 
    if exptName.lower() == "all":
        exptNameList = PE.getListOfNonexcludedExpts(parserObj)
    else:
        exptNameList = [exptName]
    for oneExptName in exptNameList:
        allItems = PE.getAllItems(parserObj, oneExptName)
        if not allItems:
            print("no data for requested section: " + oneExptName)
            return None
        epiFileNames = PE.episodeList(parserObj, oneExptName)
        firstEpisode = readDatFile(epiFileNames[0], readTraceData=False)
        if not firstEpisode:
            print("Could not access data file in expt section: " + epiFileNames[0])
            return None
        outDict = {}
        epiRootList = []
        for oneFileName in epiFileNames:
            epiPath, epiFileRoot = path.split(oneFileName)
            epiRootList.append(epiFileRoot)
        outDict["epiFilePath"] = epiPath
        outDict["epiFileNames"] = epiRootList
        outDict["msPerPoint"] = firstEpisode["msPerPoint"]
        outDict["pointsPerMs"] = 1. / firstEpisode["msPerPoint"]
        exportList = PE.getOneValue(parserObj, oneExptName, "export")
        if exportList:
            exportChannels = PE.getOneValue(parserObj, oneExptName, "channels").split(",")
            if not exportChannels:
                print("You need to indicate specific traces as: channels = xx, yy")
                return None
            for oneExportItem in exportList.split("|"):
                subparts = oneExportItem.strip().split(" ")
                cmd = subparts[0].lower()
                if cmd in ["events", "onsettime", "onsettimes"]:
                    eventListPath = PE.getOneValue(parserObj, oneExptName, "eventListPath")
                    if not eventListPath:
                        print("Cannot generate event data for selected expts because eventListPath = key was not indicated")
                        return None
                    eventsDict = {} # a Dict to hold all the subDicts from each episode
                    for oneEpi in epiFileNames:
                        print("getting event data from " + path.split(oneEpi)[1])
                        epiEvents = {} # a Dict to hold just event times from this one episode with channels as keys
                        for oneChan in exportChannels:
                            events = PE.readEventTimes(oneEpi, oneChan.strip(), eventListPath)
                            if events:
                                epiEvents[oneChan.strip()] = events
                        dots = findOccurences(oneEpi, ".")
                        dictKey = oneEpi[dots[-2] + 1:dots[-1]] # text between second-to-last and last dot is E15 etc
                        if len(epiEvents) > 0:
                            eventsDict[dictKey] = epiEvents
                        else:
                            print("No events found in file: " + path.split(oneEpi)[1])
                    if len(eventsDict) > 0:
                        outDict["events"] = eventsDict
                    else:
                        print("Looked for event csv files but found no events")
                if cmd in ["trace", "traces"]:
                    tracesDict = {}
                    for oneEpi in epiFileNames:
                        print("getting trace data from " + path.split(oneEpi)[1])
                        epiTraces = {}
                        tempEpi = readDatFile(oneEpi)
                        for oneChan in exportChannels:
                            epiTraces[oneChan.strip()] = tempEpi["traces"][oneChan.strip()]
                        dots = findOccurences(oneEpi, ".")
                        dictKey = oneEpi[dots[-2] + 1:dots[-1]]
                        if len(epiTraces) > 0:
                            tracesDict[dictKey] = epiTraces
                    if len(tracesDict) > 0:
                        outDict["traces"] = tracesDict
                    else:
                        print("Looked for trace data but did not find for selected channels: " + str(exportChannels))
        for key,value in allItems:
            outDict[key] = value
        saveMAT = PE.getOneValue(parserObj, oneExptName, "saveMAT")
        if saveMAT:
            if saveMAT not in falseStrings:
                if not outputFolder:
                    outputFolder = Common.getCleanTempFolder()
                outputFileName = outputFolder + "/" + oneExptName + ".mat"
                sio.savemat(outputFileName, outDict)
                print("getExptData exported data to: " + outputFileName)
    return outDict

def findOccurences(s, ch):
    return [i for i, letter in enumerate(s) if letter == ch]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("You need to supply both the INI file name and the expt/section name")
    else:
        getExptData(sys.argv[1], sys.argv[2])