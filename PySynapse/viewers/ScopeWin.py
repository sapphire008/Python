# -*- coding: utf-8 -*-
"""Scope window for viewing electrophysiology traces.

This module represents the core of the Synapse sytem. It defines the clsScopeWin class that can
be used repeatedly to generate multiple viewing windows. Each window is self-contained and so can 
display the same data on differnet time or vertical scales.

last revised 17 July 2015 BWS

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import csv as csv
import os.path as path
import os
from operator import itemgetter
from itertools import groupby
from FileIO.CoreDatFileIO import readDatFile
import FileIO.ProcessEpisodes as PE
import pyperclip as clip

class clsScopeWin(QtGui.QMainWindow):
    """The core scope window class 

    The class is initiated with two variables: a reference to the main Synapse.py window and an integer 
    value that is the number assigned to this particular scope window. Typically the first window 
    that the Synapse program creates is assigned a winNum of 1. The clsScopeWin class needs the reference
    to the calling Synapse.py instance because some functions executed within the scope window need to 
    pass information back to parent program.

    """
    def __init__(self, tempSyanpseInstance, winNum):
        super(clsScopeWin, self).__init__()
        self.callingSynapseInstance = tempSyanpseInstance
        self.thisScopeWinNumber = winNum
        self.initUI(winNum)

    # core plotting functions
    
    def plotEpisode(self, epiList, datafolder=None, exptListFile=None):
        # can be either called with None or with a list of episodeObjects
        if datafolder is not None:
            # used to transfer where the data files were just loaded from
            if "//" in datafolder:
                self.curDatafolder = datafolder.replace("//", "/") 
            else:
                self.curDatafolder = datafolder
        if exptListFile:
            # used to transfer what expt desc INI file was used, if any
            if "//" in exptListFile:
                self.curExptListFile = exptListFile.replace("//", "/")
            else:
                self.curExptListFile = exptListFile
        if epiList:
            self.mEpiList = epiList # make a local copy of the reference to list
            if self.debugMode:
                print("starting plotEpisode function with epiList specified")
        else:
            if self.debugMode:
                print("starting plotEpisode function with None entered")
        if self.debugMode and self.verticalScaleMode:
            print("Starting vertScale: " + str(self.verticalScaleMode[0]))
        # setup global vars to describe data and plot conditions
        self.maxTime = self.mEpiList[0]["sweepWindow"]
        numPoints = self.mEpiList[0]["numPoints"]
        msPerPoint = self.mEpiList[0]["msPerPoint"]
        tracesSimilar = True
        for oneEpi in self.mEpiList:
            if oneEpi["sweepWindow"] != self.maxTime:
                tracesSimilar = False
            if oneEpi["msPerPoint"] != msPerPoint:
                tracesSimilar = False
        if not tracesSimilar:
            print("Can not plot a cluster of episodes with different time bases")
            return None
        currentTracesOkay = True
        if self.traceNames:
            for oneCurrentDisplayedChan in self.traceNames:
                if oneCurrentDisplayedChan not in self.mEpiList[0]["chanNames"]:
                    currentTracesOkay = False
        if not currentTracesOkay:
            self.traceNames = None # reset traceNames if new epi list contains different chan names
            self.resetTimeAxis("Full")
            self.intializeMainSettings()
        if not self.traceNames:
            self.updateTraceList(self.guessChannelsToDisplay(self.mEpiList[0]))
        if self.externalScopeCmd:
            self.processGenericScopeCommand(self.externalScopeCmd)
        (tempFileRoot, fileExt) = path.splitext(path.basename(self.mEpiList[0]["loadedFileName"]))
        if len(self.mEpiList) > 1:
            allEpiNums = []
            for ii in range(len(self.mEpiList)):
                allEpiNums.append(PE.getEpisodeNumFromFilename(self.mEpiList[ii]["loadedFileName"]))
            sAllEpiNums = sorted(allEpiNums)
            pos = (jj - kk for kk, jj in enumerate(sAllEpiNums))
            t = 0
            eIndex = tempFileRoot.rfind('E', 0, len(tempFileRoot)-1)
            outList = tempFileRoot[0:eIndex+1]
            for kk, els in groupby(pos):
                l = len(list(els))
                el = sAllEpiNums[t]
                t += l
                if l > 1:
                    outList += str(el) + "-" + str((el + l) - 1) + ", "
                else:
                    outList += str(el) + ", "
            outList = outList[0:-2]
            tempFileRoot = outList
        self.fullFileRoot = tempFileRoot.strip()
        # fileRoot is for window title while fullFileRoot is for export information
        if len(self.mEpiList) > 20:
            self.fileRoot = tempFileRoot + " (+" + str(len(self.mEpiList) - 1) + " episodes)"
        else:
            self.fileRoot = tempFileRoot
        self.pointsPerMs = 1./msPerPoint
        time = np.linspace(0., numPoints*msPerPoint, numPoints, endpoint=False)
        if self.curTimeRange is None:
            self.curTimeRange = (0, self.maxTime)
        if self.blankingEnable:
            tempBlankingExtent = self.blankingExtent
        else:
            tempBlankingExtent = None

        # do actual plot
        self.lay = pg.GraphicsLayout()
        self.view.setCentralWidget(self.lay)
        self.pList = [];
        self.mLines = []
        self.secLines = []
        verticalIndex = 0
        for chanName in self.traceNames:
            # the next line returns a python list of numpy vectors
            self.processedTracesContainAverage = False # overwrite below if containing averages
            startIndex = int(self.curTimeRange[0] * self.pointsPerMs)
            stopIndex = int(self.curTimeRange[1] * self.pointsPerMs)
            tempTraces = PE.getTracesFromEpiList(self.mEpiList, chanName, self.verticalNullTimeRange, self.stimTimes, tempBlankingExtent, self.filterCmd)
            if self.showAverageMode == 1:
                # only show average
                tempTraces = [PE.averageTraces(tempTraces)] # make list containing one vector, the average of all traces of that chan type
                self.processedTracesContainAverage = True
            if self.showAverageMode == 2:
                tempTraces.append(PE.averageTraces(tempTraces))
                self.processedTracesContainAverage = True
            self.processedTracesDict[chanName] = tempTraces
            parts = self.verticalScaleMode[verticalIndex].split(" ")
            verticalCmd = parts[0].lower()
            newYrange = (-100., 100.)
            if verticalCmd in ["default"]:
                newYrange = self.getDefaultScale(chanName)
            if verticalCmd in ["freeze"]:
                newYrange = self.processedTracesVertScaleDict[chanName]
            if verticalCmd in ["auto", "autotrace", "autoscale"]:
                overallMin = np.amin(tempTraces[0][startIndex:stopIndex])
                overallMax = np.amax(tempTraces[0][startIndex:stopIndex])
                if len(tempTraces) > 1:
                    for oneTrace in tempTraces:
                        tempD = np.amin(oneTrace[startIndex:stopIndex])
                        if tempD < overallMin:
                            overallMin = tempD
                        tempD = np.amax(oneTrace[startIndex:stopIndex])
                        if tempD > overallMax:
                            overallMax = tempD
                newYrange = (overallMin, overallMax)
                if self.debugMode:
                    print("New autoscale: " + self.prettyPrintTuple((overallMin, overallMax)))
            if verticalCmd in ["fp", "floatpos", "floatpositive"]: # FloatPositive bias
                bottomRange = np.mean(tempTraces[0][startIndex + 0: startIndex + 9]) - (0.2 * float(parts[1]))
                topRange = bottomRange + float(parts[1])
                newYrange = (bottomRange, topRange)
            if verticalCmd in ["fn", "floatneg", "floatnegative"]: # FloatNegative bias
                bottomRange = np.mean(tempTraces[0][startIndex + 0: startIndex + 9]) - (0.8 * float(parts[1]))
                topRange = bottomRange + float(parts[1])
                newYrange = (bottomRange, topRange)
            if verticalCmd in ["fb", "floatbase"]: # FloatBase with no pos/neg bias
                bottomRange = np.mean(tempTraces[0][startIndex + 0: startIndex + 9]) - (0.5 * float(parts[1]))
                topRange = bottomRange + float(parts[1])
                newYrange = (bottomRange, topRange)
            if verticalCmd in ["man", "manual"]:
                newYrange = (float(parts[1]), float(parts[2]))
            self.processedTracesVertScaleDict[chanName] = newYrange # save vertical scale for each channel for exporting
            self.pList.append(self.lay.addPlot(verticalIndex, 0, title = chanName))  # create plot widget for this channel type
            firstOneOfChanType = True
            if verticalIndex == self.lastChanIndex:
                # at bottom most channel so allow time axis to be plotted on first plot of this channel type
                showTimeAxis = True
            else:
                showTimeAxis = False
            if self.showAverageMode == 0:
                # now look for events to display if requested and only displaying one trace
                if self.displayEventMarkers and len(tempTraces) == 1 and self.eventListFolder:
                    tempName = self.mEpiList[0]["loadedFileName"]
                    if self.enableEventProcessing:
                        eventTimes = PE.readEventTimes(tempName, chanName, self.eventListFolder, procCmd=self.eventProcCommand, stimTimes=self.stimTimes)
                    else:
                        eventTimes = PE.readEventTimes(tempName, chanName, self.eventListFolder)
                else:
                    eventTimes = None
                # just show traces
                for tempTrace in tempTraces:
                    self.displayOneTrace(time, tempTrace, newYrange, (0,0,0), firstOneOfChanType, verticalIndex, events=eventTimes)
                    if firstOneOfChanType:
                        firstOneOfChanType = False
            elif self.showAverageMode == 1:
                # just show average in blue assume list of traces in tempTrace has been replaced with a one-length list containing the average
                self.displayOneTrace(time, tempTraces[0], newYrange, (0,0,255), firstOneOfChanType, verticalIndex)
            elif self.showAverageMode == 2:
                # show traces and average and assume average trace is last item in list of trace vectors
                for i in range(len(tempTraces)):
                    if i < len(tempTraces) - 1:
                        self.displayOneTrace(time, tempTraces[i], newYrange, (0,0,0), firstOneOfChanType, verticalIndex)
                    else:
                        self.displayOneTrace(time, tempTraces[i], newYrange, (0,0,255), firstOneOfChanType, verticalIndex)
                if firstOneOfChanType:
                    firstOneOfChanType = False
            else:
                print("Unknown showAverageMode: " + str(self.showAverageMode))
            verticalIndex += 1
        self.pList[-1].showAxis("bottom") # to reveal time axis only on last plot in bottom channel
        self.proxy = pg.SignalProxy(self.pList[-1].scene().sigMouseMoved, rateLimit=60, slot=self.mouseMoved)
        if self.clipboardMode and len(self.mEpiList) > 0:
            if self.clipboardMode == "ini":
                self.dumpCurrentEpiInfoToClipboard()
            elif self.clipboardMode == "epi":
                clip.copy(self.mEpiList[0]["loadedFileName"])
            elif self.clipboardMode == "view":
                descStr = self.createVisibleDesc()
                clip.copy(descStr)
        return self.pList

    def displayOneTrace(self, time, trace, newYrange, penColor, firstOneOfChanType, curVerticalIndex, events=None):
        if penColor == (0,0,0):
            thisWidth = 2 # thin for regular black traces
        else:
            thisWidth = 20 # thicker for highlighted or average traces
        self.pList[-1].plot(time, trace, pen=penColor)
        if len(self.pList) > 1:
            self.pList[-1].setXLink(self.pList[0])
        self.pList[-1].hideAxis("bottom") # later change flag to enable only on last one on pList in caller function
        if self.timePadding:
            self.pList[-1].setRange(xRange = self.curTimeRange)
        else:
            self.pList[-1].setRange(xRange = self.curTimeRange, padding=0)
        self.pList[-1].setRange(yRange = newYrange)
        self.pList[-1].setMouseEnabled(x=True, y=False)
        if firstOneOfChanType:
            self.pList[-1].ctrlMenu = [self.scaleMenu, self.xMenu, self.optionMenu]
            self.secLines.append(pg.InfiniteLine(angle=90, pen=(255,255,255), movable=False, pos=0))
            self.mLines.append(pg.InfiniteLine(angle=90, pen=(0,0,0), movable=False, pos=0))
            self.pList[-1].addItem(self.secLines[curVerticalIndex], ignoreBounds=True)
            self.pList[-1].addItem(self.mLines[curVerticalIndex], ignoreBounds=True)
        if events:
            markerColor = (255,0,0) # red
            markerTop = trace[self.curTimeRange[0] * self.pointsPerMs] # 1.5 below trace value at beginning of display window
            eventY = None
            if self.eventMarkerMode.lower() == "cc":
                markerTop = markerTop - 1.5
                eventY = [markerTop, markerTop - 5.]
            elif self.eventMarkerMode.lower() == "vc":
                markerTop = markerTop + 25
                eventY = [markerTop, markerTop + 40.]
            else:
                print("Unknown eventMarkerMode: " + self.eventMarkerMode + " (should be cc or vc)")
            if eventY:
                for eventTime in events:
                    if eventTime > self.curTimeRange[0] and eventTime <= self.curTimeRange[1]:
                        eventX = [eventTime, eventTime]
                        self.pList[-1].plot(eventX, eventY, pen=markerColor)

    # helper functions for core plotting routines

    def guessChannelsToDisplay(self, mEpi):
        if mEpi["clampModeStr"] == "CC":
            chanList = ["VoltA", "CurA"]
        else:
            chanList = ["CurA", "VoltA"]
        return chanList

    def getCurPos(self, pos):
        for ii in range(len(self.traceNames)): # first group of pList entries should be one of each chan type
            if self.pList[ii].sceneBoundingRect().contains(pos):
                activeChanTemp = ii
                mousePoint = self.pList[self.lastChanIndex].vb.mapSceneToView(pos)
                if (mousePoint.x() >= 0.) & (mousePoint.x() < self.maxTime):
                    curPosTemp = mousePoint.x()
                    curIndexTemp = int(curPosTemp/self.mEpiList[0]["msPerPoint"])
                    return activeChanTemp, curPosTemp, curIndexTemp

    def updateTraceList(self, newChanList):
        self.traceNames = newChanList
        self.lastChanIndex = len(newChanList) - 1
        self.verticalScaleMode = []
        for ii in range(len(newChanList)):
            self.verticalScaleMode.append("default")
        
    def resetTimeAxis(self, inScaleMode, newDirectRange = None):
        scaleMode = inScaleMode.strip().lower()
        if scaleMode == "full":
            newRange = (0, self.maxTime)
        elif scaleMode == "betweencursors":
            newRange = (self.lastFixedPos, self.lastMovingPos)
        elif scaleMode == "centeroncursor":
            timeWindowMs = 1000.0
            startMs = round(self.lastMovingPos - (timeWindowMs/2.0))
            stopMs = startMs + timeWindowMs
            newRange = (startMs, stopMs)
        elif scaleMode == "offsetandduration":
            if newDirectRange[0] >= 0:
                newRange = (newDirectRange[0], newDirectRange[0] + newDirectRange[1])
            else:
                # negative offset time means relative to first stim time
                if self.stimTimes:
                    newRange = (self.stimTimes[0] + newDirectRange[0], self.stimTimes[0] + newDirectRange[0] + newDirectRange[1])
        elif scaleMode == "directentry":
            if newDirectRange:
                newRange = newDirectRange
            else:
                newText = "Enter min/max ms separated by a space"
                if self.curTimeRange:
                    defaultText = str(int(self.curTimeRange[0])) + " " + str(int(self.curTimeRange[1]))
                else:
                    defaultText = ""
                newCmd = self.msgBox("Time axis control", newText, defaultText)
                if newCmd:
                    parts = newCmd.split(" ")
                    if len(parts) == 2:
                        newRange = float(parts[0]), float(parts[1])
        elif scaleMode == "commandentry":
            newRange = newDirectRange
        elif scaleMode == "shiftright":
            increment = self.curTimeRange[1] - self.curTimeRange[0]
            newRange = (self.curTimeRange[0] + increment, self.curTimeRange[1] + increment)
        elif scaleMode == "shiftleft":
            increment = self.curTimeRange[1] - self.curTimeRange[0]
            newRange = (self.curTimeRange[0] - increment, self.curTimeRange[1] - increment)
        elif scaleMode == "plus":
            center = (self.curTimeRange[0] + self.curTimeRange[1]) / 2. # keep same center time value
            newDur = (self.curTimeRange[1] - self.curTimeRange[0]) * 2. # double window duration
            offset = center - (newDur / 2.)
            newRange = (offset, offset + newDur)
        elif scaleMode == "minus":
            center = (self.curTimeRange[0] + self.curTimeRange[1]) / 2. # keep same center time value
            newDur = (self.curTimeRange[1] - self.curTimeRange[0]) / 2. # half of the original window duration
            offset = center - (newDur / 2.)
            newRange = (offset, offset + newDur)
        else:
            print("Could not processed requested time mode: " + scaleMode)
            return None
        # now store new time range
        if newRange[0] < 0:
            newRange = (0, newRange[1])
        if newRange[1] > self.maxTime:
            newRange = (newRange[0], self.maxTime)
        self.curTimeRange = newRange
        if self.debugMode:
            print("New time range selected: " + self.prettyPrintTuple(self.curTimeRange))
        for oneTrace in self.pList:
            if self.timePadding:
                oneTrace.setRange(xRange = newRange)
            else:
                oneTrace.setRange(xRange = newRange, padding=0)
        if "auto" in self.verticalScaleMode:
            if self.debugMode:
                print("Calling plotEpisode to refresh new display time window setttings")
            self.plotEpisode(None)

    def prettyPrintTuple(self, inTuple):
        # returns a string with both input values truncated to one digit beyond the period
        tempStr = str(int(10. * inTuple[0]) / 10) + " to "
        tempStr += str(int(10. * inTuple[1]) / 10)
        return tempStr

    def getDefaultScale(self, chanName):
        firstChar = chanName[0].upper()
        if firstChar == "V":
            return -120., 40.
        elif firstChar == "C":
            return -500., 500.
        elif firstChar == "S":
            return -500., 500.
        else:
            return -100., 100.

    # Trace export functions for cvs and eps files below here

    def exportCurrentView(self):
        fN = "/Users/Ben/Dropbox/Code/Postscript/test3.eps"
        f = open(fN, "w")
        print("%!PS-Adobe-3.0 EPSF-3.0", file=f)
        print("%%Pages: 1", file=f)
        print("%%BoundingBox:   50   50   272   272", file=f)
        print("%%EndComments", file=f)
        print("/Helvetica findfont 6 scalefont setfont", file=f)
        print("/m /moveto load def", file=f)
        print("/l /lineto load def", file=f)
        print("0.567 setlinewidth", file=f)

        print("newpath", file=f)
        print("25 0 m", file=f)
        print("0 0 l", file=f)
        print("0 25 l", file=f)
        print("stroke", file=f)

        print("newpath", file=f)
        print("248 272 m", file=f)
        print("272 272 l", file=f)
        print("272 248 l", file=f)
        print("stroke", file=f)

        numChannels = 1
        tempTraceName = "VoltA"
        tempScaling = self.processedTracesVertScaleDict[tempTraceName]
        for tempTrace in self.processedTracesDict[tempTraceName]:
            self.helperExportOneTrace(tempTrace, 0, 1, tempScaling, f)
        f.close()
        if self.debugMode:
            print("finished export of current view to " + fN)

    def helperExportOneTrace(self, tempTrace, verticalIndex, numChannels, tempScaling, f):
        startIndex = int(self.pointsPerMs * self.curTimeRange[0])
        stopIndex = int(self.pointsPerMs * self.curTimeRange[1])
        incPoints = 272.0 / (stopIndex - startIndex)
        verticalScalingPoints = (272.0 / numChannels) / (tempScaling[1] - tempScaling[0]) # points per real unit
        verticalOffsetPoints = (272.0 / numChannels) * verticalIndex
        print("newpath", file=f)
        print("0 " +  str.format("{0:.6f}", (verticalOffsetPoints + ((tempTrace[startIndex] -  tempScaling[0]) * verticalScalingPoints))) + " m", file=f)
        for i in xrange(startIndex, stopIndex):
            print(str.format("{0:.6f}", (i-startIndex) * incPoints) + " " +  str.format("{0:.6f}", (verticalOffsetPoints + ((tempTrace[i] - tempScaling[0]) * verticalScalingPoints))) + " l", file=f)
        print("stroke", file=f)

    def dumpTraceValues(self, verticalIndex, timeRangeIn):
        timeRange = sorted(timeRangeIn)
        thisChan = self.traceNames[verticalIndex]
        tempTraces = PE.getTracesFromEpiList(self.mEpiList, thisChan, self.verticalNullTimeRange, self.stimTimes, self.blankTimeList, self.filterCmd)
        startIndex = int(timeRange[0] * self.pointsPerMs)
        stopIndex = int(timeRange[1] * self.pointsPerMs)
        for i in range(len(self.mEpiList)):
            thisFileName = path.basename(self.mEpiList[i]["loadedFileName"])
            thisTrace = tempTraces[i]
            print(thisFileName + " = " + str(np.mean(thisTrace[startIndex:stopIndex])))

    def dumpCurActiveChanAsCSV(self, verticalIndex, timeRangeIn):
        # only sends first trace (or average trace) to CSV file
        timeRange = sorted(timeRangeIn)
        thisChan = self.traceNames[verticalIndex]
        tempTraces = PE.getTracesFromEpiList(self.mEpiList, thisChan, self.verticalNullTimeRange, self.stimTimes, self.blankTimeList, self.filterCmd)
        if self.showAverageMode == 1:
            tempTraces = [PE.averageTraces(tempTraces)]
        startIndex = int(timeRange[0] * self.pointsPerMs)
        stopIndex = int(timeRange[1] * self.pointsPerMs)
        msPerPoint = 1. / self.pointsPerMs
        tempFile = self.exportPath + "/" + self.traceNames[verticalIndex] + ".csv"
        with open(tempFile, "w") as fCSV:
            writer = csv.writer(fCSV) # assume the first line is the header info
            writer.writerow(["Time", self.traceNames[verticalIndex]])
            for ii in range(stopIndex - startIndex):
                index = ii + startIndex
                writer.writerow(["{:.1f}".format(index * msPerPoint), "{:.3f}".format(tempTraces[0][index])])
        if self.debugMode:
            tempStr = "saved selected part of " + self.traceNames[verticalIndex] 
            if self.showAverageMode == 1:
                tempStr += " (average)"
            print(tempStr + " as " + tempFile)

    def exportVisibleDesc(self, descFileRoot):
        # creates a text file that describes how to recreate the traces in the current scope window display
        fN = self.exportPath + "/" + descFileRoot + ".txt"
        f = open(fN, "w")
        print(self.createVisibleDesc, file=f)
        f.close()
        if self.debugMode:
            print("finished export of current view to " + fN)

    def createVisibleDesc(self):
        tempStr = "CurTimeRange: " + self.prettyPrintTuple(self.curTimeRange) + "\n"
        if self.verticalNullTimeRange:
            tempStr += "BaselineNullRange: " + self.prettyPrintTuple(self.verticalNullTimeRange) + "\n"
        else:
            tempStr += "BaselineNullRange:" + "\n"
        tempStr += "ShowAverage: "
        if self.showAverageMode == 0:
            tempStr += "Traces | 0"
        elif self.showAverageMode == 1:
            tempStr += "Average | 1"
        elif self.showAverageMode == 2:
            tempStr += "Traces and average | 2"
        else:
            print("Unknown value in showAverageMode")
            tempStr += " (unknown) | -1"
        tempStr += "\n"
        if self.eventListFolder:
            tempStr += "EventPath: " + self.eventListFolder + "\n"
        else:
            tempStr += "EventPath: " + "\n"
        tempStr += "DataPath: " + self.curDatafolder + "\n"
        tempStr += "FileRoot: " + self.fileRoot + "\n"
        if self.filterCmd:
            tempStr += "FilterCmd: " + self.filterCmd + "\n"
        else:
            tempStr += "FilterCmd: " + "\n"
        tempStr += "MsPerPoint: " + str(1./self.pointsPerMs) + "\n"
        tempStr += "SweepWindow: " + str(self.mEpiList[0]["sweepWindow"]) + "\n"
        tempStr += "Comment: " + "\n"
        if self.curExptListFile:
            tempStr += "ExptListFile: " + self.curExptListFile + "\n"
        else:
            tempStr += "ExptListFile: " + "\n"
        tempStr += "<ChannelList starting>" + "\n"
        for ii in range(len(self.traceNames)):
            tempStr += self.traceNames[ii] + " | scaling: " + self.verticalScaleMode[ii] + " | " 
            tempStr += self.prettyPrintTuple(self.processedTracesVertScaleDict[self.traceNames[ii]]) + " | "
            if ii == self.activeChan:
                tempStr += "Active" + "\n"
            else:
                tempStr += "NotActive" + "\n"
        tempStr += "<ChannelList end>" + "\n"
        tempStr += "<TraceList starting>" + "\n"
        for oneEpi in self.mEpiList:
            tempStr += path.basename(oneEpi["loadedFileName"]) + "\n"
        tempStr += "<TraceList end>"
        return tempStr 

    def dumpCurrentEpiInfoToClipboard(self):
        # first get current path from first episode
        filePath, fileName = path.split(self.mEpiList[0]["loadedFileName"])
        fileRoot, fileExt = path.splitext(fileName)
        dots = [i for i, ltr in enumerate(fileRoot) if ltr == "."]
        blockName = fileRoot[0:dots[1]] # to get rid of .S1.E21 stuff
        blockName = blockName.replace("Cell", "")
        blockName = "[" + blockName.replace(" ", "") + "]"
        epiListStr = self.fullFileRoot[dots[2]+2:]
        eIndex = fileRoot.find(".E")
        restPath, deepestFolder = path.split(filePath)
        restPath, nextDeepestFolder = path.split(restPath)
        tempStr = blockName + "\n"
        tempStr += "innerPath = " + nextDeepestFolder + "/" + deepestFolder + "\n"
        tempStr += "fileRoot = " + fileRoot[0:eIndex + 2] + "\n"
        if self.verticalNullTimeRange:
            nullStr = self.prettyPrintTuple(self.verticalNullTimeRange).replace(" ", "")
            nullStr = nullStr.replace("to", ", ")
            tempStr += "nullRange = " + nullStr + "\n"
        tempStr += self.clipboardINIblockKey + " = " + epiListStr + "\n"
        clip.copy(tempStr)


    # Menu and text command processing routines below here

    def setExternalCommands(self, newScopeCmd, newStimTimes, newEventPath, newEventProcCmd):
        # allows Synapse.py to pre-load commands to execute later when plotEpisode is run
        self.externalScopeCmd = newScopeCmd
        self.eventProcCommand = newEventProcCmd
        if self.enableRemoteCommands:
            self.stimTimes = newStimTimes
            if newEventPath:
                self.eventListFolder = newEventPath
            if self.debugMode:
                print("Got external commands: " + str(newScopeCmd) + " and stimTimes of " + str(newStimTimes))
                if newEventPath:
                    print(" and event path from external source: " + self.eventListFolder)

    def processRemoteCommand(self, newCmd):
        # allows conditional running of text commands based on self.enableRemoteCommands boolean flag
        if self.enableRemoteCommands:
            self.processGenericScopeCommand(newCmd)

    def processGenericScopeCommand(self, newCmd, forcePlotRefresh=False):
        # main routine for processing all text commands
        falseStrings = ["none", "off", "false", "no", "0"]
        parts = newCmd.strip().split("|")
        for oneCmd in parts:
            cleanCmd = oneCmd.strip()
            subparts = cleanCmd.split(" ")
            actualCommand = subparts[0].lower()
            textAfterCmd = cleanCmd[len(actualCommand):].strip()
            if actualCommand in ["displaywindow", "dw"]:
                if len(subparts) == 2 and subparts[1].lower() in ["full", "auto", "automatic"]:
                    self.resetTimeAxis("Full")
                elif len(subparts) == 4 and subparts[1].lower() in ["offset", "offsetdur", "offsetduration", "offsetandduration"]:
                    self.resetTimeAxis("OffsetAndDuration", (float(subparts[2]), float(subparts[3])))
                elif len(subparts) == 4 and subparts[1].lower() in ["command", "commmandentry", "direct"]:
                    self.resetTimeAxis("CommandEntry", (float(subparts[2]), float(subparts[3])))
                else:
                    print("Could not understand ResetTimeAxis mode: " + subparts[1])
            elif actualCommand in ["displayevent", "displayevents", "showevents", "loadevents", "showe"]:
                if subparts[1].lower() in falseStrings:
                    self.displayEventMarkers = False
                else:
                    self.displayEventMarkers = True
            elif actualCommand in ["clipboarddump", "clipboard", "clip", "cb"]:
                # possible arguments: None, epi, view, ini
                if subparts[1].lower() in falseStrings:
                    self.clipboardMode = None
                elif subparts[1].lower() == "help":
                    print("Clipboard possible arguments:")
                    print("  none = disable any coping to the clipboard")
                    print("  epi = copy current episode filename and path (only for first episode)")
                    print("  view = copy all displayed filenames and the view conditions")
                    print("  ini = copy current episode as section for INI expt listing file")
                else:
                    self.clipboardMode = subparts[1].lower()
            elif actualCommand in ["blockkey", "iniblockkey", "blockname", "iniblockname"]:
                # used to change what key is used to identify current traces in INI text dump to clipboard
                self.clipboardINIblockKey = subparts[1]
            elif actualCommand in ["seteventchan", "seteventchannel", "seteventchannels"]:
                self.eventChannels = []
                for ii in range(2, len(subparts)):
                    self.eventChannels.append(subparts[ii])
            elif actualCommand in ["enableremotecommands", "enableremote"]:
                if subparts[1].lower() in falseStrings:
                    self.enableRemoteCommands = False
                else:
                    self.enableRemoteCommands = True
            elif actualCommand in ["refresh", "refreshdisplay", "refreshwindow", "ref"]:
                self.plotEpisode(None)
            elif actualCommand in ["setdisplaychan", "setdisplaychannels", "setc", "setchan"]:
                if len(subparts) >= 2:
                    tempTraceNames = []
                    for oneName in subparts[1:]:
                        cleanName = oneName.strip()
                        if len(cleanName) > 0:
                            if cleanName in str(self.mEpiList[0]["chanNames"]):
                                tempTraceNames.append(cleanName)
                            else:
                                print("Requested channel is not available: " + cleanName)
                    if len(tempTraceNames) > 0:
                        self.updateTraceList(tempTraceNames)
            elif actualCommand in ["setverticalscale", "setvert", "setscale"]:
                if len(subparts) >= 2:
                    if self.debugMode:
                        print("New vertScale from command: " + textAfterCmd)
                    for i in range(len(self.verticalScaleMode)):
                        self.verticalScaleMode[i] = textAfterCmd.lower()
            elif actualCommand in ["setprojectpath", "setproject", "projectpath"]:
                if len(subparts) == 1:
                    fPath = QtGui.QFileDialog.getExistingDirectory(self, "Please select folder for analysis output files")
                    if fPath:
                        self.projectPath = fPath
                elif len(subparts) == 2:
                    if path.exists(textAfterCmd):
                        self.projectPath = textAfterCmd
                else:
                    print("Problem with setting projectPath")
            elif actualCommand in ["seteventpath", "setevent", "eventpath"]:
                if subparts[1].lower() in falseStrings:
                    self.eventListFolder = None
                    self.displayEventMarkers = False
                else:
                    tempPath = textAfterCmd
                    if path.exists(tempPath):
                        self.eventListFolder = tempPath
                        self.displayEventMarkers = True
                        if self.debugMode:
                            print("event file checking enabled")
                    else:
                        print("Selected path does not exist: " + tempPath)
            elif actualCommand in ["setexportpath", "setexport", "exportpath"]:
                if len(subparts) == 1:
                    fPath = QtGui.QFileDialog.getExistingDirectory(self, "Please select folder for exported files")
                    if fPath:
                        self.exportPath = fPath
                elif len(subparts) == 2:
                    if path.exists(textAfterCmd):
                        self.exportPath = textAfterCmd
                else:
                    print("Problem with setting exportPath")
            elif actualCommand in ["eventmode", "eventmarkermode"]:
                if len(subparts) == 2:
                    self.eventMarkerMode = subparts[1].lower()
            elif actualCommand in ["eventproc", "eventprocessing", "eventprocessingcommand"]:
                self.eventProcCommand = textAfterCmd
            elif actualCommand in ["enableeventproc", "enableeventprocesssing"]:
                if subparts[1].lower() in falseStrings:
                    self.enableEventProcessing = False
                else:
                    self.enableEventProcessing = True
            elif actualCommand in ["null", "nullrange" "setnull", "setverticalnull"]:
                if len(subparts) == 3:
                    firstValue = float(subparts[1])
                    secondValue = float(subparts[2])
                    if firstValue >= 0:
                        self.verticalNullTimeRange = (firstValue, secondValue)
                        if self.debugMode:
                            print("New pos null times: " + self.prettyPrintTuple(self.verticalNullTimeRange))
                    else:
                        # negative null times mean times relative to first stim time
                        if self.stimTimes:
                            self.verticalNullTimeRange = (self.stimTimes[0] + firstValue, self.stimTimes[0] + secondValue)
                            if self.debugMode:
                                print("New neg null times: " + self.prettyPrintTuple(self.verticalNullTimeRange))
                        else:
                            print("Cannot use relative (negative) VerticalNullTimeRange values without loading stimTimes first")
            elif actualCommand in ["dumpview", "dumpcurrentview", "describeview", "descview"]:
                if len(subparts) == 1:
                    self.exportVisibleDesc("currentDesc")
                elif len(subparts) == 2:
                    self.exportVisibleDesc(subparts[1])
                else:
                    print("Problem with argumens for dumpCurrentView command; requires either no arguments or just the fileRoot")
            elif actualCommand in ["clearcommands", "clearsettings", "initializesettings"]:
                self.intializeMainSettings()
            elif actualCommand in ["debug", "debugmode"]:
                if len(subparts) == 1:
                    self.debugMode = not self.debugMode
                if len(subparts) == 2:
                    if subparts[1] in falseStrings:
                        self.debugMode = False
                    else:
                        self.debugMode = True
                else:
                    print("Debug command requires either no arguments (to toggle state) or one argument: True or False")
            elif actualCommand in ["diag", "showdiag", "printdiag"]:
                self.printDiagInfo()
            elif actualCommand in ["help", "showhelp", "printhelp"]:
                self.printHelp()
            else:
                print("Unknown command: " + actualCommand)
                return None
            if forcePlotRefresh:
                self.plotEpisode(None)

    def doContextOptions(self, cmd):
        # for processing commands from context menu
        if cmd:
            activeChanName = self.traceNames[self.activeChan]
            if cmd == "ChangeChannels":
                currentDislayChan = ""
                for oneChan in self.traceNames:
                    currentDislayChan += oneChan + " " # one long string
                newCmd = self.msgBox("Enter command for scaling " + str(activeChanName), 
                    "Available channels" + str(self.mEpiList[0]["chanNames"]), currentDislayChan)
                if newCmd:
                    parts = newCmd.split(" ")
                    tempTraceNames = []
                    for oneName in parts:
                        tempTraceNames.append(oneName) # a list of multiple strings
                    self.updateTraceList(tempTraceNames)
                    self.plotEpisode(None)

    def popUpCommandVerticalScaling(self):
        # this routine is triggered by option in context menu
        activeChanName = self.traceNames[self.activeChan]
        newCmd = self.msgBox("Enter command for scaling " + str(activeChanName), "FB for FloatBase, FP for positive, FN for negative")
        if newCmd:
            self.verticalScaleMode[self.activeChan] = newCmd.lower()
            if self.debugMode:
                print("New command from VertScale pop-up: " + newCmd)
            self.plotEpisode(None)

    def processCommandVerticalScaling(self, newCmd):
        # called by fixed options in context menu (not pop-up or msgBox text entry from context menu)
        self.verticalScaleMode[self.activeChan] = newCmd.lower()
        self.plotEpisode(None)

    # GUI interactions with mouse and keyboard below here

    def mouseMoved(self, evt):
        try:
            self.activeChan, curPos, curIndex = self.getCurPos(evt[0])
            self.lastMovingPos = curPos
            self.lastMovingIndex = curIndex
        except:
            return
        if self.diffState == 0:
            tempTime = curPos
        else:
            tempTime = curPos - self.lastFixedPos
        self.mw.setWindowTitle(self.fileRoot + "  *  " + (str(int(tempTime * 10.)/10.0) + " ms"))
        for mLine in self.mLines:
            mLine.setPos(curPos)
        for iChan in range(len(self.traceNames)):
            thisChanName = self.traceNames[iChan]
            oneTrace = self.processedTracesDict[thisChanName][0]
            if self.diffState == 0:
                #thisValue = self.mEpiList[0]["traces"][thisChanName][curIndex] # fix xx
                thisValue = oneTrace[curIndex]
                tempStr =  "  " + str(int(thisValue * 10.)/ 10.)
                self.lastFixedPos = curPos
                self.lastFixedIndex = curIndex
            else:
                #thisValue = self.mEpiList[0]["traces"][thisChanName][curIndex] - self.mEpiList[0]["traces"][thisChanName][self.lastFixedIndex]
                oneTrace = self.processedTracesDict[thisChanName][0]
                thisValue = oneTrace[curIndex] - oneTrace[self.lastFixedIndex]
                tempStr =  "  " + str(int(thisValue * 100.)/ 100.)
            self.pList[iChan].setTitle(thisChanName + tempStr)

    def mouseClickEvent(self, evt):
        if evt.button() == QtCore.Qt.LeftButton:
            if self.diffState == 0:
                for oneLine in self.secLines:
                    oneLine.setPen(255,0,0)
                    oneLine.setPos(self.lastFixedPos)
                self.diffState = 1
                labelStyle = {"color": "#FF0000"}
                for iChan in range(len(self.traceNames)):
                    thisChanName = self.traceNames[iChan]
                    self.pList[iChan].setTitle(thisChanName, **labelStyle)
            else:
                for oneLine in self.secLines:
                    oneLine.setPen(255,255,255)
                    oneLine.setPos(0)
                self.diffState = 0
                labelStyle = {"color": "#000000"}
                for iChan in range(len(self.traceNames)):
                    if evt.pos():
                        thisChanName = self.traceNames[iChan]
                        activeChan, curPos, curIndex = self.getCurPos(evt.pos())
                        oneTrace = self.processedTracesDict[thisChanName][0]
                        thisValue = oneTrace[curIndex]
                        tempStr =  "  " + str(int(thisValue * 10.)/ 10.)
                        self.pList[iChan].setTitle(thisChanName + tempStr, **labelStyle)

    def keyPressEvent(self, evt):
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_V):
            self.popUpCommandVerticalScaling()
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_F):
            self.resetTimeAxis("Full")
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_B):
            self.resetTimeAxis("BetweenCursors")
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Right):
            self.resetTimeAxis("ShiftRight")
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Left):
            self.resetTimeAxis("ShiftLeft")
        if evt.key() == QtCore.Qt.Key_Equal:
            self.resetTimeAxis("Plus")
        if evt.key() == QtCore.Qt.Key_Minus:
            self.resetTimeAxis("Minus")
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_N):
            if self.verticalNullTimeRange:
                if self.debugMode:
                    print("No trace nulling")
                self.verticalNullTimeRange = None
            else:
                if self.lastMovingPos > self.lastFixedPos:
                    self.verticalNullTimeRange= (self.lastFixedPos, self.lastMovingPos)
                else:
                    self.verticalNullTimeRange = (self.lastMovingPos, self.lastFixedPos)
                if self.debugMode:
                    print("Trace nulling set for: " + self.prettyPrintTuple(self.verticalNullTimeRange))
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_1):
            if self.debugMode:
                print("using default vertical scaling for " + self.traceNames[self.activeChan])
            self.verticalScaleMode[self.activeChan] = "default"
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_2):
            if self.debugMode:
                print("using floatPos 20 scaling for " + self.traceNames[self.activeChan])
            self.verticalScaleMode[self.activeChan] = "fp 20"
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_3):
            if self.debugMode:
                print("using auto scaling for " + self.traceNames[self.activeChan])
            self.verticalScaleMode[self.activeChan] = "auto"
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_4):
            if self.debugMode:
                print("Freezing time scale for " + self.traceNames[self.activeChan])
            self.verticalScaleMode[self.activeChan] = "freeze"
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_A):
            self.showAverageMode += 1
            if self.showAverageMode == 3:
                self.showAverageMode = 0 # only allow modes of 0 1 and 2
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_D):
            self.dumpTraceValues(self.activeChan, (self.lastFixedPos, self.lastMovingPos)) 
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_C):
            self.dumpCurActiveChanAsCSV(self.activeChan, (self.lastFixedPos, self.lastMovingPos)) 
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Space):
            newCmd = self.msgBox("Enter generic command", "Selected: " + self.traceNames[self.activeChan] + 
                    "  Available channels: " + str(self.mEpiList[0]["chanNames"]))
            if newCmd:
                self.processGenericScopeCommand(newCmd, forcePlotRefresh=True)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_P):
            self.timePadding = not self.timePadding
            if self.debugMode:
                print("Time axis padding flag is now " + str(self.timePadding))
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_X):
            self.blankingEnable = not self.blankingEnable
            if self.debugMode:
                print("Stimulus blankingEnable flag is now " + str(self.blankingEnable))
            self.plotEpisode(None)
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_R):
            self.plotEpisode(None) # refresh display of current loaded episodes
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_E):
            # exports eps file based on current view without event markers
            self.exportCurrentView()
        if evt.key() == (QtCore.Qt.Key_Control and QtCore.Qt.Key_Z):
            # exports txt file that describes the current view
            self.exportVisibleDesc("currentDesc")

    def msgBox(self, windowTitle, labelText, defaultText = None):
        dlg = QtGui.QInputDialog(self)
        dlg.setInputMode(QtGui.QInputDialog.TextInput)
        dlg.setLabelText(labelText)
        dlg.setWindowTitle(windowTitle)
        if defaultText:
            dlg.setTextValue(defaultText)
        dlg.resize(500,100)
        ok = dlg.exec_()
        text = str(dlg.textValue())
        if ok & (len(text.strip()) > 0):
            return str(text.strip())
        else:
            return None

    def closeEvent(self, evt):
        if self.debugMode:
            print("scope is about to close")

    # Initializations routines below here

    def intializeMainSettings(self):
        self.externalScopeCmd = None
        self.blankingExtent = (-0.2, 0.8) # in ms relative to each stim time
        self.blankingEnable = False
        self.eventListFolder = None
        self.eventMarkerMode = "cc"
        self.blankTimeList = None
        self.displayEventMarkers = True
        self.timeScaleMode = "Auto"
        self.traceNames = None
        self.verticalScaleMode = None
        self.verticalNullTimeRange = None
        self.eventChannels = ["VoltA"]
        self.enableRemoteCommands = True
        self.eventProcCommand = None  # new on 9 July 2015
        self.enableEventProcessing = True  # new on 9 July 2015
        self.clipboardMode = "ini"
        self.clipboardINIblockKey = "episodes"
        self.timePadding = True
        self.showAverageMode = 0 # 0=onlyTraces 1=onlyAverage 2=averageAndTraces
        self.curDatafolder = None
        self.curExptListFile = None
        self.filterCmd = None

    def initUI(self, winNum):
        self.activeChan = -1
        self.curTimeRange = None
        self.processedTracesDict = {}
        self.processedTracesVertScaleDict = {}
        self.processedTracesContainAverage = False
        self.debugMode = True
        self.projectPath = None
        if path.exists("/Volumes/RamDrive"):
            self.exportPath = "/Volumes/RamDrive"
        elif path.exists("R:/"):
            self.exportPath = "R:/"
        else:
            self.exportPath = None
        self.stimTimes = None
        self.lastFixedIndex = -1
        self.lastFixedPos = -1
        self.lastMovingIndex = -1
        self.lastMovingPos = -1
        self.diffState = 0
        self.intializeMainSettings()
        pg.setConfigOptions(antialias=True)
        pg.setConfigOption("background", "w")
        pg.setConfigOption("foreground", "k")
        self.mw = QtGui.QMainWindow()
        self.mw.setWindowTitle("Scope window " + str(winNum))
        self.mw.mousePressEvent = self.mouseClickEvent
        self.mw.keyPressEvent = self.keyPressEvent
        self.view = pg.GraphicsView()
        self.mw.setCentralWidget(self.view)
        self.lay = pg.GraphicsLayout()
        self.view.setCentralItem(self.lay)
        self.mw.resize(1000, 700)

        self.scaleMenu = QtGui.QMenu("Y axis scaling")
        self.cmdScaleAction = QtGui.QAction("Enter command", self.mw)
        self.cmdScaleAction.triggered.connect(lambda: self.popUpCommandVerticalScaling())
        self.scaleMenu.addAction(self.cmdScaleAction)
        self.defaultScaleAction = QtGui.QAction("Default", self.mw)
        self.defaultScaleAction.triggered.connect(lambda: self.processCommandVerticalScaling("Default"))
        self.scaleMenu.addAction(self.defaultScaleAction)
        self.autoRegionAction = QtGui.QAction("Autoscale region", self.mw)
        self.autoRegionAction.triggered.connect(lambda: self.processCommandVerticalScaling("Auto"))
        self.scaleMenu.addAction(self.autoRegionAction)
        self.autoScaleAction = QtGui.QAction("Autoscale trace", self.mw)
        self.autoScaleAction.setChecked(True)
        self.autoScaleAction.triggered.connect(lambda: self.processCommandVerticalScaling("Auto"))
        self.scaleMenu.addAction(self.autoScaleAction)
    
        self.xMenu = QtGui.QMenu("Time axis")
        self.partialXaxisAction = QtGui.QAction("Display between cursors", self.mw)
        self.partialXaxisAction.triggered.connect(lambda: self.resetTimeAxis("BetweenCursors"))
        self.xMenu.addAction(self.partialXaxisAction)
        self.wholeXaxisAction = QtGui.QAction("Whole episode", self.mw)
        self.wholeXaxisAction.triggered.connect(lambda: self.resetTimeAxis("Full"))
        self.xMenu.addAction(self.wholeXaxisAction)
        self.enterXaxisAction = QtGui.QAction("Enter new time axis limits", self.mw)
        self.enterXaxisAction.triggered.connect(lambda: self.resetTimeAxis("DirectEntry"))
        self.xMenu.addAction(self.enterXaxisAction)

        self.optionMenu = QtGui.QMenu("Options")
        self.setTraceListAction = QtGui.QAction("Set channels to display", self.mw)
        self.setTraceListAction.triggered.connect(lambda: self.doContextOptions("ChangeChannels"))
        self.optionMenu.addAction(self.setTraceListAction)

        self.mw.show()
        self.lay.layout.setSpacing(0.)
        self.lay.setContentsMargins(0., 0., 0., 0.)
        
    # Help documentation and diagnostics functions

    def printHelp(self):
        print("One help command")

    def printDiagInfo(self):
        print("  *** Diagnostics listing ***")
        print("Display time range: " + self.prettyPrintTuple(self.curTimeRange))
        print("msPerPoint: " + str(self.mEpiList[0]["msPerPoint"]))
        print("Sweep window: " + str(self.mEpiList[0]["sweepWindow"]) + " ms")
        for ii in range(len(self.traceNames)):
            tempStr = self.traceNames[ii] + " scaling: " + self.verticalScaleMode[ii]
            if ii == self.activeChan:
                tempStr += " (currently selected)"
            print("Channel " + tempStr) 
        if self.verticalNullTimeRange:
            print("Baseline null from times: " + self.prettyPrintTuple(self.verticalNullTimeRange))
        else:
            print("Baseline subtract disabled.")
        if self.blankingEnable:
            print("Stimulus artifact blanking from times: " + self.prettyPrintTuple(self.blankingExtent))
        else:
            print("Stimulus blanking disabled.")
        print("Stim times: " + str(self.stimTimes))
        if self.projectPath:
            print("Project path: " + str(self.projectPath) + "  (" + str(path.exists(self.projectPath)) + ")")
        else:
            print("Project path: none specified")
        if self.eventListFolder:
            print("Event list path: " + self.eventListFolder + "  (" + str(path.exists(self.eventListFolder)) + ")")
        else:
            print("Event list path: none specified")
        print("Display event markers: " + str(self.displayEventMarkers))
        if self.eventProcCommand:
            print("Event processing cmd: "+ self.eventProcCommand)
        print("EnableEventProcessing: " + str(self.enableEventProcessing))
        if self.exportPath:
            print("Export path: " + str(self.exportPath) + "  (" + str(path.exists(self.exportPath)) + ")")
        else:
            print("Export path: none specified")
        print("Current data file folder: " + self.curDatafolder)
        print("Clipboard: " + str(self.clipboardMode))
        print("INI file block key: " + self.clipboardINIblockKey)
        print("Enabled remote commands: " + str(self.enableRemoteCommands))
        print("Debug mode: " + str(self.debugMode))

