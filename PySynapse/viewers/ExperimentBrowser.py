#!/usr/bin/python
# -*- coding: utf-8 -*-

# revised 7 July 2015 BWS

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pyqtgraph.Qt import QtGui, QtCore
from PyQt4.QtGui import QListWidget, QListWidgetItem, QAbstractItemView
import sys, os, glob, re, os.path as path
import pyperclip
import FileIO.ProcessEpisodes as PE

class clsExperimentBrowser(QtGui.QMainWindow):
    def __init__(self, tempSynapseInstance):
        super(clsExperimentBrowser, self).__init__()
        self.curParser = None
        self.callingSynapseInstance = tempSynapseInstance
        self.initUI()

    def initUI(self):
        self.lView = QListWidget()
        self.lView.setSelectionMode(QAbstractItemView.SingleSelection)
        #self.lView.setGeometry(50, 150, 200, 350)
        self.lView.itemSelectionChanged.connect(self.newLVselection)
        self.mw = QtGui.QMainWindow()
        self.mw.setGeometry(20, 50, 500, 200)
        self.mw.setCentralWidget(self.lView)
        self.mw.setWindowTitle("Experiment List")
        self.mw.show()

    def setupExpts(self, tempParser, newTitleText):
        self.mw.setWindowTitle(newTitleText)      
        self.curParser = tempParser
        self.lView.clear()
        for oneExpt in PE.getListOfNonexcludedExpts(self.curParser):
            item = QListWidgetItem()
            episodes = PE.episodeList(self.curParser, oneExpt)
            if episodes:
                (tipText, fileExt) = path.splitext(path.basename(episodes[0]))
                tempStr = oneExpt + " (" + str(len(episodes)) + " epi)"
                tempStr += "  " + self.curParser.get(oneExpt, "comment")
                item.setText(tempStr)
                item.setWhatsThis(oneExpt)
                item.setToolTip(tipText)
                self.lView.addItem(item)
            else:
                print("Did not find any episodes in " + oneExpt)

    def newLVselection(self):
        curExptName = str(self.lView.selectedItems()[0].whatsThis()) # need string do deal with unicode mess
        episodesInExpt = PE.episodeList(self.curParser, curExptName)
        tempScopeCmd = PE.getOneValue(self.curParser, curExptName, "scopeCmd")
        tempEventPath = PE.getOneValue(self.curParser, curExptName, "eventListPath")
        tempEventProcCmd = PE.getOneValue(self.curParser, curExptName, "eventProcCmd")
        realStimTimes = None
        tempStimTimes = PE.getOneValue(self.curParser, curExptName, "stimTimes")
        if tempStimTimes:
            if "," in tempStimTimes:
                parts = tempStimTimes.strip().split(",")
            else:
                parts = tempStimTimes.strip().split(" ")
            if len(parts) > 0:
                realStimTimes = []
                for oneStimTime in parts:
                    realStimTimes.append(float(oneStimTime))
        self.callingSynapseInstance.doOpenFileList(episodesInExpt, curExptName, newScopeCmd=tempScopeCmd, newStimTimes=realStimTimes, newEventPath = tempEventPath, newEventProcCmd = tempEventProcCmd, exptName=curExptName)
