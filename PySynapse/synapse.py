#!/usr/bin/python
# -*- coding: utf-8 -*-
"""Main synapse episode browser window.

This is the main program in the Synapse package and the one that gets called by itself to start the system. This 
program also calls child forms (eg, ScopeWin.py or ExperimentBrowser.py). The calling signature for these child
forms includes a reference to the parent form (the current Synapse.py instance) so that information can be passed 
back to Synapse. This is needed primarily because ExperimentBrowser.py learns about display settings from reading
the INI file structure and needs to be able to tell each ScopeWin.py instance about those settings (eg, which 
channels to display and default vertical gain settings.)

last revised 20 July 2015 BWS

"""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from pyqtgraph.Qt import QtGui, QtCore
from PyQt4.QtGui import QListWidget, QListWidgetItem, QAbstractItemView
import pyqtgraph as pg
import sys, os, glob, re, os.path as path
import pyperclip
import FileIO.ProcessEpisodes as PE
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FileIO.CoreDatFileIO import readDatFile
from Viewers.ScopeWin import clsScopeWin
from Viewers.ExperimentBrowser import clsExperimentBrowser

class clsSynapse(QtGui.QMainWindow):
    def __init__(self):
        super(clsSynapse, self).__init__()
        self.versionNum = 1.141
        self.initUI()
        self.selectAllEpisodes = False  
        self.curEpiList = []
        self.curScopeCmd = None
        self.curStimTimes = None
        self.curEventPath = None
        self.curEventProcCmd = None
        self.curExptListFile = None
        self.curExptName = None

    def initToolbar(self):
        self.tb = self.addToolBar("Command")
        self.openAction = QtGui.QAction("OpenFolder", self)
        self.openAction.triggered.connect(self.doOpenFolder)
        self.scopeAction = QtGui.QAction("ScopeWin", self)
        self.scopeAction.triggered.connect(self.doNewScopeWin)
        self.imageAction = QtGui.QAction("ImageWin", self)
        self.imageAction.triggered.connect(self.doNewImageWin)
        self.loadExptAction = QtGui.QAction("ExptList", self)
        self.loadExptAction.triggered.connect(self.doLoadExpt)
        self.endAction = QtGui.QAction("End", self)
        self.endAction.triggered.connect(self.doEnd)
        self.tb.addAction(self.openAction)
        self.tb.addAction(self.scopeAction)
        #self.tb.addAction(self.imageAction)
        self.tb.addAction(self.loadExptAction)
        self.tb.addAction(self.endAction)
   
    def initMenuBar(self):
        mb = self.menuBar()
        mnFile = mb.addMenu("File")
        mnEdit = mb.addMenu("Edit")
        mnOptions = mb.addMenu("Options")
       
        self.toggleSelectAllAction = QtGui.QAction("toggleSelectAll", self)
        mnOptions.addAction(self.toggleSelectAllAction)
        self.toggleSelectAllAction.triggered.connect(self.toggleSelectAll)
        mnView = mb.addMenu("View")

    def initUI(self):
        macRamDrive = 'diskutil erasevolume HFS+ "RamDrive" `hdiutil attach -nomount ram://262967`'
        self.lView = QListWidget()
        self.lView.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.lView.setGeometry(5, 5, 400, 200)
        self.setCentralWidget(self.lView)
        
        self.initToolbar()
        self.initMenuBar()
        self.sb = self.statusBar()

        self.setGeometry(10, 60, 550, 300)
        #self.setGeometry(20, 290, 550, 300)
        pVer = str(sys.version_info[0]) + "." + str(sys.version_info[1])
        self.setWindowTitle("Synapse ver " + str(self.versionNum) + " python ver " + pVer)
        self.lView.itemSelectionChanged.connect(self.newLVselection)
        self.exptBrowser = None
        self.scopeWindows = []
        self.imageWindows = []

    def toggleSelectAll(self):
        self.selectAllEpisodes = not self.selectAllEpisodes

    def newLVselection(self):
        curItems = self.lView.selectedItems()
        if curItems:
            tempDataPath = path.dirname(str(curItems[0].whatsThis()))
            self.curEpiList = []
            self.curEpiNames = []
            for oneItem in curItems:
                self.curEpiList.append(readDatFile(str(oneItem.whatsThis())))
                self.curEpiNames.append(str(oneItem.whatsThis()))
            #pyperclip.copy(self.curEpiNames)  # copy all current file names with paths to clipboard
            # fix xx list works on some computers and fails on home iMac
            if len(self.scopeWindows) == 0:
                self.doNewScopeWin()
            tempExptDesc = None
            if self.curExptName and self.curExptListFile:
                tempExptDesc = self.curExptListFile + " | " + self.curExptName
            for oneWindow in self.scopeWindows:
                oneWindow.setExternalCommands(self.curScopeCmd, self.curStimTimes, self.curEventPath, self.curEventProcCmd)
                oneWindow.plotEpisode(self.curEpiList, datafolder=tempDataPath, exptListFile=tempExptDesc )
            if self.curScopeCmd:
                self.curScopeCmd = None # clear commands to pass to scopeWindows after first episode is selected
                
    def messageToSynapse(self, inMsg):
        print("Message: " + inMsg)

    def doOpenFolder(self):
        self.curScopeCmd = "initializeSettings | dw full"
        self.curExptListFile = None # clear flag used when reading episode lists from INI expt descriptions
        if os.name == "posix":
            fPath = QtGui.QFileDialog.getExistingDirectory(self, "Find data folder", "/Volumes/BenMacFiles/Lab Data")
            allFiles = self.naturalSort(glob.glob(str(fPath) + "/*.dat"))
            self.doOpenFileList(allFiles, fPath)
        else:
            fPath = QtGui.QFileDialog.getExistingDirectory(self, "Find data folder", "D:/")
            allFiles = self.naturalSort(glob.glob(str(fPath) + "/*.dat"))
            self.doOpenFileList(allFiles, fPath)
            #fDrive, temp = path.splitdrive(selFile)
            #fPath, fRoot = path.split(temp)
            #fPath = fDrive + fPath

    def doOpenFileList(self, fileNameList, newTitle, newScopeCmd=None, newStimTimes=None, newEventPath=None, newEventProcCmd=None, exptName=None):
        self.curScopeCmd = newScopeCmd
        self.curStimTimes = newStimTimes
        self.curEventPath = newEventPath
        self.curEventProcCmd = newEventProcCmd 
        self.curExptName = exptName
        self.lView.clear()
        firstEpisode = True
        for oneFile in fileNameList:
            tPath, tName = path.split(oneFile)
            tRoot, tExt = path.splitext(tName)
            item = QListWidgetItem()
            mEpi = readDatFile(str(oneFile), readTraceData=False)
            if mEpi:
                if firstEpisode:
                    self.setWindowTitle(newTitle + " (" + mEpi["clampModeStr"] + " type " + mEpi["fileVersion"] + ")")
                epiDesc = tRoot + "  " + mEpi["cellTimeStr"] + "  " + str(int(mEpi["sweepWindow"])) + " ms " + "  " 
                epiDesc += mEpi["stimDesc"] 
                if int(mEpi["drugLevel"]) > 0:
                	epiDesc += "  " + mEpi["drugLevelStr"] + "  " + mEpi["drugTimeStr"]
                item.setText(epiDesc)
                item.setWhatsThis(oneFile)
                item.setToolTip("this is a tool tip")
                self.lView.addItem(item)
        if self.selectAllEpisodes:
            for i in range(self.lView.count()):
                self.lView.item(i).setSelected(True)
        else:
            self.lView.item(0).setSelected(True)

    def naturalSort(self, l):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def doNewScopeWin(self):
        self.scopeWindows.append(clsScopeWin(self, 1 + len(self.scopeWindows)))

    def doNewImageWin(self):
        # self.imageWindows.append(clsImageWin())
        pass

    def doLoadExpt(self):
        if os.name == "posix":
            startingDir = "/Volumes/General/Dropbox/Work/Files"
        else:
            startingDir = "D:/"
        fN = QtGui.QFileDialog.getOpenFileName(self, "Find expt file", startingDir)
        if path.isfile(fN):
            self.curExptListFile = fN
            par = PE.readEpisodeList(fN)
            self.exptBrowser = clsExperimentBrowser(self)
            (fileRoot, fileExt) = path.splitext(path.basename(str(fN)))
            self.exptBrowser.setupExpts(par, path.basename(fileRoot))
        else:
            print("Expt list is not a file")
            self.curExptListFile = None

    def doEnd(self):
        sys.exit()


def main():
    app = QtGui.QApplication(sys.argv)
    ns = clsSynapse()
    ns.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
