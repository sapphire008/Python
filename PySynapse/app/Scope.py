# -*- coding: utf-8 -*-
"""
Created: Tue Mar 8 05:12:21 2016

Form implementation generated from reading ui file 'Scope.ui'

      by: PyQt4 UI code generator 4.11.4

WARNING! All changes made in this file will be lost!

Scope window

@author: Edward
"""
import sys
import os
import collections

from pdb import set_trace

import numpy as np
import pandas as pd

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget


try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

sys.path.append('D:/Edward/Documents/Assignments/Scripts/Python/PySynapse')
sys.path.append('D:/Edward/Docuemnts/Assignments/Scripts/Python/generic')
from util.ImportData import NeuroData
from util.ExportData import *

# Global variables
__version__ = "Scope Window 0.2"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd154','#17becf'] # tableau10, or odd of tableau20


# Custom helper functions
def roundto125(x, r=np.array([1,2,5,10])): # helper static function
        """5ms, 10ms, 20ms, 50ms, 100ms, 200ms, 500ms, 1s, 2s, 5s, etc.
        5mV, 10mV, 20mV, etc.
        5pA, 10pA, 20pA, 50pA, etc."""
        p = int(np.floor(np.log10(x))) # power of 10
        y = r[(np.abs(r-x/(10**p))).argmin()] # find closest value
        return(y*(10**p))
        
def scalebarlabel(x, unitstr):
    x = int(x)
    if unitstr.lower()[0] == 'm':
        return(str(x)+unitstr if x<1000 else str(int(x/1000))+
            unitstr.replace('m',''))
    elif unitstr.lower()[0] == 'p':
        return(str(x)+unitstr if x<1000 else str(int(x/1000))+
            unitstr.replace('p','n'))

class ScopeWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, maxepisodes=10):
        super(ScopeWindow, self).__init__(parent)
        self.episodes = None
        self.index = []
        # set a limit on how many episodes to cache
        self.maxepisodes = maxepisodes
        # Record state of the scope window
        self.isclosed = True
        # This keeps track of the indices of which episodes are loaded
        self._loaded_array = []
        # Check if the user decided to keep traces from another cell
        self.keepOther = False
        # if use color for traaces
        self.colorfy = False
        # layout = [channel, stream, row, col]
        self.layout = [['Voltage', 'A', 0, 0], ['Current', 'A', 1, 0]]
        # range of axis
        self.viewMode = 'default'
        # Keep track of which colors have been used
        self._usedColors = []
        # Track if scalebar is turned on or not
        self.has_scalebar=False
        # Set up the GUI window
        self.setupUi(self)
        self.setDisplayTheme()

    def setupUi(self, MainWindow):
        """This function is converted from the .ui file from the designer"""
        MainWindow.setObjectName(_fromUtf8("Scope Window"))
        MainWindow.resize(1200, 600)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))

        # Graphics layout
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))
        self.graphicsLayout = QtGui.QHBoxLayout()
        self.graphicsLayout.setObjectName(_fromUtf8("graphicsLayout"))
        self.graphicsView = GraphicsLayoutWidget(self.centralwidget)
        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        self.graphicsLayout.addWidget(self.graphicsView)
        self.horizontalLayout.addLayout(self.graphicsLayout)

        # Side panel layout: initialize as a list view
        self.sideDockPanel = QtGui.QDockWidget("Settings", self)
        self.sideDockPanel.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.sideDockPanel.setObjectName(_fromUtf8("sideDockPanel"))
        self.sideDockPanel.hide()
        # self.sidePanelLayout = QtGui.QHBoxLayout()
        # self.sidePanelLayout.setObjectName(_fromUtf8("sidePanelLayout"))
        self.listView = QtGui.QListView(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listView.sizePolicy().hasHeightForWidth())
        self.listView.setSizePolicy(sizePolicy)
        self.listView.setObjectName(_fromUtf8("listView"))
        self.sideDockPanel.setWidget(self.listView)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.sideDockPanel)
        # self.sidePanelLayout.addWidget(self.listView)
        # self.horizontalLayout.addLayout(self.sidePanelLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1225, 26))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.setMenuBarItems()
        MainWindow.setMenuBar(self.menubar)

        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ---------------- Additional main window behaviors -----------------------
    def setMenuBarItems(self):
        # File Menu
        fileMenu = self.menubar.addMenu('&File')
        # File: Export
        exportMenu = fileMenu.addMenu('&Export')
        exportWithScaleBarAction = QtGui.QAction(QtGui.QIcon('export.png'), 'Export with scalebar', self)
        exportWithScaleBarAction.setShortcut('Ctrl+Alt+E')
        exportWithScaleBarAction.setStatusTip('Export with scalebar')
        exportWithScaleBarAction.triggered.connect(lambda: self.exportWithScalebar(arrangement='overlap'))
        exportMenu.addAction(exportWithScaleBarAction)
        
        exportVerticalAction = QtGui.QAction(QtGui.QIcon('export.png'), 'Export vertical arrangement', self)
        exportVerticalAction.setStatusTip('Export the selected episodes in a vertical arrangement')
        exportVerticalAction.triggered.connect(lambda: self.exportWithScalebar(arrangement='vertical'))
        exportMenu.addAction(exportVerticalAction)
                
        exportVerticalAction = QtGui.QAction(QtGui.QIcon('export.png'), 'Export horizontal arrangement', self)
        exportVerticalAction.setStatusTip('Export the selected episodes in horizontal arrangement. Good for concatenated episodes')
        exportVerticalAction.triggered.connect(lambda: self.exportWithScalebar(arrangement='horizontal'))
        exportMenu.addAction(exportVerticalAction)

        # View Menu
        viewMenu = self.menubar.addMenu('&View')
        # View: show settings
        viewMenu.addAction(self.sideDockPanel.toggleViewAction())
        # View: Colorfy
        colorfyAction = QtGui.QAction('Color code traces', self, checkable=True, checked=False)
        colorfyAction.setShortcut('Ctrl+Alt+C')
        colorfyAction.setStatusTip('Toggle between color coded traces and black traces')
        colorfyAction.triggered.connect(lambda: self.toggleTraceColors(colorfyAction.isChecked()))
        viewMenu.addAction(colorfyAction)
        # View: Keep previous
        keepPrev = QtGui.QAction('Keep previous', self, checkable=True, checked=False)
        keepPrev.setStatusTip('Keep traces from other data set on the scope window')
        keepPrev.triggered.connect(lambda: self.toggleKeepPrev(keepPrev.isChecked()))
        viewMenu.addAction(keepPrev)

    def printme(self): # for debugging
        print('doing stuff')

    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True

    def retranslateUi(self, MainWindow):
        """Set window title and other miscellaneous"""
        MainWindow.setWindowTitle(_translate(__version__, __version__, None))

    # ------------- Episode plotting utilities --------------------------------
    def updateEpisodes(self, episodes=None, index=[]):
        """First compare episodes with self.episodes and index with self.index
        Only update the difference in the two sets"""
        if not isinstance(episodes, dict) or not isinstance(self.episodes, dict):
            bool_old_episode = False
        else:
            bool_old_episode = self.episodes['Name'] == episodes['Name']
            
        # reset the grpahicsview if user not keeping traces from older dataset
        if not self.keepOther and not bool_old_episode:
            self.graphicsView.clear()
            self._usedColors = []
            self._loaded_array = []
            

        index_insert = list(set(index) - set(self.index))
        index_remove = list(set(self.index) - set(index))

        if bool_old_episode and not index_insert and not index_remove: # same episode, same index
            return
        elif not bool_old_episode: # new item / cell
            index_insert = index
            index_remove = []
            self.episodes = episodes
            self.episodes['Data'] = [[]] * len(self.episodes['Dirs'])

        # update index
        self.index += index_insert
        for a in index_remove:
            self.index.remove(a)

        # Insert new episodes
        for i in index_insert:
            self.episodes['Data'][i] = NeuroData(dataFile=self.episodes['Dirs'][i], old=True, infoOnly=False, getTime=True)
            self._loaded_array.append(i)
            # call self.drawPlot
            self.drawEpisode(self.episodes['Data'][i], info=(self.episodes['Name'], self.episodes['Epi'][i]))

        # Remove episodes
        for j in index_remove:
            self.removeEpisode(info=(self.episodes['Name'], self.episodes['Epi'][j]))
            
        self.setDataViewRange()

    def drawEpisode(self, zData, info=None, pen=None):
        """Draw plot from 1 zData"""
        # Set up pen color
        if self.colorfy:
            availableColors = list(colors)
            for c in self._usedColors:
                availableColors.remove(c)
            pen = availableColors[0]
            self._usedColors.append(pen)
        elif pen is None:
            pen = self.theme['pen']

        # Loop through all the subplots
        for n, l in enumerate(self.layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            if p is None:
                p = self.graphicsView.addPlot(row=l[2], col=l[3])
                # Make sure later viewboxes are linked in time domain
                if n>0:
                    p.setXLink(self.graphicsView.getItem(row=0, col=0))

            # put an identifier on the trace
            if isinstance(info, tuple):
                pname = info[0]+'.'+info[1]+'.'+l[0]+'.'+l[1]
            else:
                pname = None

            p.plot(x=zData.Time, y=getattr(zData, l[0])[l[1]], pen=pen, name=pname)


    def removeEpisode(self, info=None):
        if not info:
            return

        for l in self.layout:
            # get viewbox
            p1 = self.graphicsView.getItem(row=l[2], col=l[3])
            pname = info[0]+'.'+info[1]+'.'+l[0]+'.'+l[1]

            remove_index = []
            for k, a in enumerate(p1.listDataItems()):
                if a.name() == pname: # matching
                    p1.removeItem(a)
                    remove_index.append(k)

        # recover the colors
        if remove_index and self.colorfy:
            for r in remove_index:
                del self._usedColors[r]

    # ----------------------- Layout utilities --------------------------------
    def setLayout(self):
        return
        
    def drawScaleBar(self, vb, xrange, yrange, xunit, yunit, color='k',\
                         linewidth=None, fontsize=None):
        if self.has_scalebar:
            return
        # Span of axes
        X = np.ptp(xrange)
        Y = np.ptp(yrange)
        # calculate scale bar unit length
        X, Y = roundto125(X/5), roundto125(Y/5)
        # Parse scale bar labels
        xlab, ylab = scalebarlabel(X, xunit), scalebarlabel(Y, yunit)
         # Calculate positions of points in the scale bar
        xi = np.max(xrange) + X/2.0
        yi = np.mean(yrange)
        positions = [[xi,yi], [xi+X, yi], [xi+X, yi+Y]]
        # calculate position of text
        xtext1, ytext1 = xi+X/2.0, yi-Y/10.0 # horizontal
        xtext2, ytext2 = xi+X+X/10.0, yi+Y/2.0 # vertical
        # Draw the scalebar line
        scalebar_line = pg.PolyLineROI(positions=positions, closed=False, pen=pg.mkPen(color), movable=False, removable=True)
        [p.setVisible(False) for p in scalebar_line.getHandles()]
        vb.addItem(scalebar_line)
        # Add the text
        htext = pg.TextItem(xlab, color=color, anchor=(0.5,0.4))
        htext.setPos(xtext1, ytext1)
        vb.addItem(htext)
        vtext = pg.TextItem(ylab, color=color, anchor=(0.6,0.5))
        vtext.setPos(xtext2, ytext2)
        vb.addItem(vtext)
        
    # ----------------------- Option utilities ----------------------------------
    def toggleTraceColors(self, checked):
        """Change traces from black to color coded"""
        # if already painted in color, paint in default pen again
        if not checked:
            self.colorfy = False
            self._usedColors = [] # reset used colors
        else:
            self.colorfy = True
        
        for l in self.layout:
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            for k, a in enumerate(p.listDataItems()):
                if not checked:
                    pen = self.theme['pen']
                else:
                    pen = colors[k%len(colors)]
                    if pen not in self._usedColors:
                        self._usedColors.append(pen)
                pen = pg.mkPen(pen)
                a.setPen(pen)
                
    def toggleKeepPrev(self, checked):
        if checked:
            self.keepOther = True
        else:
            self.keepOther = False

        
    def setDisplayTheme(self, theme='whiteboard'):
        self.theme = {'blackboard':{'background':'k', 'pen':'w'}, \
                 'whiteboard':{'background':'w', 'pen':'k'}\
                }.get(theme)

        self.graphicsView.setBackground(self.theme['background'])
        # self.graphicsView.setForegroundBrush
        # change color / format of all objects
        
    def setDataViewRange(self, viewMode=None, xRange=None, yRange=None):
        # print('view range %s'%self.viewMode)
        if self.viewMode == viewMode:
            return
        self.viewMode = viewMode if viewMode is not None else self.viewMode
        self.viewRange = collections.OrderedDict()
        # Loop through all the subplots
        for n, l in enumerate(self.layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            #print(a.tickValues())
            if self.viewMode == 'default':
                yRange = {'Voltage':(-100, 40), 'Current': (-500, 500), 
                          'Stimulus':(-500, 500)}.get(l[0])
                p.setRange(yRange=yRange, padding=0)
            elif self.viewMode == 'auto':
                p.autoRange()
            elif self.viewMode == 'manual':
                return # not implemented
            else:
                raise(TypeError('Unrecognized view mode'))
                    
    def exportWithScalebar(self, arrangement='overlap', savedir="R:/tmp.svg"):
        viewRange = collections.OrderedDict()
        channels = []
        for n, l in enumerate(self.layout):
            if (l[0],l[1]) not in viewRange.keys():
                # get viewbox
                p = self.graphicsView.getItem(row=l[2], col=l[3])
                viewRange[(l[0],l[1])] = p.viewRange()
            if l[1] not in channels:
                # getting list of channels displayed
                channels.append(l[1])
#        # draw scalebar
#        for n, l in enumerate(self.layout):
#            p = self.graphicsView.getItem(row=l[2], col=l[3])
#            self.drawScaleBar(p.getViewBox(), viewRange[(l[0],l[1])][0], viewRange[(l[0],l[1])][1],xunit='ms',yunit='mV' if l[0]=='Voltage' else 'pA')
#        self.has_scalebar=True
#        return
        # Make strings for exporting
        self.episodes['Notes'] = [[]] * len(self.episodes['Dirs'])
        self.episodes['InitVal'] = [[]] * len(self.episodes['Dirs'])
        notestr = "{} Initial: {} WCTime: {} min"
        channeldescripstr = "Channel {} {:0.1f} mV {:0.0f} pA"
        
        for i in self.index: # iterate over episodes
            channelstr = []
            if self.episodes['Notes'][i]:
                continue # skip if notes already existed
            # zData = NeuroData(dataFile=self.episodes['Dir'][i], old=True, infoOnly=True)
            # iterate over all the streams
            for c in channels:
                initVolt = self.episodes['Data'][i].Voltage[c][0]
                initCur = self.episodes['Data'][i].Current[c][0]
                try:
                    initStim = self.episodes['Data'][i].Stimulus[c][0]
                except:
                    initStim = 0 
                self.episodes['InitVal'][i]={('Voltage', c):'{:0.0f}mV'.format(initVolt), ('Current',c):'{:0.0f}pA'.format(initCur), ('Stimulus',c):'{:0.0f}pA'.format(initStim)}
                channelstr.append(channeldescripstr.format(c,initVolt, initCur))

            channelstr = " ".join(channelstr)               
            self.episodes['Notes'][i] = notestr.format(self.episodes['Dirs'][i], channelstr, 
                                                       self.episodes['Data'][i].Protocol.WCtimeStr)
        
        # Do the plotting once all the necessary materials are gathered
        if arrangement == 'overlap':
            PlotTraces(self.episodes, self.index, viewRange, saveDir=savedir, colorfy=self._usedColors)
        elif arrangement == 'vertical':
            PlotTracesVertically(self.episodes, self.index, viewRange, saveDir=savedir, colorfy=self._usedColors)
        elif arrangement == 'horizontal':
            PlotTracesHorizontally(self.episodes, self.index, viewRange, saveDir=savedir, colorfy=self._usedColors)
        else:
            raise(TypeError('Unrecognized export arragement'))
        

run_example = False


if __name__ == '__main__' and not run_example:
    episodes = {'Duration': [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 90000, 90000, 90000, 50000, 50000], 'Name': 'Neocortex F.08Feb16', 'Drug Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '3.4 sec', '2:03', '3:40', '5:37', '8:29'], 'Drug Level': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'Comment': ['', 'DAC0: PulseB 200', 'DAC0: PulseB -50', 'DAC0: PulseB -75', 'DAC0: PulseB -50', 'DAC0: PulseB 50', 'DAC0: PulseB 100', 'DAC0: PulseB 150', 'DAC0: PulseB 200', 'DAC0: PulseB 200', '', '', '', '', 'DAC0: PulseB 200', 'DAC0: PulseB 200'], 'Dirs': ['D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E1.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E2.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E3.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E4.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E5.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E6.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E7.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E8.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E9.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E10.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E11.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E12.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E13.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E14.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E15.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E16.dat'], 'Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '6:43', '8:42', '10:19', '12:16', '15:08'], 'Drug Name': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'Epi': ['S1.E1', 'S1.E2', 'S1.E3', 'S1.E4', 'S1.E5', 'S1.E6', 'S1.E7', 'S1.E8', 'S1.E9', 'S1.E10', 'S1.E11', 'S1.E12', 'S1.E13', 'S1.E14', 'S1.E15', 'S1.E16'], 'Sampling Rate': [0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001]}
    index = [2]
    app = QtGui.QApplication(sys.argv)
    w = ScopeWindow()
    w.updateEpisodes(episodes=episodes, index=index)
    w.show()
    # Connect upon closing
    # app.aboutToQuit.connect(restartpyshell)
    # Make sure the app stays on the screen
    sys.exit(app.exec_())

if run_example:
    import pyqtgraph.examples
    pyqtgraph.examples.run()
