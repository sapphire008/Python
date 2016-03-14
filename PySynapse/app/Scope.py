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

from pdb import set_trace

import numpy as np
import pandas as pd

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

from PyQt4 import QtCore, QtGui

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

# sys.path.append('..')
from util.ImportData import NeuroData

# Global variables
__version__ = "Scope Window 0.2"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd154','#17becf'] # tableau10, or odd of tableau20


class ScopeWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, maxepisodes=10):
        super(ScopeWindow, self).__init__(parent)
        self.episodes = None
        self.index = []
        # set a limit on how many episodes to memorize
        self.maxepisodes = maxepisodes
        # Record state of the scope window
        self.isclosed = True
        # This keeps track of the indices of which episodes are loaded
        self._loaded_array = []
        # Default options
        self.options = {'colorfy':False,\
        'layout':[['Voltage', 'A', 0, 0], ['Current', 'A', 1, 0]]} # layout = [Stream, Channel, row, col]
        # Keep track of which colors have been used
        self._usedColors = []
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
        exportWithScaleBarAction.triggered.connect(self.printme)
        exportMenu.addAction(exportWithScaleBarAction)

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

    def printme(self):
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

        index_insert = list(set(index) - set(self.index))
        index_remove = list(set(self.index) - set(index))

        if bool_old_episode and not index_insert and not index_remove: # same episode, same index
            return
        elif not bool_old_episode: # new item / cell
            index_insert = index
            index_remove = []
            self.episodes = episodes
            self.episodes['Data'] = [[]] * len(self.episodes['Dirs'])
            self._loaded_array = []

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

    def drawEpisode(self, zData, info=None, pen=None):
        """Draw plot from 1 zData"""
        # Set up pen color
        if self.options['colorfy']:
            availableColors = list(colors)
            for c in self._usedColors:
                availableColors.remove(c)
            pen = availableColors[0]
            self._usedColors.append(pen)
        elif pen is None:
            pen = self.options['theme']['pen']

        # Loop through all the subplots
        for n, l in enumerate(self.options['layout']):
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

        for l in self.options['layout']:
            # get viewbox
            p1 = self.graphicsView.getItem(row=l[2], col=l[3])
            pname = info[0]+'.'+info[1]+'.'+l[0]+'.'+l[1]

            remove_index = []
            for k, a in enumerate(p1.listDataItems()):
                if a.name() == pname: # matching
                    p1.removeItem(a)
                    remove_index.append(k)

        # recover the colors
        if remove_index and self.options['colorfy']:
            for r in remove_index:
                del self._usedColors[r]

    # ----------------------- Layout utilities --------------------------------
    def setLayout(self):
        return


    # ----------------------- Option utilities ----------------------------------
    def toggleTraceColors(self, checked):
        """Change traces from black to color coded"""
        # if already painted in color, paint in default pen again
        if not checked:
            self.options['colorfy'] = False
            self._usedColors = [] # reset used colors
        else:
            self.options['colorfy'] = True
        
        for l in self.options['layout']:
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            for k, a in enumerate(p.listDataItems()):
                if not checked:
                    pen = self.options['theme']['pen']
                else:
                    pen = colors[k%len(colors)]
                    if pen not in self._usedColors:
                        self._usedColors.append(pen)
                pen = pg.mkPen(pen)
                a.setPen(pen)
        
    def setDisplayTheme(self, theme='whiteboard'):
        self.options['theme'] = {'blackboard':{'background':'k', 'pen':'w'}, \
                 'whiteboard':{'background':'w', 'pen':'k'}\
                }.get(theme)

        self.graphicsView.setBackground(self.options['theme']['background'])
        # self.graphicsView.setForegroundBrush
        # change color / format of all objects

run_example = False


if __name__ == '__main__' and not run_example:
    episodes = {'Duration': [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 90000, 90000, 90000, 50000, 50000], 'Name': 'Neocortex F.08Feb16', 'Drug Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '3.4 sec', '2:03', '3:40', '5:37', '8:29'], 'Drug Level': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], 'Comment': ['', 'DAC0: PulseB 200', 'DAC0: PulseB -50', 'DAC0: PulseB -75', 'DAC0: PulseB -50', 'DAC0: PulseB 50', 'DAC0: PulseB 100', 'DAC0: PulseB 150', 'DAC0: PulseB 200', 'DAC0: PulseB 200', '', '', '', '', 'DAC0: PulseB 200', 'DAC0: PulseB 200'], 'Dirs': ['D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E1.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E2.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E3.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E4.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E5.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E6.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E7.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E8.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E9.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E10.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E11.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E12.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E13.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E14.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E15.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E16.dat'], 'Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '6:43', '8:42', '10:19', '12:16', '15:08'], 'Drug Name': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''], 'Epi': ['S1.E1', 'S1.E2', 'S1.E3', 'S1.E4', 'S1.E5', 'S1.E6', 'S1.E7', 'S1.E8', 'S1.E9', 'S1.E10', 'S1.E11', 'S1.E12', 'S1.E13', 'S1.E14', 'S1.E15', 'S1.E16'], 'Sampling Rate': [0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001]}
    index = [1]
    app = QtGui.QApplication(sys.argv)
    w = ScopeWindow(episodes=episodes, index=index)
    w.show()
    # Connect upon closing
    # app.aboutToQuit.connect(restartpyshell)
    # Make sure the app stays on the screen
    sys.exit(app.exec_())

if run_example:
    import pyqtgraph.examples
    pyqtgraph.examples.run()
