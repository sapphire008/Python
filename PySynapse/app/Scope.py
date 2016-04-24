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
# from pyqtgraph.flowchart import Flowchart


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
from util.MATLAB import *
from util.spk_util import *
from app.AccordionWidget import AccordionWidget

# Global variables
__version__ = "Scope Window 0.3"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd154','#17becf'] # tableau10, or odd of tableau20

class SideDockPanel(QtGui.QWidget):
    """Collapsible dock widget that displays settings and analysis results for
    the Scope window
    """
    # Keep track of position of the widget added
    _widget_index = 0
    _sizehint = None

    def __init__(self, parent=None, friend=None):
        super(SideDockPanel, self).__init__(parent)
        self.parent = parent
        self.friend = friend
        self.setupUi()

    def setupUi(self):
        self.verticalLayout = self.parent.layout()
        # self.setLayout(self.verticalLayout)
        self.accWidget = AccordionWidget(self)

        # Add various sub-widgets, which interacts with Scope, a.k.a, friend
        self.accWidget.addItem("Channels", self.layoutWidget(), collapsed=True)
        self.accWidget.addItem("Analysis", self.analysisWidget(), collapsed=True)

        self.accWidget.setRolloutStyle(self.accWidget.Maya)
        self.accWidget.setSpacing(0) # More like Maya but I like some padding.
        self.verticalLayout.addWidget(self.accWidget)

    # ------- Layout control -------------------------------------------------
    def layoutWidget(self):
        """Setting layout of the graphicsview of the scope"""
        # Generate a list of available channels and streams
        all_layouts = self.friend.getAvailableStreams(warning=False)
        if not all_layouts: # if no data loaded
            return self.buildTextFrame(text="No Data Loaded")

        # Initialize the layout widget
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("LayoutWidgetFrame"))
        widgetFrame.layout().setSpacing(0)
        all_streams = sorted(set([l[0] for l in all_layouts]))
        all_streams = [s for s in ['Voltage', 'Current','Stimulus'] if s in all_streams]
        all_channels = sorted(set([l[1] for l in all_layouts]))
        # Layout setting table
        self.setLayoutTable(all_streams, all_channels)
        # Buttons for adding and removing channels and streams
        addButton = QtGui.QPushButton("Add") # Add a channel
        addButton.clicked.connect(lambda: self.addLayoutRow(all_streams=all_streams, all_channels=all_channels))
        removeButton = QtGui.QPushButton("Remove") # Remove a channel
        removeButton.clicked.connect(self.removeLayoutRow)
        # Add the exisiting channels and streams to the table
        widgetFrame.layout().addWidget(addButton, 1, 0)
        widgetFrame.layout().addWidget(removeButton, 1, 1)
        widgetFrame.layout().addWidget(self.layout_table, 2, 0, self.layout_table.rowCount(), 2)
        return widgetFrame

    def setLayoutTable(self, all_streams, all_channels):
        self.layout_table = QtGui.QTableWidget(0, 2) # (re)initialize
        self.layout_table.verticalHeader().setVisible(False)
        self.layout_table.horizontalHeader().setVisible(False)
        self.layout_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        for l in self.friend.layout: # current layout from scope
            self.addLayoutRow(all_streams=all_streams, all_channels=all_channels,\
                                current_stream=l[0], current_channel=l[1])

    def addLayoutRow(self, all_streams=['Voltage','Current','Stimulus'], \
                           all_channels=['A','B','C','D'], \
                           current_stream='Voltage', current_channel='A'):
        """Create a row of 2 combo boxes, one for stream, one for channel"""
        scomb = QtGui.QComboBox()
        scomb.addItems(all_streams)
        scomb.setCurrentIndex(all_streams.index(current_stream))
        ccomb = QtGui.QComboBox()
        ccomb.addItems(all_channels)
        ccomb.setCurrentIndex(all_channels.index(current_channel))
        row = self.layout_table.rowCount()
        self.layout_table.insertRow(row)
        self.layout_table.setCellWidget(row, 0, scomb) # Stream
        self.layout_table.setCellWidget(row, 1, ccomb) # Channel
        if row+1 > len(self.friend.layout): # update layout
            self.friend.addSubplot(layout=[current_stream, current_channel, row, 0])
        scomb.currentIndexChanged.connect(lambda: self.friend.updateStream(old_layout=['stream', 'channel', row, 0], new_layout=[str(scomb.currentText()), str(ccomb.currentText()), row, 0]))
        ccomb.currentIndexChanged.connect(lambda: self.friend.updateStream(old_layout=['stream', 'channel', row, 0], new_layout=[str(scomb.currentText()), str(ccomb.currentText()), row, 0]))

    def removeLayoutRow(self):
        row = self.layout_table.rowCount()-1
        if row < 1:
            return
        self.layout_table.removeRow(row)
        self.friend.removeSubplot(layout = self.friend.layout[-1])

    def buildTextFrame(self, text="Not Available"):
        """Simply displaying some text inside a frame"""
        someFrame = QtGui.QFrame(self)
        someFrame.setLayout(QtGui.QVBoxLayout())
        someFrame.setObjectName("Banner")
        labelarea = QtGui.QLabel(text)
        someFrame.layout().addWidget(labelarea)
        return someFrame

    # ------- Analysis tools -------------------------------------------------
    def analysisWidget(self):
        # Initalize the widget
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("AnalysisWidgetFrame"))
        widgetFrame.layout().setSpacing(10)
        # Detect spikes button
        detectSpikesButton = QtGui.QPushButton("Detect Spikes")
        # Detect PSPs button
        detectPSPsButton = QtGui.QPushButton("Detect PSP")
        detectPSPComboBox = QtGui.QComboBox()
        detectPSPComboBox.addItems(['EPSP', 'IPSP', 'EPSC', 'IPSC'])
        # Curve fitting
        curveFitButton = QtGui.QPushButton("Curve Fitting")
        curveFitComboBox = QtGui.QComboBox()
        curveFitComboBox.addItems(['Single Exponential', 'Double Exponential', 'Linear', 'Polynomial', 'Power', 'Custom'])
        return widgetFrame

    # ------- Other utilities ------------------------------------------------
    def replaceWidget(self, widget=None, index=0):
        old_widget = self.accWidget.takeAt(index)
        self.accWidget.addItem(title=old_widget.title(), widget=widget, collapsed=old_widget._collapsed, index=index)

    def sizeHint(self):
        """Helps with initial dock window size"""
        return QtCore.QSize(self.friend.frameGeometry().width()/6, 20)

class ScopeWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, maxepisodes=10, layout=None):
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
        self.layout =[['Voltage', 'A', 0, 0]] if not layout else layout# [['Voltage', 'A', 0, 0], ['Current', 'A', 1, 0], ['Stimulus', 'A', 1,0]]
        # range of axes
        self.viewMode = 'default'
        # view region
        self.viewRegionOn = False
        # self.linkViewRegion = True
        # Data tip
        self.dataCursorOn = False
        # self.linkCrossHair = True
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
        self.graphicsView.artists = [] # Auxillary graphics items
        self.graphicsLayout.addWidget(self.graphicsView)
        self.horizontalLayout.addLayout(self.graphicsLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        # Side panel layout: initialize as a list view
        self.dockWidget = QtGui.QDockWidget("Toolbox", self)
        self.dockWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dockWidget.setObjectName(_fromUtf8("dockwidget"))
        # self.dockWidget.hide() # keep the dock hidden by default
        # Dock content, containing widgets
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        # Layout of the dock content
        self.dockLayout = QtGui.QVBoxLayout(self.dockWidgetContents)
        self.dockLayout.setObjectName(_fromUtf8("dockLayout"))
        # listview
        # self.listView = QtGui.QListView(self.dockWidgetContents)
        self.dockPanel = SideDockPanel(self.dockWidgetContents, self)
        # self.listView.setObjectName(_fromUtf8("listView"))
        self.dockPanel.setObjectName(_fromUtf8("dockPanel"))
        # Add listview to layout
        # self.dockLayout.addWidget(self.listView)
        self.dockLayout.addWidget(self.dockPanel)
        self.dockWidget.setWidget(self.dockWidgetContents)
        # Add the dock to the MainWindow
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(2), self.dockWidget)

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
        exportWithScaleBarAction.setShortcut('Alt+E')
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
        # View: Default view range
        defaultViewAction = QtGui.QAction('Default Range', self)
        defaultViewAction.setShortcut('Alt+D')
        defaultViewAction.setStatusTip('Reset view to default range')
        defaultViewAction.triggered.connect(lambda: self.setDataViewRange(viewMode='default'))
        viewMenu.addAction(defaultViewAction)
        # View: Colorfy
        colorfyAction = QtGui.QAction('Color code traces', self, checkable=True, checked=False)
        colorfyAction.setShortcut('Alt+C')
        colorfyAction.setStatusTip('Toggle between color coded traces and black traces')
        colorfyAction.triggered.connect(lambda: self.toggleTraceColors(colorfyAction.isChecked()))
        viewMenu.addAction(colorfyAction)

        # View: view region
        viewRegionAction = QtGui.QAction('Region Selection', self, checkable=True, checked=False)
        viewRegionAction.setShortcut('Alt+R')
        viewRegionAction.setStatusTip('Show view region selection')
        viewRegionAction.triggered.connect(lambda: self.toggleRegionSelection(viewRegionAction.isChecked()))
        viewMenu.addAction(viewRegionAction)
        # View: cross hair
        dataCursorAction = QtGui.QAction('Data cursor', self, checkable=True, checked=False)
        dataCursorAction.setShortcut('Alt+T')
        dataCursorAction.setStatusTip('Show data cursor on the traces')
        dataCursorAction.isChecked()
        dataCursorAction.triggered.connect(lambda: self.toggleDataCursor(dataCursorAction.isChecked()))
        viewMenu.addAction(dataCursorAction)
        # View: Keep previous
        keepPrev = QtGui.QAction('Keep previous', self, checkable=True, checked=False)
        keepPrev.setStatusTip('Keep traces from other data set on the scope window')
        keepPrev.triggered.connect(lambda: self.toggleKeepPrev(keepPrev.isChecked()))
        viewMenu.addAction(keepPrev)
        # View: show toolbox
        viewMenu.addAction(self.dockWidget.toggleViewAction())

    def printme(self, msg='doing stuff'): # for debugging
        print(msg)

    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True

    def retranslateUi(self, MainWindow):
        """Set window title and other miscellaneous"""
        MainWindow.setWindowTitle(_translate(__version__, __version__, None))

    # ------------- Episode plotting utilities --------------------------------
    def updateEpisodes(self, episodes=None, index=[], updateLayout=False):
        """First compare episodes with self.episodes and index with self.index
        Only update the difference in the two sets. The update does not sort
        the index; i.e. it will be kept as the order of insert / click
        """
        if not isinstance(episodes, dict) or not isinstance(self.episodes, dict):
            bool_old_episode = False
        else:
            bool_old_episode = self.episodes['Name'] == episodes['Name']

        # reset the grpahicsview if user not keeping traces from older dataset
        if not self.keepOther and not bool_old_episode:
            self.graphicsView.clear()
            self._usedColors = []
            self._loaded_array = []
            self.index = []

        index_insert = list(set(index) - set(self.index))
        index_remove = list(set(self.index) - set(index))

        if bool_old_episode and not index_insert and not index_remove and not updateLayout: # same episode, same index
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
            if not self.episodes['Data'][i]: # load if not already loaded
                self.episodes['Data'][i] = NeuroData(dataFile=self.episodes['Dirs'][i], old=True, infoOnly=False, getTime=True)
            self._loaded_array.append(i)
            # call self.drawPlot
            self.drawEpisode(self.episodes['Data'][i], info=(self.episodes['Name'], self.episodes['Epi'][i]))

        # Remove episodes
        for j in index_remove:
            self.removeEpisode(info=(self.episodes['Name'], self.episodes['Epi'][j]))

        self.setDataViewRange()
        # print(self.index)
        if not bool_old_episode: # artists have been cleared. Add it back
            self.reissueArtists()

        # Update the companion, Toolbox: Layout
        if self.dockPanel.accWidget.widgetAt(0).objectName() == 'Banner':
            # Replace the banner widget with real widget
            layoutWidget = self.dockPanel.layoutWidget()
            self.dockPanel.replaceWidget(widget=layoutWidget, index=0)

    def drawEpisode(self, zData, info=None, pen=None, layout=None):
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

        layout = self.layout if not layout else layout
        # Loop through all the subplots
        for n, l in enumerate(layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            if p is None:
                p = self.graphicsView.addPlot(row=l[2], col=l[3])
                # Make sure later viewboxes are linked in time domain
                if l[2]>0 or l[3]>0:
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
    def setLayout(self, stream, channel, row, col, index=None):
    	l = [stream, channel, row, col]
    	if not index:
    		self.layout.append(l)
    	elif all(hasattr(index, attr) for attr in ['__add__', '__sub__', '__mul__', '__abs__', '__pow__']): # is numeric
    		if index>=len(self.layout):
    			self.layout.append(l)
    		else:
    			self.layout[index] = l

    def indexStreamChannel(self, stream, channel):
    	"""Return the index of given stream & channel pair in self.layout"""
    	index = [n for n, sc in enumerate(self.layout) if sc[0]==stream and sc[1]==channel]
    	if len(index)==1:
    		index = index[0]

    	return index

    def getAvailableStreams(self, warning=True):
        """Construct all available data streams"""
        # Assuming when the scope window is called, there is at least one data
        try:
            zData = self.episodes['Data'][self.index[0]]
        except:
            if warning:
                print('No data loaded')
            return None
        streams = []
        for s in ['Voltage', 'Current', 'Stimulus']:
            if not hasattr(zData, s):
                continue
            for c in getattr(zData, s).keys():
                streams.append([s, c])
        # sort by channel
        streams.sort(key=lambda x: x[1])

        return streams

    def addSubplot(self, layout, check_duplicates=False):
        """Append another data stream into the display"""
        if check_duplicates and layout in self.layout:
            return # Check for duplicates.
        self.layout.append(layout)
        # Sort data
        self.layout = sorted(self.layout, key=lambda x: (x[2], x[3]))
        # Plot this new data stream
        for n, i in enumerate(self.index):
            zData = self.episodes['Data'][i]
            self.drawEpisode(zData, info=(self.episodes['Name'], self.episodes['Epi'][i]), pen=self._usedColors[n] if self.colorfy else 'k', layout=[layout])

        # Set the proper view range
        self.setDataViewRange()
        # Redraw the artists
        self.reissueArtists()

    def removeSubplot(self, layout, exact_match=False):
        """Remove a data stream from the display"""
        if exact_match and layout not in self.layout: # nothing to remove
            return
        l = self.graphicsView.getItem(row=layout[2], col=layout[3])
        if not l:
            return # if item not found
        self.graphicsView.removeItem(l)
        if exact_match:
            self.layout.remove(layout)
        else:
            for lo in self.layout:
                if lo[2] == layout[2] and lo[3] == layout[3]:
                    self.layout.remove(lo)

    def updateStream(self, old_layout, new_layout):
        """Replace one stream with another stream"""
        self.removeSubplot(old_layout, exact_match=False)
        self.addSubplot(new_layout, check_duplicates=False)

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
        self.keepOther = checked

    def setDisplayTheme(self, theme='whiteboard'):
        self.theme = {'blackboard':{'background':'k', 'pen':'w'}, \
                 'whiteboard':{'background':'w', 'pen':'k'}\
                }.get(theme)

        self.graphicsView.setBackground(self.theme['background'])
        # self.graphicsView.setForegroundBrush
        # change color / format of all objects

    def setDataViewRange(self, viewMode=None, xRange=None, yRange=None):
        # print('view range %s'%self.viewMode)
        self.viewMode = viewMode if viewMode is not None else self.viewMode
        self.viewRange = collections.OrderedDict()
        # Loop through all the subplots
        for n, l in enumerate(self.layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            #print(a.tickValues())
            if self.viewMode == 'default':
                # Make everything visible first
                p.autoRange()
                yRange = {'Voltage':(-100, 40), 'Current': (-500, 500),
                          'Stimulus':(-500, 500)}.get(l[0])
                p.setYRange(yRange[0], yRange[1], padding=0)
            elif self.viewMode == 'auto':
                p.autoRange()
            elif self.viewMode == 'manual':
                return # not implemented
            else:
                raise(TypeError('Unrecognized view mode'))

    def reissueArtists(self):
        """In case artists are removed, toggle back on the artists"""
        if self.viewRegionOn: # remove, then redraw
            self.toggleRegionSelection(checked=False)
            self.toggleRegionSelection(checked=True, rng=self.selectedRange, rememberRange=True, cmd='add')
        # Add cursor if selection checked
        if self.dataCursorOn: # remove, then redraw
            self.toggleDataCursor(checked=False)
            self.toggleDataCursor(checked=True)

    def toggleRegionSelection(self, checked, plotitem=None, rng=(0, 500), rememberRange=True, cmd=None):
        """Add linear view region. Region selection for data analysis
        rememberRange: when toggling, if set to True, when checked again, the
                       region was set to the region before the user unchecked
                       the selection.
        """
        if not plotitem:
             # if did not specify which viewbox, update on all views
            plotitem = [self.graphicsView.getItem(row=l[2], col=l[3]) for l in self.layout]
        if (self.viewRegionOn and not checked) or cmd == 'remove': # remove
            # Remove all the linear regions aand data tip labels
            def removeRegion(pm):
                for r in pm.items:
                    if "LinearRegionItem" in str(type(r)) and r in self.graphicsView.artists:
                        pm.removeItem(r)
            # vectorize
            removeRegion = np.frompyfunc(removeRegion, 1, 1)
            # Do the removal
            removeRegion(plotitem)
            for n, r in enumerate(self.graphicsView.artists):
                if "LinearRegionItem" in str(type(r)):
                    self.graphicsView.artists[n] = None
                if 'LabelItem' in str(type(r)) and r in self.graphicsView.items() and 'Start' in r.text and 'End' in r.text and 'Diff' in r.text:
                    self.graphicsView.removeItem(r)
                    self.graphicsView.artists[n] = None
            self.graphicsView.artists = [r for r in self.graphicsView.artists if r]

        elif (not self.viewRegionOn and checked) or cmd == 'add': # add
            # Record selection range
            if not hasattr(self, 'selectedRange'):
                self.selectedRange = rng
            # Add a data tip label
            label = pg.LabelItem(justify='right')
            label.setText(self.regionDataTip(rng=self.selectedRange))
            self.graphicsView.addItem(label)
            self.graphicsView.artists.append(label)
            # Add the view region on top of the viewbox
            def addRegion(pm):
                # Initialize the region
                region = pg.LinearRegionItem()
                region.setZValue(len(self.index)+10) # make sure it is on top
                 # initial range of the region
                region.setRegion(self.selectedRange if rememberRange else rng)
                region.sigRegionChanged.connect(lambda: self.onRegionChanged(region, plotitem, label))
                pm.addItem(region, ignoreBounds=True)
                # add these items in a collection
                self.graphicsView.artists.append(region)

            # vectorize
            addRegion = np.frompyfunc(addRegion, 1, 1) # (minX, maxX)
            # Add the view region
            addRegion(plotitem)
        else:
            raise(Exception('toggleRegionSelection fell through'))

        # update attribute
        self.viewRegionOn = checked

    def onRegionChanged(self, region, pm, label):
        """Called if region selection changed"""
        # update the current range
        self.selectedRange = region.getRegion()
        # Modify LinearRegion items from other viewbox
        for p in pm:
            for r in p.items:
                if r is region:
                    continue
                if 'LinearRegionItem' in str(type(r)):
                    r.setRegion(self.selectedRange)
        # Set label only once
        label_text = self.regionDataTip(self.selectedRange)
        if not label_text:
            return
        label.setText(label_text) # chagne data tip content

    def regionDataTip(self, rng):
        """Print the data tip in HTML format"""
        if not rng:
            return
        table_HTML = '<table align="center" width=200><tr><th></th><th><span style="font-style: italic">Start</span></th><th><span style="font-style: italic">End</span></th><th><span style="font-style: italic">Diff</span>{}</table>' # header, {row1, row2, ...}
        row_HTML = '<tr><th><span style="font-style: italic">{}</span></th><td align="center">{:0.1f}</td><td align="center">{:0.1f}</td><td align="center">{:0.1f}</td></tr>' # {Stream+Channel, data, data, data}

        # Add in time stream first
        final_HTML = row_HTML.format('Time', rng[0], rng[1], rng[1]-rng[0])
        # Add in other displayed streams / channelstr
        for l in self.layout:
            zData = self.episodes['Data'][self.index[-1]] # get the most recently clicked episode
            try:
                ind = time2ind(np.asarray(rng), ts=zData.Protocol.msPerPoint)
                # Get the extremes within the boundaries of data
                ind[0] = min(max(ind[0], 0), zData.Protocol.numPoints-1)
                ind[1] = max(min(ind[1], zData.Protocol.numPoints-1), 0)
                ymin = float(getattr(zData, l[0])[l[1]][ind[0]])
                ymax = float(getattr(zData, l[0])[l[1]][ind[1]])
            except:
                return None
            final_HTML += row_HTML.format(l[0]+' '+l[1], ymin, ymax, ymax-ymin)

        final_HTML = table_HTML.format(final_HTML)

        return final_HTML


    def toggleDataCursor(self, checked, plotitem=None, cmd=None):
        """Add cross hair to display data points at cursor point"""
        if self.dataCursorOn != checked and not plotitem:
            #if did not specify which viewbox, update on all views
            plotitem = [self.graphicsView.getItem(row=l[2], col=l[3]) for l in self.layout]
        if (self.dataCursorOn and not checked) or cmd == 'remove': # remove
            # Remove all the vertical line objects and data tip labels
            def removeCursor(pm):
                for r in pm.items:
                    if 'InfiniteLine' in str(type(r)) and r in self.graphicsView.artists:
                        pm.removeItem(r)
            # Vectorize
            removeCursor = np.frompyfunc(removeCursor, 1, 1)
            # Remove the cursor
            removeCursor(plotitem)
            # Remove labels and records in artists
            for n, r in enumerate(self.graphicsView.artists):
                if 'InfiniteLine' in str(type(r)):
                    self.graphicsView.artists[n] = None
                # set_trace()
                if 'LabelItem' in str(type(r)) and r in self.graphicsView.items() and 'Start' not in r.text and 'End' not in r.text and 'Diff' not in r.text:
                    self.graphicsView.removeItem(r)
                    self.graphicsView.artists[n] = None
            self.graphicsView.artists = [r for r in self.graphicsView.artists if r]

        elif (not self.dataCursorOn and checked) or cmd == 'add': # add
            # Add a data tip label
            label = pg.LabelItem(justify='right')
            label.setText(self.cursorDataTip(x=0))
            self.graphicsView.addItem(label)
            self.graphicsView.artists.append(label)
            # Add the data cursor
            def addCursor(pm):
                # Initialize the cursor
                cursor = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k'))
                pm.addItem(cursor, ignoreBounds=False)
                # Modify cursor and label upon mouse moving
                pm.scene().sigMouseMoved.connect(lambda pos: self.onMouseMoved(pos, cursor, plotitem, label))
                # proxy = pg.SignalProxy(pm.scene().sigMouseMoved, rateLimit=60, slot=lambda evt: self.onMouseMoved(evt, cursor, pm, label)) # proxy =, limit refreshment
                # add these items in a collection
                self.graphicsView.artists.append(cursor)

            # vectorize
            addCursor = np.frompyfunc(addCursor, 1, 1)
            # Do the adding
            addCursor(plotitem)
        else:
            raise(Exception('toggleDataCursor fell through'))

        # Update attribute
        self.dataCursorOn = checked

    def onMouseMoved(self, pos, cursor, pm, label):
        """pm: plot item
           evt: event
           label: data tip label
           vLine: vertical data cursor
        """
        # pos = pos[0]
        if not self.dataCursorOn:
            return
        # pos = evt # using signal proxy turns original arguments into a tuple
        xpos = None
        for p in pm:
            if p.sceneBoundingRect().contains(pos):
                mousePoint = p.getViewBox().mapSceneToView(pos)
                xpos = mousePoint.x() # modify xpos
                cursor.setPos(mousePoint.x())

        # Set label only once
        label_text = self.cursorDataTip(xpos)
        if not label_text:
            return
        label.setText(label_text) # change data tip content

    def cursorDataTip(self, x):
        """Print the data tip in HTML format"""
        if x is None:
            return
        table_HTML = '<table align="center" width=100>{}</table>' # header, {row1, row2, ...}
        row_HTML = '<tr><th><span style="font-style: italic">{}</span></th><td align="center">{}</td></tr>' # {Stream+Channel, data}
        # Add in time stream first
        final_HTML = row_HTML.format('Time', '{:0.1f}'.format(x))
        # Add in other displayed streams / channels
        for l in self.layout:
            zData = self.episodes['Data'][self.index[-1]] # get the most recently clicked episode
            try:
                ind = time2ind(x, ts=zData.Protocol.msPerPoint)
                y = 'NaN' if ind < 0 or ind > zData.Protocol.numPoints else '{:0.1f}'.format(float(getattr(zData, l[0])[l[1]][ind]))
            except:
                return None
            final_HTML += row_HTML.format(l[0]+' '+l[1], y)

        final_HTML = table_HTML.format(final_HTML)

        return final_HTML

    def exportWithScalebar(self, arrangement='overlap', savedir="R:/temp.eps"):
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
    # w.toggleRegionSelection(checked=True)
    # w.toggleDataCursor(checked=True)
    # w.addSubplot(layout=['Stimulus','A',1,0])
    # w.removeSubplot(layout=['Stimulus', 'A', 1, 0])
    w.show()
    # Connect upon closing
    # app.aboutToQuit.connect(restartpyshell)
    # Make sure the app stays on the screen
    sys.exit(app.exec_())

if run_example:
    import pyqtgraph.examples
    pyqtgraph.examples.run()
