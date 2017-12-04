# -*- coding: utf-8 -*-
"""
Created: Tue Mar 8 05:12:21 2016

Form implementation generated from reading ui file 'Scope.ui'

      by: PyQt4 UI code generator 4.11.4

WARNING! All changes made in this file will be lost!

Scope window.

@author: Edward
"""
import sys
import os
import re
import collections

from pdb import set_trace

# Global variables
__version__ = "Scope Window 0.4"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] # tableau10, or odd of tableau20
old = True # load old data format

import numpy as np
import pandas as pd

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

# from pyqtgraph.flowchart import Flowchart
sys.path.append(os.path.join(__location__, '..'))
from util.spk_util import *
from util.ImportData import NeuroData
from util.ExportData import *
from util.MATLAB import *
from app.Settings import *
from app.Toolbox import *

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

class ScopeWindow(QtGui.QMainWindow):
    def __init__(self, parent=None, partner=None, maxepisodes=30, layout=None, hideDock=True):
        super(ScopeWindow, self).__init__(parent)
        self.episodes = None
        self.index = []
        self.partner = partner
        # Initialization setting file
        self.iniPath = os.path.join(__location__,'../resources/config.ini')
        # set a limit on how many episodes to cache
        self.maxepisodes = maxepisodes
        # Record state of the scope window
        self.isclosed = True
        # Hide side dock panel
        self.hideDock = hideDock
        # This keeps track of the indices of which episodes are loaded
        self._loaded_array = []
        # Check if the user decided to keep traces from another cell
        self.keepOther = False
        # if use color for traaces
        self.colorfy = False
        # layout = [channel, stream, row, col]
        self.layout =[['Voltage', 'A', 0, 0]] if not layout else layout# [['Voltage', 'A', 0, 0], ['Current', 'A', 1, 0], ['Stimulus', 'A', 1,0]]
        # range of axes
        self.viewMode = 'keep'
        # Recording view Range
        self.viewRange = dict()
        # Null the baseline
        self.isnull = False
        # Range of baseline for null
        self.nullRange = None
        # Null baseline
        self.nullBaseline = None
        # view region
        self.viewRegionOn = False
        # self.linkViewRegion = True
        # Data tip
        self.dataCursorOn = False
        # crosshair
        self.crosshairOn = False
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
        MainWindow.setWindowIcon(QtGui.QIcon('resources/icons/activity.png'))
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
        if self.hideDock:
            self.dockWidget.hide() # keep the dock hidden by default
        # Dock content, containing widgets
        self.dockWidgetContents = QtGui.QWidget()
        self.dockWidgetContents.setObjectName(_fromUtf8("dockWidgetContents"))
        # Layout of the dock content
        self.dockLayout = QtGui.QVBoxLayout(self.dockWidgetContents)
        self.dockLayout.setObjectName(_fromUtf8("dockLayout"))
        # listview
        # self.listView = QtGui.QListView(self.dockWidgetContents)
        self.dockPanel = Toolbox(self.dockWidgetContents, self)
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

        exportVerticalAction = QtGui.QAction(QtGui.QIcon('export.png'), 'Export grid arrangement', self)
        exportVerticalAction.setStatusTip('Export the selected episodes in a vertical arrangement')
        exportVerticalAction.triggered.connect(lambda: self.exportWithScalebar(arrangement='vertical'))
        exportMenu.addAction(exportVerticalAction)

        exportVerticalAction = QtGui.QAction(QtGui.QIcon('export.png'), 'Export concatenated arrangement', self)
        exportVerticalAction.setStatusTip('Export the selected episodes in arrangement concatenated over time.')
        exportVerticalAction.triggered.connect(lambda: self.exportWithScalebar(arrangement='concatenate'))
        exportMenu.addAction(exportVerticalAction)

        # File: Settings
        settingsAction = QtGui.QAction("Settings", self)
        settingsAction.setStatusTip('Configure settings of PySynapse')
        settingsAction.triggered.connect(self.openSettingsWindow)
        fileMenu.addAction(settingsAction)

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
        # View: data cursor
        dataCursorAction = QtGui.QAction('Data cursor', self, checkable=True, checked=False)
        dataCursorAction.setShortcut('Alt+T')
        dataCursorAction.setStatusTip('Show data cursor on the traces')
        dataCursorAction.triggered.connect(lambda: self.toggleDataCursor(dataCursorAction.isChecked()))
        viewMenu.addAction(dataCursorAction)
        # View: crosshair
        crosshairAction = QtGui.QAction('Crosshair', self, checkable=True, checked=False)
        crosshairAction.setShortcut('Alt+X')
        crosshairAction.setStatusTip('Show crosshair on the plots')
        crosshairAction.triggered.connect(lambda: self.toggleCrosshair(crosshairAction.isChecked()))
        viewMenu.addAction(crosshairAction)
        # View: Keep previous
        keepPrev = QtGui.QAction('Keep previous', self, checkable=True, checked=False)
        keepPrev.setStatusTip('Keep traces from other data set on the scope window')
        keepPrev.triggered.connect(lambda: self.toggleKeepPrev(keepPrev.isChecked()))
        viewMenu.addAction(keepPrev)
        # View: show toolbox
        viewMenu.addAction(self.dockWidget.toggleViewAction())

    # ------------- Helper functions ----------------------------------------
    def printme(self, msg='doing stuff'): # for debugging
        print(msg)

    def openSettingsWindow(self):
        if self.partner is not None:
            if not hasattr(self.partner, 'settingsWidget'):
                self.partner.settingsWidget = Settings()
            self.settingsWidget = self.partner.settingsWidget
        else:
            self.settingsWidget = Settings()

        if self.settingsWidget.isclosed:
            self.settingsWidget.show()
            self.settingsWidget.isclosed = False

    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        self.isclosed = True

    def getNullBaseline(self, Y, ts, nullRange=None):
        """Get the baseline of null"""
        if nullRange is None:
            nullRange = self.nullRange
        if isinstance(self.nullRange, list):
            self.nullBaseline = np.mean(spk_window(Y, ts, nullRange))
        else: # a single number
            self.nullBaseline = Y[time2ind(nullRange, ts)][0]

        return self.nullBaseline

    def retranslateUi(self, MainWindow):
        """Set window title and other miscellaneous"""
        MainWindow.setWindowTitle(_translate(__version__, __version__, None))

    # ------------- Episode plotting utilities --------------------------------
    def updateEpisodes(self, episodes=None, index=[], updateLayout=False):
        """First compare episodes with self.episodes and index with self.index
        Only update the difference in the two sets. The update does not sort
        the index; i.e. it will be kept as the order of insert / click
        updateLayout:
        """
        if not isinstance(episodes, dict) or not isinstance(self.episodes, dict):
            bool_old_episode = False # upon startup
        else:
            bool_old_episode = self.episodes['Name'] == episodes['Name']

        # reset the grpahicsview if user not keeping traces from older dataset
        if not self.keepOther and not bool_old_episode:
            self.getDataViewRange() # Get the data range before clearing
            self.graphicsView.clear() # TODO: This changes the data view range from previous cell
            self._usedColors = []
            self._loaded_array = []
            self.index = []
            # print('reset view')

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
            
        # Remove episodes
        for j in index_remove:
            self.removeEpisode(info=(self.episodes['Name'], self.episodes['Epi'][j]))

        # Insert new episodes
        for i in index_insert:
            if not self.episodes['Data'][i]: # load if not already loaded
                self.episodes['Data'][i] = NeuroData(dataFile=self.episodes['Dirs'][i], old=old, infoOnly=False, getTime=True)
            self._loaded_array.append(i)
            # Draw the episode
            self.drawEpisode(self.episodes['Data'][i], info=(self.episodes['Name'], self.episodes['Epi'][i], i))

        # print(self.index)
        if not bool_old_episode:
            self.reissueArtists() # artists have been cleared. Add it back
            self.setDataViewRange(viewMode='reset')
        else:
            self.setDataViewRange(viewMode='keep')

        # Update the companion, Toolbox: Layout
        layout_index = self.dockPanel.accWidget.indexOfTitle('Channels')
        if self.dockPanel.accWidget.widgetAt(layout_index).objectName() == 'Banner':
            # Replace the banner widget with real widget
            layoutWidget = self.dockPanel.layoutWidget()
            self.dockPanel.replaceWidget(widget=layoutWidget, index=layout_index)

        if not bool_old_episode:
            self.dockPanel.updateLayoutComboBox()

        self.dockPanel.updateTTL()

        # print(self.layout)

    def drawEpisode(self, zData, info=None, pen=None, layout=None):
        """Draw plot from 1 zData"""
        # Set up pen color if not already specified.
        # Specified pen will not go to _usedColors list
        if pen is None: # go on automatic mode
            if self.colorfy:
                availableColors = list(colors)
                for c in self._usedColors:
                    availableColors.remove(c)
                pen = availableColors[0]
                self._usedColors.append(pen)
            else: # monocolor
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

            Y = getattr(zData, l[0])[l[1]]
            if self.isnull and self.nullRange is not None:
                Y = Y - self.getNullBaseline(Y, zData.Protocol.msPerPoint)
            cl = p.plot(x=zData.Time, y=Y, pen=pen, name=pname)
            cl._ts = zData.Protocol.msPerPoint # save the sampling rate to dataitem

    def removeEpisode(self, info=None):
        if not info:
            return

        for j, l in enumerate(self.layout):
            # get viewbox
            p1 = self.graphicsView.getItem(row=l[2], col=l[3])
            pname = info[0]+'.'+info[1]+'.'+l[0]+'.'+l[1]

            for k, a in enumerate(p1.listDataItems()):
                if a.name() == pname: # matching
                    # Remove color first from _usedColor list
                    if self.colorfy and j==0:
                        current_pen = a.opts['pen']
                        if isinstance(current_pen, str):
                            self._usedColors.remove(current_pen)
                        else: # Assume it is PyQt4.QtGui.QPen
                            self._usedColors.remove(current_pen.color().name())
                    # Remove the actual trace
                    p1.removeItem(a)


    def drawEvent(self, eventTime, which_layout, info=None, color='r', linesize=None, drawat='bottom', iteration=0):
        """Draw events occurring at specific times"""
        p = None
        for l in self.layout:
            if which_layout[0] in l and which_layout[1] in l:
                # get graphics handle
                p = self.graphicsView.getItem(row=l[2], col=l[3])
                break
        if not p:
            return

        # yRange = self.viewRange[l[0], l[1]][1]
        # set_trace()
        yRange = p.viewRange()[1]
        if linesize is None:
            linesize = abs((yRange[1]-yRange[0]) / 15.0)
        if drawat == 'bottom': # Stacked backwards
            ypos_0, ypos_1 = yRange[0] + iteration * linesize * 1.35, yRange[0] + linesize - iteration * linesize * 1.35
        else: # top
            ypos_0, ypos_1 = yRange[1] - iteration * linesize * 1.35, yRange[1] - linesize - iteration * linesize * 1.35

        # Cell + Episode + Even type + Stream + Channel
        pname = ".".join(info)+'.'+l[0]+'.'+l[1]
        for t in eventTime:
            p.plot(x=[t,t], y=[ypos_0, ypos_1], pen=color, name=pname)

        eventArtist = {'eventTime': eventTime, 'y': [ypos_0, ypos_1], 'layout': which_layout, 'name': pname,
                               'linecolor': color, 'type': 'event'}

        return eventArtist

    def removeEvent(self, info=None, which_layout=None, event_type='event'):
        """Remove one event type from specified layout (which_layout) or all
        members of the layout, if not specifying which_layout"""
        if not info:
            return

        for l in self.layout:
            if which_layout and (which_layout[0] not in l or which_layout[1] not in l):
                continue # specified which_layout but does not match current iteration
            # Get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            # Remove based on item name
            pname = ".".join(info)+'.'+l[0]+'.'+l[1]
            if event_type == 'event':
                for k, a in enumerate(p.listDataItems()):
                    if pname in a.name(): # matching
                        p.removeItem(a)
            else:
                index = 0
                iteration = len(p.items)
                for k in range(iteration):
                    try:
                        if p.items[index].name() in pname:
                            p.removeItem(p.items[index])
                        else:
                            index = index + 1
                            #continue
                    except:
                        if p.items[index].name in pname:
                            p.removeItem(p.items[index])
                        else:
                            index = index + 1
                            #continue

            if which_layout: # if specified which_layout
                break

    def drawROI(self, artist, which_layout, **kwargs):
        """Draw a specified ROI on the canvas"""
        def setPenStyle(pen, linestyle):
            if linestyle == '-':
                pass
            elif linestyle == '--':
                pen.setStyle(QtCore.Qt.DashLine)
            elif linestyle == '-.':
                pen.setStyle(QtCore.Qt.DashDotLine)
            elif linestyle == ':':
                pen.setStyle(QtCore.Qt.DotLine)
            else:
                pass

            return pen
        # Add the artist
        if artist['type'] == 'box':
            roi = QtGui.QGraphicsRectItem(float(artist['x0']), float(artist['y0']), \
                                        float(artist['width']), float(artist['height']))
            if artist['fill']:
                roi.setBrush(pg.mkBrush(artist['fillcolor']))
            if artist['line']:
                pen = pg.mkPen(artist['linecolor'])
                pen.setWidth(float(artist['linewidth']))
                pen = setPenStyle(pen, artist['linestyle'])
                roi.setPen(pen)
        elif artist['type'] == 'line':
            roi = QtGui.QGraphicsLineItem(float(artist['x0']), float(artist['y0']), \
                                          float(artist['x1']), float(artist['y1']))
            pen = pg.mkPen(artist['linecolor'])
            pen.setWidth(float(artist['linewidth']))
            pen = setPenStyle(pen, artist['linestyle'])
            roi.setPen(pen)
        elif artist['type'] == 'rectROI':
            roi = pg.RectROI([artist['x0'], artist['y0']], [artist['width'], artist['height']], \
                             pen=artist['linecolor'], **kwargs)
           # if not resizable:
           #     [roi.removeHandle(h) for h in roi.getHandles()]
        else:
            raise(NotImplementedError("'{}' annotation object has not been implemented yet".format(artist['type'])))

        # Get which graphicsview to draw the ROI on
        p = None
        for l in self.layout:
            if which_layout[0] in l and which_layout[1] in l:
                # get graphics handle
                p = self.graphicsView.getItem(row=l[2], col=l[3])
                break
        if not p:
            return
        # Set properties of the ROI artist
        # Stream + channel + Name of the artist
        pname = artist['name'] + '.' + l[0] + '.' + l[1]
        roi.name = pname
        p.addItem(roi)


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

        # TODO: The new plot will likely change the view

        # Set the proper view range
        self.setDataViewRange('keep')
        # Redraw the artists
        self.reissueArtists()
        # Force the new subplot to start with default view range
        # self.setDataViewRange(viewMode='default')

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
        if '' in new_layout:
            return
        if old_layout == new_layout:
            return

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
                #a._pen = pen
                #pen = pg.mkPen(pen)
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

    def setDataViewRange(self, viewMode='default', xRange=None, yRange=None):
        # print('view range %s'%self.viewMode)
        self.viewMode = viewMode
        if not self.viewRange.keys():
            self.viewMode = 'default'

        # Loop through all the subplots
        for n, l in enumerate(self.layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            if l[2]>0 or l[3]>0:
                p.setXLink(self.graphicsView.getItem(row=0, col=0))
            #print(a.tickValues())
            if self.viewMode == 'default':
                # Make everything visible first, may help pervent failure 
                # if the subsequent range calculation fails
                p.autoRange()
                # Find out the default yrange options
                options = readini(self.iniPath)  # Read it real time
                default_yRange_dict = {'Voltage': (options['voltRangeMin'], options['voltRangeMax']),
                                       'Current': (options['curRangeMin'], options['curRangeMax']),
                                       'Stimulus': (options['stimRangeMin'], options['stimRangeMax'])}
                yRange = default_yRange_dict.get(l[0])
                p.setYRange(yRange[0], yRange[1], padding=0)
                default_xRange = [options['timeRangeMin'], options['timeRangeMax']]
                if default_xRange[0] is None or isinstance(default_xRange[0], str):
                    default_xRange[0] = 0
                if default_xRange[1] is None or isinstance(default_xRange[1], str):
                    default_xRange[1] = max([max(s.xData) for s in p.dataItems])
                p.setXRange(default_xRange[0], default_xRange[1], padding=0)
                # Update current viewRange
                self.viewRange[l[0], l[1]] = p.viewRange()
                if n == len(self.layout)-1: # update only after iterating through
                    self.viewMode = viewMode
            elif self.viewMode == 'auto':
                p.autoRange()
                # Update current viewRange
                self.viewRange[l[0], l[1]] = p.viewRange()
            elif self.viewMode == 'keep':
                # Update current viewRange
                self.viewRange[l[0], l[1]] = p.viewRange()
                # no chnage in viewRange, but still link the views
            elif self.viewMode == 'reset':
                if p.viewRange() != self.viewRange[l[0], l[1]]:
                    X, Y = self.viewRange[l[0], l[1]]
                    p.setXRange(X[0], X[1], padding=0)
                    p.setYRange(Y[0], Y[1], padding=0)
                    self.viewMode = 'keep'
            elif self.viewMode == 'manual':
                if xRange is not None:
                    p.setXRange(xRange)
                if yRange is not None:
                    p.setYRange(yRange)
            else:
                raise(TypeError('Unrecognized view mode'))

    def getDataViewRange(self):
        for l in self.layout:
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            if p is None:
                return
            self.viewRange[l[0], l[1]] = p.viewRange()

    def reissueArtists(self):
        """In case artists are removed, toggle back on the artists"""
        if self.viewRegionOn: # remove, then redraw
            self.toggleRegionSelection(checked=False)
            self.toggleRegionSelection(checked=True, rng=self.selectedRange, rememberRange=True, cmd='add')
        # Add cursor if selection checked
        if self.dataCursorOn: # remove, then redraw
            self.toggleDataCursor(checked=False)
            self.toggleDataCursor(checked=True)

        if self.crosshairOn: # remove, then redraw
            self.toggleCrosshair(checked=False)
            self.toggleCrosshair(checked=True)

    def toggleRegionSelection(self, checked, plotitem=None, rng=None, rememberRange=False, cmd=None):
        """Add linear view region. Region selection for data analysis
        rememberRange: when toggling, if set to True, when checked again, the
                       region was set to the region before the user unchecked
                       the selection.
        """
        if not plotitem:
             # if did not specify which viewbox, update on all views
            plotitem = [self.graphicsView.getItem(row=l[2], col=l[3]) for l in self.layout]
        if (self.viewRegionOn and not checked) or cmd == 'remove': # remove
            # Remove all the linear regions and data tip labels
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
                    self.graphicsView.artists.remove(r)
            self.graphicsView.artists = [r for r in self.graphicsView.artists if r]

        elif (not self.viewRegionOn and checked) or cmd == 'add': # add
            # Update current data view range
            # print(rng)
            rng = None
            if rng is None:
                self.getDataViewRange()
                xRange = np.array(list(self.viewRange.values())[0][0])
                rng = (xRange[0], xRange[0] + np.diff(xRange)[0] / 10.0)
            # Record selection range
            if not hasattr(self, 'selectedRange'):
                self.selectedRange = rng

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

            # Add a data tip label
            label = pg.LabelItem(justify='right')
            label.setText(self.regionDataTip(rng=self.selectedRange))
            self.graphicsView.addItem(label)
            self.graphicsView.artists.append(label)

            # Add right click menu item for selection region, allowing precise definition of selection region
            # self.addMenu
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
                if self.isnull:
                    ymin -= self.nullBaseline
                    ymax -= self.nullBaseline
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
            # Remove all the line objects from the view first
            def removeCursor(pm):
                indices = []
                for nr, r in enumerate(pm.items):
                    if 'InfiniteLine' in str(type(r)) and r in self.graphicsView.artists:
                        indices.append(nr)

                for ind in sorted(indices, reverse=True):
                        pm.removeItem(pm.items[ind])

            # Vectorize
            removeCursor = np.frompyfunc(removeCursor, 1, 1)
            # Remove the cursor
            removeCursor(plotitem)

            # Then, remove the items from graphicsView.artists
            for n, r in enumerate(self.graphicsView.artists):
                if 'InfiniteLine' in str(type(r)):
                    self.graphicsView.artists[n] = None
                if 'LabelItem' in str(type(r)) and 'Start' not in r.text and 'End' not in r.text and 'Diff' not in r.text:
                    try:
                        self.graphicsView.removeItem(r)
                    except:
                        pass
                    self.graphicsView.artists[n] = None

            self.graphicsView.artists = [r for r in self.graphicsView.artists if r] # Get rid of None


        elif (not self.dataCursorOn and checked) or cmd == 'add': # add
            # Add a data tip label
            label = pg.LabelItem(justify='right')
            label.setText(self.cursorDataTip(x=0))
            self.graphicsView.addItem(label)
            self.graphicsView.artists.append(label)
            # Add the data cursor
            def addCursor(pm):
                # Initialize the cursor
                cursorV = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('k'), name='verticalCursor')
                pm.addItem(cursorV, ignoreBounds=False)
                # Modify cursor and label upon mouse moving
                pm.scene().sigMouseMoved.connect(lambda pos: self.onMouseMoved(pos, cursorV, plotitem, label))
                # proxy = pg.SignalProxy(pm.scene().sigMouseMoved, rateLimit=60, slot=lambda evt: self.onMouseMoved(evt, cursor, pm, label)) # proxy =, limit refreshment
                # add these items in a collection
                self.graphicsView.artists.append(cursorV)
                # Add horizontal cursor
                if self.crosshairOn:
                    cursorH = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('k'), name='horizontalCursor')
                    pm.addItem(cursorH, ignoreBounds=False)
                    pm.scene().sigMouseMoved.connect(lambda pos: self.onMouseMoved(pos, cursorH, plotitem, label=None))
                    self.graphicsView.artists.append(cursorH)
            # vectorize
            addCursor = np.frompyfunc(addCursor, 1, 1)
            # Do the adding
            addCursor(plotitem)
        else:
            return
            # raise(Exception('toggleDataCursor fell through'))

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
                mousePoint = p.vb.mapSceneToView(pos)
                xpos = mousePoint.x() # modify xpos
                if cursor.name()[0] == 'v':
                    cursor.setPos(mousePoint.x())
                elif cursor.name()[0] == 'h':
                    cursor.setPos(mousePoint.y())
                else:
                    raise(ValueError('Unrecognized cursor type'))

        # Set label only once
        if label is None:
            return
        label_text = self.cursorDataTip(xpos)
        #label_text = label_text + "\n{}, {}".format(str(self.graphicsView.items()[1]), str(self.graphicsView.artists[0]))
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
                if ind < 0 or ind > zData.Protocol.numPoints:
                    y = 'NaN'
                else:
                    y = float(getattr(zData, l[0])[l[1]][ind])
                    if self.isnull: # remove baseline
                        y = y - self.nullBaseline # should have been calculated already

                    y = '{:0.1f}'.format(y)
            except:
                return None
            final_HTML += row_HTML.format(l[0]+' '+l[1], y)

        final_HTML = table_HTML.format(final_HTML)

        return final_HTML

    def toggleCrosshair(self, checked=False, plotitem=None):
        if checked: # from unchecked to checked
            if self.dataCursorOn:
                self.toggleDataCursor(checked=False) # turn off cursor first
            self.crosshairOn = True
            self.toggleDataCursor(checked=True) # turn it back on with crosshair
        else: # from checked to unchecked
            self.toggleDataCursor(checked=False) # turn it off
            self.crosshairOn = False

    def exportWithScalebar(self, arrangement='overlap'):
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
            # zData = NeuroData(dataFile=self.episodes['Dir'][i], old=old, infoOnly=True)
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

        # Get the options before saving the figure
        options = readini(self.iniPath)
        # figure out the figure size
        nchannels = len(viewRange.keys())
        # Annotation artists
        annotationArtists = self.dockPanel.getArtists()
        # Do the plotting once all the necessary materials are gathered
        if arrangement == 'overlap':
            PlotTraces(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors, artists=annotationArtists,
                       fig_size=(options['figSizeW'], options['figSizeH']), adjustFigW=options['figSizeWMulN'], adjustFigH=options['figSizeHMulN'],
                       dpi=options['dpi'], nullRange=None if not self.isnull else self.nullRange, annotation=options['annotation'], showInitVal=options['showInitVal'],
                       setFont=options['fontName'], fontSize=options['fontSize'], linewidth=options['linewidth'], monoStim=options['monoStim'],
                       stimReflectCurrent=options['stimReflectCurrent'])
        elif arrangement == 'concatenate':
            PlotTracesConcatenated(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors, artists=annotationArtists,
                                 dpi=options['dpi'], fig_size=(options['figSizeW'], options['figSizeH']), nullRange=None if not self.isnull else self.nullRange, hSpaceType=options['hSpaceType'], hFixedSpace=options['hFixedSpace'],
                                 adjustFigW= options['figSizeWMulN'],adjustFigH= options['figSizeHMulN'], annotation=options['annotation'], showInitVal=options['showInitVal'],
                                 setFont=options['fontName'], fontSize=options['fontSize'], linewidth=options['linewidth'], monoStim=options['monoStim'],
                                 stimReflectCurrent=options['stimReflectCurrent'])
        elif arrangement in ['vertical', 'horizontal', 'channels x episodes', 'episodes x channels']:
            PlotTracesAsGrids(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors, artists=annotationArtists,
                                 dpi=options['dpi'], fig_size=(options['figSizeW'], options['figSizeH']),adjustFigW=options['figSizeWMulN'],adjustFigH=options['figSizeHMulN'],
                                 nullRange=None if not self.isnull else self.nullRange, annotation=options['annotation'],setFont=options['fontName'], fontSize=options['fontSize'],
                                 scalebarAt=options['scalebarAt'], gridSpec=options['gridSpec'], linewidth=options['linewidth'], monoStim=options['monoStim'], showInitVal=options['showInitVal'],
                                 stimReflectCurrent=options['stimReflectCurrent'])
        else:
            raise(ValueError('Unrecognized arragement:{}'.format(arrangement)))

run_example = False


if __name__ == '__main__' and not run_example:
#    episodes = {'Duration': [4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 4000, 90000, 90000, 90000, 50000, 50000],
#    'Name': 'Neocortex F.08Feb16',
#    'Drug Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '3.4 sec', '2:03', '3:40', '5:37', '8:29'],
#    'Drug Level': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
#    'Comment': ['', 'DAC0: PulseB 200', 'DAC0: PulseB -50', 'DAC0: PulseB -75', 'DAC0: PulseB -50', 'DAC0: PulseB 50', 'DAC0: PulseB 100', 'DAC0: PulseB 150', 'DAC0: PulseB 200', 'DAC0: PulseB 200', '', '', '', '', 'DAC0: PulseB 200', 'DAC0: PulseB 200'],
#    'Dirs': ['D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E1.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E2.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E3.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E4.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E5.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E6.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E7.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E8.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E9.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E10.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E11.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E12.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E13.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E14.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E15.dat', 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016/Neocortex F.08Feb16.S1.E16.dat'],
#    'Time': ['0.0 sec', '58.8 sec', '1:08', '1:22', '1:27', '1:37', '1:49', '1:56', '2:03', '3:38', '4:41', '6:43', '8:42', '10:19', '12:16', '15:08'], 'Drug Name': ['', '', '', '', '', '', '', '', '', '', '', '', '', '', '', ''],
#    'Epi': ['S1.E1', 'S1.E2', 'S1.E3', 'S1.E4', 'S1.E5', 'S1.E6', 'S1.E7', 'S1.E8', 'S1.E9', 'S1.E10', 'S1.E11', 'S1.E12', 'S1.E13', 'S1.E14', 'S1.E15', 'S1.E16'],
#    'Sampling Rate': [0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001, 0.20000000000000001]}
#    episodes = {'Duration': [40000], 'Name': 'Neocortex A.10Aug16', 'Drug Time': ['17:34'], 'Drug Level': [1], 'Comment':[''],
#                'Dirs':['D:/Data/Traces/2015/08.August/Data 10 Aug 2015/Neocortex A.10Aug15.S1.E16.dat'],'Time':['18:35'], 'Epi':['S1.E16'], 'Sampling Rate': [0.1]}
#    index = [0]

#    episodes = {'Duration': [50000], 'Name': 'Neocortex B.13Oct15', 'Drug Time': ['10:29'], 'Drug Level': [3], 'Comment':[''],
#                'Dirs':['D:/Data/Traces/2015/10.October/Data 13 Oct 2015/Neocortex B.13Oct15.S1.E24.dat'],'Time':['1:04:31'], 'Epi':['S1.E24'], 'Sampling Rate': [0.1]}
#    index = [0]

#    episodes = {'Drug Name': ['', '', ''], 'Epi': ['S1.E3', 'S1.E7', 'S1.E13'],
#    'Duration': [4000, 4000, 4000], 'Drug Level': [0, 0, 0], 'Time': ['31.7 sec', '2:03', '2:54'],
#    'Name': 'Neocortex I.03Aug16', 'Drug Time': ['31.7 sec', '2:03', '2:54'], 'Sampling Rate': [0.1, 0.1, 0.1],
#    'Comment': ['DAC0: PulseA -50 PulseB 75', 'DAC0: PulseA -50 PulseB 75', 'DAC0: PulseA -50 PulseB 75'],
#    'Dirs': ['D:/Data/Traces/2016/08.August/Data 3 Aug 2016/Neocortex I.03Aug16.S1.E3.dat', 'D:/Data/Traces/2016/08.August/Data 3 Aug 2016/Neocortex I.03Aug16.S1.E7.dat', 'D:/Data/Traces/2016/08.August/Data 3 Aug 2016/Neocortex I.03Aug16.S1.E13.dat']}
#
#    index = [0,1,2]

#    episodes = {'Drug Name': [''], 'Epi': ['S1.E4'],
#    'Duration': [10000], 'Drug Level': [0], 'Time': ['31.7 sec'],
#    'Name': 'Neocortex E.09Jun16', 'Drug Time': ['31.7 sec', '2:03', '2:54'], 'Sampling Rate': [0.1],
#    'Comment': ['DAC0: PulseA -50 PulseB 50'],
#    'Dirs': ['D:/Data/Traces/2016/06.June/Data 9 Jun 2016/Neocortex E.09Jun16.S1.E4.dat']}
#    index = [0]
#    episodes = {'Drug Name': [''], 'Epi': ['S1.E13'],
#    'Duration': [10000], 'Drug Level': [0], 'Time': ['31.7 sec'],
#    'Name': 'Neocortex G.07Jun16', 'Drug Time': ['31.7 sec'], 'Sampling Rate': [0.1],
#    'Comment': ['DAC0: Step -40 (1000 to 5000 ms) PulseA -10'],
#    'Dirs': ['D:/Data/Traces/2016/06.June/Data 7 Jun 2016/Neocortex G.07Jun16.S1.E13.dat']}
#    index = [0]

    # # Cell attached recording
    # episodes = {'Drug Name': [''], 'Epi': ['S1.E33', 'S1.E34', 'S1.E35'],
    # 'Duration': [40000,40000,40000], 'Drug Level': [1,1,1], 'Time': ['25:06','25:54','26:43'],
    # 'Name': 'Neocortex E.09Jun16', 'Drug Time': ['24:27','25:15','26:03'], 'Sampling Rate': [0.1,0.1,0.1],
    # 'Comment': ['TTL3: SIU train','',''],
    # 'Dirs': ['D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E33.dat',
    #          'D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E34.dat',
    #          'D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E35.dat']}

    # Optogenetics
    episodes = {'Drug Name': [''], 'Epi': ['S1.E22', 'S1.E23', 'S1.E24'],
    'Duration': [50000,50000,50000], 'Drug Level': [0,0,0], 'Time': ['30:03','31:03','32:03'],
    'Name': 'NeocortexChRNBM D.09Nov16', 'Drug Time': ['24:27','25:15','26:03'], 'Sampling Rate': [0.1,0.1,0.1],
    'Comment': ['TTL3: SIU train','',''],
    'Dirs': ['D:/Data/Traces/2016/11.November/Data 9 Nov 2016/NeocortexChRNBM D.09Nov16.S1.E22.dat',
             'D:/Data/Traces/2016/11.November/Data 9 Nov 2016/NeocortexChRNBM D.09Nov16.S1.E23.dat',
             'D:/Data/Traces/2016/11.November/Data 9 Nov 2016/NeocortexChRNBM D.09Nov16.S1.E24.dat']}

    index = [0]
    app = QtGui.QApplication(sys.argv)
    w = ScopeWindow(hideDock=False, layout=[['Voltage', 'A', 0, 0],  ['Stimulus', 'A', 1, 0]])
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
