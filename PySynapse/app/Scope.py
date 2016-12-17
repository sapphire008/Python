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
import re
import collections

from pdb import set_trace

# Global variables
__version__ = "Scope Window 0.3"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] # tableau10, or odd of tableau20
old = True # load old data format

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from PyQt4 import QtGui, QtCore

import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget

# from pyqtgraph.flowchart import Flowchart
sys.path.append(os.path.join(__location__, '..'))
from util.spk_util import *
from util.ImportData import NeuroData
from util.ExportData import *
from util.MATLAB import *
from util.spk_util import *
from app.AccordionWidget import AccordionWidget
from app.Settings import *

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


class SideDockPanel(QtGui.QWidget):
    """Collapsible dock widget that displays settings and analysis results for
    the Scope window
    """
    # Keep track of position of the widget added
    _widget_index = 0
    _sizehint = None
    # used for replace formula variables, total allow 52 replacements, from a-zA-Z
    _newvarsList = [chr(i) for i in 65+np.arange(26)]+[chr(i) for i in 97+np.arange(26)]
    
    def __init__(self, parent=None, friend=None):
        super(SideDockPanel, self).__init__(parent)
        self.parent = parent
        self.friend = friend
        self.detectedEvents = []
        self.annotationArtists = []
        self.setupUi()

    def setupUi(self):
        self.verticalLayout = self.parent.layout()
        # self.setLayout(self.verticalLayout)
        self.accWidget = AccordionWidget(self)

        # Add various sub-widgets, which interacts with Scope, a.k.a, friend
        self.accWidget.addItem("Arithmetic", self.arithmeticWidget(), collapsed=True)
        # self.accWidget.addItem("Annotation", self.annotationWidget(), collapsed=True)
        self.accWidget.addItem("Channels", self.layoutWidget(), collapsed=True)
        self.accWidget.addItem("Curve Fit", self.curvefitWidget(), collapsed=True)
        self.accWidget.addItem("Event Detection", self.eventDetectionWidget(), collapsed=True)

        self.accWidget.setRolloutStyle(self.accWidget.Maya)
        self.accWidget.setSpacing(0) # More like Maya but I like some padding.
        self.verticalLayout.addWidget(self.accWidget)

    # --------- Trace arithmetic tools ---------------------------------------
    def arithmeticWidget(self):
        """Setting widget for trace manipulation"""
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("ArithmeticWidgetFrame"))
        # widgetFrame.layout().setSpacing(0)

        calculateButton = QtGui.QPushButton("Calculate")
        # Remove baseline from the trace check box
        nullCheckBox = QtGui.QCheckBox("Null")
        nullCheckBox.setToolTip("Remove baseline")
        # null baseline range
        rangeTextBox = QtGui.QLineEdit()
        rangeTextBox.setToolTip("Range of baseline.\nEnter a single number or a range [min, max] in ms")
        rangeTextBox.setText("0")
        # Range unit label
        rangeUnitLabel = QtGui.QLabel("ms")

        # Formula
        formulaTextBox = QtGui.QLineEdit()
        formulaTextBox.setPlaceholderText("Formula")
        Tooltips = "Examples:\n"
        Tooltips += "Mean: (S1.E1 + S1.E2 + S1.E3) / 3\n"
        Tooltips += "Diff between episodes: S1.E1-S1.E2\n"
        Tooltips += "Calculation between regions: S1.E1:[500, 700] - S1.E2:[800, 1000]\n"
        Tooltips += "Multiple manipulations: {S1.E1 - S1.E2; S1.E3 - S1.E4; S1.E5 - S1.E6}"
        formulaTextBox.setToolTip(Tooltips)

        # Report box
        arithReportBox = QtGui.QLabel("Arithmetic Results")
        arithReportBox.setStyleSheet("background-color: white")
        arithReportBox.setWordWrap(True)

        # Connect all the items to calculationevents
        nullCheckBox.stateChanged.connect(lambda checked: self.nullTraces(checked, rangeTextBox))
        calculateButton.clicked.connect(lambda: self.calculateTraces(formulaTextBox.text(), nullCheckBox.checkState(), arithReportBox))
        formulaTextBox.returnPressed.connect(lambda: self.calculateTraces(formulaTextBox.text(), nullCheckBox.checkState(), arithReportBox))

        # Organize all the items in the frame
        widgetFrame.layout().addWidget(calculateButton, 0, 0, 1, 3)
        widgetFrame.layout().addWidget(nullCheckBox, 1, 0)
        widgetFrame.layout().addWidget(rangeTextBox, 1, 1)
        widgetFrame.layout().addWidget(rangeUnitLabel, 1, 2)
        widgetFrame.layout().addWidget(formulaTextBox, 2, 0, 1, 3)
        widgetFrame.layout().addWidget(arithReportBox, 3, 0, 1, 3)

        return widgetFrame

    def nullTraces(self, checked, rangeTextBox):
        self.friend.isnull = checked
        # parse the range
        r = rangeTextBox.text()
        if "[" not in r: # presumbaly a single number
            self.friend.nullRange = float(r)
        else: # parse the range
            r=r.replace("[","").replace("]","").replace(","," ")
            self.friend.nullRange = [float(k) for k in r.split()]

        # Redraw episodes
        index = list(self.friend.index) # keep the current index. Make a copy
        episodes = self.friend.episodes # keep the current episode
        self.friend.updateEpisodes(episodes=episodes, index=[], updateLayout=False) # clear all the episodes
        self.friend.updateEpisodes(episodes=episodes, index=index, updateLayout=False) # redraw all the episodes

    def calculateTraces(self, formula, isNulled, arithReportBox):
        arithReportBox.setText('') # clear any previous error message first
        if isNulled:
            r = self.friend.nullRange # should have been already calculated before
        else:
            r = None
            
        def parseTilda(f):
            """Turn "S1.E2~4" into
            "(S1.E2+S1.E3+S1.E4)"
            """
            
            if "~" not in f:
                return f
                
            # Assuming the S#.E# structure
            ep_ranges = re.findall('S(\d+)\.E(\d+)~(\d+)', f)
            for m, ep in enumerate(ep_ranges):
                epsl = ["S{}.E{:d}".format(ep[0], i) for i in np.arange(int(ep[1]), int(ep[2])+1, 1)]
                epsl = "("+"+".join(epsl)+")"
                f = re.sub('S(\d+)\.E(\d+)~(\d+)', epsl, f, count=1)
            
            return f
                
        def parseSimpleFormula(f):
            """Simple linear basic four operations
            e.g. f = "S1.E1 + S1.E2 - S1.E3 / 2 + S1.E4 * 3 / 8 +5" -->
            D = [S1.E1, S1.E2, S1.E3, S1.E4], K = [1, 1, -0.5, 0.375]
            C = 5 (constant term)
            """
            # separate the formula first
            groups = [s.replace(" ","") for s in filter(None, re.split(r"(\+|-)", f))]
            D = [] # data variable
            K = [] # scale factors
            C = 0 # constant

            for n, g in enumerate(groups):
                # initialize scale factor
                if n==0 or groups[n-1] == '+':
                    k = 1
                elif groups[n-1] == '-':
                    k = -1
                    
                if g == "-" or g == "+":
                    continue
                elif isstrnum(g): # constants
                    C += k * str2numeric(g)
                elif "/" not in g and "*" not in g: # single episodes
                    D.append(g)
                    K.append(k) # scale factor
                elif "/" in g or "*" in g:
                    hubs = [s.replace(" ","") for s in filter(None, re.split(r"(\*|/)", g))]
                    for m, h in enumerate(hubs):
                        if h == '*' or h == '/':
                            continue
                        elif isstrnum(h):
                            # examine the operator before
                            if m == 0 or hubs[m-1] == '*':
                                k *= str2numeric(h)
                            elif hubs[m-1] == '/':
                                k = k/str2numeric(h)
                            else:
                                arithReportBox.setText("Unrecognized operation " + hubs[m-1])
                                return
                        else: # Data variable
                            D.append(h)
                    K.append(k)
                else: # fall through for some reason. Need check
                    arithReportBox.setText("Unexpected formula")
                    return

            return D, K, C

        def simpleMath(f, stream, channel, **kwargs):
            """" f = "S1.E1 + S1.E2 - S1.E3 / 2 + S1.E4 * 3 / 8"
            Additional variables can be provided by **kwargs"""
            D, K, Y = parseSimpleFormula(f)

            for d, k in zip(D, K):
                if d not in kwargs.keys():
                    # load episodes
                    try:
                        yind = self.friend.episodes['Epi'].index(d)
                    except:
                        # arithReportBox.setText(g + " is not a valid episode")
                        return

                    if not self.friend.episodes['Data'][yind]: # if empty
                        self.friend.episodes['Data'][yind] = NeuroData(dataFile=self.friend.episodes['Dirs'][yind], old=old, infoOnly=False)

                    y = getattr(self.friend.episodes['Data'][yind], stream)[channel] # get the time series
                    # null the time series
                    if r is not None:
                        y = y - self.friend.getNullBaseline(y, self.friend.episodes['Data'][yind].Protocol.msPerPoint, r)
                else:
                    y = kwargs[d] # assume everything is processed

                # final assembly
                # taking care of uneven Y length
                try:
                    if len(Y)==1:
                        y_len = len(y)
                    else:
                        y_len = min([len(Y), len(y)])
                    Y = Y[0:y_len]
                    y = y[0:y_len]  
                except: # object not iterable, like int
                    pass                    

                Y += y * k

            return Y
            
        def callback(match):
            return next(callback.v)
        
#        mmdict = {}
#        for kk, vv in self.friend.episodes.items():
#            if kk == 'Name':
#                mmdict[kk] = vv
#                continue
#            mmdict[kk] = list(np.array(vv)[np.array(self.friend.index)])
#            
#        print(mmdict)
#        return

        # parse formulac
        if "{" in formula:
            # separate each formula
            formula = formula.replace("{","").replace("}","")
            formula = formula.split(";")
        else:
            formula = [formula]
        
        # parse each formula
        for f0 in formula:
            if ":" in f0: # has range. Assume each formula hsa only 1 range
                set_trace()
                f, rng = f0.split(":")
                f = parseTilda(f)
                rng = str2num(rng)
            else:
                f = parseTilda(f0)
                rng = None
            # if has parenthesis
            y = dict()
            try:
                if "(" in f:
                    # to be safe, remove any duplicate parentheses
                    f = re.sub("(\()+", "(", f)
                    f = re.sub("(\))+", ")", f)
                    for s, c, _, _ in self.friend.layout:
                        # separate into a list of simple ones
                        fSimpleList = re.findall('\(([^()]*)\)', f)
                        # for each simple ones, do calculation
                        YList = [simpleMath(fSimple, s, c) for fSimple in fSimpleList]
        
                        newvars = self._newvarsList[:len(fSimpleList)] # ['A','B','C',...]
                        callback.v = iter(newvars)
                        # new formula: replace all parentheses with a new variable
                        nf = re.sub(r'\(([^()]*)\)', callback, f)
                        # build a dictionary between the parentheses values and new variables
                        nfdict = {}
                        for nn, v in enumerate(newvars):
                            nfdict[v] = YList[nn]
                        # use the new variable, together with episode names that was not 
                        # in the parentheses to calculate the final Y
                        y[(s,c)] = simpleMath(nf, s, c, **nfdict)
                else:
                    for s, c, _, _ in self.friend.layout:
                        y[(s,c)] = simpleMath(f, s, c)
            except Exception as err:
                arithReportBox.setText("{}".format(err))
                return

            # Subset of the time series if range specified
            ts = self.friend.episodes['Sampling Rate'][0]
            if rng is not None:
                for s, c, _, _ in self.friend.layout:
                    y[(s,c)] = spk_window(y[(s,c)], ts, rng)
                    
            y_len = len(y[s,c]) # length of time series
            
            # Append the data to friend's episodes object
            self.friend.episodes['Duration'].append(ind2time(y_len-1,ts)[0])
            self.friend.episodes['Drug Time'].append('00:00')
            self.friend.episodes['Drug Name'].append('')
            self.friend.episodes['Drug Level'].append(-1)
            self.friend.episodes['Comment'].append('PySynapse Arithmetic Data')
            self.friend.episodes['Dirs'].append(f)
            self.friend.episodes['Time'].append('00:00')
            self.friend.episodes['Epi'].append(f)
            self.friend.episodes['Sampling Rate'].append(ts)
            # Make up fake data. Be more complete so that it can be exported correctly
            zData = NeuroData()
            for s, c, _, _ in self.friend.layout:
                setattr(zData, s, {c: y[s,c]})
            
            # fill in missing data
            stream_list,_,_,_ = zip(*self.friend.layout)
            stream_all = ['Voltage', 'Current', 'Stimulus']
            for _, c, _, _ in self.friend.layout:
                for s in stream_all:
                    if s not in stream_list:
                        setattr(zData, s, {c: np.zeros(y_len)})

            zData.Time = np.arange(y_len) * ts
            zData.Protocol.msPerPoint = ts
            zData.Protocol.WCtimeStr = ""
            zData.Protocol.readDataFrom = self.friend.episodes['Name'] + " " + f0
            zData.Protocol.numPoints = y_len
            zData.Protocol.acquireComment = 'PySynapse Arithmetic Data'
            self.friend.episodes['Data'].append(zData)
            
         # Redraw episodes with new calculations
        episodes = self.friend.episodes # keep the current episode
        index = list(range(len(episodes['Epi'])-len(formula), len(episodes['Epi']))) # keep the current index. Make a copy
        self.friend.updateEpisodes(episodes=episodes, index=[], updateLayout=False) # clear all the episodes
        # temporarily disable isnull
        self.friend.isnull = False
        # Draw the episodes
        self.friend.updateEpisodes(episodes=episodes, index=index, updateLayout=False) # redraw all the episodes
        # Turn back isnull
        self.friend.isnull = isNulled
    
    # ----------- Annotation widget ------------------------------------------
    def annotationWidget(self):
        """Adding annotation items on the graph"""
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("AnnotationWidgetFrame"))
        widgetFrame.layout().setSpacing(0)
        
        return widgetFrame
        
        avail_obj = ['box', # [x1, y1, x2, y2, linewidth, linestyle, color]
                     'line',  # [x1, y1, x2, y2, linewidth, linestyle, color]
                     'circle', # [center_x, center_y, a, b, rotation, linewidth, linestyle, color]
                     'arrow',  # [x, y, x_arrow, y_arrow, linewidth, linestyle, color]
                     'symbol' # ['symbol', x, y, markersize, color]
                     ]
#        addButton = QtGui.QPushButton("Add") # Add a channel
#        addButton.clicked.connect(lambda: self.addAnnotationRow(avail_obj=avail_obj))
#        removeButton = QtGui.QPushButton("Remove") # Remove a channel
#        removeButton.clicked.connect(self.removeAnnotationRow)
#        # Add the buttons
#        widgetFrame.layout().addWidget(addButton, 1, 0)
#        widgetFrame.layout().addWidget(removeButton, 1, 1)
#        # Add the exisiting channels and streams to the table
#        widgetFrame.layout().addWidget(self.annotation_table, 2, 0, self.annotation_table.rowCount(), 2)
        
        
    def addAnnotationRow(self, avail_obj=None):
        return
        
    def removeAnnotationRow(self):
        row = self.annotation_table.rowCount()-1
        if row < 1:
            return
        self.annotation_table.removeRow(row)
        self.removeArtists(artists=[self.annotationArtists[-1]])
    
    def removeArtists(artists):
        return


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
        # Add the buttons
        widgetFrame.layout().addWidget(addButton, 1, 0)
        widgetFrame.layout().addWidget(removeButton, 1, 1)
        # Add the exisiting channels and streams to the table
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
        # self.layout_comboBox = {'stream':scomb, 'channel':ccomb}
    
    def updateLayoutComboBox(self):
        all_layouts = self.friend.getAvailableStreams(warning=False)
        all_streams = sorted(set([l[0] for l in all_layouts]))
        all_streams = [s for s in ['Voltage', 'Current','Stimulus'] if s in all_streams]
        all_channels = sorted(set([l[1] for l in all_layouts]))
        for r in range(self.layout_table.rowCount()):
            current_stream = self.layout_table.cellWidget(r, 0).currentText()
            self.layout_table.cellWidget(r, 0).clear() # clear all streams
            self.layout_table.cellWidget(r, 0).addItems(all_streams) # add back all streams
            if current_stream in all_streams: # Set original stream back
                self.layout_table.cellWidget(r,0).setCurrentIndex(all_streams.index(current_stream))
            
            current_channel = self.layout_table.cellWidget(r, 1).currentText()
            self.layout_table.cellWidget(r, 1).clear() # clear all channels
            self.layout_table.cellWidget(r, 1).addItems(all_channels)
            if current_channel in all_channels:
                self.layout_table.cellWidget(r, 1).setCurrentIndex(all_channels.index(current_channel))


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
        
    # -------- Curve fitting tools -------------------------------------------
    def curvefitWidget(self):
        """This returns the initialized curve fitting widget
        """
        # initialize the widget
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("CurveFittingWidgetFrame"))
        widgetFrame.layout().setSpacing(10)
        # Curve fitting button
        fitButton = QtGui.QPushButton("Curve Fit")
        # Type of curve to fit dropdown box
        curveTypeComboBox = QtGui.QComboBox()
        curveTypeComboBox.addItems(['Exponential', 'Polynomial', 'Power'])
        # Center and scale
        # csCheckBox = QtGui.QCheckBox("Center and scale")
        # Report box
        cfReportBox = QtGui.QLabel("Curve Fit Results")
        cfReportBox.setStyleSheet("background-color: white")
        cfReportBox.setWordWrap(True)
        
        # Arrange the widget
        widgetFrame.layout().addWidget(fitButton, 0, 0, 1,3)
        widgetFrame.layout().addWidget(curveTypeComboBox, 1, 0, 1, 3)
        
        # Settings of curve fitting
        self.setCFSettingWidgetFrame(widgetFrame, cfReportBox, curveTypeComboBox.currentText())
        
        # Refresh setting section when cf type changed
        curveTypeComboBox.currentIndexChanged.connect(lambda: self.setCFSettingWidgetFrame(widgetFrame, cfReportBox, curveTypeComboBox.currentText()))
        
        # Summary box behavior
        fitButton.clicked.connect(lambda : self.curveFit(curveTypeComboBox.currentText(), cfReportBox))#, csCheckBox.checkState()))

        return widgetFrame
        
    def setCFSettingWidgetFrame(self, widgetFrame, cfReportBox, curve):
        # Remove everthing at and below the setting rows: rigid setting
        nrows = widgetFrame.layout().rowCount()
        if nrows>2:
            for row in range(3,nrows):
                for col in range(widgetFrame.layout().columnCount()):
                    currentItem = widgetFrame.layout().itemAtPosition(row, col)
                    if currentItem is not None:
                        if currentItem.widget() is not cfReportBox:
                            currentItem.widget().deleteLater()
                        else:
                            widgetFrame.layout().removeItem(currentItem)

        # Get the setting table again
        self.getCFSettingTable(curve)
        for key, val in self.CFsettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1])
        # Report box
        widgetFrame.layout().addWidget(cfReportBox, widgetFrame.layout().rowCount(), 0, 1, 3)
        return
        
    def getCFSettingTable(self, curve):
        if curve == 'Exponential':
            eqLabel = QtGui.QLabel("Equation:")
            eqComboBox = QtGui.QComboBox()
            eqComboBox.addItems(['a*exp(b*x)+c','a*exp(b*x)', 'a*exp(b*x)+c*exp(d*x)']) 
            self.CFsettingTable = {(3,0): eqLabel, (3,1): eqComboBox}
        elif curve == 'Power':
            eqLabel = QtGui.QLabel("Equation")
            eqComboBox = QtGui.QComboBox()
            eqComboBox.addItems(['a*x^b', 'a*x^b+c'])
            self.CFsettingTable = {(3,0): eqLabel, (3,1): eqComboBox}
        elif curve == 'Polynomial':
            degLabel = QtGui.QLabel("Degree:")
            degText = QtGui.QLineEdit("1")
            self.CFsettingTable = {(3,0):degLabel, (3,1): degText}
                        
    def curveFit(self, curve, cfReportBox):#, centerAndScale):
        # get view
        currentView = [0, 0]
        p = self.friend.graphicsView.getItem(row=currentView[0], col=currentView[1])
        # clear previous fit artists
        count_fit = 0
        for k, a in enumerate(p.listDataItems()):
            if 'fit' in a.name():
                count_fit = count_fit + 1
 
        if len(p.listDataItems())-count_fit > 1:
            cfReportBox.setText("Can only fit curve at 1 trace at a time. Please select only 1 trace")
            return
        
        # Get the plotted data
        d = p.listDataItems()[0]        
                    
        if self.friend.viewRegionOn: # fit between region selection
            xdata, ydata = spk_window(d.xData, d._ts, self.friend.selectedRange), spk_window(d.yData, d._ts, self.friend.selectedRange)
        else: # fit within the current view
            xdata, ydata = spk_window(d.xData, d._ts, p.viewRange()[0]), spk_window(d.yData, d._ts, p.viewRange()[0])
            
        # remove baseline: -= and += can be tricky. Use carefully
        xoffset = xdata[0]
        xdata = xdata - xoffset
        yoffset = min(ydata)
        ydata = ydata - yoffset
            
        f0 = None
        if curve == 'Exponential':
            eqText = self.CFsettingTable[(3,1)].currentText()
            if eqText == 'a*exp(b*x)+c':
                f0 = lambda x, a, b, c: a*np.exp(b*x)+c
                p0 = [max(ydata), -0.015 if ydata[-1]<ydata[0] else 0.025, 0]
                # bounds = [(-max(abs(ydata))*1.1, -10, -np.inf),  (max(abs(ydata))*1.1, 10, np.inf)]
                ptext = ['a','b','c']
            elif eqText == 'a*exp(b*x)':
                f0 = lambda x, a, b: a*np.exp(b*x)
                p0 = [max(ydata), -0.015 if ydata[-1]<ydata[0] else 0.025]
                # bounds = [(-max(abs(ydata))*1.1, -10), (max(abs(ydata))*1.1, 10)]
                ptext = ['a','b']
            elif eqText == 'a*exp(b*x)+c*exp(d*x)':
                f0 = lambda x, a, b, c, d: a*np.exp(b*x) + c*np.exp(d*x)
                p0 = [max(ydata), -0.015 if ydata[-1]<ydata[0] else 0.025, max(ydata), -0.015 if ydata[-1]<ydata[0] else 0.025]
                # bounds = [(-max(abs(ydata))*1.1, -10, -max(abs(ydata))*1.1, -10),  (max(abs(ydata))*1.1, 10, max(abs(ydata))*1.1, 10)]
                ptext = ['a','b','c','d']
        elif curve == 'Power':
            eqText = self.CFsettingTable[(3,1)].currentText()
            if eqText == 'a*x^b':
                f0 = lambda x, a, b: a*(x**b)
                p0 = np.ones(2,)
                # bounds = [(-np.inf, -np.inf), (np.inf, np.inf)]
                ptext = ['a','b']
            elif eqText == 'a*x^b+c':
                f0 = lambda x, a, b, c: a*(x**b)+c
                p0 = np.ones(3,)
                # bounds = [(-np.inf, -np.inf, -np.inf), (np.inf, np.inf, np.inf)]
                ptext = ['a','b','c']
        elif curve == 'Polynomial':
            eqText = self.CFsettingTable[(3,1)].text()
            def f0(x, *p):
                poly = 0.
                for i, n in enumerate(p):
                    poly += n * x**i
                return poly
            deg = int(eqText)
            p0 = np.ones(deg+1, )
            ptext = ['p'+str(i) for i in range(deg+1)]
            # bounds = [tuple([-np.inf]*deg), tuple([np.inf]*deg)]
            eqText = []
            for m, ppt in enumerate(ptext):
                if m == 0:
                    eqText.append(ptext[0])
                elif m==1:
                    eqText.append(ptext[1] + "*" + "x")
                elif m>=2:
                    if len(ptext)>3:
                        eqText.append("...")
                    eqText.append(ptext[-1] + "*" + "x^{:d}".format(len(ptext)-1))
                    break
                
            eqText = "+".join(eqText)
        
        if f0 is None: # shouldn't go here. For debug only
            raise(ValueError('Unrecognized curve equation %s: %s'%(curve, eqText)))
        
        # Fit the curve
        try:
            popt, pcov = curve_fit(f0, xdata, ydata, p0=p0, method='trf')
        except Exception as err:
            cfReportBox.setText("{}".format(err))
            return
            
        # Generate fitted data
        yfit = f0(xdata, *popt)
        # Do some calculations on the fitting before reporting
        SSE = np.sum((yfit - ydata)**2)
        RMSE = np.sqrt(SSE/len(yfit))
        SS_total = np.poly1d(np.polyfit(xdata, ydata, 1))
        SS_total = np.sum((SS_total(xdata) - ydata)**2)
        R_sq = 1.0 - SSE / SS_total
        R_sq_adj = 1.0 - (SSE/(len(xdata)-len(p0))) / (SS_total/(len(xdata)-1))# Adjusted R_sq
        # Draw the fitted data
        for a in p.listDataItems():
            if 'fit' in a.name():
                a.setData(xdata+xoffset, yfit+yoffset)
            else:
                p.plot(xdata+xoffset, yfit+yoffset, pen='r', name='fit: ' + eqText)
        # Report the curve fit
        final_text = "Model: {}\nEquation:\n\t{}\n".format(curve, eqText)
        final_text += "Parameters:\n"
        for ppt, coeff in zip(ptext, popt): # report fitted parameters
            final_text += "\t" + ppt + ": " + "{:.4g}".format(coeff) + "\n"
        if curve == 'Exponential':
            final_text += "Time Constants:\n"
            if eqText in ['a*exp(b*x)+c', 'a*exp(b*x)']:
                tau = -1.0/popt[1]
                final_text += "\ttau: " + "{:.4g} ms".format(tau) + "\n"
            elif eqText == 'a*exp(b*x)+c*exp(d*x)':
                tau1, tau2 = -1.0/popt[1], -1.0/popt[3]
                final_text += "\ttau1: " + "{:.4g} ms".format(tau1) + "\n"
                final_text += "\ttau2: " + "{:.4g} ms".format(tau2) + "\n"
        
        final_text += "\nGoodness of fit:\n\tSSE: {:.4g}\n\tR-squared: {:.4g}\n\tAdjusted R-squared: {:.4g}\n\tRMSE: {:.4g}".format(SSE, R_sq, R_sq_adj, RMSE)
        cfReportBox.setText(final_text)

    # ------- Analysis tools -------------------------------------------------
    def eventDetectionWidget(self):
        """This returns the initialized event detection widget"""
        # Initalize the widget
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("EventDetectionWidgetFrame"))
        widgetFrame.layout().setSpacing(10)
        # Detect spikes button
        detectButton = QtGui.QPushButton("Detect")
        # Type of Event detection to run
        # Summary box
        detectReportBox = QtGui.QLabel("Event Detection Results")
        detectReportBox.setStyleSheet("background-color: white")
        detectReportBox.setWordWrap(True)
        # Even type selection
        eventTypeComboBox = QtGui.QComboBox()
        eventTypeComboBox.addItems(['Action Potential', 'Cell Attached Spike', 'EPSP', 'IPSP', 'EPSC','IPSC'])
        # Asking to draw on the plot
        drawCheckBox = QtGui.QCheckBox("Mark Events")
        drawCheckBox.stateChanged.connect(self.clearEvents)

        # Arrange the widget
        widgetFrame.layout().addWidget(detectButton, 0, 0, 1, 3)
        widgetFrame.layout().addWidget(eventTypeComboBox, 1, 0, 1, 1)
        widgetFrame.layout().addWidget(drawCheckBox, 1, 1, 1,1)

        # Settings of event detection
        self.setEDSettingWidgetFrame(widgetFrame, detectReportBox, eventTypeComboBox.currentText())

        # Refresh setting section when event type changed
        eventTypeComboBox.currentIndexChanged.connect(lambda: self.setEDSettingWidgetFrame(widgetFrame, detectReportBox, eventTypeComboBox.currentText()))
        # Summary box behavior
        detectButton.clicked.connect(lambda : self.detectEvents(eventTypeComboBox.currentText(), detectReportBox, drawCheckBox.checkState()))

        return widgetFrame

    def setEDSettingWidgetFrame(self, widgetFrame, detectReportBox, event):
        # Remove everthing at and below the setting rows: rigid setting
        nrows = widgetFrame.layout().rowCount()
        if nrows>2:
            for row in range(2,nrows):
                for col in range(widgetFrame.layout().columnCount()):
                    currentItem = widgetFrame.layout().itemAtPosition(row, col)
                    if currentItem is not None:
                        if currentItem.widget() is not detectReportBox:
                            currentItem.widget().deleteLater()
                        else:
                            widgetFrame.layout().removeItem(currentItem)

        # Get the setting table again
        self.getEDSettingTable(event)
        for key, val in self.EDsettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1])
        # Report box
        widgetFrame.layout().addWidget(detectReportBox, widgetFrame.layout().rowCount(), 0, 1, 3)

    def getEDSettingTable(self, event='Action Potential'):
        """return a table for settings of each even detection"""
        if event == 'Action Potential':
            minHeightLabel = QtGui.QLabel("Min Height")
            minHeightLabel.setToolTip("Minimum amplitude of the AP")
            minHeightTextEdit = QtGui.QLineEdit("-10")
            minHeightUnitLabel = QtGui.QLabel("mV")
            minDistLabel = QtGui.QLabel("Min Dist")
            minDistLabel.setToolTip("Minimum distance between detected APs")
            minDistTextEdit = QtGui.QLineEdit("1")
            minDistUnitLabel = QtGui.QLabel("ms")
            self.EDsettingTable = {(3,0): minHeightLabel, (3,1): minHeightTextEdit,
                            (3,2): minHeightUnitLabel, (4,0):minDistLabel,
                            (4,1): minDistTextEdit, (4,2): minDistUnitLabel}
        elif event in ['EPSP', 'IPSP', 'EPSC','IPSC']:
            ampLabel = QtGui.QLabel("Amplitude")
            ampLabel.setToolTip("Minimum amplitude of the event")
            ampTextEdit =  QtGui.QLineEdit("0.5")
            ampUnitLabel = QtGui.QLabel("mV")
            riseTimeLabel = QtGui.QLabel("Rise Time")
            riseTimeLabel.setToolTip("Rise time of PSP template")
            riseTimeTextEdit = QtGui.QLineEdit("1")
            riseTimeUnitLabel = QtGui.QLabel("ms")
            decayTimeLabel = QtGui.QLabel("Decay Time")
            decayTimeLabel.setToolTip("Decay time of the PSP template")
            decayTimeTextEdit =  QtGui.QLineEdit("4")
            decayTimeUnitLabel = QtGui.QLabel("ms")
            criterionLabel = QtGui.QLabel("Criterion")
            criterionLabel.setToolTip("Detection statistical criterion: \n'se': standard error\n'corr': correlation")
            criterionTextEdit =  QtGui.QLineEdit("se")
            criterionUnitLabel = QtGui.QLabel("")
            threshLabel = QtGui.QLabel("Threshold")
            threshLabel.setToolTip("Threshold of statistical criterion")
            threshTextEdit =  QtGui.QLineEdit("3")
            threshUnitLabel = QtGui.QLabel("")
            stepLabel = QtGui.QLabel("Step")
            stepLabel.setToolTip("Step size to convolve the template with the trace")
            stepTextEdit =  QtGui.QLineEdit("20")
            stepUnitLabel = QtGui.QLabel("")

            self.EDsettingTable = {(3,0):ampLabel, (3,1):ampTextEdit, (3,2):ampUnitLabel,
                                   (4,0):riseTimeLabel, (4,1):riseTimeTextEdit, (4,2):riseTimeUnitLabel,
                                   (5,0):decayTimeLabel, (5,1):decayTimeTextEdit, (5,2):decayTimeUnitLabel,
                                   (6,0):criterionLabel, (6,1):criterionTextEdit, (6,2):criterionUnitLabel,
                                   (7,0):threshLabel, (7,1):threshTextEdit, (7,2):threshUnitLabel,
                                   (8,0):stepLabel, (8,1):stepTextEdit, (8,2):stepUnitLabel
                                   }


        elif event == 'Cell Attached Spike':
            minHeightLabel = QtGui.QLabel("Min Height")
            minHeightLabel.setToolTip("Minimum amplitude of the spike")
            minHeightTextEdit = QtGui.QLineEdit("30")
            minHeightUnitLabel = QtGui.QLabel("pA")
            
            maxHeightLabel = QtGui.QLabel("Min Height")
            maxHeightLabel.setToolTip("Minimum amplitude of the spike")
            maxHeightTextEdit = QtGui.QLineEdit("300")
            maxHeightUnitLabel = QtGui.QLabel("pA")
            
            minDistLabel = QtGui.QLabel("Min Dist")
            minDistLabel.setToolTip("Minimum distance between detected spikes")
            minDistTextEdit = QtGui.QLineEdit("10")
            minDistUnitLabel = QtGui.QLabel("ms")
            
            basefiltLabel = QtGui.QLabel("Filter Window")
            basefiltLabel.setToolTip("median filter preprocessing window")
            basefiltTextEdit = QtGui.QLineEdit("20")
            basefiltUnitLabel = QtGui.QLabel("ms")
            
            self.EDsettingTable = {(3,0): minHeightLabel, (3,1): minHeightTextEdit, (3,2): minHeightUnitLabel, 
                                   (4,0): maxHeightLabel, (4,1): maxHeightTextEdit, (4,2): maxHeightUnitLabel,
                                   (5,0):minDistLabel, (5,1): minDistTextEdit, (5,2): minDistUnitLabel,
                                   (6,0):basefiltLabel, (6,1): basefiltTextEdit, (6,2): basefiltUnitLabel
                                   }
        else:
            raise(ValueError('Unrecognized event type %s'%(event)))

    def detectEvents(self, event='Action Potential', detectReportBox=None, drawEvents=False, *args, **kwargs):
        self.detectedEvents.append(event)
        if event == 'Action Potential':
            msh = float(self.EDsettingTable[(3,1)].text())
            msd = float(self.EDsettingTable[(4,1)].text())
            self.detectAPs(detectReportBox, drawEvents, msh, msd)
        elif event in ['EPSP', 'IPSP', 'EPSC', 'IPSC']:
            amp = float(self.EDsettingTable[(3,1)].text())
            riseTime = float(self.EDsettingTable[(4,1)].text())
            decayTime = float(self.EDsettingTable[(5,1)].text())
            criterion = self.EDsettingTable[(6,1)].text()
            thresh = float(self.EDsettingTable[7,1].text())
            step = float(self.EDsettingTable[(8,1)].text())
            self.detectPSPs(detectReportBox, drawEvents, event, riseTime, decayTime, amp, step, criterion, thresh)
        elif event == 'Cell Attached Spike':
            msh = float(self.EDsettingTable[(3,1)].text())
            maxsh = float(self.EDsettingTable[(4,1)].text())
            msd = float(self.EDsettingTable[(5,1)].text())
            basefilt = float(self.EDsettingTable[(6,1)].text())
            self.detectCellAttachedSpikes(detectReportBox, drawEvents, msh, msd, basefilt, maxsh)
            

    def clearEvents(self, checked, eventTypes=None, which_layout=None):
        """Wraps removeEvent. Clear all event types if not specified event
        type. Connected to checkbox state"""
        if checked or not self.detectedEvents:
            return

        if not eventTypes:
            eventTypes = self.detectedEvents

        if isinstance(eventTypes, str):
            eventTypes = [eventTypes]

        for evt in eventTypes:
            self.friend.removeEvent(info=[evt], which_layout=which_layout)
            self.detectedEvents.remove(evt)

    def detectAPs(self, detectReportBox, drawEvent=False, msh=-10, msd=1):
        """detectAPs(detectReportBox, drawEvent, 'additional settings',...)"""
        if not self.friend.index or len(self.friend.index)>1:
            detectReportBox.setText("Can only detect spikes in one episode at a time")

        zData = self.friend.episodes['Data'][self.friend.index[-1]]
        ts = zData.Protocol.msPerPoint
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]
        final_label_text = ""
        for c, Vs in zData.Voltage.items():
            Vs = spk_window(Vs, ts,selectedWindow, t0=0)
            num_spikes, spike_time, spike_heights = spk_count(Vs, ts, msh=msh, msd=msd)
            final_label_text = final_label_text + c + " : \n"
            final_label_text = final_label_text + "  # spikes: " + str(num_spikes) + "\n"
            final_label_text = final_label_text + "  mean ISI: "
            final_label_text += "{:0.2f}".format(np.mean(np.diff(spike_time))) if len(spike_time)>1 else "NaN"
            final_label_text += "\n"
            # Draw event markers
            if drawEvent:
                if selectedWindow[0] is not None:
                    spike_time += selectedWindow[0]
                self.friend.drawEvent(spike_time, which_layout = ['Voltage', c], info=[self.detectedEvents[-1]])

        detectReportBox.setText(final_label_text[:-1])

    def detectPSPs(self, detectReportBox, drawEvent=False, event='EPSP', riseTime=1, decayTime=4, amp=1, step=20, criterion='se', thresh=3.0):
        if not self.friend.index or len(self.friend.index)>1:
            detectReportBox.setTexxt("Can only detect spikes in one episode at a time")

        zData = self.friend.episodes['Data'][self.friend.index[-1]]
        ts = zData.Protocol.msPerPoint
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]
        final_label_text = ""
        if event in ['EPSP', 'IPSP']:
            stream = 'Voltage'
        else: # ['EPSC', 'IPSC']
            stream = 'Current'

        # Get events
        for c, S in getattr(zData, stream).items():
            S = spk_window(S, ts, selectedWindow, t0=0)
            event_time, pks, _, _ = detectPSP_template_matching(S, ts, event=event, \
                                            w=200, tau_RISE=riseTime, tau_DECAY=decayTime, \
                                            mph=amp, step=step, criterion=criterion, thresh=thresh)
            final_label_text = final_label_text + c + ": \n"
            final_label_text = final_label_text + "  # " + event + ": " + str(len(event_time)) +  "\n"
            final_label_text += "  mean IEI: "
            final_label_text += "{:0.2f}".format(np.mean(np.diff(event_time))) if len(event_time)>1 else "NaN"
            final_label_text += "\n"
            # Draw event markers
            if drawEvent:
                if selectedWindow[0] is not None:
                    event_time += selectedWindow[0]
                self.friend.drawEvent(event_time, which_layout = [stream, c], info=[self.detectedEvents[-1]])

        detectReportBox.setText(final_label_text[:-1])
        
    def detectCellAttachedSpikes(self, detectReportBox, drawEvent=False, msh=30, msd=10, basefilt=20, maxsh=300):
        if not self.friend.index or len(self.friend.index)>1:
            detectReportBox.setTexxt("Can only detect spikes in one episode at a time")
            
        zData = self.friend.episodes['Data'][self.friend.index[-1]]
        ts = zData.Protocol.msPerPoint
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]
        final_label_text = ""
        for c, Is in zData.Current.items():
            Is = spk_window(Is, ts, selectedWindow, t0=0)
            num_spikes, spike_time, spike_heights = detectSpikes_cell_attached(Is, ts, msh=msh, msd=msd, \
                                                                               basefilt=basefilt, maxsh=maxsh, removebase=False)
            final_label_text = final_label_text + c + " : \n"
            final_label_text = final_label_text + "  # spikes: " + str(num_spikes) + "\n"
            final_label_text = final_label_text + "  mean ISI: "
            final_label_text += "{:0.2f}".format(np.mean(np.diff(spike_time))) if len(spike_time)>1 else "NaN"
            final_label_text += "\n"
            # Draw event markers
            if drawEvent:
                if selectedWindow[0] is not None:
                    spike_time += selectedWindow[0]
                self.friend.drawEvent(spike_time, which_layout = ['Current', c], info=[self.detectedEvents[-1]])

        detectReportBox.setText(final_label_text[:-1])

    # ------- Other utilities ------------------------------------------------
    def replaceWidget(self, widget=None, index=0):
        old_widget = self.accWidget.takeAt(index)
        self.accWidget.addItem(title=old_widget.title(), widget=widget, collapsed=old_widget._collapsed, index=index)

    def sizeHint(self):
        """Helps with initial dock window size"""
        return QtCore.QSize(self.friend.frameGeometry().width()/4.95, 20)

# %%
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
            bool_old_episode = False
        else:
            bool_old_episode = self.episodes['Name'] == episodes['Name']

        # reset the grpahicsview if user not keeping traces from older dataset
        if not self.keepOther and not bool_old_episode:
            self.getDataViewRange() # Get the data range before clearing
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
                self.episodes['Data'][i] = NeuroData(dataFile=self.episodes['Dirs'][i], old=old, infoOnly=False, getTime=True)
            self._loaded_array.append(i)
            # Draw the episode
            self.drawEpisode(self.episodes['Data'][i], info=(self.episodes['Name'], self.episodes['Epi'][i], i))

        # Remove episodes
        for j in index_remove:
            self.removeEpisode(info=(self.episodes['Name'], self.episodes['Epi'][j]))

        # print(self.index)
        if not bool_old_episode:
            self.reissueArtists() # artists have been cleared. Add it back
            self.setDataViewRange(viewMode='reset')
        else:
            self.setDataViewRange(viewMode='keep')

        # Update the companion, Toolbox: Layout
        if self.dockPanel.accWidget.widgetAt(1).objectName() == 'Banner':
            # Replace the banner widget with real widget
            layoutWidget = self.dockPanel.layoutWidget()
            self.dockPanel.replaceWidget(widget=layoutWidget, index=1)
        
        if not bool_old_episode:
            self.dockPanel.updateLayoutComboBox()
            
        # print(self.layout)

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

            Y = getattr(zData, l[0])[l[1]]
            if self.isnull and self.nullRange is not None:
                Y = Y - self.getNullBaseline(Y, zData.Protocol.msPerPoint)
            cl = p.plot(x=zData.Time, y=Y, pen=pen, name=pname)
            cl._ts = zData.Protocol.msPerPoint # save the sampling rate to dataitem

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

    def drawEvent(self, eventTime, which_layout, info=None, color='r', linesize=None, drawat='bottom'):
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
        if drawat == 'bottom':
            ypos_0, ypos_1 = yRange[0], yRange[0] + linesize
        else: # top
            ypos_0, ypos_1 = yRange[1], yRange[1] - linesize

        # Cell + Episode + Even type + Stream + Channel
        pname = ".".join(info)+'.'+l[0]+'.'+l[1]
        for t in eventTime:
            p.plot(x=[t,t], y=[ypos_0, ypos_1], pen=color, name=pname)

    def removeEvent(self, info=None, which_layout=None):
        """Remove one event type from specified layout (which_layout) or all
        members of thelayout, if not specifying which_layout"""
        if not info:
            return

        for l in self.layout:
            if which_layout and (which_layout[0] not in l or which_layout[1] not in l):
                continue # specified which_layout but does not match current iteration
            # Get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            # Remove based on item name
            pname = ".".join(info)+'.'+l[0]+'.'+l[1]
            for k, a in enumerate(p.listDataItems()):
                if pname in a.name(): # matching
                    p.removeItem(a)

            if which_layout: # if specified which_layout
                break

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

    def setDataViewRange(self, viewMode='default', xRange=None, yRange=None):
        # print('view range %s'%self.viewMode)
        self.viewMode = viewMode
        if not self.viewRange.keys():
            self.viewMode = 'default'
                
        default_yRange_dict = {'Voltage':(-100, 40), 'Current': (-500, 500),
                          'Stimulus':(-500, 500)}

        # Loop through all the subplots
        for n, l in enumerate(self.layout):
            # get viewbox
            p = self.graphicsView.getItem(row=l[2], col=l[3])
            #print(a.tickValues())
            if self.viewMode == 'default':
                # Make everything visible first
                p.autoRange()
                yRange = default_yRange_dict.get(l[0])
                p.setYRange(yRange[0], yRange[1], padding=0)
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
                # no chnage in viewRange
                return
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
                    self.graphicsView.artists.remove(r)
            self.graphicsView.artists = [r for r in self.graphicsView.artists if r]

        elif (not self.viewRegionOn and checked) or cmd == 'add': # add
            # Update current data view range
            print(rng)
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
                mousePoint = p.vb.mapSceneToView(pos)
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
        # Do the plotting once all the necessary materials are gathered
        if arrangement == 'overlap':
            PlotTraces(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors, 
                       fig_size=(options['figSizeW'], options['figSizeH']), adjustFigW=options['figSizeWMulN'], adjustFigH=options['figSizeHMulN'],
                       dpi=options['dpi'], nullRange=None if not self.isnull else self.nullRange, annotation=options['annotation'],
                       setFont=options['fontName'], fontSize=options['fontSize'], linewidth=options['linewidth'], monoStim=options['monoStim'],
                       stimReflectCurrent=options['stimReflectCurrent'])
        elif arrangement == 'concatenate':
            PlotTracesConcatenated(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors,
                                 dpi=options['dpi'], fig_size=(options['figSizeW'], options['figSizeH']), nullRange=None if not self.isnull else self.nullRange, hSpaceType=options['hSpaceType'], hFixedSpace=options['hFixedSpace'],
                                 adjustFigW= options['figSizeWMulN'],adjustFigH= options['figSizeHMulN'], annotation=options['annotation'], 
                                 setFont=options['fontName'], fontSize=options['fontSize'], linewidth=options['linewidth'], monoStim=options['monoStim'],
                                 stimReflectCurrent=options['stimReflectCurrent'])
        elif arrangement in ['vertical', 'horizontal', 'channels x episodes', 'episodes x channels']:
            PlotTracesAsGrids(self.episodes, self.index, viewRange, saveDir=options['saveDir'], colorfy=self._usedColors,
                                 dpi=options['dpi'], fig_size=(options['figSizeW'], options['figSizeH']),adjustFigW=options['figSizeWMulN'],adjustFigH=options['figSizeHMulN'],
                                 nullRange=None if not self.isnull else self.nullRange, annotation=options['annotation'],setFont=options['fontName'], fontSize=options['fontSize'], 
                                 scalebarAt=options['scalebarAt'], gridSpec=options['gridSpec'], linewidth=options['linewidth'], monoStim=options['monoStim'],
                                 stimReflectCurrent=options['stimReflectCurrent'])
        else:
            raise(ValueError('Unrecognized arragement:{}'.format(arragement)))

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
    
    episodes = {'Drug Name': [''], 'Epi': ['S1.E33', 'S1.E34', 'S1.E35'], 
    'Duration': [40000,40000,40000], 'Drug Level': [1,1,1], 'Time': ['25:06','25:54','26:43'], 
    'Name': 'Neocortex E.09Jun16', 'Drug Time': ['24:27','25:15','26:03'], 'Sampling Rate': [0.1,0.1,0.1], 
    'Comment': ['TTL3: SIU train','',''], 
    'Dirs': ['D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E33.dat',
             'D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E34.dat',
             'D:/Data/Traces/2016/04.April/Data 21 Apr 2016/NeocortexCA F.21Apr16.S1.E35.dat']}
    index = [0]
    app = QtGui.QApplication(sys.argv)
    w = ScopeWindow(hideDock=False, layout=[['Voltage', 'A', 0, 0],  ['Current', 'A', 1, 0]])
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
