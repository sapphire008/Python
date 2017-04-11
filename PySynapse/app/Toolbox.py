# -*- coding: utf-8 -*-
"""
Created: Fri Apr 7 22:12:21 2017

Side dock toolbox for Scope window.
Methods that call self.friend assumes that the Scope window is already running (instance created)

@author: Edward
"""

# Global variables
old = True # load old data format
colors = ['#1f77b4','#ff7f0e', '#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'] # tableau10, or odd of tableau20

from app.AccordionWidget import AccordionWidget
from app.Annotations import *
from util.spk_util import *
from util.ImportData import NeuroData


from pdb import set_trace

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


class Toolbox(QtGui.QWidget):
    """Collapsible dock widget that displays settings and analysis results for the Scope window
    """
    _widget_index = 0 # Keep track of position of the widget added
    _sizehint = None
    # used for replace formula variables, total allow 52 replacements, from a-zA-Z
    _newvarsList = [chr(i) for i in 65+np.arange(26)]+[chr(i) for i in 97+np.arange(26)]

    def __init__(self, parent=None, friend=None):
        super(Toolbox, self).__init__(parent)
        self.parent = parent
        self.friend = friend
        self.detectedEvents = []
        self.annotationArtists = [] # list of IDs
        self.setupUi()

    def setupUi(self):
        self.verticalLayout = self.parent.layout()
        # self.setLayout(self.verticalLayout)
        self.accWidget = AccordionWidget(self)

        # Add various sub-widgets, which interacts with Scope, a.k.a, friend
        self.accWidget.addItem("Arithmetic", self.arithmeticWidget(), collapsed=True)
        self.accWidget.addItem("Annotation", self.annotationWidget(), collapsed=False)
        self.accWidget.addItem("Channels", self.layoutWidget(), collapsed=True)
        self.accWidget.addItem("Curve Fit", self.curvefitWidget(), collapsed=True)
        self.accWidget.addItem("Event Detection", self.eventDetectionWidget(), collapsed=True)

        self.accWidget.setRolloutStyle(self.accWidget.Maya)
        self.accWidget.setSpacing(0) # More like Maya but I like some padding.
        self.verticalLayout.addWidget(self.accWidget)

    # <editor-fold desc="Trace arithmetic tools">
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
                        # arithReportBox.setText(d + " is not a valid episode")
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

        # parse formula
        if "{" in formula:
            # separate each formula
            formula = formula.replace("{","").replace("}","")
            formula = formula.split(";")
        else:
            formula = [formula]

        # parse each formula
        for f0 in formula:
            if ":" in f0: # has range. Assume each formula hsa only 1 range
                # set_trace()
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

    # </editor-fold>

    # <editor-fold desc="Annotation widget">
    # ----------- Annotation widget ------------------------------------------
    def annotationWidget(self):
        """Adding annotation items on the graph"""
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("AnnotationWidgetFrame"))
        widgetFrame.layout().setSpacing(0)
        self.setAnnotationTable()
        addButton = QtGui.QPushButton("Add") # Add an annotation object
        addButton.clicked.connect(self.addAnnotationRow)
        removeButton = QtGui.QPushButton("Remove") # Remove an annotation object
        removeButton.clicked.connect(self.removeAnnotationRow)
        editButton = QtGui.QPushButton("Edit") # Edit an annotation object
        editButton.clicked.connect(self.editAnnotationArtist)
        # Add the buttons
        widgetFrame.layout().addWidget(addButton, 1, 0)
        widgetFrame.layout().addWidget(removeButton, 1, 1)
        widgetFrame.layout().addWidget(editButton, 1, 2)
        # Add the exisiting annotations to the table
        widgetFrame.layout().addWidget(self.annotation_table, 2, 0, 1, 3)

        return widgetFrame

    def setAnnotationTable(self):
        """"(Re)initialize the annotation table"""
        self.annotation_table = QtGui.QTableWidget(0, 2)
        self.annotation_table.verticalHeader().setVisible(False)
        # self.annotation_table.horizontalHeader().setVisible(False)
        self.annotation_table.setHorizontalHeaderLabels(['Artist', 'Notes'])
        self.annotation_table.horizontalHeader().setResizeMode(QtGui.QHeaderView.Stretch)
        self.annotation_table.horizontalHeader().highlightSections()
        self.annotation_table.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        # Change style sheet a little for Windows 10
        if sys.getwindowsversion().major == 10: # fix the problem that in Windows 10, bottom border of header is not displayed
            self.annotation_table.setStyleSheet("""
                QHeaderView::section{
                border-top:0px solid #D8D8D8;
                border-left:0px solid #D8D8D8;
                border-right:1px solid #D8D8D8;
                border-bottom: 1px solid #D8D8D8;
                background-color:white;
                padding:4px;
                font: bold 10px;}
                QTableCornerButton::section{
                border-top:0px solid #D8D8D8;
                border-left:0px solid #D8D8D8;
                border-right:1px solid #D8D8D8;
                border-bottom: 1px solid #D8D8D8;
                background-color:white;}""")
        else:
            self.annotation_table.setStyleSheet("""QHeaderView::section{font: bold 10px;}""")

        self.annotation_table.itemChanged.connect(self.onArtistChecked)

    def addAnnotationRow(self):
        """Add annotation into teh table"""
        # Pop up the annotation settings window to get the properties of the annotation settings
        annSet = AnnotationSetting()
        # annotationSettings.show()
        def append_num_to_repeated_str(l, s, recycle=False):
            # rep = sum(1 if s in a else 0 for a in l) # count
            rep = [a for a in l if s in a] # gather
            nums = [int(r[len(s):]) for r in rep]

            # Extract the numbers appended
            if not recycle:
                if isinstance(nums, list) and not nums:
                    nums = [0]
                s = s + str(max(nums)+1)
            else: # smallest avaiable number starting from 1
                if isinstance(nums, list) and not nums:
                    s = s + '1'
                else:
                    s = s + str(min(list(set(range(1, max(nums)+1)) - set(nums))))

            l = l + [s]

            return l, s

        if annSet.exec_(): # Need to wait annotationSettings has been completed
            if annSet.artist.keys(): # if properties are properly specified, draw the artist
                # Set Artist table item
                self.annotationArtists, artist_name = append_num_to_repeated_str(self.annotationArtists, annSet.type)
                AT_item = QtGui.QTableWidgetItem(artist_name)
                AT_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable) # can be checked | can be selected
                AT_item.setCheckState(QtCore.Qt.Checked) # newly added items are always checked
                # Add to the table
                row = self.annotation_table.rowCount()
                self.annotation_table.blockSignals(True)
                self.annotation_table.insertRow(row)
                self.annotation_table.setItem(row, 0, AT_item)
                self.annotation_table.blockSignals(False)
                # Get artist property
                artistProperty = annSet.artist
                artistProperty['type'] = annSet.type
                artistProperty['position'] = row
                artistProperty['name'] = artist_name
                AT_item._artistProp = artistProperty
                # Draw the artist
                self.drawAnnotationArtist(artist=artistProperty)

    def removeAnnotationRow(self):
        numRows = self.annotation_table.rowCount()
        if numRows < 1:
            return
        row = self.annotation_table.currentRow()
        # print(row)
        if row is None or row < 0: # if no currently selected row, remove the last row / annotation item
            row = numRows - 1

        item = self.annotation_table.item(row, 0)
        # print(self.annotationArtists)
        # print('removing: %s'%item._artistProp['name'])
        self.annotationArtists.remove(item._artistProp['name']) # something would be wrong if there is no artist of this name to remove
        # print(self.annotationArtists)
        self.eraseAnnotationArtist(artist=item._artistProp)
        self.annotation_table.removeRow(row)

    def onArtistChecked(self, item=None):
        """Respond if click state was changed for pre-existing artists"""
        if item.column() > 0: # editing comments, ignore
            return

        if item.checkState() == 0: # remove the artist if it is present
            self.eraseAnnotationArtist(artist=item._artistProp)
        else: # assume checkstate > 0, likely 2, redraw the artist
            self.drawAnnotationArtist(artist=item._artistProp)

    def drawAnnotationArtist(self, artist=None, which_layout=None):
        print('draw annotation artist')
        if which_layout is None:
            which_layout = self.friend.layout[0]
        if artist['type'] == 'box':
            self.friend.drawBox(artist=artist, which_layout=which_layout)
        elif artist['type'] == 'ttl':
            # Get additional information about TTL from data: a list of OrderedDict
            artist['TTL'] = self.friend.episodes['Data'][self.friend.index[-1]].Protocol.ttlDict # TODO
            # set_trace()

    def eraseAnnotationArtist(self, artist=None, which_layout=None):
        self.friend.removeEvent(info=[artist['name']], which_layout=which_layout, event_type='annotation')

    def editAnnotationArtist(self):
        """ Redraw a modified artist"""
        numRows = self.annotation_table.rowCount()
        if numRows < 1:
            return
        row =self.annotation_table.currentRow()
        if row is None or row < 0: # if no currently selected row, edit the last row
            row = numRows - 1
        # Get the item
        item = self.annotation_table.item(row, 0)
        # Prompt for new information
        annSet = AnnotationSetting(artist=item._artistProp)
        if annSet.exec_(): # Need to wait annotationSettings has been completed
            if annSet.artist.keys(): # Draw new artist
                # Remove the old artist
                self.eraseAnnotationArtist(artist=item._artistProp)
                artistProperty = annSet.artist
                AT_item = QtGui.QTableWidgetItem(artistProperty['name'])
                AT_item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable)  # can be checked | can be selected
                AT_item.setCheckState(QtCore.Qt.Checked)  # newly added items are always checked
                AT_item._artistProp = artistProperty
                self.drawAnnotationArtist(artist=artistProperty)

    def clearAnnotationArtists(self):
        print("clear all artist")

    # </editor-fold>

    # <editor-fold desc="Layout control">
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
        # (Re)initialize the layout table
        self.layout_table = QtGui.QTableWidget(0, 2)
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
        """Called when changing a different dataset"""
        all_layouts = self.friend.getAvailableStreams(warning=False)
        all_streams = sorted(set([l[0] for l in all_layouts]))
        all_streams = [s for s in ['Voltage', 'Current','Stimulus'] if s in all_streams]
        all_channels = sorted(set([l[1] for l in all_layouts]))
        for r in range(self.layout_table.rowCount()):
            current_stream = self.layout_table.cellWidget(r, 0).currentText()
            # IMPORTANT: Need to block the signal from this combobox, otherwise, whatever function connected to this
            # combobox will be called, which we want to avoid
            self.layout_table.cellWidget(r, 0).blockSignals(True)
            self.layout_table.cellWidget(r, 0).clear() # clear all streams
            self.layout_table.cellWidget(r, 0).addItems(all_streams) # add back all streams
            self.layout_table.cellWidget(r, 0).blockSignals(False)
            if current_stream in all_streams: # Set original stream back
                self.layout_table.cellWidget(r,0).setCurrentIndex(all_streams.index(current_stream))

            current_channel = self.layout_table.cellWidget(r, 1).currentText()
            self.layout_table.cellWidget(r, 1).blockSignals(True)
            self.layout_table.cellWidget(r, 1).clear() # clear all channels
            self.layout_table.cellWidget(r, 1).addItems(all_channels)
            self.layout_table.cellWidget(r, 1).blockSignals(False)

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

    # </editor-fold>

    # <editor-fold desc="Curve fitting tools">
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

    # </editor-fold>

    # <editor-fold desc="Analysis tools">
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
        # Remove everything at and below the setting rows: rigid setting
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

    # </editor-fold>


    # <editor-fold desc="Other utilities">
    #------- Other utilities ------------------------------------------------
    def replaceWidget(self, widget=None, index=0):
        old_widget = self.accWidget.takeAt(index)
        self.accWidget.addItem(title=old_widget.title(), widget=widget, collapsed=old_widget._collapsed, index=index)

    def sizeHint(self):
        """Helps with initial dock window size"""
        return QtCore.QSize(self.friend.frameGeometry().width()/4.95, 20)

    # </editor-fold>
