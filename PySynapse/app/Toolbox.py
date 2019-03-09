# -*- coding: utf-8 -*-
"""
Created: Fri Apr 7 22:12:21 2017

Side dock toolbox for Scope window.
Methods that call self.friend assumes that the Scope window is already running (instance created)

@author: Edward
"""

# Global variables
old = True # load old data format
# colors = readini(os.path.join(__location__,'../resources/config.ini'))['colors']
ignoreFirstTTL = True # Ignore the first set of TTL Data when parsing TTL pulse protocols

from app.AccordionWidget import AccordionWidget
from app.Annotations import *
from util.spk_util import *
from util.ImportData import NeuroData

from scipy.signal import butter, filtfilt
from scipy.optimize import curve_fit

from pdb import set_trace

try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtCore.QCoreApplication.translate(context, text, disambig)


class Toolbox(QtWidgets.QWidget):
    """Collapsible dock widget that displays settings and analysis results for the Scope window
    """
    _widget_index = 0 # Keep track of position of the widget added
    _sizehint = None
    # used for replace formula variables, total allow 52 replacements, from a-zA-Z
    _newvarsList = [chr(i) for i in 65+np.arange(26)]+[chr(i) for i in 97+np.arange(26)]
    # Annotatable objects
    def __init__(self, parent=None, friend=None):
        super(Toolbox, self).__init__(parent)
        self.parent = parent
        self.friend = friend
        self.detectedEvents = []
        self.eventArtist = [] # list of IDs
        self.annotationArtists = [] # list of IDs
        self.setupUi()

    def setupUi(self):
        self.verticalLayout = self.parent.layout()
        # self.setLayout(self.verticalLayout)
        self.accWidget = AccordionWidget(self)

        # Add various sub-widgets, which interacts with Scope, a.k.a, friend
        self.accWidget.addItem("Arithmetic", self.arithmeticWidget(), collapsed=True)
        self.accWidget.addItem("Annotation", self.annotationWidget(), collapsed=True)
        self.accWidget.addItem("Channels", self.layoutWidget(), collapsed=True)
        self.accWidget.addItem("Curve Fit", self.curvefitWidget(), collapsed=True)
        self.accWidget.addItem("Event Detection", self.eventDetectionWidget(), collapsed=True)
        self.accWidget.addItem("Filter", self.filterWidget(), collapsed=True)
        self.accWidget.addItem("Function", self.functionWidget(), collapsed=False)

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

        # Apply filtering before calculation
        filtCheckBox = QtGui.QCheckBox("Apply filter before calculation")
        filtCheckBox.setToolTip('Apply a filter, defined in the "Filter" tool onto each episode, before doing any calculation')

        # Formula
        formulaTextBox = QtGui.QLineEdit()
        formulaTextBox.setPlaceholderText("Formula")
        Tooltips = "Examples:\n"
        Tooltips += "Mean: (S1.E1 + S1.E2 + S1.E3) / 3\n"
        Tooltips += "Diff between episodes: S1.E1-S1.E2\n"
        Tooltips += "Calculation between regions: S1.E1[500,700] - S1.E2[800,1000]\n"
        Tooltips += "Multiple manipulations: S1.E1 - S1.E2; S1.E3 - S1.E4; S1.E5 - S1.E6\n"
        Tooltips += "Short hand of S1.E1+S1.E2+S1.E3+S1.E4: S1.E1~4"
        formulaTextBox.setToolTip(Tooltips)

        # Report box
        arithReportBox = QtGui.QLabel("Arithmetic Results")
        arithReportBox.setStyleSheet("background-color: white")
        arithReportBox.setWordWrap(True)

        # Connect all the items to calculationevents
        nullCheckBox.stateChanged.connect(lambda checked: self.nullTraces(checked, rangeTextBox))
        calculateButton.clicked.connect(lambda: self.calculateTraces(formulaTextBox.text(), nullCheckBox.checkState(), filtCheckBox.checkState(), arithReportBox))
        formulaTextBox.returnPressed.connect(lambda: self.calculateTraces(formulaTextBox.text(), nullCheckBox.checkState(), filtCheckBox.checkState(), arithReportBox))

        # Organize all the items in the frame
        widgetFrame.layout().addWidget(calculateButton, 0, 0, 1, 3)
        widgetFrame.layout().addWidget(nullCheckBox, 1, 0)
        widgetFrame.layout().addWidget(rangeTextBox, 1, 1)
        widgetFrame.layout().addWidget(rangeUnitLabel, 1, 2)
        widgetFrame.layout().addWidget(filtCheckBox, 2, 0, 1, 3)
        widgetFrame.layout().addWidget(formulaTextBox, 3, 0, 1, 3)
        widgetFrame.layout().addWidget(arithReportBox, 4, 0, 1, 3)

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

    def calculateTraces(self, formula, isNulled, isfilt, arithReportBox):
        if not formula or formula=="Formula":
            return
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
            If each episode is followed by a range, e.g.
            f = "S1.E1[100,500] + S1.E2[600,1000] - S1.E3[200,600]/ 2 + S1.E4[700,1100] * 3 / 8 +5",
            also return the range R = [[100,500], [600,1000], [200,600], [700,1100]]. Otherwis, R = None
            """
            # separate the formula first
            groups = [s.replace(" ","") for s in filter(None, re.split(r"(\+|-)", f))]
            D = [] # data variable
            K = [] # scale factors
            C = 0 # constant
            R = []

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

            # Further separate D and R
            bool_has_range = False
            for count_g, g in enumerate(D):
                if "[" in g:
                    if not bool_has_range:
                        bool_has_range = True
                    g, rng_tmp = re.split("\[", g)
                    D[count_g] = g
                    R.append(str2num("+"+rng_tmp))

            # Double check the length of D and R matches
            if bool_has_range and len(D) != len(R):
                arithReportBox.setText("Specified ranges must follow each episode.")
                return

            return D, K, C, R

        def simpleMath(f, stream, channel, **kwargs):
            """" f = "S1.E1 + S1.E2 - S1.E3 / 2 + S1.E4 * 3 / 8"
            Additional variables can be provided by **kwargs"""
            D, K, Y, R = parseSimpleFormula(f)
            if not R:
                R = [[]] * len(K)

            for d, k, w in zip(D, K, R):
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
                    # Window the episode if R is not empty
                    if w:
                        y = spk_window(y, self.friend.episodes['Data'][yind].Protocol.msPerPoint, w)
                    # null the time series
                    if r is not None:
                        y = y - self.friend.getNullBaseline(y, self.friend.episodes['Data'][yind].Protocol.msPerPoint, r)

                    if isfilt: # apply a filter based on "Filter" tool specification
                        filterType = self.filtertype_comboBox.currentText()
                        self.getFiltSettingTable(filterType) # update
                        y = self.inplaceFiltering(True, filterType, yData=y)
                        if y is None:
                            print('filtered y became None')

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
        if ";" in formula: # a list of formulas
            # separate each formula
            formula = formula.split(";")
        elif "\n" in formula: # a list of formulas separated by newline character
            formula = formula.split("\n")
        else:
            formula = [formula]

        # parse each formula
        for f0 in formula:
            f = parseTilda(f0)
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

            y_len = len(y[s,c]) # length of time series

            # Append the data to friend's episodes object
            self.friend.episodes['Name'].append(self.friend.episodes['Name'][-1])
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
            zData.Protocol.readDataFrom = self.friend.episodes['Name'][0] + " " + f0 + ".dat"
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
    # TODO: when add a TTL object in the table, also display the detailed description of the object
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
        if sys.platform[:3]== 'win' and sys.getwindowsversion().major == 10: # fix the problem that in Windows 10, bottom border of header is not displayed
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

        # Pop up the annotation settings window to get the properties of the annotation settings
        annSet = AnnotationSetting()
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
                # Draw the artist
                artistProperty = self.drawAnnotationArtist(artist=artistProperty)
                AT_item._artistProp = artistProperty

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

    def getArtists(self):
        """Return a dictionary of artists from annotationTable"""
        artist_dict = OrderedDict()
        # Annotation table
        for r in range(self.annotation_table.rowCount()):
            item = self.annotation_table.item(r, 0)
            # Get annotation artist
            for k, v in item._artistProp.items():
                if isstrnum(v):
                    item._artistProp[k] = str2numeric(v)

            artist_dict[item._artistProp['name']] = item._artistProp
        # fitted curve
        if hasattr(self, 'fittedCurve'):
            artist_dict['fit'] = self.fittedCurve
        # Detected events
        if hasattr(self, 'eventArtist'):
            for evt in self.eventArtist:
                artist_dict[evt['name']] = evt

        return artist_dict

    def onArtistChecked(self, item=None):
        """Respond if click state was changed for pre-existing artists"""
        if item.column() > 0: # editing comments, ignore
            return

        if item.checkState() == 0: # remove the artist if it is present
            self.eraseAnnotationArtist(artist=item._artistProp)
        else: # assume checkstate > 0, likely 2, redraw the artist
            self.drawAnnotationArtist(artist=item._artistProp)

    def drawAnnotationArtist(self, artist=None, which_layout=None):
        """
        :param artist: artist properties
        :param which_layout: allows only 1 layout
        :return: artist
        """
        if which_layout is None:
            which_layout = self.friend.layout[0]
        artist['layout'] = which_layout
        if artist['type'] in ['box', 'line']:
            self.friend.drawROI(artist=artist, which_layout=which_layout)
        elif artist['type'] == 'ttl':
            # Get additional information about TTL from data: a list of OrderedDict
            artist['TTL'] = self.friend.episodes['Data'][self.friend.index[-1]].Protocol.ttlDict
            # Get SIU duration
            artist['SIU_Duration'] =self.friend.episodes['Data'][self.friend.index[-1]].Protocol.genData[53] # microsec
            artist['TTLROIs'] = [[]] * len(artist['TTL']) # used to store properties of TTL annotation shapes
            # Looping through each TTL data
            for n, TTL in enumerate(artist['TTL']):
                if ignoreFirstTTL and n == 0:
                    continue

                if not TTL['is_on']: # global enable
                    continue

                TTL_art = []
                if TTL['Step_is_on']:
                    TTL_art.append({'start': np.array([TTL['Step_Latency']]),
                                    'dur': np.array([TTL['Step_Duration']])})

                if TTL['SIU_Single_Shocks_is_on']:
                    TTL_art.append({'start': np.array([TTL['SIU_A'], TTL['SIU_B'], TTL['SIU_C'], TTL['SIU_D']]),
                                   'dur': (artist['SIU_Duration']/1000.0 if not artist['bool_pulse2step'] else 25)})

                if TTL['SIU_Train_is_on']:
                    if artist['bool_pulse2step']:
                        TTL_art.append({'start': np.array([TTL['SIU_Train_Start']]),
                                        'dur': np.array([(TTL['SIU_Train_Number'] -1) * TTL['SIU_Train_Interval'] + artist['SIU_Duration']/1000 +\
                                               ((TTL['SIU_Train_Burst_Number'] - 1) * TTL['SIU_Train_Burst_Internal'] if TTL['SIU_Train_of_Bursts_is_on'] else 0)])})
                    else:
                        start_mat = np.arange(int(TTL['SIU_Train_Number'])) * TTL['SIU_Train_Interval'] + TTL['SIU_Train_Start']
                        if TTL['SIU_Train_of_Bursts_is_on']:
                            burst_mat = np.arange(int(TTL['SIU_Train_Burst_Number'])) * TTL['SIU_Train_Burst_Interval']
                            burst_mat = burst_mat[:, np.newaxis]
                            start_mat = start_mat + burst_mat # broadcasting
                            start_mat = start_mat.flatten(order='F')
                        TTL_art.append({'start': start_mat, 'dur': artist['SIU_Duration']/1000})

                artist['TTLROIs'][n] = TTL_art

            # Draw the ROIs once we know the start and the end
            if artist['bool_realpulse']:
                p = None
                for l in self.friend.layout:
                    if which_layout[0] in l and which_layout[1] in l:
                        # get graphics handle
                        p = self.friend.graphicsView.getItem(row=l[2], col=l[3])
                        break
                if not p:
                    return
                yRange = p.viewRange()[1]
                yheight = abs((yRange[1] - yRange[0]) / 20.0)

            iteration = 0
            self.TTL_final_artist = [] # self.getArtist will get artist from here
            for n, TTL in enumerate(artist['TTLROIs']): # for each TTL channel
                if not TTL: # continue if empty
                    continue
                for m, ROIs in enumerate(TTL): # for each ROI in the TTL
                    if artist['bool_realpulse']: # draw a box
                        y0 = yRange[0] + iteration * yheight * 1.35
                        for x0 in ROIs['start']:
                            evt = {'type': 'box', 'x0': x0, 'y0': y0, 'width': ROIs['dur'], 'height': yheight, 'fill': True,
                                   'fillcolor': 'k', 'line': False, 'linecolor': 'k', 'linewidth': 0, 'linstyle': '-',
                                   'name': artist['name']}
                            self.friend.drawROI(artist=evt, which_layout=which_layout)
                            self.TTL_final_artist.append(evt)
                    else: # draw as events
                        evt = self.friend.drawEvent(ROIs['start'], which_layout=which_layout, info=[artist['name']], color='k', drawat='bottom', iteration=iteration)
                        self.TTL_final_artist.append(evt)
                    iteration = iteration + 1 # incerase iteration wen drawing everytime

        else:
            raise(NotImplementedError("'{}' annotation object has not been implemented yet".format(artist['type'])))

        return artist

    def eraseAnnotationArtist(self, artist=None, which_layout=None):
        if artist['type'] == 'ttl' and not artist['bool_realpulse']:
            self.friend.removeEvent(info=[artist['name']], which_layout=which_layout, event_type='event')
        else:
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
        # check item type
        if not hasattr(item, '_artistProp') or item._artistProp['type'] not in AnnotationSetting.ann_obj:
            notepad = QtGui.QTableWidgetItem()
            notepad.setText("This item is not editable")
            self.annotation_table.setItem(row, 1, notepad)
            return

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


    def updateTTL(self):
        # Get TTL artist
        # print('update TTL')
        TTL = None
        for r in range(self.annotation_table.rowCount()):
            if 'ttl' in self.annotation_table.item(r, 0)._artistProp['name'] and \
                    self.annotation_table.item(r, 0).checkState():
                TTL = self.annotation_table.item(r, 0)._artistProp
                break
        if TTL is None:
            return

        self.eraseAnnotationArtist(artist=TTL)
        self.drawAnnotationArtist(artist=TTL)

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
        self.updateLayoutComboBox()
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

    # <editor-fold desc="Curve Fitting tools">
    # -------- Curve Fitting tools -------------------------------------------
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
        cfReportBox = QtGui.QTextEdit("Curve Fit Results")
        cfReportBox.setStyleSheet("background-color: white")

        # Arrange the widget
        widgetFrame.layout().addWidget(fitButton, 0, 0, 1,3)
        widgetFrame.layout().addWidget(curveTypeComboBox, 1, 0, 1, 3)

        # Settings of curve fitting
        self.setCFSettingWidgetFrame(widgetFrame, cfReportBox, curveTypeComboBox.currentText())

        # Refresh setting section when cf type changed
        curveTypeComboBox.currentIndexChanged.connect(lambda: self.setCFSettingWidgetFrame(widgetFrame, cfReportBox, curveTypeComboBox.currentText()))

        # Summary box behavior
        fitButton.clicked.connect(lambda: self.curveFit(curveTypeComboBox.currentText(), cfReportBox))#, csCheckBox.checkState()))

        return widgetFrame


    def setCFSettingWidgetFrame(self, widgetFrame, cfReportBox, curve):
        # Remove everything at and below the setting rows: rigid setting
        widgetFrame = self.removeFromWidget(widgetFrame, cfReportBox, row=3)
        # Get the setting table again
        self.getCFSettingTable(widgetFrame, cfReportBox, curve)
        for key, val in self.CFsettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1])
        # Report box
        widgetFrame.layout().addWidget(cfReportBox, widgetFrame.layout().rowCount(), 0, 1, 3)

    def getCFSettingTable(self, widgetFrame, cfReportBox, curve):
        if curve == 'Exponential':
            eqLabel = QtGui.QLabel("Equation:")
            eqComboBox = QtGui.QComboBox()
            eqComboBox.addItems(['a*exp(b*x)+c','a*exp(b*x)', 'a*exp(b*x)+c*exp(d*x)'])
            eqComboBox.currentIndexChanged.connect(lambda: self.setExpCFInitializationParams(widgetFrame, cfReportBox, eqComboBox.currentText()))
            self.CFsettingTable = {(3,0): eqLabel, (3,1): eqComboBox}
            # Call it once at startup to get initialization parameters
            self.setExpCFInitializationParams(widgetFrame, cfReportBox, eqComboBox.currentText())
        elif curve == 'Power':
            eqLabel = QtGui.QLabel("Equation")
            eqComboBox = QtGui.QComboBox()
            eqComboBox.addItems(['a*x^b', 'a*x^b+c'])
            self.CFsettingTable = {(3,0): eqLabel, (3,1): eqComboBox}
        elif curve == 'Polynomial':
            degLabel = QtGui.QLabel("Degree:")
            degText = QtGui.QLineEdit("1")
            self.CFsettingTable = {(3,0):degLabel, (3,1): degText}

    def setExpCFInitializationParams(self, widgetFrame, cfReportBox, equation='a*exp(b*x)+c'):
        # Remove everything at and below the setting rows:
        widgetFrame = self.removeFromWidget(widgetFrame, reportBox=cfReportBox, row=4)
        # Get the setting table
        self.getExpCFParamTable(equation=equation)
        for key, val in self.CFsettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1])
        # Report box
        widgetFrame.layout().addWidget(cfReportBox, widgetFrame.layout().rowCount(), 0, 1, 3)

    def getExpCFParamTable(self, equation='a*exp(b*x)+c'):
        if equation == 'a*exp(b*x)+c':
            a0_label = QtGui.QLabel('a0')
            a0_text = QtGui.QLineEdit('auto')
            b0_label = QtGui.QLabel('b0')
            b0_text = QtGui.QLineEdit('auto')
            c0_label = QtGui.QLabel('c0')
            c0_text = QtGui.QLineEdit('0')
            self.CFsettingTable[(4, 0)] = a0_label
            self.CFsettingTable[(4, 1)] = a0_text
            self.CFsettingTable[(5, 0)] = b0_label
            self.CFsettingTable[(5, 1)] = b0_text
            self.CFsettingTable[(6, 0)] = c0_label
            self.CFsettingTable[(6, 1)] = c0_text
        elif equation == 'a*exp(b*x)':
            a0_label = QtGui.QLabel('a0')
            a0_text = QtGui.QLineEdit('auto')
            b0_label = QtGui.QLabel('b0')
            b0_text = QtGui.QLineEdit('auto')
            self.CFsettingTable[(4, 0)] = a0_label
            self.CFsettingTable[(4, 1)] = a0_text
            self.CFsettingTable[(5, 0)] = b0_label
            self.CFsettingTable[(5, 1)] = b0_text
        elif equation == 'a*exp(b*x)+c*exp(d*x)':
            a0_label = QtGui.QLabel('a0')
            a0_text = QtGui.QLineEdit('auto')
            b0_label = QtGui.QLabel('b0')
            b0_text = QtGui.QLineEdit('auto')
            c0_label = QtGui.QLabel('c0')
            c0_text = QtGui.QLineEdit('auto')
            d0_label = QtGui.QLabel('d0')
            d0_text = QtGui.QLineEdit('auto')
            self.CFsettingTable[(4, 0)] = a0_label
            self.CFsettingTable[(4, 1)] = a0_text
            self.CFsettingTable[(5, 0)] = b0_label
            self.CFsettingTable[(5, 1)] = b0_text
            self.CFsettingTable[(6, 0)] = c0_label
            self.CFsettingTable[(6, 1)] = c0_text
            self.CFsettingTable[(7, 0)] = d0_label
            self.CFsettingTable[(7, 1)] = d0_text
        else:
            pass

    def getExpCFDefaultParams(self, xdata, ydata, equation='a*exp(b*x)+c'):
        if equation == 'a*exp(b*x)+c':
            p0 = list(fit_exp_with_offset(xdata, ydata, sort=False))
        elif equation == 'a*exp(b*x)':
            p0 = [max(ydata), -0.015 if ydata[-1] < ydata[0] else 0.025]
        elif equation == 'a*exp(b*x)+c*exp(d*x)':
            p0 = [max(ydata), -0.015 if ydata[-1] < ydata[0] else 0.025, max(ydata),
                  -0.015 if ydata[-1] < ydata[0] else 0.025]
        else:
            return

        for m in range(len(p0)): # replacing with user custom values
            if self.CFsettingTable[(4+m, 1)].text() == 'auto':
                pass
            elif not isstrnum(self.CFsettingTable[(4+m, 1)].text()):
                pass
            else:
                p0[m] = str2numeric(self.CFsettingTable[(4+m, 1)].text())

        #print('Initial fitted parameters')
        #print(p0)
        return p0

    def curveFit(self, curve, cfReportBox, currentView=(0,0)):#, centerAndScale):
        # get view
        p = self.friend.graphicsView.getItem(row=currentView[0], col=currentView[1])
        # clear previous fit artists
        # count_fit = 0
        for k, a in enumerate(p.listDataItems()):
            if 'fit' in a.name():
                # count_fit = count_fit + 1
                # Erase the older fits
                p.removeItem(a)

        # if len(p.listDataItems())-count_fit > 1:
        #     cfReportBox.setText("Can only fit curve at 1 trace at a time. Please select only 1 trace")
        #     return

        # Get only the plotted data of first channel / stream
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
            p0 = self.getExpCFDefaultParams(xdata, ydata, equation=eqText)
            if eqText == 'a*exp(b*x)+c':
                f0 = lambda x, a, b, c: a*np.exp(b*x)+c
                # bounds = [(-max(abs(ydata))*1.1, -10, -np.inf),  (max(abs(ydata))*1.1, 10, np.inf)]
                ptext = ['a','b','c']
            elif eqText == 'a*exp(b*x)':
                f0 = lambda x, a, b: a*np.exp(b*x)
                # bounds = [(-max(abs(ydata))*1.1, -10), (max(abs(ydata))*1.1, 10)]
                ptext = ['a','b']
            elif eqText == 'a*exp(b*x)+c*exp(d*x)':
                f0 = lambda x, a, b, c, d: a*np.exp(b*x) + c*np.exp(d*x)
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
                p.plot(xdata+xoffset, yfit+yoffset, pen='r', name='fit: '+eqText)
        # Add fitted curve to annotation artist
        self.fittedCurve = {'x': xdata+xoffset, 'y': yfit+yoffset, 'linecolor': 'r', 'name': 'fit: '+eqText, \
                            'layout': self.friend.layout[currentView[0]], 'type': 'curve'}
        # Report the curve fit
        final_text = "Model: {}\nEquation:\n\t{}\n".format(curve, eqText)
        final_text += "Parameters:\n"
        for ppt, coeff in zip(ptext, popt): # report fitted parameters
            final_text += "\t" + ppt + ": " + "{:.4g}".format(coeff) + "\n"
        if curve == 'Exponential':
            final_text += "Time Constants:\n"
            if eqText in ['a*exp(b*x)+c', 'a*exp(b*x)']:
                tau = -1.0/popt[1]
                final_text += "\ttau: " + "{:.5f} ms".format(tau) + "\n"
            elif eqText == 'a*exp(b*x)+c*exp(d*x)':
                tau1, tau2 = -1.0/popt[1], -1.0/popt[3]
                final_text += "\ttau1: " + "{:.5f} ms".format(tau1) + "\n"
                final_text += "\ttau2: " + "{:.5f} ms".format(tau2) + "\n"

        final_text += "\nGoodness of fit:\n\tSSE: {:.4g}\n\tR-squared: {:.4g}\n\tAdjusted R-squared: {:.4g}\n\tRMSE: {:.4g}".format(SSE, R_sq, R_sq_adj, RMSE)
        cfReportBox.setText(final_text)

    # </editor-fold>

    # <editor-fold desc="Event Detection tools">
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
        widgetFrame = self.removeFromWidget(widgetFrame, reportBox=detectReportBox, row=2)
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
            threshLabel = QtGui.QLabel("Threshold")
            threshLabel.setToolTip("Finds peaks that are at least greater than both adjacent samples by the threshold, TH. TH is a real-valued scalar greater than or equal to zero. The default value of TH is zero.")
            threshTextEdit = QtGui.QLineEdit("0")
            threshTextUnitLabel = QtGui.QLabel("mV")
            self.EDsettingTable = {(3,0): minHeightLabel, (3,1): minHeightTextEdit,
                            (3,2): minHeightUnitLabel, (4,0):minDistLabel,
                            (4,1): minDistTextEdit, (4,2): minDistUnitLabel,
                            (5,0): threshLabel, (5,1): threshTextEdit, (5,2): threshTextUnitLabel}
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
            thresh = float(self.EDsettingTable[(5,1)].text())
            self.detectAPs(detectReportBox, drawEvents, msh, msd, thresh)
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

    def detectAPs(self, detectReportBox, drawEvent=False, msh=-10, msd=1, thresh=0):
        """detectAPs(detectReportBox, drawEvent, 'additional settings',...)"""
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]
        final_label_text = ""
        iteration = 0
        for i in self.friend.index:
            zData = self.friend.episodes['Data'][i]
            ts = zData.Protocol.msPerPoint
            for c, Vs in zData.Voltage.items(): # iterate over channels
                Vs = spk_window(Vs, ts, selectedWindow, t0=0)
                num_spikes, spike_time, spike_heights = spk_count(Vs, ts, msh=msh, msd=msd, threshold=thresh)
                if len(self.friend.index)>0:
                    final_label_text = final_label_text + os.path.basename(self.friend.episodes['Dirs'][i]) + "\n"
                final_label_text = final_label_text + c + " : \n"
                final_label_text = final_label_text + "  # spikes: " + str(num_spikes) + "\n"
                final_label_text = final_label_text + "  mean ISI: "
                final_label_text += "{:0.2f}".format(np.mean(np.diff(spike_time))) if len(spike_time)>1 else "NaN"
                final_label_text += "\n"
                # Draw event markers
                if drawEvent:
                    if selectedWindow[0] is not None:
                        spike_time += selectedWindow[0]
                    # find out the layout
                    which_layout = self.friend.layout[self.friend.indexStreamChannel('Voltage', c)]
                    color = self.friend._usedColors[iteration] if self.friend.colorfy else 'r'
                    eventArtist = self.friend.drawEvent(spike_time, which_layout = which_layout, info=[self.detectedEvents[-1]], color=color, iteration=iteration)
                    iteration = iteration + 1
                    self.eventArtist.append(eventArtist)
        detectReportBox.setText(final_label_text[:-1])

    def detectPSPs(self, detectReportBox, drawEvent=False, event='EPSP', riseTime=1, decayTime=4, amp=1, step=20, criterion='se', thresh=3.0):
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]
        final_label_text = ""
        if event in ['EPSP', 'IPSP']:
            stream = 'Voltage'
        else: # ['EPSC', 'IPSC']
            stream = 'Current'

        iteration = 0
        for i in self.friend.index:
            zData = self.friend.episodes['Data'][i]
            ts = zData.Protocol.msPerPoint
            # Get events
            for c, S in getattr(zData, stream).items():
                S = spk_window(S, ts, selectedWindow, t0=0)
                event_time, pks, _, _ = detectPSP_template_matching(S, ts, event=event, \
                                                w=200, tau_RISE=riseTime, tau_DECAY=decayTime, \
                                                mph=amp, step=step, criterion=criterion, thresh=thresh)
                if len(self.friend.index)>0:
                    final_label_text = final_label_text + os.path.basename(self.friend.episodes['Dirs'][i]) + "\n"
                final_label_text = final_label_text + c + ": \n"
                final_label_text = final_label_text + "  # " + event + ": " + str(len(event_time)) +  "\n"
                final_label_text += "  mean IEI: "
                final_label_text += "{:0.2f}".format(np.mean(np.diff(event_time))) if len(event_time)>1 else "NaN"
                final_label_text += "\n"
                # Draw event markers
                if drawEvent:
                    if selectedWindow[0] is not None:
                        event_time += selectedWindow[0]

                    # find out the layout
                    which_layout = self.friend.layout[self.friend.indexStreamChannel(stream, c)]
                    color = self.friend._usedColors[iteration] if self.friend.colorfy else 'r'
                    eventArtist = self.friend.drawEvent(event_time, which_layout = which_layout, info=[self.detectedEvents[-1]], color=color, iteration=iteration)
                    iteration = iteration + 1
                    self.eventArtist.append(eventArtist)
        detectReportBox.setText(final_label_text[:-1])

    def detectCellAttachedSpikes(self, detectReportBox, drawEvent=False, msh=30, msd=10, basefilt=20, maxsh=300):
        if self.friend.viewRegionOn:
            selectedWindow = self.friend.selectedRange
        else:
            selectedWindow = [None, None]

        final_label_text = ""
        iteration = 0
        for i in self.friend.index:
            zData = self.friend.episodes['Data'][i]
            ts = zData.Protocol.msPerPoint
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

                    which_layout = self.friend.layout[self.friend.indexStreamChannel('Current', c)]
                    color = self.friend._usedColors[iteration] if self.friend.colorfy else 'r'
                    eventArtist = self.friend.drawEvent(spike_time, which_layout = which_layout, color=color, info=[self.detectedEvents[-1]], iteration=iteration)
                    iteration = iteration + 1
                    self.eventArtist.append(eventArtist)
        detectReportBox.setText(final_label_text[:-1])

    # </editor-fold>

    #<editor-fold desc="Filter widget">
    def filterWidget(self):
        """Inplace Filter traces"""
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("Filter"))

        filter_checkbox = QtGui.QCheckBox('Apply Filter')
        filter_checkbox.setToolTip('Apply inplace filtering to current trace')
        self.filtertype_comboBox = QtGui.QComboBox()
        self.filtertype_comboBox.addItems(['Butter'])

        widgetFrame.layout().addWidget(filter_checkbox, 0, 0, 1, 2)
        widgetFrame.layout().addWidget(self.filtertype_comboBox, 1, 0, 1, 2)

        # Settings of filter
        self.setFiltSettingWidgetFrame(widgetFrame, self.filtertype_comboBox.currentText())

        # Refresh setting section when filter type changed
        self.filtertype_comboBox.currentIndexChanged.connect(lambda: self.setFilterSettingWidgetFrame(widgetFrame, self.filtertype_comboBox.currentText()))

        # When "Apply Filter" checkbox is clicked
        filter_checkbox.stateChanged.connect(lambda checked: self.inplaceFiltering(checked, self.filtertype_comboBox.currentText()))


        return widgetFrame

    def setFiltSettingWidgetFrame(self, widgetFrame, filterType):
        self.getFiltSettingTable(filterType)
        for key, val in self.FiltSettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1])

    def getFiltSettingTable(self, filterType):
        if filterType.lower() == 'butter':
            order_label = QtGui.QLabel("Order")
            order_label.setToolTip("Filter order")
            order_text = QtGui.QLineEdit("3")
            Wn_label = QtGui.QLabel("Wn")
            Wn_label.setToolTip("Normalized cutoff frequency, between 0 and 1")
            Wn_text = QtGui.QLineEdit("0.2")
            Btype_label = QtGui.QLabel("Type")
            Btype_combobox = QtGui.QComboBox()
            Btype_combobox.addItems(["low","high", "band"])
            self.FiltSettingTable = {(3,0): order_label, (3,1): order_text, (4,0): Wn_label, (4,1): Wn_text,
                                     (5,0): Btype_label, (5,1): Btype_combobox}
        else:
            pass

    def inplaceFiltering(self, checked, filterType, currentView=(0,0), yData=None):
        p = self.friend.graphicsView.getItem(row=currentView[0], col=currentView[1])
        # Get only the plotted data of first channel / stream
        data = p.listDataItems()

        if checked: # assuming changed from unchecked to checked state, apply the filter
            if filterType.lower() == 'butter':
                Order = str2numeric(self.FiltSettingTable[(3,1)].text())
                Wn = str2num(self.FiltSettingTable[(4,1)].text())
                Btype = self.FiltSettingTable[(5,1)].currentText()
                if yData is None: # inplace
                    for d in data:
                        y = self.butterFilter(d.yData, Order, Wn, Btype)
                        d.original_yData = d.yData
                        d.setData(d.xData, y)
                else:
                    y = self.butterFilter(yData, Order, Wn, Btype)
                    return y
        else: # inplace only: assuming changed from checked to unchecked state, recover original data
            for d in data:
                if not hasattr(d, 'original_yData'):
                    print('Data is not currently filtered, cannot recover original data')
                    return
                else:
                    d.setData(d.xData, d.original_yData)

    def butterFilter(self, y, Order, Wn, Btype="low"):
        b, a = butter(Order, Wn, btype=Btype, analog=False, output='ba')
        y_filt = filtfilt(b, a, y)
        return y_filt


    # </editor-fold>

    # <editor-fold desc="Function widget">
    def functionWidget(self):
        """Apply a function to selected regions and print out the summary"""
        widgetFrame = QtGui.QFrame(self)
        widgetFrame.setLayout(QtGui.QGridLayout())
        widgetFrame.setObjectName(_fromUtf8("FunctionWidgetFrame"))
        widgetFrame.layout().setSpacing(10)
        # Apply button
        applyButton = QtGui.QPushButton("Apply")
        # Select from a list of pre-existing tools, or enter a custom function
        functionComboBox = QtGui.QComboBox()
        functionComboBox.addItems(['mean', 'std', 'diff', 'rms', 'series resistance', 'Rin', 'Rin2']) #  'custom'
        # Summary box
        functionReportBox = QtGui.QLabel("Apply a function")
        functionReportBox.setStyleSheet("background-color: white")
        functionReportBox.setWordWrap(True)
        # Arrange the widget
        widgetFrame.layout().addWidget(applyButton, 0, 0, 1, 3)
        widgetFrame.layout().addWidget(functionComboBox, 1, 0, 1, 3)

        # Get setting for each function
        self.setAFSettingWidgetFrame(widgetFrame, functionReportBox, functionComboBox.currentText())

        # Refresh setting section when function changed
        functionComboBox.currentIndexChanged.connect(lambda: self.setAFSettingWidgetFrame(widgetFrame, functionReportBox, functionComboBox.currentText()))
        # Summary box behavior
        applyButton.clicked.connect(lambda: self.applyFunction(functionComboBox.currentText(), functionReportBox))
        return widgetFrame

    def setAFSettingWidgetFrame(self, widgetFrame, functionReportBox, func):
        # Remove everything at and below the setting rows: rigid setting
        widgetFrame = self.removeFromWidget(widgetFrame, reportBox=functionReportBox, row=2)
        self.getFASettingTable(func, functionReportBox)
        for key, val in self.FASettingTable.items():
            widgetFrame.layout().addWidget(val, key[0], key[1], 1, 3)
        # Report box
        widgetFrame.layout().addWidget(functionReportBox, widgetFrame.layout().rowCount(), 0, 1, 3)
        if func == 'Rin':
            functionReportBox.setText("Calculate Rin from negative pulse in the trace")
        elif func == 'Rin2':
            functionReportBox.setText("Calculate Rin from two episodes")
        elif func == 'series resistance':
            functionReportBox.setText("Calculate series resistance from current trace in voltage clamp")
        elif func ==' diff':
            functionReportBox.setText("Calculate the average difference between two trace")
        else:
            functionReportBox.setText("Apply a function")

    def getFASettingTable(self, func='mean', functionReportBox=None):
        """Return a table for settings of each function to be applied"""
        if func == "Rin":
            useCurrent_checkBox = QtGui.QCheckBox("Use current instead")
            useCurrent_checkBox.setToolTip("Check to use current to calculate input resistance instead")
            windowSize_label = QtGui.QLabel("Window (ms)")
            windowSize_textbox = QtGui.QLineEdit("25")
            self.FASettingTable = {(2,0):useCurrent_checkBox, (3,0):windowSize_label, (4,0):windowSize_textbox}

        elif func =='custom':
            customFuncTextEdit = QtGui.QLineEdit()
            customFuncTextEdit.setPlaceholderText("Custom Function")
            customFuncTextEdit.setToolTip("Enter a custom function to be applied")
            self.FASettingTable = {(2,0):customFuncTextEdit}

        else:
            self.FASettingTable = {}


    def applyFunction(self, func='mean', functionReportBox=None, *args, **kwargs):
        if not self.friend.index:
            functionReportBox.setText("Select episode to apply function to")
            return
        layout = self.friend.layout[0] # Apply only onto the trace in the first / top layout
        zData = self.friend.episodes['Data'][self.friend.index[-1]]
        ts = zData.Protocol.msPerPoint
        final_label_text = ""

        if func in ['mean', 'std', 'rms']:
            Y = getattr(zData, layout[0])[layout[1]]
            if self.friend.viewRegionOn:
                Y = spk_window(Y, ts, self.friend.selectedRange)

        if func == 'mean':
            if layout[0][0].lower() == 'v':
                unit_suffix = 'mV'
            elif layout[0][0].lower() == 'C':
                unit_suffix = 'pA'
            else: # Stimulus
                unit_suffix = ''
            final_label_text = "mean: {:.9f} {}".format(np.mean(Y), unit_suffix)
        elif func == 'std':
            final_label_text = "std: {:.9f}".format(np.std(Y))
        elif func == 'rms':
            final_label_text = "rms: {:.9f}".format(rms(Y))
        elif func == 'series resistance':
            Vs = zData.Stimulus[layout[1]]
            Is = zData.Current[layout[1]]
            series_resistance, tau, adjrsquare = spk_vclamp_series_resistance(Is, Vs, ts)
            final_label_text = "R series: {:.3f} MOhm\nadjrsquare: {:.9f}".format(series_resistance, adjrsquare)
        elif func == 'Rin': # Calculate Rin with negative step
            if not self.friend.viewRegionOn:
                final_label_text = "Select a region to calculate Rin"
            else:
                useCurrent = self.FASettingTable[(2,0)].isChecked()
                window_size = str2numeric(self.FASettingTable[(4,0)].text())

                V1 = np.mean(spk_window(zData.Voltage[layout[1]], ts, self.friend.selectedRange[0] + window_size * np.asarray([-1,1])))
                V2 = np.mean(spk_window(zData.Voltage[layout[1]], ts, self.friend.selectedRange[1] + window_size * np.asarray([-1,1])))

                if useCurrent:
                    S1 = np.mean(spk_window(zData.Current[layout[1]], ts, self.friend.selectedRange[0] + window_size * np.asarray([-1,1])))
                    S2 = np.mean(spk_window(zData.Current[layout[1]], ts, self.friend.selectedRange[1] + window_size * np.asarray([-1,1])))
                else:
                    S1 = np.mean(spk_window(zData.Stimulus[layout[1]], ts, self.friend.selectedRange[0] + window_size * np.asarray([-1,1])))
                    S2 = np.mean(spk_window(zData.Stimulus[layout[1]], ts, self.friend.selectedRange[1] + window_size * np.asarray([-1,1])))

                Rin = (V2-V1)/(S2-S1)*1000
                final_label_text = "Rin = {:.9f} MOhm;".format(Rin)
        elif func == 'Rin2': # Calculating for 2 episodes with holding current change
            if not self.friend.viewRegionOn:
                final_label_text = "Select a region to calculate Rin"
            elif len(self.friend.index)<2:
                final_label_text = "Select two episodes to calculate Rin"
            else:
                V1 = np.mean(spk_window(zData.Voltage[layout[1]], ts, self.friend.selectedRange))
                S1 = np.mean(spk_window(zData.Current[layout[1]], ts, self.friend.selectedRange))

                zData2 = self.friend.episodes['Data'][self.friend.index[-2]]
                V2 = np.mean(spk_window(zData2.Voltage[layout[1]], ts, self.friend.selectedRange))
                S2 = np.mean(spk_window(zData2.Current[layout[1]], ts, self.friend.selectedRange))

                Rin = (V2 - V1) / (S2 - S1) * 1000
                final_label_text = "Rin = {:.5f} MOhm;".format(Rin)

        elif func == 'diff': # Calculating average difference between two episodes
            if len(self.friend.index)<2:
                final_label_text = "Select two episodes to calculate diff"
            else:
                zData1 = self.friend.episodes['Data'][self.friend.index[-1]]
                zData2 = self.friend.episodes['Data'][self.friend.index[-2]]

                Y1 = getattr(zData1, layout[0])[layout[1]]
                Y2 = getattr(zData2, layout[0])[layout[1]]
                if self.friend.viewRegionOn:
                    Y1 = spk_window(Y1, ts, self.friend.selectedRange)
                    Y2 = spk_window(Y2, ts, self.friend.selectedRange)

                final_label_text = 'Diff = {:.9f}'.format(np.mean(Y1) - np.mean(Y2))

        else: # custom function
            pass
        functionReportBox.setText(final_label_text.strip())

    # </editor-fold>

    # <editor-fold desc="Other utilities 2">
    # ------- Other utilities ------------------------------------------------
    def replaceWidget(self, widget=None, index=0):
        old_widget = self.accWidget.takeAt(index)
        self.accWidget.addItem(title=old_widget.title(), widget=widget, collapsed=old_widget._collapsed,
                               index=index)
        return

    def removeFromWidget(self, widgetFrame, reportBox, row=0):
        """Remove widgets from a widgetFrame below row, excluding a reportBox"""
        nrows = widgetFrame.layout().rowCount()
        if nrows>row:
            for r in range(row, nrows):
                for col in range(widgetFrame.layout().columnCount()):
                    currentItem = widgetFrame.layout().itemAtPosition(r, col)
                    if currentItem is not None:
                        if currentItem.widget() is not reportBox:
                            currentItem.widget().deleteLater()
                        else:
                            widgetFrame.layout().removeItem(currentItem)

        return widgetFrame

    def sizeHint(self):
        """Helps with initial dock window size"""
        return QtCore.QSize(self.friend.frameGeometry().width() / 4.95, 20)

    # </editor-fold>

