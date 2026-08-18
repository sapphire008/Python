# -*- coding: utf-8 -*-
"""
Created: Sat Apr 18 21:40:21 2015

Form implementation generated from reading ui file 'SynapseQt.ui'

      by: PyQt4 UI code generator 4.10.4

WARNING! All changes made in this file will be lost!

Main window of Synapse

@author: Edward
"""

import os
import sys
import re
import csv
import signal
import numpy as np
from pdb import set_trace
import subprocess
import pandas as pd


# sys.path.append('D:/Edward/Documents/Assignments/Scripts/Python/PySynapse')
# sys.path.append('D:/Edward/Docuemnts/Assignments/Scripts/Python/generic')
from util.ImportData import NeuroData, get_cellpath
from util.spk_util import *
from app.Scope import ScopeWindow
from app.Settings import *
from app.config import settings

import sip
sip.setapi('QVariant', 2)

# Routines for Qt import errors
from PyQt5 import QtGui, QtCore, QtWidgets
#from pyqtgraph.Qt import QtGui, QtCore
try:
    from PyQt5.QtCore import QString
except ImportError:
    QString = str

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

# Set some global variables
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__version__ = "PySynapse 0.5"
APP_ICON_FILE = os.path.join(__location__, 'resources', 'icons', 'Synapse-resizeimage.png')
if not os.path.isfile(APP_ICON_FILE):
    APP_ICON_FILE = os.path.join(__location__, 'resources', 'icons', 'Synapse.png')


def icon_file(name):
    return os.path.join(__location__, 'resources', 'icons', name)


def apply_app_icon(app):
    """Set the process icon. On macOS, window.setWindowIcon does not appear in
    the title bar; the Dock icon must be set on QApplication (and NSApp)."""
    icon = QtGui.QIcon(APP_ICON_FILE)
    app.setWindowIcon(icon)
    app.setApplicationName("PySynapse")
    app.setApplicationDisplayName("PySynapse")
    if sys.platform == 'darwin':
        try:
            from AppKit import NSApplication, NSImage
            ns_img = NSImage.alloc().initWithContentsOfFile_(APP_ICON_FILE)
            if ns_img is not None:
                NSApplication.sharedApplication().setApplicationIconImage_(ns_img)
        except Exception:
            pass
    return icon

# Custom helper functions
def sort_nicely(l):
    """ Sort the given list in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def alphanum_key(s):
    """Key for human-friendly sort of cell / episode names."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'([0-9]+)', str(s))]

def my_excepthook(type, value, tback):
    """This helps prevent program crashing upon an uncaught exception"""
    sys.__excepthook__(type, value, tback)

# Custom File system
class Node(object):
    """Reimplement Node object"""
    def __init__(self, name, path=None, parent=None, info=None):
        super(Node, self).__init__()

        self.name = name
        self.children = []
        self.parent = parent
        self.info = info

        self.is_dir = False
        self.is_sequence = False
        self.type = "" #drive, directory, file, link, sequence
        self.path = path
        self.is_traversed = False

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

    def insert_child(self, position, child):
        if position < 0 or position > self.child_count():
            return False

        self.children.insert(position, child)
        child.parent = self

        return True
        
    def remove_child(self, position, child):
        if position < 0 or position > self.child_count():
            return False
        
        if child in self.children:
            self.children.remove(child)
        
        return True
        
    def child(self, row):
        return self.children[row]

    def child_count(self):
        return(len(self.children))

    def row(self):
        if self.parent is not None:
            return self.parent.children.index(self)
        return(0)

class FileSystemTreeModel(QtCore.QAbstractItemModel):
    """Reimplement custom FileSystemModel"""
    FLAG_DEFAULT = QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable

    def __init__(self, path=None, parent=None, root='FileName'):
        super(FileSystemTreeModel, self).__init__()
        self.root = Node(root)
        self.parent = parent
        self.path = path
        if not self.path: # if startup path is not provided
            self.initialNode(sys.platform)
        else:
            self.getChildren(self.path, startup=True)

    def initialNode(self, running_os):
        """create initial node based on OS
        On Windows, list all the drives
        On Mac, start at "/Volumes"
        On Linux, start at "/"
        """
        if running_os[0:3] == 'win':
            hasLabel = True
            try:
                drives = subprocess.check_output('wmic logicaldisk get name, volumename', stderr=subprocess.STDOUT, timeout=3)
            except:
                hasLabel = False
                drives = subprocess.check_output('wmic logicaldisk get name', stderr=subprocess.STDOUT)
            if not drives: # final check
                raise(Exception('Cannot locate drives from wmic logicaldisk'))
            drives = drives.decode('utf-8')
            drives = drives.split('\n') # split by lines
            for d in drives:
                if 'Name' in d or not d:
                    continue
                dpath = re.split('[\s]+',d)[:-1]
                if not dpath or not dpath[0]: # if empty string
                    continue
                if hasLabel:
                    label = " ".join(dpath[1:])
                    dpath = dpath[0]
                    label += " ({})".format(dpath)
                else:
                    cmd = 'wmic volume where "name=' + "'{}\\\\'".format(dpath) + '" get label'
                    try:
                        label = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=2)
                        label = label.decode('utf-8')
                        if "No Instance" in label:
                            label = dpath
                        else:
                            label = re.split('[\s]+', label)[1:-1]
                            if isinstance(label, list):
                                label = " ".join(label)
                            label += " ({})".format(dpath)
                    except:
                        label = dpath

                # Modify dpath to include slash
                dpath += "/"

                node = Node(label, dpath, parent=self.root)
                node.is_dir = True
                node.type = "drive" # drive
        elif running_os[0:3] == 'dar' or running_os[0:3] == 'mac':
            self.getChildren("/Volumes/", startup=True)
        elif running_os[0:3] == 'lin':
            self.getChildren("/", startup=True)
        else:
            self.getChildren("/", startup=True)
            print("Warning: Unrecognized OS. Starting at '/' directory")

    def getNode(self, index):
        if index.isValid():
            return(index.internalPointer())
        else:
            return(self.root)

    ## - dynamic row insertion starts here
    def canFetchMore(self, index):
        node = self.getNode(index)

        if node.is_dir and not node.is_traversed:
            return(True)

        return(False)

    ## this is where you put custom logic for handling your special nodes
    def fetchMore(self, index):
        parent = self.getNode(index)
        self.ucwd = parent.path

        nodes = self.getChildren(parent.path, startup=False)

        # insert the newly fetched files
        self.insertNodes(0, nodes, index)
        parent.is_traversed = True


    def hasChildren(self, index):
        node = self.getNode(index)

        if node.is_dir:
            return(True)

        return(super(FileSystemTreeModel, self).hasChildren(index))

    def getChildren(self, path, startup=False):
        dat_files, other_files, img_files = [], [], []
        # first separate files into two categories
        for file in os.listdir(path):
            if str(os.path.splitext(file)[1]).lower() == '.dat' and re.findall('.S(\d+).E(\d+).dat', file):
                dat_files.append(file)
            elif str(os.path.splitext(file)[1].lower()) == '.img':
                img_files.append(file)
            else:
                other_files.append(file)

        # Make the sequence for dat files
        sequence = self.createSequence(path=path, files=dat_files)
        # Make the stack for img files
        stack = self.createStack(path=path, files=img_files)

        # insert the nodes
        nodes = []
        parent = self.root if startup else None
        # Sort other files as human expect
        other_files = sort_nicely(other_files)
        # insert other files first
        for file in other_files:
            file_path = os.path.join(path, file)
            node = Node(file, file_path, parent=parent)
            if os.path.isdir(file_path):
                node.is_dir = True
                node.type = "directory" # directory
            elif os.path.islink(file_path):
                node.type = "link"
            else:
                node.type = "file"

            nodes.insert(0, node)

        # insert custom sequence
        for s in sequence:
            file_path = os.path.join(path, s['Name']+'.{}.dat')
            node = Node("{} ({:d})".format(s['Name'], len(s['Dirs'])), file_path, parent=parent, info=s)
            node.is_dir = False
            node.type = "sequence"
            nodes.insert(0, node)

        # insert custom stack
        for t in stack:
            file_path = os.path.join(path, s['Name']+'.{}.IMG')
            node = Node("{} ({:d})".format(s['Name'], len(s['Dirs'])), file_path, parent=parent, info=s)
            node.is_dir = False
            node.type = "stack"
            node.insert(0, node)

        return(nodes)

    def rowCount(self, parent):
        node = self.getNode(parent)
        return(node.child_count())

    ## dynamic row insert ends here
    def columnCount(self, parent):
        return(1)

    def flags(self, index):
        return(FileSystemTreeModel.FLAG_DEFAULT)

    def parent(self, index):
        node = self.getNode(index)

        parent = node.parent
        if parent == self.root:
            return(QtCore.QModelIndex())

        return(self.createIndex(parent.row(), 0, parent))

    def index(self, row, column, parent):
        node = self.getNode(parent)

        child = node.child(row)

        if not child:
            return(QtCore.QModelIndex())

        return(self.createIndex(row, column, child))

    def headerData(self, section, orientation, role):
        return(self.root.name)

    def data(self, index, role):
        if not index.isValid():
            return(None)

        node = index.internalPointer()

        if role == QtCore.Qt.DisplayRole:
            return(node.name)
        elif role == QtCore.Qt.DecorationRole: # insert icon here
            if node.type == 'drive':
                iconimg = 'drive.png'
            elif node.type == 'directory':
                iconimg = 'folder.png'
            elif node.type == 'file':
                iconimg = 'file.png'
            elif node.type == 'sequence':
                iconimg = 'activity.png'
            elif node.type == 'stack':
                iconimg = 'setting.png'
            else: # for debugging, should not reach this
                raise(TypeError('Unrecognized node type'))
            return QtGui.QIcon(QtGui.QPixmap(icon_file(iconimg)))
        elif role == QtCore.Qt.BackgroundRole: # insert highlight color here
            return(QtGui.QBrush(QtCore.Qt.transparent))
        else:
            return(None)

    def insertNodes(self, position, nodes, parent=QtCore.QModelIndex()):
        node = self.getNode(parent)
        success = False

        self.beginInsertRows(parent, position, position + len(nodes) - 1)

        for child in nodes:
            success = node.insert_child(position, child)

        self.endInsertRows()

        return success
        
    def refreshNode(self, parent=QtCore.QModelIndex()):
        node = self.getNode(parent)
        # set_trace()
        # Remove old items
        self.beginRemoveRows(parent, 0, len(node.children))        
        node.children = []        
        self.endRemoveRows()
        # Add new items
        self.fetchMore(parent)
        
    def fileName(self, index):
        return(self.getNode(index))

    def filePath(self, index):
        return(os.path.dirname(self.getNode(index)))

    def setRootPath(self, path):
        self.path = path

    def createSequence(self, path, files=None):
        """Extract episode information in order to create a table
           Set name of the sequence based on the list of files.
           Return True if successfully made the sequence."""
        if not files:
            return([])
        Z = ['S%s.E%s'%re.findall('.S(\d+).E(\d+).dat', f)[0] for f in files]
        Q = [re.split('.S(\d+).E(\d+).dat', f)[0] for f in files] # name
        # get unique IDs
        names, _, inverse, counts = np.unique(Q, return_index=True, return_inverse=True, return_counts=True)
        sequence = []

        for n, nm in enumerate(names):
            sequence.append({'Name':('%s'%(nm)),
                'Dirs': [os.path.join(path, pp).replace('\\','/') for ii, pp in zip(inverse==n, files) if ii],
                'Epi': [zz for ii, zz in zip(inverse==n, Z) if ii],
                'Time':[],
                'Sampling Rate': [],
                'Duration':[],
                'Drug Level':[],
                'Drug Name': [],
                'Drug Time': [],
                'Comment': []
                })
            # load episode info
            for d in sequence[n]['Dirs']:
                # zData = readDatFile(d, readTraceData = False)
                zData = NeuroData(d, old=True, infoOnly=True)

                sequence[n]['Time'].append(zData.Protocol.WCtimeStr)
                sequence[n]['Sampling Rate'].append(zData.Protocol.msPerPoint)
                sequence[n]['Duration'].append(int(zData.Protocol.sweepWindow))
                sequence[n]['Drug Level'].append(zData.Protocol.drug)
                sequence[n]['Drug Name'].append(zData.Protocol.drugName)
                sequence[n]['Drug Time'].append(zData.Protocol.drugTimeStr)
                sequence[n]['Comment'].append(zData.Protocol.stimDesc)

        return(sequence)

    def createStack(self, path, files=None):
        """For images"""
        return([])

# Episode Table
class EpisodeTableModel(QtCore.QAbstractTableModel):
    def __init__(self, dataIn=None, parent=None, *args):
        super(EpisodeTableModel, self).__init__()
        self.datatable = dataIn
        self.selectedRow = None

    def update(self, dataIn):
        # print('Updating Model')
        self.datatable = dataIn # pandas dataframe
        # print('Datatable : {0}'.format(self.datatable))

    def rowCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        return len(self.datatable.columns.values)

    def headerData(self, section, orientation, role=QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole and orientation == QtCore.Qt.Horizontal:
            return self.datatable.columns[section]
        return QtCore.QAbstractTableModel.headerData(self, section, orientation, role)

    def data(self, index, role=QtCore.Qt.DisplayRole):
        i = index.row()
        j = index.column()
        if role == QtCore.Qt.DisplayRole:
            # return the data got as a string
            return '{0}'.format(self.datatable.iat[i, j])
        elif role == QtCore.Qt.BackgroundRole:
            return QtGui.QBrush(QtCore.Qt.transparent)
        else:
            return None

    def flags(self, index):
        return QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsSelectable


class EpisodeFilterProxy(QtCore.QSortFilterProxyModel):
    """Filter episode rows by Drug Name using a Python regex."""

    def __init__(self, parent=None):
        super(EpisodeFilterProxy, self).__init__(parent)
        self._regex = None
        self._column_name = "Drug Name"

    def setDrugNameFilter(self, pattern):
        self._regex = None
        if pattern:
            try:
                self._regex = re.compile(pattern)
            except re.error:
                self._regex = None
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row, source_parent):
        if self._regex is None:
            return True
        model = self.sourceModel()
        if model is None or model.datatable is None:
            return True
        df = model.datatable
        if self._column_name not in df.columns:
            return True
        val = df.iat[source_row, df.columns.get_loc(self._column_name)]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            val = ""
        return self._regex.search(str(val)) is not None


class DrugNameFilterDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, pattern=""):
        super(DrugNameFilterDialog, self).__init__(parent)
        self.setWindowTitle("Filter Drug Name")
        self.setModal(True)
        self._pattern = pattern
        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(QtWidgets.QLabel("Match Drug Name with regex:"))
        self.edit = QtWidgets.QLineEdit(pattern)
        self.edit.setPlaceholderText("e.g. CCh|ML297")
        self.edit.setClearButtonEnabled(True)
        self.edit.selectAll()
        layout.addWidget(self.edit)
        self.status = QtWidgets.QLabel("")
        layout.addWidget(self.status)
        buttons = QtWidgets.QDialogButtonBox()
        buttons.addButton("Apply", QtWidgets.QDialogButtonBox.AcceptRole)
        self.clear_btn = buttons.addButton("Clear", QtWidgets.QDialogButtonBox.ResetRole)
        buttons.addButton(QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(self.tryAccept)
        buttons.rejected.connect(self.reject)
        self.clear_btn.clicked.connect(self.clearAndAccept)
        self.edit.returnPressed.connect(self.tryAccept)

    def pattern(self):
        return self._pattern

    def tryAccept(self):
        text = self.edit.text().strip()
        if not text:
            self._pattern = ""
            self.accept()
            return
        try:
            re.compile(text)
        except re.error as err:
            self.status.setText("Invalid regex: {}".format(err))
            self.status.setStyleSheet("color: #c0392b;")
            return
        self._pattern = text
        self.accept()

    def clearAndAccept(self):
        self._pattern = ""
        self.accept()


class EpisodeFilterHeader(QtWidgets.QHeaderView):
    """Show a filter funnel next to Drug Name on hover; click opens a dialog."""

    filterRequested = QtCore.pyqtSignal(int, str)
    FILTERABLE = {"Drug Name"}

    def __init__(self, orientation, parent=None):
        super(EpisodeFilterHeader, self).__init__(orientation, parent)
        self.setSectionsClickable(True)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self._hover_section = -1
        self._active_filters = set()
        self._icon_size = 12
        self._icon_gap = 5

    def setActiveFilter(self, column_name, active):
        if active:
            self._active_filters.add(column_name)
        else:
            self._active_filters.discard(column_name)
        self.viewport().update()

    def _sectionName(self, logicalIndex):
        model = self.model()
        if model is None:
            return ""
        return str(model.headerData(logicalIndex, self.orientation(), QtCore.Qt.DisplayRole) or "")

    def _isFilterable(self, logicalIndex):
        return logicalIndex >= 0 and self._sectionName(logicalIndex) in self.FILTERABLE

    def _sectionRect(self, logicalIndex):
        return QtCore.QRect(
            self.sectionViewportPosition(logicalIndex),
            0,
            self.sectionSize(logicalIndex),
            self.height(),
        )

    def _iconRect(self, section_rect, logicalIndex):
        text = self._sectionName(logicalIndex)
        fm = self.fontMetrics()
        text_w = fm.width(text)
        align = self.defaultAlignment()
        if align & QtCore.Qt.AlignHCenter:
            text_x = section_rect.center().x() - text_w // 2
        else:
            text_x = section_rect.left() + 6
        x = text_x + text_w + self._icon_gap
        max_x = section_rect.right() - self._icon_size - 3
        x = min(x, max_x)
        x = max(x, section_rect.left() + 2)
        y = section_rect.center().y() - self._icon_size // 2
        return QtCore.QRect(int(x), int(y), self._icon_size, self._icon_size)

    def _shouldShowIcon(self, logicalIndex):
        if not self._isFilterable(logicalIndex):
            return False
        name = self._sectionName(logicalIndex)
        return logicalIndex == self._hover_section or name in self._active_filters

    def paintSection(self, painter, rect, logicalIndex):
        super(EpisodeFilterHeader, self).paintSection(painter, rect, logicalIndex)
        if not self._shouldShowIcon(logicalIndex):
            return
        icon_rect = self._iconRect(rect, logicalIndex)
        active = self._sectionName(logicalIndex) in self._active_filters
        painter.save()
        painter.setClipRect(rect)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        color = QtGui.QColor(31, 119, 180) if active else QtGui.QColor(90, 90, 90)
        painter.setPen(QtGui.QPen(color, 1.2))
        painter.setBrush(QtGui.QBrush(color))
        x, y, w, h = icon_rect.x(), icon_rect.y(), float(icon_rect.width()), float(icon_rect.height())
        path = QtGui.QPainterPath()
        path.moveTo(x + 0.5, y + 1.5)
        path.lineTo(x + w - 0.5, y + 1.5)
        path.lineTo(x + w * 0.58, y + h * 0.55)
        path.lineTo(x + w * 0.58, y + h - 1)
        path.lineTo(x + w * 0.42, y + h - 1)
        path.lineTo(x + w * 0.42, y + h * 0.55)
        path.closeSubpath()
        painter.drawPath(path)
        painter.restore()

    def mouseMoveEvent(self, event):
        logical = self.logicalIndexAt(event.pos())
        hover = logical if self._isFilterable(logical) else -1
        if hover != self._hover_section:
            old = self._hover_section
            self._hover_section = hover
            if old >= 0:
                self.updateSection(old)
            if self._hover_section >= 0:
                self.updateSection(self._hover_section)
        if self._hover_section >= 0:
            icon_rect = self._iconRect(self._sectionRect(self._hover_section), self._hover_section)
            if icon_rect.adjusted(-4, -4, 4, 4).contains(event.pos()):
                self.setCursor(QtCore.Qt.PointingHandCursor)
                self.setToolTip("Filter Drug Name")
            else:
                self.unsetCursor()
                self.setToolTip("")
        else:
            self.unsetCursor()
            self.setToolTip("")
        super(EpisodeFilterHeader, self).mouseMoveEvent(event)

    def leaveEvent(self, event):
        if self._hover_section >= 0:
            old = self._hover_section
            self._hover_section = -1
            self.updateSection(old)
        self.unsetCursor()
        super(EpisodeFilterHeader, self).leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            logical = self.logicalIndexAt(event.pos())
            if self._isFilterable(logical) and self._shouldShowIcon(logical):
                icon_rect = self._iconRect(self._sectionRect(logical), logical)
                if icon_rect.adjusted(-4, -4, 4, 4).contains(event.pos()):
                    self.filterRequested.emit(logical, self._sectionName(logical))
                    event.accept()
                    return
        super(EpisodeFilterHeader, self).mousePressEvent(event)

    def sectionSizeFromContents(self, logicalIndex):
        size = super(EpisodeFilterHeader, self).sectionSizeFromContents(logicalIndex)
        if self._isFilterable(logicalIndex):
            size.setWidth(size.width() + self._icon_size + self._icon_gap + 4)
        return size


# Episode Tableview delegate for selection and highlighting
class TableviewDelegate(QtWidgets.QItemDelegate):
    def __init__(self, parent=None, *args):
        QtWidgets.QItemDelegate.__init__(self, parent, *args)

    def paint(self, painter, option, index):
        # print('here painter delegates')
        painter.save()
        # set background color
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        if (option.state & QtWidgets.QStyle.State_Selected):
            grid_color = QtGui.QColor(31,119,180,225)
            text_color = QtCore.Qt.white
        else:
            grid_color = QtCore.Qt.transparent
            text_color = QtCore.Qt.black

        # color the grid
        painter.setBrush(QtGui.QBrush(grid_color))
        painter.drawRect(option.rect)

        # color the text
        painter.setPen(QtGui.QPen(text_color))
        value = index.data(QtCore.Qt.DisplayRole)
        painter.drawText(option.rect, QtCore.Qt.AlignVCenter |QtCore.Qt.AlignHCenter, value)

        painter.restore()

# %%
class Synapse_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, startpath=None, hideScopeToolbox=True, layout=None):
        super(Synapse_MainWindow, self).__init__(parent)
        # Set up the GUI window
        self.setupUi(self)
        # Set the treeview model for directory
        self.setDataBrowserTreeView(startpath=startpath)
        self.hideScopeToolbox = hideScopeToolbox
        self.scopeLayout = layout
        self.startpath=startpath
        self.loaded_database_path = None
        self.table_from_database = False

    def setupUi(self, MainWindow):
        """This function is converted from the .ui file from the designer"""
        # Set up basic layout of the main window
        MainWindow.setObjectName(_fromUtf8("Synpase TreeView"))
        MainWindow.resize(1000, 500)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))

        # Set splitter for two panels
        self.splitter = QtWidgets.QSplitter(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))

        # Set treeview
        self.treeview = QtWidgets.QTreeView(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeview.sizePolicy().hasHeightForWidth())
        self.treeview.setSizePolicy(sizePolicy)
        # self.treeview.setTextElideMode(QtCore.Qt.ElideNone)
        self.treeview.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.treeview.header().setStretchLastSection(False)
        self.treeview.setObjectName(_fromUtf8("treeview"))

        # Set up Episode list table view
        self.tableview = QtWidgets.QTableView(self.splitter)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableview.sizePolicy().hasHeightForWidth())
        self.tableview.setSizePolicy(sizePolicy)
        self.tableview.setObjectName(_fromUtf8("tableview"))
        # additional tableview customizations
        self.tableview.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.tableview.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.tableview.setItemDelegate(TableviewDelegate(self.tableview))
        self.tableview.headers = []
        self.tableview.hiddenColumnList = []
        self.tableview.proxy = EpisodeFilterProxy(self)
        self.drug_name_filter = ""
        header = EpisodeFilterHeader(QtCore.Qt.Horizontal, self.tableview)
        header.setStretchLastSection(True)
        header.filterRequested.connect(self.openDrugNameFilterDialog)
        self.tableview.setHorizontalHeader(header)
        # self.tableview.setShowGrid(False)
        self.tableview.setStyleSheet("""QTableView{border : 20px solid white}""")
        self.tableview.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.tableview.customContextMenuRequested.connect(self.onTableContextMenu)
        delete_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence.Delete, self.tableview)
        delete_shortcut.activated.connect(self.deleteSelectedDatabaseRows)
        backspace_shortcut = QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Backspace), self.tableview)
        backspace_shortcut.activated.connect(self.deleteSelectedDatabaseRows)
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        # Set up menu bar
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 638, 100))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        self.setMenuBarItems() # call function to set menubar
        MainWindow.setMenuBar(self.menubar)

        # Set up status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        self.filter_status = QtWidgets.QLabel("")
        self.statusbar.addPermanentWidget(self.filter_status)
        MainWindow.setStatusBar(self.statusbar)

        # Execution
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ---------------- Additional main window behaviors -----------------------
    def setMenuBarItems(self):
        # File Menu
        fileMenu = self.menubar.addMenu('&File')

        # File: Load csv
        loadDBAction = QtWidgets.QAction('Load Database', self)
        loadDBAction.setStatusTip('Load a database table from a .csv, .xlsx, or .xls file')
        loadDBAction.triggered.connect(self.loadDatabase)
        fileMenu.addAction(loadDBAction)

        # File: Reload the last loaded database from disk
        reloadDBAction = QtWidgets.QAction('Reload Database', self)
        reloadDBAction.setShortcut('Ctrl+R')
        reloadDBAction.setStatusTip('Reload the currently loaded database file from disk')
        reloadDBAction.triggered.connect(self.reloadDatabase)
        fileMenu.addAction(reloadDBAction)
        
        # File: Refresh. Refresh currently selected item/directory and the loaded database
        refreshAction = QtWidgets.QAction('Refresh', self)
        refreshAction.setShortcut('F5')
        refreshAction.setStatusTip('Refresh the selected directory and reload the current database')
        refreshAction.triggered.connect(self.refreshCurrentBranch)
        fileMenu.addAction(refreshAction)
        
        # File: Settings
        settingsAction = QtWidgets.QAction("Settings", self)
        settingsAction.setStatusTip('Configure settings of PySynapse')
        settingsAction.triggered.connect(self.openSettingsWindow)
        fileMenu.addAction(settingsAction)
        
        # File: Exit
        exitAction = QtWidgets.QAction(QtGui.QIcon('exit.png'),'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit Synapse')
        exitAction.triggered.connect(self.close)
        fileMenu.addAction(exitAction)
        
        # View Menu
        viewMenu = self.menubar.addMenu('&View')

        # View: Column
        columnMenu = viewMenu.addMenu('&Additional Columns')
        drugNameAction = QtWidgets.QAction('Drug Name', self, checkable=True, checked=False)
        drugNameAction.triggered.connect(lambda: self.toggleTableViewColumnAction(4, drugNameAction))
        columnMenu.addAction(drugNameAction)

        drugTimeAction = QtWidgets.QAction('Drug Time', self, checkable=True, checked=False)
        drugTimeAction.triggered.connect(lambda: self.toggleTableViewColumnAction(5, drugTimeAction))
        columnMenu.addAction(drugTimeAction)

        dirsAction = QtWidgets.QAction('Directory', self, checkable=True, checked=False)
        dirsAction.triggered.connect(lambda: self.toggleTableViewColumnAction(7, dirsAction))
        columnMenu.addAction(dirsAction)

    def toggleTableViewColumnAction(self, column, action):
        if self.tableview.isColumnHidden(column):
            self.tableview.showColumn(column)
            action.setChecked(True)
            if column in self.tableview.hiddenColumnList:
                self.tableview.hiddenColumnList.remove(column)
        else:
            self.tableview.hideColumn(column)
            action.setChecked(False)
            if column not in self.tableview.hiddenColumnList:
                self.tableview.hiddenColumnList.append(column)

    def _columnIndex(self, name):
        headers = getattr(self.tableview, "headers", [])
        try:
            return headers.index(name)
        except ValueError:
            return None

    def _sourceRow(self, index):
        proxy = getattr(self.tableview, "proxy", None)
        if proxy is not None and index.model() is proxy:
            return proxy.mapToSource(index).row()
        return index.row()

    def openDrugNameFilterDialog(self, section, column_name):
        dlg = DrugNameFilterDialog(self, pattern=self.drug_name_filter)
        header = self.tableview.horizontalHeader()
        dlg.adjustSize()
        pos = header.mapToGlobal(QtCore.QPoint(
            header.sectionViewportPosition(section),
            header.height(),
        ))
        dlg.move(pos)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        self.drug_name_filter = dlg.pattern()
        self.applyDrugNameFilter()

    def applyDrugNameFilter(self):
        pattern = getattr(self, "drug_name_filter", "") or ""
        proxy = getattr(self.tableview, "proxy", None)
        if proxy is not None:
            proxy.setDrugNameFilter(pattern)
        header = self.tableview.horizontalHeader()
        if isinstance(header, EpisodeFilterHeader):
            header.setActiveFilter("Drug Name", bool(pattern))
        self.updateFilterStatus()
        col = self._columnIndex("Drug Name")
        if col is None:
            return
        if pattern:
            self.tableview.showColumn(col)
        elif col in getattr(self.tableview, "hiddenColumnList", []):
            self.tableview.hideColumn(col)

    def updateFilterStatus(self):
        pattern = getattr(self, "drug_name_filter", "") or ""
        if not pattern:
            self.filter_status.setText("")
            return
        text = "Filter Drug Name: {}".format(pattern)
        proxy = getattr(self.tableview, "proxy", None)
        source = getattr(self.tableview, "source_model", None)
        if proxy is not None and source is not None and source.datatable is not None:
            text = "{}  ({}/{})".format(text, proxy.rowCount(), source.rowCount())
        self.filter_status.setText(text)

    def _bindEpisodeTable(self, df, hidden_columns=None):
        self.tableview.headers = df.columns.tolist()
        source = EpisodeTableModel(df.reset_index(drop=True))
        self.tableview.source_model = source
        self.tableview.proxy.setSourceModel(source)
        self.tableview.setModel(self.tableview.proxy)
        self.tableview.verticalHeader().hide()
        if hidden_columns is None:
            hidden_columns = []
        self.tableview.hiddenColumnList = list(hidden_columns)
        for cc in range(len(self.tableview.headers)):
            self.tableview.setColumnHidden(cc, cc in self.tableview.hiddenColumnList)
        self.applyDrugNameFilter()
        self.tableview.selectionModel().selectionChanged.connect(self.onItemSelected)

    def loadDatabase(self):
        start_dir = os.path.dirname(self.loaded_database_path) if self.loaded_database_path else os.path.join(__location__, 'database')
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open File',
            start_dir,
            'Spreadsheet (*.csv *.xlsx *.xls);;All Files (*)',
        )
        if not filename:
            return
        self._loadDatabaseFile(filename)

    def reloadDatabase(self):
        path = getattr(self, "loaded_database_path", None)
        if not path:
            self.statusBar().showMessage("No database loaded")
            return
        if not os.path.isfile(path):
            QtWidgets.QMessageBox.warning(self, "Reload Database", "File not found:\n{}".format(path))
            return
        self._loadDatabaseFile(path)

    def _loadDatabaseFile(self, filename):
        if filename.lower().endswith('.csv'):
            df = pd.read_csv(filename)
        elif filename.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(filename)
        else:
            return

        cols = {c.lower(): c for c in df.columns}
        if "show" in cols:
            show = df[cols["show"]]
            if show.dtype == bool:
                mask = show
            else:
                mask = show.astype(str).str.strip().str.lower().isin(["1", "true", "yes", "t"])
            df = df.loc[mask.to_numpy()].reset_index(drop=True)

        paths = df[cols["path"]].tolist() if "path" in cols else None
        if "drug_level" in cols:
            drug_levels = df[cols["drug_level"]].tolist()
        elif "drug level" in cols:
            drug_levels = df[cols["drug level"]].tolist()
        else:
            drug_levels = None

        rename_dict = {
            "Cell": "Name",
            "Episode": "Epi",
            "SweepWindow": "Duration",
            "Drug": "Drug Name",
            "DrugTime": "Drug Time",
            "WCTime": "Time",
            "StimDescription": "Comment",
        }
        keep = [c for c in df.columns if c in rename_dict]
        df = df[keep].rename(columns=rename_dict)
        df["Sampling Rate"] = 0.1
        df["Drug Level"] = drug_levels if drug_levels is not None else 0
        df.loc[df["Drug Name"].isnull(), "Drug Name"] = ""
        df["Time"] = [NeuroData.epiTime(ttt) for ttt in df["Time"]]
        df["Drug Time"] = [NeuroData.epiTime(ttt) for ttt in df["Drug Time"]]
        if paths is not None:
            df["Dirs"] = [str(p).replace("\\", "/") for p in paths]
        else:
            df["Dirs"] = [
                os.path.join(self.startpath, get_cellpath(cb, ep)).replace("\\", "/")
                for cb, ep in zip(df["Name"], df["Epi"])
            ]
        df = df.reset_index(drop=True)
        order = sorted(
            range(len(df)),
            key=lambda i: (alphanum_key(df["Name"].iat[i]), alphanum_key(df["Epi"].iat[i])),
        )
        df = df.iloc[order].reset_index(drop=True)
        self.tableview.sequence = df.to_dict('list')
        n_rows = len(df)
        df = df.reindex(
            ["Name", "Epi", "Time", "Duration", "Drug Level", "Drug Name", "Drug Time", "Comment"],
            axis=1,
        )
        self.loaded_database_path = filename
        self.table_from_database = True
        self._bindEpisodeTable(df)
        self.statusBar().showMessage("Loaded {} ({} episodes)".format(os.path.basename(filename), n_rows))

    def refreshCurrentBranch(self):
        # Get parent index
        index = self.treeview.selectionModel().currentIndex()
        if index.isValid():
            node = self.treeview.model.getNode(index)
            if node.type == "directory":
                self.treeview.model.refreshNode(index)
        if getattr(self, "loaded_database_path", None):
            self.reloadDatabase()
            
    def openSettingsWindow(self):
        if not hasattr(self, 'settingsWidget'):
            self.settingsWidget = Settings()
        if self.settingsWidget.isclosed:
            self.settingsWidget.show()
            self.settingsWidget.isclosed = False
        
    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        return
        #quit_msg = "Are you sure you want to exit the program?"
        #reply = QtWidgets.QMessageBox.question(self, 'Message', quit_msg,
        #                                   QtWidgets.QMessageBox.Yes,
        #                                   QtWidgets.QMessageBox.No)
        #if reply == QtWidgets.QMessageBox.Yes:
        #    event.accept()
        #else:
        #    event.ignore()
        # Consider if close children windows when closing Synapse main window
        # children = ['settingsWidget', 'sw']
        # for c in children:
        #     if hasattr(self, c):
        #         getattr(self, c).close()
          
    def retranslateUi(self, MainWindow):
        """Set window title and other miscellaneous"""
        MainWindow.setWindowTitle(_translate(__version__, __version__, None))
        MainWindow.setWindowIcon(QtGui.QIcon(APP_ICON_FILE))

    # ---------------- Data browser behaviors ---------------------------------
    def setDataBrowserTreeView(self, startpath=None):
        # Set file system as model of the tree view
        # self.treeview.model = QtWidgets.QFileSystemModel()
        self.treeview.model = FileSystemTreeModel(path=startpath)
        self.treeview.setModel(self.treeview.model)
        # Set behavior upon clicked
        self.treeview.clicked.connect(self.onSequenceClicked)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def onSequenceClicked(self, index):
        """ Display a list of episodes upon sequence clicked"""
        #indexItem = self.treeview.model.index(index.row(), 0, index.parent())
        self.raise_()
        node = self.treeview.model.getNode(index)
        # Check if the item clicked is sequence instead of a folder / file
        if node.type == "sequence":
            # populate the table view on the other panel
            self.setEpisodeListTableView(node.info)
        
    # --------------- Episode list behaviors ----------------------------------
    def setEpisodeListTableView(self, sequence=None):
        if not sequence:
            return # do nothing if there is no sequence information
        self.tableview.headers = ['Epi', 'Time', 'Duration', 'Drug Level', 'Drug Name', 'Drug Time', 'Comment','Dirs', 'Stimulus', 'StimDuration']
        self.tableview.hiddenColumnList = [4, 5, 7, 8, 9] # Drug Name, Drug Time, Dirs
        # Render the data frame from sequence
        df = pd.DataFrame.from_dict(sequence)
        # sort the data frame by 'Epi' column
        epi_sort = df['Epi'].tolist()
        ind = pd.DataFrame([[int(k) for k in re.findall('\d+', m)] \
                                    for m in epi_sort])
        ind = ind.sort_values([0,1], ascending=[1,1]).index.tolist()
        df = df.reindex(ind, axis=0)
        self.tableview.sequence = df.reset_index(drop=True).to_dict('list') # data information
        # self.tableview.sequence['Name'] = self.tableview.sequence['Name'][0] # remove any duplication
        # get the subset of columns based on column settings
        df = df.reindex(self.tableview.headers, axis=1)
        self.table_from_database = False
        self._bindEpisodeTable(df, hidden_columns=self.tableview.hiddenColumnList)
        # self.tableview.clicked.connect(self.onItemSelected)

    def _selectedSourceRows(self):
        sm = self.tableview.selectionModel()
        if sm is None:
            return []
        return sorted({self._sourceRow(idx) for idx in sm.selectedRows()})

    def onTableContextMenu(self, pos):
        index = self.tableview.indexAt(pos)
        if index.isValid():
            sm = self.tableview.selectionModel()
            if sm is not None and not sm.isSelected(index):
                self.tableview.selectRow(index.row())
        menu = QtWidgets.QMenu(self)
        deleteAction = menu.addAction("Delete")
        can_delete = (
            bool(self._selectedSourceRows())
            and bool(getattr(self, "table_from_database", False))
            and bool(getattr(self, "loaded_database_path", None))
        )
        deleteAction.setEnabled(can_delete)
        chosen = menu.exec_(self.tableview.viewport().mapToGlobal(pos))
        if chosen == deleteAction:
            self.deleteSelectedDatabaseRows()

    def deleteSelectedDatabaseRows(self):
        if not getattr(self, "table_from_database", False):
            return
        db_path = getattr(self, "loaded_database_path", None)
        if not db_path:
            QtWidgets.QMessageBox.information(
                self, "Delete", "Delete is available after loading a database file."
            )
            return
        rows = self._selectedSourceRows()
        if not rows:
            return
        sequence = self.tableview.sequence
        n = len(rows)
        basename = os.path.basename(db_path)
        if n == 1:
            name = sequence.get("Name", [""])[rows[0]] if "Name" in sequence else ""
            epi = sequence["Epi"][rows[0]]
            label = "{} {}".format(name, epi).strip()
            question = "Delete {} from {}?\n\nThis cannot be undone.".format(label, basename)
        else:
            question = "Delete {} episodes from {}?\n\nThis cannot be undone.".format(n, basename)
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete",
            question,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return
        paths = {
            str(sequence["Dirs"][i]).replace("\\", "/")
            for i in rows
            if "Dirs" in sequence
        }
        cell_epi = set()
        names = sequence.get("Name")
        epis = sequence.get("Epi")
        if names is not None and epis is not None:
            cell_epi = {(str(names[i]), str(epis[i])) for i in rows}
        try:
            removed = self._removeRowsFromDatabaseFile(db_path, paths, cell_epi)
        except Exception as err:
            QtWidgets.QMessageBox.warning(self, "Delete", "Could not update the file:\n{}".format(err))
            return
        if removed == 0:
            QtWidgets.QMessageBox.warning(self, "Delete", "No matching rows were found in {}.".format(basename))
            return
        self._loadDatabaseFile(db_path)
        self.statusBar().showMessage("Deleted {} episode(s) from {}".format(removed, basename))

    def _removeRowsFromDatabaseFile(self, db_path, paths, cell_epi):
        lower = db_path.lower()
        if lower.endswith(".csv"):
            with open(db_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = list(reader)
            if not fieldnames:
                return 0
            kept = []
            removed = 0
            for row in rows:
                csv_path = str(row.get("path") or "").replace("\\", "/")
                cell = str(row.get("Cell") or "")
                epi = str(row.get("Episode") or "")
                if (csv_path and csv_path in paths) or ((cell, epi) in cell_epi):
                    removed += 1
                    continue
                kept.append(row)
            with open(db_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(kept)
            return removed
        if lower.endswith((".xlsx", ".xls")):
            df = pd.read_excel(db_path)
            cols = {c.lower(): c for c in df.columns}
            drop = pd.Series(False, index=df.index)
            if "path" in cols:
                drop = drop | df[cols["path"]].astype(str).str.replace("\\", "/", regex=False).isin(paths)
            if "cell" in cols and "episode" in cols:
                pairs = list(zip(df[cols["cell"]].astype(str), df[cols["episode"]].astype(str)))
                drop = drop | pd.Series([p in cell_epi for p in pairs], index=df.index)
            removed = int(drop.sum())
            df.loc[~drop.to_numpy()].to_excel(db_path, index=False)
            return removed
        raise ValueError("Unsupported database file type")

    @QtCore.pyqtSlot(QtCore.QItemSelection, QtCore.QItemSelection)
    def onItemSelected(self, selected, deselected):
        """Executed when an episode in the tableview is clicked"""
        # Get the information of last selected item
        if not selected and not deselected:
            return
        try:
            ind = self._sourceRow(selected.indexes()[-1])
        except:
            ind = self._sourceRow(deselected.indexes()[-1])
        sequence = self.tableview.sequence
        drugName = sequence['Drug Name'][ind]
        if not drugName: # in case of empty string
            drugName = str(sequence['Drug Level'][ind])
        ep_info_str = "ts: {:0.1f} ms; Drug: {} ({})".format(sequence['Sampling Rate'][ind], drugName, sequence['Drug Time'][ind])
        pattern = getattr(self, "drug_name_filter", "") or ""
        if pattern:
            ep_info_str = "{}; Filter Drug Name: {}".format(ep_info_str, pattern)
        self.statusBar().showMessage(ep_info_str)
        self.setWindowTitle("{}  {}".format(__version__, sequence['Dirs'][ind]))
        # Get selected row
        indexes = self.tableview.selectionModel().selectedRows()
        rows = [self._sourceRow(index) for index in sorted(indexes)]
        # if not rows: # When nothing is selected, keep the last selected item on the Scope
        #     return
        # Call scope window
        if not hasattr(self, 'sw'): # Start up a new window
            # self.sw = ScopeWindow(parent=self)
            self.sw = ScopeWindow(partner=self, hideDock=self.hideScopeToolbox, layout=self.scopeLayout) # new window
        if self.sw.isclosed:
            self.sw.show()
            self.sw.isclosed = False
        # update existing window
        self.sw.updateEpisodes(episodes=sequence, index=rows)


if __name__ == '__main__':
    sys.excepthook = my_excepthook # helps prevent uncaught exception crashing the GUI
    app = QtWidgets.QApplication(sys.argv)
    apply_app_theme(app)
    apply_app_icon(app)
    # Ctrl+C (SIGINT) is otherwise stuck in Qt's C++ loop until some Python slot runs
    signal.signal(signal.SIGINT, lambda *args: QtWidgets.QApplication.quit())
    _wakeup = QtCore.QTimer()
    _wakeup.start(250)
    _wakeup.timeout.connect(lambda: None)
    running_os = sys.platform[:3]
    startpath = settings.startpath.get(running_os)
    w = Synapse_MainWindow(
        startpath=startpath,
        hideScopeToolbox=settings.hide_scope_toolbox,
    )
    w.show()
    # Connect upon closin
    # app.aboutToQuit.connect(restartpyshell)
    # Make sure the app stays on the screen
    sys.exit(app.exec_())
