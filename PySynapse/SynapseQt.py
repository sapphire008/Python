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
import glob
import six
import numpy as np
from pdb import set_trace
import subprocess
import pandas as pd

from util.ImportData import NeuroData
from app.Scope import Scope

import sip
sip.setapi('QVariant', 2)

# Routines for Qt import errors
from PyQt4 import QtCore, QtGui
try:
    from PyQt4.QtCore import QString
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
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)

# Set some global variables
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__version__ = "PySynapse 0.2"


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
            return(False)

        self.children.insert(position, child)
        child.parent = self

        return(True)

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
        path = 'D:/Data/Traces/2016/02.February/Data 8 Feb 2016'
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
            drives = subprocess.check_output('wmic logicaldisk get name', stderr=subprocess.STDOUT)
            drives = drives.decode('utf-8')
            drives = drives.split('\n') # split by lines
            for d in drives:
                if 'Name' in d:
                    continue
                dpath = re.split('[\s]+',d)[0]
                if not dpath: # if empty string
                    continue
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
            if str(os.path.splitext(file)[1]).lower() == '.dat':
                dat_files.append(file)
            elif str(os.path.splittext(file)[1].lower()) == '.img':
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
            file_path = os.path.join(path, s['Name']+'.{}.dat')
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
            return(None)
        elif role == QtCore.Qt.BackgroundRole: # insert highlight color here
            return(QtGui.QBrush(QtCore.Qt.transparent))
        else:
            return(None)

    def insertNodes(self, position, nodes, parent=QtCore.QModelIndex()):
        node = self.getNode(parent)

        self.beginInsertRows(parent, position, position + len(nodes) - 1)

        for child in nodes:
            success = node.insert_child(position, child)

        self.endInsertRows()

        return(success)

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
        # set_trace()
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
                'Comment': [],
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
        
# Episode Tableview delegate for selection and highlighting
class TablviewDelegate(QtGui.QItemDelegate):
    def __init__(self, parent=None, *args):
        QtGui.QItemDelegate.__init__(self, parent, *args)
    
    def paint(self, painter, option, index):
        # print('here painter delegates')
        painter.save()
        # set background color
        painter.setPen(QtGui.QPen(QtCore.Qt.NoPen))
        if (option.state & QtGui.QStyle.State_Selected):
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
        painter.drawText(option.rect, QtCore.Qt.AlignLeft, value)
            
        painter.restore()
            

class Ui_MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # Set up the GUI window
        self.setupUi(self)
        # Set the treeview model for directory
        self.setDataBrowserTreeView()

    def setupUi(self, MainWindow):
        """This function is converted from the .ui file from the designer"""
        # Set up basic layout of the main window
        MainWindow.setObjectName(_fromUtf8("Synpase TreeView"))
        MainWindow.resize(1000, 500)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.horizontalLayout = QtGui.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName(_fromUtf8("horizontalLayout"))

        # Set splitter for two panels
        self.splitter = QtGui.QSplitter(self.centralwidget)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.splitter.sizePolicy().hasHeightForWidth())
        self.splitter.setSizePolicy(sizePolicy)
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
        self.splitter.setObjectName(_fromUtf8("splitter"))

        # Set treeview
        self.treeview = QtGui.QTreeView(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.treeview.sizePolicy().hasHeightForWidth())
        self.treeview.setSizePolicy(sizePolicy)
        self.treeview.setTextElideMode(QtCore.Qt.ElideNone)
        self.treeview.setObjectName(_fromUtf8("treeview"))

        # Set up Episode list table view
        self.tableview = QtGui.QTableView(self.splitter)
        sizePolicy = QtGui.QSizePolicy(QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(3)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableview.sizePolicy().hasHeightForWidth())
        self.tableview.setSizePolicy(sizePolicy)
        self.tableview.setObjectName(_fromUtf8("tableview"))
        # additional tableview customizations
        self.tableview.setSelectionMode(QtGui.QAbstractItemView.ExtendedSelection)
        self.tableview.setSelectionBehavior(QtGui.QAbstractItemView.SelectRows)
        self.tableview.setItemDelegate(TablviewDelegate(self.tableview))
        
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        # Additional actions
        exitAction = QtGui.QAction(QtGui.QIcon('exit.png'),'Exit', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit Synapse')
        exitAction.triggered.connect(self.close)

        columnAction = QtGui.QAction(QtGui.QIcon('exit23.png'), 'Column', self)
        columnAction.setShortcut('Ctrl+E')
        columnAction.setStatusTip('Set Episode display columns')
        columnAction.triggered.connect(self.columnCheckList)

        # Set up menu bar
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 638, 100))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        fileMenu = self.menubar.addMenu('&File')
        fileMenu.addAction(exitAction)
        EpisodeMenu = self.menubar.addMenu('&Episode')
        EpisodeMenu.addAction(columnAction)
        MainWindow.setMenuBar(self.menubar)

        # Set up status bar
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        # Execution
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ---------------- Additional Main window behaviors -----------------------
    def closeEvent(self, event):
        """Override default behavior when closing the main window"""
        #quit_msg = "Are you sure you want to exit the program?"
        #reply = QtGui.QMessageBox.question(self, 'Message', quit_msg,
        #                                   QtGui.QMessageBox.Yes,
        #                                   QtGui.QMessageBox.No)
        #if reply == QtGui.QMessageBox.Yes:
        #    event.accept()
        #else:
        #    event.ignore()

    # ---------------- Data browser behaviors ---------------------------------
    def setDataBrowserTreeView(self):
        # Set file system as model of the tree view
        # self.treeview.model = QtGui.QFileSystemModel()
        self.treeview.model = FileSystemTreeModel()
        self.treeview.setModel(self.treeview.model)
        # Set behavior upon clicked
        self.treeview.clicked.connect(self.onSequenceClicked)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def onSequenceClicked(self, index):
        """ Display a list of episodes upon sequence clicked"""
        #indexItem = self.treeview.model.index(index.row(), 0, index.parent())
        node = self.treeview.model.getNode(index)
        # Check if the item clicked is sequence instead of a folder / file
        if node.type == "sequence":
            # populuate the table view on the other panel
            self.setEpisodeListTableView(node.info)

    # --------------- Episode list behaviors ----------------------------------
    def setEpisodeListTableView(self, sequence=None):
        if not sequence:
            return # do nothing if there is no sequence information
        # Render the data frame from sequence
        df = pd.DataFrame.from_dict(sequence)
        # sort the data frame by 'Epi' column
        epi_sort = df['Epi'].tolist()
        ind = pd.DataFrame([[int(k) for k in re.findall('\d+', m)] \
                                    for m in epi_sort])
        ind = ind.sort_values([0,1], ascending=[1,1]).index.tolist()
        df = df.reindex_axis(ind, axis=0)
        self.tableview.sequence = df.reset_index(drop=True).to_dict('list') # data information
        # get the subset of columns based on column settings
        headers = ['Epi', 'Time', 'Duration', 'Drug Level', 'Comment']
        df = df.reindex_axis(headers, axis=1)
        self.tableview.model = EpisodeTableModel(df)
        self.tableview.setModel(self.tableview.model)
        self.tableview.verticalHeader().hide()
        # Set behavior upon selection
        self.tableview.clicked.connect(self.onItemClicked)

    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def onItemClicked(self, index):
        """Executed when an episode in the tableview is clicked"""
        # self.tableview.model.selectedRow = index.row()
        # print('clicked row:%d, col:%d'%(index.row(), index.column()))
        # Get the information of currently selected item
        ind = index.row()
        sequence = self.tableview.sequence
        drugName = sequence['Drug Name'][ind]
        if not drugName: # in case of empty string
            drugName = str(sequence['Drug Level'][ind])
        ep_info_str = "ts: {:0.1f} ms; Drug: {} ({})".format(sequence['Sampling Rate'][ind], drugName, sequence['Drug Time'][ind])
        self.statusBar().showMessage(ep_info_str)
        self.setWindowTitle("{}  {}".format(__version__, sequence['Dirs'][ind]))
        # Get selected row
        indexes = self.tableview.selectionModel().selectedRows()
        rows = [index.row() for index in sorted(indexes)]
        # Call scope window
        Scope(sequence, index=rows)

    def columnCheckList(self):
        print("Column changed")
        model = QtGui.QStandardItemModel()
        item = QtGui.QStandardItem("Item")
        item.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        item.setData(QtCore.QVariant(QtCore.Qt.Checked), QtCore.Qt.CheckStateRole)
        model.appendRow(item)
        view = QtGui.QListView()
        view.setModel(model)
        view.show()

    def retranslateUi(self, MainWindow):
        """Set window title and other miscellaneous"""
        MainWindow.setWindowTitle(_translate(__version__, __version__, None))

if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    w = Ui_MainWindow()
    w.show()
    # Connect upon closing
    # app.aboutToQuit.connect(restartpyshell)
    # Make sure the app stays on the screen
    sys.exit(app.exec_())
