# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'SynapseQt.ui'
#
# Created: Sat Apr 18 21:40:21 2015
#      by: PyQt4 UI code generator 4.10.4
#
# WARNING! All changes made in this file will be lost!

import os
import sys
import re
import glob
import numpy as np

from PyQt4 import QtCore, QtGui
from FileIO.CoreDatFileIO import readDatFile

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

class Sequence(object):
    """Sequence object"""
    def __init__(self, P):
        self.make_sequence(P)

    def make_sequence(P):
        """Extract episode information in order to create a table
           Set name of the sequence based on the list of files"""
        Q = [os.path.basename(p) for p in P] # path
        Z = ['S%s.E%s'%re.findall('.S(\d+).E(\d+).dat', q)[0] for q in Q]
        Q = [re.split('.S(\d+).E(\d+).dat', q)[0] for q in Q] # name
        # get unique IDs
        names, _, inverse, counts = np.unique(Q, return_index=True, return_inverse=True, return_counts=True)
        self.sequence = []

        for n, nm in enumerate(names):
            self.sequence.append({'Name':('%s (%d)'%(nm, counts[n])), 
                'Dirs':list(P[list(np.where(inverse==n)[0])]), 
                'Epi':list(Z[list(np.where(inverse==n)[0])]), 
                'Time':[],
                'Duration':[],
                'Drug Level':[],
                'Comment': []
                })
            # load episode info
            for d in self.sequence[n]['Dirs']:
                zData = readDatFile(d, readTraceData = False)
                self.sequence[n]['Time'].append = zData['cellTimeStr']
                self.sequence[n]['Duration'].append = zData['sweepWindow']
                self.sequence[n]['Drug Level'].append = zData['drugLevel']
                self.sequence[n]['Comment'].append = mEpi["stimDesc"]


# Listing episodes from a sequence
class SequenceListingTree(QtGui.QApplication):
    def __init__(self, parent=None):
        """Initialize a table view for sequence browser"""
        super(SequenceListingTree, self).__init__(parent)


class Ui_MainWindow(QtGui.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # Set up the GUI window
        self.setupUi(self)
        # Set the treeview model for directory
        self.setDataBrowser_treeview()

    def setupUi(self, MainWindow):
        """This function is converted from the .ui file from the designer"""
        # Set up basic layout of the main window
        MainWindow.setObjectName(_fromUtf8("Synpase TreeView"))
        MainWindow.resize(638, 275)
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
        #self.treeview.setSizeAdjustPolicy(QtGui.QAbstractScrollArea.AdjustToContents)
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
        self.horizontalLayout.addWidget(self.splitter)
        MainWindow.setCentralWidget(self.centralwidget)

        # Set up menu bar
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 638, 21))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        MainWindow.setMenuBar(self.menubar)

        # Set up status bar
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)

        # Execution
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    # ---------------- Data browser behaviors ---------------------------------
    def setDataBrowser_treeview(self):
        # Set file system as model of the tree view
        self.treeview.model = QtGui.QFileSystemModel()
        self.treeview.model.setRootPath( QtCore.QDir.currentPath() )
        self.treeview.setModel(self.treeview.model)
        # Hide columns in file system model
        for x in range(0, self.treeview.model.columnCount()):
            self.treeview.hideColumn(x+1)
        #self.treeview.setColumnWidth(0, 200)
        # Set behavior upon clicked
        self.treeview.clicked.connect(self.on_sequence_clicked)
        # Set behavior upon expanded
        self.treeview.expanded.connect(self.on_treeview_expanded)
        
    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_treeview_expanded(self, index):
        """Return file path and file name upon tree expansion"""
        indexItem = self.treeview.model.index(index.row(), 0, index.parent())
        # path or filename selected
        self.current_fileName = self.treeview.model.fileName(indexItem)
        # full path/filename selected
        self.current_filePath = self.treeview.model.filePath(indexItem)
        if os.path.isdir(self.current_filePath): # clikced on folder
            # list desired files / sequences; modify display if found targets
            self.file_sequence_list(self.current_filePath)
        if self.available_files:
            # replace the tree items in the expansion with Sequence
            self.modify_treeview_model()
    
    @QtCore.pyqtSlot(QtCore.QModelIndex)
    def on_sequence_clicked(self, index):
        """ Display a list of episodes upon sequence clicked"""
        indexItem = self.treeview.model.index(index.row(), 0, index.parent())
        # Check if the item clicked is sequence instead of a folder / file
        if isinstance(indexItem, Sequence):
            print('is sequence')
        else:
            print('not a sequence')
 
    def file_sequence_list(self, P, delimiter='.', ext='*.dat'):
        """List files and extract common names as sequence
        P is the full path that contains the file sequence
        """
        import glob
        P = P.encode('ascii','ignore')
        # Make sure only files, not folders are used
        self.available_files = glob.glob(os.path.join(P,ext))
        if not self.available_files:
            return
        self.available_files.sort() # sort lexicologically
        # Get sequence
        self.available_sequences = [os.path.basename(f).split(delimiter)[0] for f in self.available_files]
        # get indices of list
        self.available_sequences, self.available_indices = np.unique(self.available_sequences, return_inverse=True)
        
    def modify_treeview_model(self):
        # list of sequences
        S = Sequence(self.available_files)
        S = S.sequence # get the list
        for n, s in enumerate(S): # iterate over the sequence of cells
            t = QtGui.QTreeWidgetItem(self.treeview)
            font = QtGui.QFont()
            font.setWeight(QFont.Bold)
            t.setFont(0, font)
            t.setText(0, s['Name'])
            QtGui.QtreeWidgetItem(self.treeview).setText(0, "World")
        
    # --------------- Episode list behaviors --------------------------------------    
    def setEpisodeList_tableview(self):
        # Set behavior upon selection
        self.tableview.clicked.connect(self.on_sequence_clicked)        

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))

if __name__ == '__main__':
    import sys
    app = QtGui.QApplication(sys.argv)
    w = Ui_MainWindow()
    w.show()
    # Make sure the app stays on the screen
    sys.exit(app.exec_())