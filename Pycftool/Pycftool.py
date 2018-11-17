# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\Edward\Documents\Assignments\Scripts\Python\Pycftool\resources\ui_designer\Pycftool_3.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!


import os
import sys
import numpy as np
from scipy.optimize import curve_fit
from pdb import set_trace

from PyQt5 import QtCore, QtGui, QtWidgets
from QCodeEdit import QCodeEdit
from ElideQLabel import ElideQLabel
from FitOptions import *


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__version__ = "PySynapse 0.4"

def my_excepthook(type, value, tback):
    """This helps prevent program crashing upon an uncaught exception"""
    sys.__excepthook__(type, value, tback)


import sip
sip.setapi('QVariant', 2)

# Routines for Qt import errors
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsLayoutWidget
import pyqtgraph.opengl as gl
#view = gl.GLViewWidget()
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

# try:
#     _encoding = QtGui.QApplication.UnicodeUTF8
#     def _translate(context, text, disambig):
#         return QtCore.QCoreApplication.translate(context, text, disambig, _encoding)
# except AttributeError:
#     def _translate(context, text, disambig):
#         return QtCore.QCoreApplication.translate(context, text, disambig)

_translate = QtCore.QCoreApplication.translate

class cftool_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, vars=[locals(), globals()]):
        super(cftool_MainWindow, self).__init__(parent)
        self.vars = self.filterVars(vars)
        self.xdata = np.nan
        self.ydata = np.nan
        self.zdata = np.nan
        self.wdata = np.nan
        self.availableData = {}
        self.currentDataType = 'None'
        self.varnames = {}
        self.autofit = 2
        self.centerscale = 0
        self.params = {}
        self.methods = {}
        # Display
        self.graphicsView = None
        # Set up fit options
        self.options = FitOptions(friend=self, method='2D: curve_fit')
        self.DEBUG = False
        # Set up GUI window
        self.setupUi(self)

    def setupUi(self, MainWindow):
        """This function is converted from the .ui file from the designer"""
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)

        # Grid layout to configure the tab and dock widget
        self.gridLayoutMain = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayoutMain.setObjectName("gridLayoutMain")

        self.gridLayoutTab = QtWidgets.QGridLayout()
        self.gridLayoutTab.setObjectName("gridLayoutTab")

        # Initialize tab
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabsClosable(True)
        self.tabWidget.setMovable(True)
        self.tabWidget.tabCloseRequested.connect(self.closeTab)
        self.tabWidget.setObjectName("tabWidget")

        # First tab
        self.tab = QtWidgets.QWidget() # current tab
        self.tab.setObjectName("tab")
        self.tabWidget.addTab(self.tab, "")
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Untitled Fit 1"))
        self.tabWidget.tabIndex = 1
        # Initialize the content of the tab
        self.initializeTabContents(self.tab)

        # Tab that will add another tab
        self.tabButton = QtWidgets.QToolButton()
        self.tabButton.setObjectName("tabButton")
        self.tabButton.setText("+")
        font = self.tabButton.font()
        font.setBold(True)
        self.tabButton.setFont(font)
        self.tabWidget.setCornerWidget(self.tabButton)
        self.tabButton.clicked.connect(lambda: self.newFit())

        # Adding the tab widget to the layout
        self.gridLayoutTab.addWidget(self.tabWidget, 0, 0, 1, 1)
        # Add the tab layout to the main layout
        self.gridLayoutMain.addLayout(self.gridLayoutTab, 0, 0, 1, 1)


        # <editor-fold desc="Configure dockWidget Fit Table">
        # Table of fits dockWidget
        self.fitTable_dockWidget = QtWidgets.QDockWidget(MainWindow)
        self.fitTable_dockWidget.setObjectName("fitTable_dockWidget")
        self.fitTable_dockWidgetContents = QtWidgets.QWidget()
        self.fitTable_dockWidgetContents.setObjectName("fitTable_dockWidgetContents")
        self.fitTable_dockWidget.hide()

        # Layout for table of fits
        self.gridLayoutFitsTable = QtWidgets.QGridLayout(self.fitTable_dockWidgetContents)
        self.gridLayoutFitsTable.setObjectName("gridLayoutFitsTable")
        self.gridLayoutTable = QtWidgets.QGridLayout()
        self.gridLayoutTable.setSpacing(7)
        self.gridLayoutTable.setObjectName("gridLayoutTable")
        self.tableWidget = QtWidgets.QTableWidget(self.fitTable_dockWidgetContents)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.gridLayoutTable.addWidget(self.tableWidget, 0, 0, 1, 1)
        self.gridLayoutFitsTable.addLayout(self.gridLayoutTable, 0, 0, 1, 1)

        # DockWidget table of fits
        self.fitTable_dockWidget.setWidget(self.fitTable_dockWidgetContents)
        MainWindow.addDockWidget(QtCore.Qt.DockWidgetArea(8), self.fitTable_dockWidget)
        # </editor-fold>

        # Menu
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        self.setMenuBarItems()   # Call function to set menubar
        MainWindow.setMenuBar(self.menubar)

        # Set up status bar
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        # Run
        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def newFit(self, title=None):
        """Insert a new tab"""
        tab = QtWidgets.QWidget()   # Create a new tab
        self.tabWidget.addTab(tab, "")  # Insert the new tab right before the tab_plus
        if title is None:
            self.tabWidget.tabIndex = self.tabWidget.tabIndex + 1
            index = self.tabWidget.tabIndex
            title = "Untitled Fit " + str(index)

        tab_index =self.tabWidget.indexOf(tab)
        self.tabWidget.setTabText(tab_index, _translate("MainWindow", title))
        self.tabWidget.setCurrentIndex(tab_index)  # switch to that tab
        self.initializeTabContents(tab)  # initialize the contents

    def initializeTabContents(self, tab):
        """Tab content"""
        gridLayoutFits = QtWidgets.QGridLayout(tab)

        # Method
        method_groupBox = QtWidgets.QGroupBox(tab)
        method_groupBox.setTitle("")
        method_groupBox.setObjectName("method_groupBox")
        self.method_groupBox = method_groupBox
        self.initialize_method(method_groupBox)
        gridLayoutFits.addWidget(method_groupBox, 0, 4, 1, 8)

        # Display
        display_groupBox = QtWidgets.QGroupBox(tab)
        display_groupBox.setTitle("")
        display_groupBox.setObjectName("displayGroupBox")
        display_groupBox.setMinimumSize(500, 280)
        self.initialize_display(display_groupBox)
        self.display_groupBox = display_groupBox
        gridLayoutFits.addWidget(display_groupBox, 1, 3, 1, 11)

        # AutoFit
        autofit_groupBox = QtWidgets.QGroupBox(tab)
        autofit_groupBox.setTitle("")
        self.initialize_autofit(autofit_groupBox)
        gridLayoutFits.addWidget(autofit_groupBox, 0, 12, 1, 2)

        # Results
        results_groupBox = QtWidgets.QGroupBox(tab)
        results_groupBox.setTitle(_translate("MainWindow", "Results"))
        self.initialize_results(results_groupBox)
        results_groupBox.setMinimumWidth(300)
        gridLayoutFits.addWidget(results_groupBox, 1, 0, 1, 3)

        # Data: Initialize data after everything, in case of a call of the class with an input
        data_groupBox = QtWidgets.QGroupBox(tab)
        data_groupBox.setTitle("")
        # data_groupBox.setObjectName("dataGroupBox")
        self.initialize_data(data_groupBox)
        gridLayoutFits.addWidget(data_groupBox, 0, 0, 1, 4)

        return gridLayoutFits

    def initialize_data(self, gbox):
        """Initialize the data groupbox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())
        fitname_label = QtWidgets.QLabel("Fit Name:")
        fitname_text  = QtWidgets.QLineEdit()
        fitName = self.tabWidget.tabText(self.tabWidget.indexOf(self.tabWidget.currentWidget()))
        fitname_text.setText(fitName)
        fitname_text.editingFinished.connect(lambda: self.changeTabTitle(fitname_text.text()))

        comboList = ['(none)']+list(self.vars.keys())
        x_label = QtWidgets.QLabel("X data:")
        x_comboBox  = QtWidgets.QComboBox()
        x_comboBox.addItems(comboList)
        x_comboBox.currentIndexChanged.connect(lambda: self.onDataChanged('xdata', x_comboBox.currentText()))
        x_comboBox.setCurrentIndex(1)

        y_label = QtWidgets.QLabel("Y data:")
        y_comboBox  = QtWidgets.QComboBox()
        y_comboBox.addItems(comboList)
        y_comboBox.currentIndexChanged.connect(lambda: self.onDataChanged('ydata', y_comboBox.currentText()))
        y_comboBox.setCurrentIndex(2)

        z_label = QtWidgets.QLabel("Z data:")
        z_comboBox  = QtWidgets.QComboBox()
        z_comboBox.addItems(comboList)
        z_comboBox.currentIndexChanged.connect(lambda: self.onDataChanged('zdata', z_comboBox.currentText()))
        #z_comboBox.setCurrentIndex(3)

        w_label = QtWidgets.QLabel("Weights:")
        w_comboBox  = QtWidgets.QComboBox()
        w_comboBox.addItems(comboList)
        w_comboBox.currentIndexChanged.connect(lambda: self.onDataChanged('wdata', w_comboBox.currentText()))
        #w_comboBox.setCurrentIndex(0)

        gbox.layout().addWidget(fitname_label, 0, 0, 1, 1)
        gbox.layout().addWidget(fitname_text, 0, 1, 1, 1)
        gbox.layout().addWidget(x_label, 1, 0, 1, 1)
        gbox.layout().addWidget(x_comboBox, 1, 1, 1, 1)
        gbox.layout().addWidget(y_label, 2, 0, 1, 1)
        gbox.layout().addWidget(y_comboBox, 2, 1, 1, 1)
        gbox.layout().addWidget(z_label, 3, 0, 1, 1)
        gbox.layout().addWidget(z_comboBox, 3, 1, 1, 1)
        gbox.layout().addWidget(w_label, 4, 0, 1, 1)
        gbox.layout().addWidget(w_comboBox, 4, 1, 1, 1)

    def changeTabTitle(self, new_text):
        tab_index = self.tabWidget.indexOf(self.tabWidget.currentWidget())
        self.tabWidget.setTabText(tab_index, _translate("MainWindow", new_text))

    def onDataChanged(self, key, valname):
        """Change, then graph, then fit the data upon data combobox index changed"""
        # Set data
        if valname == '(none)':
            setattr(self, key, np.nan)
            del self.availableData[key]
        else:
            setattr(self, key, self.vars[valname])
            self.availableData[key] = len(self.vars[valname])
            self.varnames[key] = valname
            if len(self.vars[valname]) < 3:
                # warn for data less than 3 elements
                popup_messageBox = QtWidgets.QMessageBox()
                popup_messageBox.setWindowTitle('Warning')
                popup_messageBox.setText('Warning: Array length cannot be less than 3')
                popup_messageBox.exec_()

        data_list, data_len = self.availableData.keys(), self.availableData.values()
        and_join = lambda x: ", ".join(x)[::-1].replace(" ,", " dna ", 1)[::-1].replace(
                    " and", ", and" if len(x) > 2 else " and")

        # Check to see if the data all has the same length
        if len(set(data_len)) != 1:
            samelength_warngBox = QtWidgets.QMessageBox()
            samelength_warngBox.setWindowTitle('Warning')
            samelength_warngBox.setText('Warning: {} array length are not the same'.format(and_join(data_list)))
            samelength_warngBox.setInformativeText(str(self.availableData))
            samelength_warngBox.exec_()

        # Change the method list comboBox
        if set(data_list) == set(['xdata']) or \
            set(data_list) == set(['xdata', 'ydata']) or \
            set(data_list) == set(['xdata', 'ydata', 'wdata']):
            newDataType = '2D'
        else:
            newDataType = '3D'

        if self.currentDataType != newDataType:
            self.currentDataType = newDataType
            method_comboBox = self.method_groupBox.layout().itemAt(0).widget()
            if newDataType == '3D':
                self.setMethodLists(method_comboBox, methodList=1)
            else: # 2D
                self.setMethodLists(method_comboBox, methodList=2)

        # Issue scatter plots
        self.graphData(data_list)

        # Do the fitting
        self.curveFit()

    def initialize_method(self, gbox, method_list='Method 2', current_method='Custom Equation'):
        """Initialize the method groupBox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())
        if method_list == 'Method 1': # 3D data
            method_set_list = ['Custom Equation', 'Interpolant', 'Lowess', 'Polynomial']
        else: # 'Method 2', # 2D data
            method_set_list = ['Custom Equation', 'Exponential', 'Fourier', 'Gaussian', 'Interpolant',
                                'Linear Fitting', 'Polynomial', 'Power', 'Rational', 'Smoothing Spline',
                                'Sum of Sine', 'Weibull']
        method_comboBox = QtWidgets.QComboBox()
        method_comboBox.setObjectName(method_list)
        method_comboBox.addItems(method_set_list)
        method_comboBox.setCurrentIndex(method_set_list.index(current_method))
        method_comboBox.currentIndexChanged.connect(lambda: self.toggleMethods(method_comboBox.currentText()))
        gbox.layout().addWidget(method_comboBox, 0, 0, 1, 8)
        self.toggleMethods(method=current_method)

    def setMethodLists(self,  method_comboBox, methodList=1, currentMethod=None):
        """Called when data type changed"""
        # Block the signal during reset
        method_comboBox.blockSignals(True)
        method_comboBox.clear()

        if methodList == 1:  # 3D data
            method_comboBox.setObjectName("Method 1")
            method_list = ['Custom Equation', 'Interpolant', 'Lowess', 'Polynomial']
            currentMethod = 'Custom Equation' if currentMethod is None else currentMethod
        else:  # 2D data
            method_comboBox.setObjectName("Method 2")
            method_list = ['Custom Equation', 'Exponential', 'Fourier', 'Gaussian', 'Interpolant',
                                      'Linear Fitting', 'Polynomial', 'Power', 'Rational', 'Smoothing Spline',
                                      'Sum of Sine', 'Weibull']
            currentMethod = 'Exponential' if currentMethod is None else currentMethod
        # Adding method list
        method_comboBox.addItems(method_list)
        # re-enable the signal
        method_comboBox.blockSignals(False)
        # Reset method
        try:
            current_index = method_list.index(currentMethod)
            method_comboBox.setCurrentIndex(current_index)
        except:
            raise(ValueError('Cannot set method "{}", which is not in the list of valid methods'.format(currentMethod)))

    def toggleMethods(self, method, block_signal=False):
        gbox = self.method_groupBox
        method_comboBox = gbox.layout().itemAt(0).widget()
        if block_signal: method_comboBox.blockSignals(True)
        # Get the setting table
        if method_comboBox.objectName() == 'Method 1':
            methods_layout, methods_dict = self.methodSet1(method=method)
        else:
            methods_layout, methods_dict = self.methodSet2(method=method)
        # Remove everything at and below the setting rows: rigid setting
        gbox = self.removeFromWidget(gbox, row=1)

        for key, val in methods_layout.items():
            gbox.layout().addWidget(val, *key[0], *key[1]) # widget, (x, y), (w, l)

        self.methods = methods_dict

        if self.autofit and hasattr(self, 'result_textBox') and self.result_textBox.toPlainText():
            print('redo fitting upon method change')
            self.curveFit() # redo the fitting

        if block_signal: method_comboBox.blockSignals(False)

    def removeFromWidget(self, widgetFrame, row=0):
        """Remove widgets from a widgetFrame below row"""
        nrows = widgetFrame.layout().rowCount()
        if nrows > row:
            for r in range(row, nrows):
                for col in range(widgetFrame.layout().columnCount()):
                    currentItem = widgetFrame.layout().itemAtPosition(r, col)
                    if currentItem is not None:
                        currentItem.widget().deleteLater()
                        widgetFrame.layout().removeItem(currentItem)
                        #widgetFrame.layout().removeWidget(currentItem.widget())
        return widgetFrame

    def clearWidget(self, widgetFrame):
        """clear everything from a widget"""
        for i in reversed(range(widgetFrame.layout().count())):
            widgetFrame.layout().itemAt(i).widget().setParent(None)

    def methodSet1(self, method='Interpolant'):
        """For 3D data"""
        methods_layout, methods_dict = {}, {'dim': '3D', 'method': method}
        if method == "Custom Equation":
            x_lineEdit = QtWidgets.QLineEdit("x")
            y_lineEdit = QtWidgets.QLineEdit("y")
            z_lineEdit = QtWidgets.QLineEdit("z")
            eqs_textEdit = QCodeEdit("""a + b * np.sin(m * pi * x * y)\n   + c * np.exp(-(w * y)**2)""")
            eqs_textEdit.setMaximumHeight(50)
            f_label = QtWidgets.QLabel("f(")
            endp_label = QtWidgets.QLabel(")")
            eql_label = QtWidgets.QLabel("=")
            eql_label2= QtWidgets.QLabel("=")
            comma_label = QtWidgets.QLabel(",")
            methods_layout[(1, 0), (1, 1)] = z_lineEdit
            methods_layout[(1, 1), (1, 1)] = eql_label
            methods_layout[(1, 2), (1, 1)] = f_label
            methods_layout[(1, 3), (1, 1)] = x_lineEdit
            methods_layout[(1, 4), (1, 1)] = comma_label
            methods_layout[(1, 5), (1, 1)] = y_lineEdit
            methods_layout[(1, 6), (1, 1)] = endp_label
            methods_layout[(2, 1), (1, 1)] = eql_label2
            methods_layout[(2, 2), (1, 6)] = eqs_textEdit
            # raise(NotImplementedError("Custom Equation currently not supported"))
        elif method == "Interpolant":
            type_label = QtWidgets.QLabel("Method:")
            type_comboBox = QtWidgets.QComboBox()
            type_comboBox.addItems(["Nearest Neighbor", "Linear", "Cubic", "Biharmonic", "Thin-plate Spline"])
            type_comboBox.setCurrentIndex(1)
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 1)] = type_label
            methods_layout[(1, 1), (1, 7)] = type_comboBox
            methods_layout[(2, 0), (1, 3)] = centerScale_checkBox
        elif method == "Lowess":
            poly_label = QtWidgets.QLabel("Polynomial:")
            poly_comboBox = QtWidgets.QComboBox()
            poly_comboBox.addItems(["Linear", "Quadratic"])
            span_label = QtWidgets.QLabel("Span:")
            span_lineEdit = QtWidgets.QLineEdit("25")
            span_pecent_label = QtWidgets.QLabel("%")
            robust_label = QtWidgets.QLabel("Robust:")
            robust_comboBox = QtWidgets.QComboBox()
            robust_comboBox.addItems(["Off", "LAR", "Bisquare"])
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 1)] = poly_label
            methods_layout[(1, 1), (1, 7)] = poly_comboBox
            methods_layout[(2, 0), (1, 1)] = span_label
            methods_layout[(2, 1), (1, 6)] = span_lineEdit
            methods_layout[(2, 7), (1, 1)] = span_pecent_label
            methods_layout[(3, 0), (1, 1)] = robust_label
            methods_layout[(3, 1), (1, 7)] = robust_comboBox
            methods_layout[(4, 0), (1, 3)] = centerScale_checkBox
        elif method == "Polynomial":
            deg_label = QtWidgets.QLabel("Degrees:")
            x_label = QtWidgets.QLabel("x:")
            x_label.setFixedWidth(8)
            x_comboBox = QtWidgets.QComboBox()
            x_comboBox.addItems(["1", "2", "3", "4", "5"])
            y_label = QtWidgets.QLabel("y:")
            y_label.setFixedWidth(8)
            y_comboBox = QtWidgets.QComboBox()
            y_comboBox.addItems(["1", "2", "3", "4", "5"])
            robust_label = QtWidgets.QLabel("Robust:")
            robust_comboBox = QtWidgets.QComboBox()
            robust_comboBox.addItems(["Off", "LAR", "Bisquare"])
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 2)] = deg_label
            methods_layout[(1, 2), (1, 1)] = x_label
            methods_layout[(1, 3), (1, 2)] = x_comboBox
            methods_layout[(1, 5), (1, 1)] = y_label
            methods_layout[(1, 6), (1, 2)] = y_comboBox
            methods_layout[(2, 0), (1, 2)] = robust_label
            methods_layout[(2, 2), (1, 6)] = robust_comboBox
            methods_layout[(3, 0), (1, 6)] = centerScale_checkBox
        else:
            raise(NotImplementedError("Unrecognized method: {}".format(method)))

        return methods_layout, methods_dict

    def methodSet2(self, method="Polynomial"):
        """For 2D data"""
        def on_centerscale_changed(isChecked):
            self.centerscale = isChecked
            self.curveFit() # reissue curve fitting

        def on_numterms_changed(index, eqs_label, text_dict, terms_dict):
            eqs_label.setText(text_dict.get(index))
            # Change the initialization parameters
            self.options.setInitializationParameters(terms_dict.get(index))
            self.curveFit() # reissue curve fitting

        methods_layout, methods_dict = {}, {'dim': '2D', 'method': method}
        if method == "Custom Equation":
            x_lineEdit = QtWidgets.QLineEdit("x")
            x_lineEdit.setMaximumWidth(30)
            y_lineEdit = QtWidgets.QLineEdit("y")
            y_lineEdit.setMaximumWidth(30)
            eqs_textEdit = QCodeEdit("""a*np.exp(-b*x)+c""")
            eqs_textEdit.setMaximumHeight(50)
            f_label = QtWidgets.QLabel("f(")
            f_label.setMaximumWidth(20)
            endp_label = QtWidgets.QLabel(")")
            eql_label = QtWidgets.QLabel("=")
            eql_label.setMaximumWidth(20)
            eql_label2 = QtWidgets.QLabel("=")
            eql_label.setMaximumWidth(20)
            self.options.setMethod(method='2D: curve_fit')
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            fitopt_Button.clicked.connect(lambda: self.options.show())
            methods_layout[(1, 0), (1, 1)] = y_lineEdit
            methods_layout[(1, 1), (1, 1)] = eql_label
            methods_layout[(1, 2), (1, 1)] = f_label
            methods_layout[(1, 3), (1, 1)] = x_lineEdit
            methods_layout[(1, 4), (1, 1)] = endp_label
            methods_layout[(2, 1), (1, 1)] = eql_label2
            methods_layout[(2, 2), (1, 6)] = eqs_textEdit
            methods_layout[(3, 7), (1, 1)] = fitopt_Button
            #raise (NotImplementedError("Custom Equation currently not supported"))
        elif method == "Exponential":
            eqs_eqs_label = QtWidgets.QLabel("Equation:")
            eqs_label = ElideQLabel("a*exp(b*x)")
            eqs_label.setMinimumWidth(200)
            numterms_label = QtWidgets.QLabel("Number of parameters:")
            numterms_comboBox = QtWidgets.QComboBox()
            numterms_comboBox.addItems(["2: a*exp(b*x)", "3: a*exp(b*x)+c", "4: a*exp(b*x) + c*exp(d*x)"])
            numterms_comboBox.currentIndexChanged.connect(lambda index: on_numterms_changed(index, eqs_label, \
                {0: "a*exp(b*x)", 1: "a*exp(b*x)+c",2: "a*exp(b*x) + c*exp(d*x)"}, \
                {0:['a','b'],1:['a','b','c'], 2:['a','b','c', 'd']}))
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            centerScale_checkBox.stateChanged.connect(lambda isChecked: on_centerscale_changed(isChecked))
            self.options.setMethod(method='2D: curve_fit')
            self.options.setInitializationParameters(coefficients=['a','b'])
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            fitopt_Button.clicked.connect(lambda: self.options.show())
            methods_layout[(1, 0), (1, 1)] = numterms_label
            methods_layout[(1, 1), (1, 7)] = numterms_comboBox
            methods_layout[(2, 0), (1, 1)] = eqs_eqs_label
            methods_layout[(2, 1), (1, 7)] = eqs_label
            methods_layout[(3, 0), (1, 7)] = centerScale_checkBox
            methods_layout[(4, 7), (1, 1)] = fitopt_Button
            methods_dict.update({'terms': eqs_label, 'center_and_scale': centerScale_checkBox})
        elif method == "Fourier": # Mixture of Fourier
            eqs_eqs_label = QtWidgets.QLabel("Equation:")
            eqs_label = ElideQLabel("a0 + a1*cos(x*w) + b1*sin(x*w)")
            eqs_label.setMinimumWidth(200)
            numterms_label = QtWidgets.QLabel("Number of parameters:")
            numterms_comboBox = QtWidgets.QComboBox()
            numterms_comboBox.addItems(["1", "2", "3", "4", "5", "6", "7", "8"])
            numterms_comboBox.currentIndexChanged.connect(lambda index:
                                                          eqs_label.setText({0: "a0 + a1*cos(x*w) + b1*sin(x*w)",
                                                                             1: "a0 + a1*cos(x*w) + b1*sin(x*w) + a2*cos(2*x*w) + b2*sin(2*x*w)",
                                                                             2: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a3*cos(3*x*w) + b3*sin(3*x*w)",
                                                                             3: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a4*cos(4*x*w) + b4*sin(4*x*w)",
                                                                             4: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a5*cos(5*x*w) + b5*sin(5*x*w)",
                                                                             5: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a6*cos(6*x*w) + b6*sin(6*x*w)",
                                                                             6: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a7*cos(7*x*w) + b7*sin(7*x*w)",
                                                                             7: "a0 + a1*cos(x*w) + b1*sin(x*w) + ... + a8*cos(8*x*w) + b8*sin(8*x*w)"
                                                                             }.get(index)))
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 1)] = numterms_label
            methods_layout[(1, 1), (1, 7)] = numterms_comboBox
            methods_layout[(2, 0), (1, 1)] = eqs_eqs_label
            methods_layout[(2, 1), (1, 7)] = eqs_label
            methods_layout[(3, 0), (1, 7)] = centerScale_checkBox
            methods_layout[(4, 7), (1, 1)] = fitopt_Button
        elif method == "Gaussian": # mixture of Gaussian
            eqs_eqs_label = QtWidgets.QLabel("Equation:")
            eqs_label = ElideQLabel("a1*exp(-((x-b1)/c1)^2")
            eqs_label.setMinimumWidth(200)
            numterms_label = QtWidgets.QLabel("Number of parameters:")
            numterms_comboBox = QtWidgets.QComboBox()
            numterms_comboBox.addItems(["1", "2", "3", "4", "5", "6", "7", "8"])
            numterms_comboBox.currentIndexChanged.connect(lambda index:
                                                          eqs_label.setText({0: "a1*exp(-((x-b1)/c1)^2",
                                                                             1: "a1*exp(-((x-b1)/c1)^2 + a2*exp(-((x-b2)/c2)^2",
                                                                             2: "a1*exp(-((x-b1)/c1)^2 + ... + a3*exp(-((x-b3)/c3)^2",
                                                                             3: "a1*exp(-((x-b1)/c1)^2 + ... + a4*exp(-((x-b4)/c4)^2",
                                                                             4: "a1*exp(-((x-b1)/c1)^2 + ... + a5*exp(-((x-b5)/c5)^2",
                                                                             5: "a1*exp(-((x-b1)/c1)^2 + ... + a6*exp(-((x-b6)/c6)^2",
                                                                             6: "a1*exp(-((x-b1)/c1)^2 + ... + a7*exp(-((x-b7)/c7)^2",
                                                                             7: "a1*exp(-((x-b1)/c1)^2 + ... + a8*exp(-((x-b8)/c8)^2",
                                                                             }.get(index)))
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 1)] = numterms_label
            methods_layout[(1, 1), (1, 7)] = numterms_comboBox
            methods_layout[(2, 0), (1, 1)] = eqs_eqs_label
            methods_layout[(2, 1), (1, 7)] = eqs_label
            methods_layout[(3, 0), (1, 7)] = centerScale_checkBox
            methods_layout[(4, 7), (1, 1)] = fitopt_Button
        elif method == "Interpolant":
            type_label = QtWidgets.QLabel("Method:")
            type_comboBox = QtWidgets.QComboBox()
            type_comboBox.addItems(["Nearest Neighbor", "Linear", "Cubic", "Shape-preserving (PCHIP)"])
            type_comboBox.setCurrentIndex(1)
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 1)] = type_label
            methods_layout[(1, 1), (1, 7)] = type_comboBox
            methods_layout[(2, 0), (1, 3)] = centerScale_checkBox
        elif method == "Linear Fitting":
            x_lineEdit = QtWidgets.QLineEdit("x")
            x_lineEdit.setMaximumWidth(30)
            y_lineEdit = QtWidgets.QLineEdit("y")
            y_lineEdit.setMaximumWidth(30)
            eqs_textEdit = QCodeEdit("""a*(sin(x-pi))+b*((x-10)^2)+c*(1)""")
            f_label = QtWidgets.QLabel("f(")
            f_label.setMaximumWidth(20)
            endp_label = QtWidgets.QLabel(")")
            eql_label = QtWidgets.QLabel("=")
            eql_label.setMaximumWidth(20)
            eql_label2 = QtWidgets.QLabel("=")
            eql_label.setMaximumWidth(20)
            edit_Button = QtWidgets.QPushButton("Edit") # pops up a window to edit a list of equations, linearly summed together
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 1)] = y_lineEdit
            methods_layout[(1, 1), (1, 1)] = eql_label
            methods_layout[(1, 2), (1, 1)] = f_label
            methods_layout[(1, 3), (1, 1)] = x_lineEdit
            methods_layout[(1, 4), (1, 1)] = endp_label
            methods_layout[(2, 1), (1, 1)] = eql_label2
            methods_layout[(2, 2), (1, 6)] = eqs_textEdit
            methods_layout[(3, 6), (1, 1)] = edit_Button
            methods_layout[(3, 7), (1, 1)] = fitopt_Button
        elif method == "Polynomial":
            deg_label = QtWidgets.QLabel("Degree:")
            deg_spinBox = QtWidgets.QSpinBox()
            deg_spinBox.setValue(1)
            deg_spinBox.setMinimum(1)
            deg_spinBox.valueChanged.connect(lambda: self.curveFit())
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 2)] = deg_label
            methods_layout[(1, 2), (1, 6)] = deg_spinBox
            methods_layout[(2, 0), (1, 6)] = centerScale_checkBox
            methods_dict.update({'degree': deg_spinBox})
        elif method == "Power":
            eqs_eqs_label = QtWidgets.QLabel("Equation:")
            eqs_label = QtWidgets.QLabel("a*x**b")
            numterms_label = QtWidgets.QLabel("Number of terms:")
            numterms_comboBox = QtWidgets.QComboBox()
            numterms_comboBox.addItems(["1", "2"])
            numterms_comboBox.currentIndexChanged.connect(lambda index:
                                    eqs_label.setText("a*x^b") if index == 0 else eqs_label.setText("a*x^b+c"))
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 1)] = numterms_label
            methods_layout[(1, 1), (1, 7)] = numterms_comboBox
            methods_layout[(2, 0), (1, 1)] = eqs_eqs_label
            methods_layout[(2, 1), (1, 6)] = eqs_label
            methods_layout[(3, 7), (1, 1)] = fitopt_Button
        elif method == "Rational":
            numdeg_label = QtWidgets.QLabel("Numerator Degree:")
            numdeg_lineEdit = QtWidgets.QLineEdit("0")
            dendeg_label = QtWidgets.QLabel("Denominator Degree:")
            dendeg_lineEdit = QtWidgets.QLineEdit("1")
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 2)] = numdeg_label
            methods_layout[(1, 2), (1, 6)] = numdeg_lineEdit
            methods_layout[(2, 0), (1, 2)] = dendeg_label
            methods_layout[(2, 2), (1, 6)] = dendeg_lineEdit
            methods_layout[(3, 0), (1, 6)] = centerScale_checkBox
            methods_layout[(4, 7), (1, 1)] = fitopt_Button
        elif method == "Smoothing Spline":
            smooth_param_label = QtWidgets.QLabel("Smoothing Parameter")
            default_radioButton = QtWidgets.QRadioButton("Default")
            specify_radioButton = QtWidgets.QRadioButton("Specify:")
            dec_button = QtWidgets.QPushButton("<")
            dec_button.setToolTip("Smoother")
            inc_button = QtWidgets.QPushButton(">")
            inc_button.setToolTip("Rougher")
            specify_lineEdit = QtWidgets.QLineEdit("0.999888")
            specify_lineEdit.setMinimumWidth(100)
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            methods_layout[(1, 0), (1, 6)] = smooth_param_label
            methods_layout[(2, 0), (1, 6)] = default_radioButton
            methods_layout[(3, 0), (1, 1)] = specify_radioButton
            methods_layout[(3, 1), (1, 1)] = dec_button
            methods_layout[(3, 2), (1, 5)] = specify_lineEdit
            methods_layout[(3, 7), (1, 1)] = inc_button
            methods_layout[(4, 0), (1, 6)] = centerScale_checkBox
        elif method == "Sum of Sine":
            numterms_label = QtWidgets.QLabel("Equation:")
            numterms_comboBox = QtWidgets.QComboBox()
            numterms_comboBox.addItems(["1", "2", "3", "4", "5", "6", "7", "8"])
            eqs_eqs_label = QtWidgets.QLabel("Equation:")
            eqs_label = ElideQLabel("a1*sin(b1*x+c1)")
            eqs_label.setMinimumWidth(200)
            numterms_comboBox.currentIndexChanged.connect(lambda index:
                                                          eqs_label.setText({0: "a1*sin(b1*x+c1)",
                                                                             1: "a1*sin(b1*x+c1) + a2*sin(b2*x+c2)",
                                                                             2: "a1*sin(b1*x+c1) + ... + a3*sin(b3*x+c3)",
                                                                             3: "a1*sin(b1*x+c1) + ... + a4*sin(b4*x+c3)",
                                                                             4: "a1*sin(b1*x+c1) + ... + a5*sin(b5*x+c3)",
                                                                             5: "a1*sin(b1*x+c1) + ... + a6*sin(b6*x+c3)",
                                                                             6: "a1*sin(b1*x+c1) + ... + a7*sin(b7*x+c3)",
                                                                             7: "a1*sin(b1*x+c1) + ... + a8*sin(b8*x+c3)",
                                                                             }.get(index)))
            centerScale_checkBox = QtWidgets.QCheckBox("Center and scale")
            centerScale_checkBox.setCheckState(self.centerscale)
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            methods_layout[(1, 0), (1, 1)] = numterms_label
            methods_layout[(1, 1), (1, 7)] = numterms_comboBox
            methods_layout[(2, 0), (1, 1)] = eqs_eqs_label
            methods_layout[(2, 1), (1, 7)] = eqs_label
            methods_layout[(3, 0), (1, 7)] = centerScale_checkBox
            methods_layout[(4, 7), (1, 1)] = fitopt_Button
        elif method == "Weibull":
            eqs_label_label = QtWidgets.QLabel("Equation:")
            eqs_label = QtWidgets.QLabel("a*b*x^(b-1)*exp(-a*x^b)")
            fitopt_Button = QtWidgets.QPushButton("Fit Options...")
            fitopt_Button.clicked.connect(lambda: self.options.show())
            self.options.setMethod(method='2D: curve_fit')
            self.options.setInitializationParameters(coefficients=['a','b'])
            methods_layout[(1, 0), (1, 1)] = eqs_label_label
            methods_layout[(1, 1), (1, 1)] = eqs_label
            methods_layout[(2, 7), (1, 1)] = fitopt_Button
            methods_dict.update({'terms': eqs_label})
        else:
            raise(NotImplementedError("Unrecognized method: {}".format(method)))
        return methods_layout, methods_dict

    def initialize_autofit(self, gbox):
        """Initialize the autofit groupBox in the tab"""
        gbox.setLayout(QtWidgets.QVBoxLayout())
        autofit_checkBox = QtWidgets.QCheckBox("Auto fit")
        autofit_checkBox.setCheckState(self.autofit)
        fit_pushButton = QtWidgets.QPushButton("Fit")
        fit_pushButton.clicked.connect(lambda: self.onFitButtonClicked())
        fit_pushButton.setEnabled(False)
        stop_pushButton = QtWidgets.QPushButton("Stop")
        stop_pushButton.clicked.connect(lambda: self.onStopButtonClicked())
        stop_pushButton.setEnabled(False)
        autofit_checkBox.stateChanged.connect(lambda checked: self.onAutoFitCheckBoxToggled(checked, fit_pushButton, stop_pushButton))

        gbox.layout().addWidget(autofit_checkBox)
        gbox.layout().addWidget(fit_pushButton)
        gbox.layout().addWidget(stop_pushButton)

    def onAutoFitCheckBoxToggled(self, checked, fit_pushButton, stop_pushButton):
        self.autofit = checked
        if checked:
            fit_pushButton.setEnabled(False)
            stop_pushButton.setEnabled(False)
        else:
            fit_pushButton.setEnabled(True)
            stop_pushButton.setEnabled(True)

    def onFitButtonClicked(self):
        pass

    def onStopButtonClicked(self):
        pass

    def initialize_results(self, gbox):
        """Initialize the results groupBox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())
        self.result_textBox = QtWidgets.QTextEdit()
        self.result_textBox.setReadOnly(True)
        gbox.layout().addWidget(self.result_textBox)

    def initialize_display(self, gbox, method='2D'):
        """Initialize the display groupBox in the tab"""
        method_comboBox = self.method_groupBox.layout().itemAt(0).widget()
        if method == '2D':
            if not isinstance(self.graphicsView, pg.graphicsItems.PlotItem.PlotItem):
                self.graphicsView = GraphicsLayoutWidget()
                self.graphicsView.setObjectName('2D')
                self.graphicsView.setBackground('w')
                self.graphicsView.addPlot(row=0, col=0) # add a plot item
                self.setMethodLists(method_comboBox, methodList=2)
        else: # 3D
            if not isinstance(self.graphicsView, pg.opengl.GLViewWidget):
                self.graphicsView = gl.GLViewWidget()
                self.graphicsView.setObjectName('3D')
                #self.graphicsView.setBackgroundColor('k')
                self.setMethodLists(method_comboBox, methodList=1)

        self.graphicsView.setObjectName(_fromUtf8("graphicsView"))
        if gbox.layout() is None:
            gbox.setLayout(QtWidgets.QGridLayout())
        self.clearWidget(gbox) # remove other widgets
        gbox.layout().addWidget(self.graphicsView)

    def graphData(self, data_list):
        """Scatter plots of data points"""
        if set(data_list) == set(['xdata']):  # 1D data
            s1 = pg.ScatterPlotItem(size=5)
            s1.addPoints(x=np.arange(0, len(self.xdata)), y=self.xdata, pen='k', brush='k')
            self.initialize_display(self.display_groupBox, method='2D')
            p = self.graphicsView.getItem(row=0, col=0)
            p.clear()
            p.showGrid(x=True, y=True)
            p.setLabels(left='{}'.format(self.varnames['xdata']), bottom='Index')
            p.addItem(s1)
            p.autoRange()
        elif set(data_list) == set(['xdata', 'ydata']):  # 2D data
            s1 = pg.ScatterPlotItem(size=5)
            s1.addPoints(x=self.xdata, y=self.ydata, pen='k', brush='k')
            self.initialize_display(self.display_groupBox, method='2D')
            p = self.graphicsView.getItem(row=0, col=0)
            p.clear()
            p.showGrid(x=True, y=True)
            p.setLabels(left='{}'.format(self.varnames['ydata']), bottom='{}'.format(self.varnames['xdata']))
            p.addItem(s1)
            p.autoRange()
        elif set(self.availableData) == set(['xdata', 'ydata', 'wdata']):  # 2D data
            s1 = pg.ScatterPlotItem(pxMode=False)  ## Set pxMode=False to allow spots to transform with the view
            spots = []
            for xx, yy, ww in zip(self.xdata, self.ydata, self.wdata):
                spots.append({'pos': (xx, yy), 'size': 5 * ww, 'pen': {'color': None},
                              'brush': (0, 0, 0, 120)})
            s1.addPoints(spots)
            self.initialize_display(self.display_groupBox, method='2D')
            p = self.graphicsView.getItem(row=0, col=0)
            p.clear()
            p.showGrid(x=True, y=True)
            p.setLabels(left='{}'.format(self.varnames['ydata']), bottom='{}'.format(self.varnames['xdata']))
            p.addItem(s1)
            p.autoRange()
        elif set(self.availableData) == set(['xdata', 'ydata', 'zdata']):  # 3D data
            # Set the graphics view to 3D
            spots = gl.GLScatterPlotItem(pos=np.c_[self.xdata, self.ydata, self.zdata],
                                         color=np.tile([1, 1, 1, 0.5], (len(self.xdata), 1)),
                                         size=5*np.ones(len(self.xdata)))
            self.initialize_display(self.display_groupBox, method='3D')
            # Remove everything
            for item in self.graphicsView.items:
                item._setView(None)
            self.graphicsView.items = []
            self.graphicsView.update()
            self.graphicsView.addItem(spots)
            self.graphicsView.addItem(gl.GLGridItem())
        elif set(self.availableData) == set(['xdata', 'ydata', 'zdata', 'wdata']):  # 3D data
            spots = gl.GLScatterPlotItem(pos=np.c_[self.xdata, self.ydata, self.zdata],
                                         color=np.tile([1, 1, 1, 0.5], (len(self.xdata), 1)),
                                         size=5*self.wdata)
            self.initialize_display(self.display_groupBox, method='3D')
            for item in self.graphicsView.items:
                item._setView(None)
            self.graphicsView.items = []
            self.graphicsView.update()
            self.graphicsView.addItem(spots)
            self.graphicsView.addItem(gl.GLGridItem())
        else:
            return  # do nothing. Data not expected

    def graphFit(self, f0=None, popt=None):
        """Plot the fitted function"""
        if self.params['dim'] == '2D':
            p = self.graphicsView.getItem(row=0, col=0)
            # check if a fit already exist. If so, remove it
            for k, a in enumerate(p.listDataItems()):
                if a.name() == 'fit': # matching
                    # Remove the trace
                    p.removeItem(a)

        if f0 is None: return
        if self.params['dim'] == '2D':
            p_xrange, p_yrange = p.viewRange()
            x0_fit = np.linspace(p_xrange[0], p_xrange[1], 100)
            y0_fit = f0(x0_fit, *popt)
            # Replot the fit
            cl = p.plot(x=x0_fit, y=y0_fit, pen='r', name='fit')
            # Make sure to set the original view back
            p.setXRange(p_xrange[0], p_xrange[1], padding=0)
            p.setYRange(p_yrange[0], p_yrange[1], padding=0)
        else: # 3D
            self.graphicsView # 3D GL

    def outputResultText(self, model):
        if 'final_text' in model.keys():
            self.result_textBox.setText(model['final_text'])
            return

        final_text = """
        {}:
            f(x) = {}
        """.format(model['type'], model['formula'])

        # Coefficients
        coef_format = "    {} = {:.3g} ({:.3g}, {:.3g})\n        "   # var = mean (lower, upper)
        coef_text = """
        Coefficeints 
        (95% confidence interval):
        """
        for n, kk in enumerate(model['ci']):
            coef_text = coef_text + coef_format.format(kk['name'], kk['mean'], kk['lower'], kk['upper'])
        final_text = final_text + coef_text

        final_text = final_text + """
        Goodness of fit:
            SSE: {:.5f}
            RMSE: {:.5f}
            R-square: {:.5f}
            Adjusted R-square: {:.5f}
        """.format(model['SSE'], model['RMSE'],  model['rsquare'], model['adjrsquare'])
        self.result_textBox.setText(final_text)

    def whatTab(self):
        """For reference"""
        self.currentTab = self.tabWidget.currentWidget()
        pass

    def closeTab(self, currentIndex):
        if self.tabWidget.count() < 2:
            return  # Do not close the last tab
        currentQWidget = self.tabWidget.widget(currentIndex)
        currentQWidget.deleteLater()
        self.tabWidget.removeTab(currentIndex)

    def setMenuBarItems(self):
        # <editor-fold desc="File menu">
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.actionClear_Session = QtWidgets.QAction(self)
        self.actionClear_Session.setObjectName("actionClear_Session")
        self.actionClear_Session.setText(_translate("MainWindow", "Clear Session"))
        self.menuFile.addAction(self.actionClear_Session)

        self.actionLoad_Session = QtWidgets.QAction(self)
        self.actionLoad_Session.setObjectName("actionLoad_Session")
        self.actionLoad_Session.setText(_translate("MainWindow", "Load Session..."))
        self.menuFile.addAction(self.actionLoad_Session)

        self.actionSave_Session = QtWidgets.QAction(self)
        self.actionSave_Session.setObjectName("actionSave_Session")
        self.actionSave_Session.setText(_translate("MainWindow", "Save Session"))
        self.menuFile.addAction(self.actionSave_Session)

        self.actionSave_Session_As = QtWidgets.QAction(self)
        self.actionSave_Session_As.setObjectName("actionSave_Session_As")
        self.actionSave_Session_As.setText(_translate("MainWindow", "Save Session As..."))
        self.menuFile.addAction(self.actionSave_Session_As)

        self.actionGenerate_Code = QtWidgets.QAction(self)
        self.actionGenerate_Code.setObjectName("actionGenerate_Code")
        self.actionGenerate_Code.setText(_translate("MainWindow", "Generate Code"))
        self.menuFile.addAction(self.actionGenerate_Code)

        self.actionPrint_to_Figure = QtWidgets.QAction(self)
        self.actionPrint_to_Figure.setObjectName("actionPrint_to_Figure")
        self.actionPrint_to_Figure.setText(_translate("MainWindow", "Print to Figure"))
        self.menuFile.addAction(self.actionPrint_to_Figure)

        self.menuFile.addSeparator()
        self.actionClose_Curve_Fitting = QtWidgets.QAction(self)
        self.actionClose_Curve_Fitting.setObjectName("actionClose_Curve_Fitting")
        self.menuFile.addAction(self.actionClose_Curve_Fitting)
        self.actionClose_Curve_Fitting.setText(_translate("MainWindow", "Close Curve Fitting"))
        # </editor-fold>

        # <editor-fold desc="Fit menu">
        self.menuFit = QtWidgets.QMenu(self.menubar)
        self.menuFit.setObjectName("menuFit")

        self.actionNew_Fit = QtWidgets.QAction(self)
        self.actionNew_Fit.setObjectName("actionNew_Fit")
        self.actionNew_Fit.setText(_translate("MainWindow", "New Fit"))
        self.actionNew_Fit.triggered.connect(lambda: self.newFit())
        self.menuFit.addAction(self.actionNew_Fit)

        self.actionOpen_Fit = QtWidgets.QAction(self)
        self.actionOpen_Fit.setObjectName("actionOpen_Fit")
        self.actionOpen_Fit.setText(_translate("MainWindow", "Open Fit"))
        self.menuFit.addAction(self.actionOpen_Fit)

        self.actionClose = QtWidgets.QAction(self)
        self.actionClose.setObjectName("actionClose")
        self.actionClose.setText(_translate("MainWindow", "Close \"\""))
        self.menuFit.addAction(self.actionClose)

        self.actionDelete = QtWidgets.QAction(self)
        self.actionDelete.setObjectName("actionDelete")
        self.actionDelete.setText(_translate("MainWindow", "Delete \"\""))
        self.menuFit.addAction(self.actionDelete)

        self.actionDuplicate = QtWidgets.QAction(self)
        self.actionDuplicate.setObjectName("actionDuplicate")
        self.actionDuplicate.setText(_translate("MainWindow", "Duplicate \"\""))
        self.menuFit.addAction(self.actionDuplicate)
        # </editor-fold>

        # <editor-fold desc="View menu">
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")

        self.actionFit_Settings = QtWidgets.QAction(self)
        self.actionFit_Settings.setObjectName("actionFit_Settings")
        self.actionFit_Settings.setText(_translate("MainWindow", "Fit Settings"))
        self.menuView.addAction(self.actionFit_Settings)

        self.actionFit_Results = QtWidgets.QAction(self)
        self.actionFit_Results.setObjectName("actionFit_Results")
        self.actionFit_Results.setText(_translate("MainWindow", "Fit Results"))
        self.menuView.addAction(self.actionFit_Results)

        self.menuView.addSeparator()
        self.actionMain_Plot = QtWidgets.QAction(self)
        self.actionMain_Plot.setObjectName("actionMain_Plot")
        self.actionMain_Plot.setText(_translate("MainWindow", "Main Plot"))
        self.menuView.addAction(self.actionMain_Plot)

        self.actionResidual_Plot = QtWidgets.QAction(self)
        self.actionResidual_Plot.setObjectName("actionResidual_Plot")
        self.actionResidual_Plot.setText(_translate("MainWindow", "Residual Plot"))
        self.menuView.addAction(self.actionResidual_Plot)

        self.actionContour_Plot = QtWidgets.QAction(self)
        self.actionContour_Plot.setObjectName("actionContour_Plot")
        self.actionContour_Plot.setText(_translate("MainWindow", "Contour Plot"))
        self.menuView.addAction(self.actionContour_Plot)

        self.menuView.addSeparator()
        self.actionTable_of_Fits = QtWidgets.QAction(self)
        self.actionTable_of_Fits.setObjectName("actionTable_of_Fits")
        self.actionTable_of_Fits.setText(_translate("MainWindow", "Table of Fits"))
        self.menuView.addAction(self.actionTable_of_Fits)
        # </editor-fold>

        # <editor-fold desc="Tools menu">
        self.menuTools = QtWidgets.QMenu(self.menubar)
        self.menuTools.setObjectName("menuTools")

        self.actionZoom_In = QtWidgets.QAction(self)
        self.actionZoom_In.setObjectName("actionZoom_In")
        self.menuTools.addAction(self.actionZoom_In)
        self.actionZoom_In.setText(_translate("MainWindow", "Zoom In"))

        self.actionZoom_Out = QtWidgets.QAction(self)
        self.actionZoom_Out.setObjectName("actionZoom_Out")
        self.actionZoom_Out.setText(_translate("MainWindow", "Zoom Out"))
        self.menuTools.addAction(self.actionZoom_Out)

        self.actionPan = QtWidgets.QAction(self)
        self.actionPan.setObjectName("actionPan")
        self.actionPan.setText(_translate("MainWindow", "Pan"))
        self.menuTools.addAction(self.actionPan)

        self.actionData_Cursor = QtWidgets.QAction(self)
        self.actionData_Cursor.setObjectName("actionData_Cursor")
        self.actionData_Cursor.setText(_translate("MainWindow", "Data Cursor"))
        self.menuTools.addAction(self.actionData_Cursor)

        self.menuTools.addSeparator()
        self.actionLegend = QtWidgets.QAction(self)
        self.actionLegend.setObjectName("actionLegend")
        self.actionLegend.setText(_translate("MainWindow", "Legend"))
        self.menuTools.addAction(self.actionLegend)

        self.actionGrid = QtWidgets.QAction(self)
        self.actionGrid.setObjectName("actionGrid")
        self.actionGrid.setText(_translate("MainWindow", "Grid"))
        self.menuTools.addAction(self.actionGrid)

        self.menuTools.addSeparator()
        self.actionPrediction_Bounds = QtWidgets.QAction(self)
        self.actionPrediction_Bounds.setObjectName("actionPrediction_Bounds")
        self.actionPrediction_Bounds.setText(_translate("MainWindow", "Prediction Bounds"))
        self.menuTools.addAction(self.actionPrediction_Bounds)

        self.menuTools.addSeparator()
        self.actionAxes_Limits = QtWidgets.QAction(self)
        self.actionAxes_Limits.setObjectName("actionAxes_Limits")
        self.actionAxes_Limits.setText(_translate("MainWindow", "Axes Limits"))
        self.menuTools.addAction(self.actionAxes_Limits)

        self.actionExclude_By_Rule = QtWidgets.QAction(self)
        self.actionExclude_By_Rule.setObjectName("actionExclude_By_Rule")
        self.actionExclude_By_Rule.setText(_translate("MainWindow", "Exclude By Rule"))
        self.menuTools.addAction(self.actionExclude_By_Rule)
        # </editor-fold>

        # <editor-fold desc="Add actions">
        self.menubar.addAction(self.menuFile.menuAction())
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menubar.addAction(self.menuFit.menuAction())
        self.menuFit.setTitle(_translate("MainWindow", "Fit"))
        self.menubar.addAction(self.menuView.menuAction())
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menubar.addAction(self.menuTools.menuAction())
        self.menuTools.setTitle(_translate("MainWindow", "Tools"))
        # </editor-fold>

    def filterVars(self, vars_list):
        """Determine what to data variables to load"""
        vars_final = {}
        for kk in vars_list:
            for key, value in kk.items():
                if isinstance(value, np.ndarray):
                    vars_final[key] = value
                elif isinstance(value,  (list, tuple)):
                    vars_final[key] = np.array(value)
        return vars_final

    def getParams(self):
        for key, widget in self.methods.items():
            if isinstance(widget, str):
                self.params[key] = widget
            elif isinstance(widget, QtWidgets.QLineEdit):
                self.params[key] = str2numericHandleError(widget.text())
            elif isinstance(widget, QtWidgets.QComboBox):
                self.params[key] = str2numericHandleError(widget.currentText())
            elif isinstance(widget, QtWidgets.QLabel):
                self.params[key] = str2numericHandleError(widget.text())
            elif isinstance(widget, ElideQLabel):
                self.params[key] = str2numericHandleError(widget.text())
            elif isinstance(widget, QCodeEdit):
                self.params[key] = widget.text()
            elif isinstance(widget, QtWidgets.QCheckBox):
                self.params[key] = True if widget.checkState() > 0 else False
            elif isinstance(widget, QtWidgets.QSpinBox):
                self.params[key] = widget.value()
            else:
                raise(TypeError('Unrecognized type of item'))

    def curveFit(self):
        """Interface the curve fitting"""
        self.getParams() # get parameters from the current method
        self.options.getParams() # get parameters if there is fit options

        # Preprocessing
        if self.centerscale:
            xdata0 = (self.xdata - np.mean(self.xdata)) / np.std(self.xdata)
            ydata0 = (self.ydata - np.mean(self.ydata)) / np.std(self.ydata)
            zdata0 = (self.zdata - np.mean(self.zdata)) / np.std(self.zdata)
            wdata0 = self.wdata
        else:
            xdata0, ydata0, zdata0, wdata0 = self.xdata, self.ydata, self.zdata, self.wdata

        # Initialize a dict for result outputs
        model = {'type': self.params['dim'] + ": " + self.params['method']}
        # Fit data case by case
        print(self.params['dim'])
        if self.params['dim'] == '2D':
            #if ~np.isnan(wdata0):
                #ydata0 = ydata0 * wdata0
            if self.params['method'] in ['Exponential', 'Weibull']:
                popt, pcov, f0, fitted_params = self.fitExponential(xdata0, ydata0, terms=self.params['terms'])
                self.graphFit(f0, popt) # Plot the fit
                model.update({'formula': self.params['terms']}) # Output fitting results to the result box
            elif self.params['method'] == 'Polynomial':
                popt, pcov = np.polyfit(xdata0, ydata0, deg=self.params['degree'],
                                        w=None if np.isnan(wdata0) else wdata0, cov=True)
                f0 = lambda x, *args: np.polyval(args, x)
                self.graphFit(f0, popt) # Plot the fit
                model['type'] = model['type'] + ' (deg={:d})'.format(self.params['degree'])
                # Make the polynomial string
                model['formula'] = {1: 'p0 + p1 * x', 2: 'p0 + p1 * x + p2 * x^2'}.get(self.params['degree'],
                                    'p0 + p1 * x + ... + p{0} * x^{0}'.format(self.params['degree']))
                # Fitted params
                fitted_params = ['p{:d}'.format(deg) for deg in range(self.params['degree']+1)]
            else:
                return
        else: # 3D
            return

        if pcov is not None:
            # Calculate goodness of fit
            gof = goodness_of_fit(xdata0, ydata0, popt, pcov, f0)
            model.update(gof)
            # Calcualte confidence interval
            ci_list = confidence_interval(ydata0, popt, pcov, alpha=0.05, parameter_names=fitted_params)
            model['ci'] = ci_list
            if self.params['method'] == 'Polynomial': pass

            self.outputResultText(model)
        else:
            self.outputResultText({'final_text': str(popt)})

    def fitExponential(self, x0, y0, terms='a*exp(b*x)+c'):
        if (isnumber(x0) and np.isnan(x0)) or (isnumber(y0) and np.isnan(y0)):
            return "Need at least 2 dimensional data to fit", None, None, None

        if terms == 'a*exp(b*x)':
            f0 = lambda x, a, b: a * np.exp(b * x)
            fitted_params = ['a', 'b']
            fitted_params = ['a', 'b']
        elif terms == 'a*exp(b*x)+c':
            f0 = lambda x, a, b, c: a * np.exp(b * x) + c
            fitted_params = ['a', 'b', 'c']
        elif terms == 'a*exp(b*x) + c*exp(d*x)':
            f0 = lambda x, a, b, c, d: a * np.exp(b * x) + c * np.exp(d * x)
            fitted_params = ['a', 'b', 'c', 'd']
        elif terms == 'a*b*x^(b-1)*exp(-a*x^b)':  # Weibull
            f0 = lambda x, a, b: a * b * x ** (b - 1) * np.exp(-a * x ** b)
            fitted_params = ['a', 'b']

        params = self.options.params
        p0 = [params['coefficients'][l][0] for l in fitted_params]
        bounds0 = (
        [params['coefficients'][l][1] for l in fitted_params], [params['coefficients'][l][2] for l in fitted_params])
        try:
            popt, pcov = curve_fit(f0, x0, y0, p0=p0, bounds=bounds0,
                                   method={'Trust-Region Reflective': 'trf', 'Levenberg-Marquardt': 'lm',
                                           'Dog-Box': 'dogbox'}.get(params['algorithm']),
                                   max_nfev=params['maxfev'], ftol=params['ftol'], loss=params['loss'],
                                   xtol=params['xtol'], gtol=params['gtol'])
            return popt, pcov, f0, fitted_params
        except Exception as err:
            print(err)
            # set_trace()
            return err, None, None, None

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))


def cftool(xdata=None, ydata=None, zdata=None, wdata=None):
    """
    Wrapper function to call cftool GUI
    :param xdata: x data
    :param ydata: y data
    :param zdata: z data (if 3D)
    :param wdata: weights
    :return:
    """
    pass

if __name__ == "__main__":
    np.random.seed(42)
    f0 = lambda x, a, b: a * b * x ** (b - 1) * np.exp(-a * x ** b)
    Xdata = np.random.randn(1000)/100 + np.arange(0, 10, 0.01)+1
    Ydata = np.random.randn(1000)/100 + f0(Xdata, 0.2, 0.5)  #5 * np.exp(-0.2*Xdata)+1
    Zdata = np.random.randn(1000)
    Wdata = np.random.randn(1000)*5
    X_small = np.array([1, 2, 3, 4])
    sys.excepthook = my_excepthook   # helps prevent uncaught exception crashing the GUI
    app = QtWidgets.QApplication(sys.argv)
    w = cftool_MainWindow()
    w.show()

    sys.exit(app.exec_())