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


__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))
__version__ = "PySynapse 0.4"

def my_excepthook(type, value, tback):
    """This helps prevent program crashing upon an uncaught exception"""
    sys.__excepthook__(type, value, tback)


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
        self.xdata = None
        self.ydata = None
        self.zdata = None
        self.wdata = None
        self.autofit = 2
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

        # Data
        data_groupBox = QtWidgets.QGroupBox(tab)
        data_groupBox.setTitle("")
        self.initialize_data(data_groupBox)
        gridLayoutFits.addWidget(data_groupBox, 0, 0, 1, 4)

        # Method
        method_groupBox = QtWidgets.QGroupBox(tab)
        method_groupBox.setTitle("")
        self.initialize_method(method_groupBox)
        gridLayoutFits.addWidget(method_groupBox, 0, 4, 1, 8)

        # AutoFit
        autofit_groupBox = QtWidgets.QGroupBox(tab)
        autofit_groupBox.setTitle("")
        self.initialize_autofit(autofit_groupBox)
        gridLayoutFits.addWidget(autofit_groupBox, 0, 12, 1, 2)

        # Results
        results_groupBox = QtWidgets.QGroupBox(tab)
        results_groupBox.setTitle(_translate("MainWindow", "Results"))
        self.initialize_results(results_groupBox)
        gridLayoutFits.addWidget(results_groupBox, 1, 0, 1, 3)

        # Display
        display_groupBox = QtWidgets.QGroupBox(tab)
        display_groupBox.setTitle("")
        self.initialize_display(display_groupBox)
        gridLayoutFits.addWidget(display_groupBox, 1, 3, 1, 11)

        return gridLayoutFits

    def initialize_data(self, gbox):
        """Initialize the data groupbox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())
        fitname_label = QtWidgets.QLabel("Fit Name")
        fitname_text  = QtWidgets.QLineEdit()
        fitName = self.tabWidget.tabText(self.tabWidget.indexOf(self.tabWidget.currentWidget()))
        fitname_text.setText(fitName)
        fitname_text.editingFinished.connect(lambda: self.changeTabTitle(fitname_text.text()))

        comboList = ['(none)']+list(self.vars.keys())
        x_label = QtWidgets.QLabel("X data")
        x_comboBox  = QtWidgets.QComboBox()
        x_comboBox.addItems(comboList)
        x_comboBox.currentIndexChanged.connect(lambda: self.setData('xdata', x_comboBox.currentText()))

        y_label = QtWidgets.QLabel("Y data")
        y_comboBox  = QtWidgets.QComboBox()
        y_comboBox.addItems(comboList)
        y_comboBox.currentIndexChanged.connect(lambda: self.setData('ydata', y_comboBox.currentText()))

        z_label = QtWidgets.QLabel("Z data:")
        z_comboBox  = QtWidgets.QComboBox()
        z_comboBox.addItems(comboList)
        z_comboBox.currentIndexChanged.connect(lambda: self.setData('zdata', z_comboBox.currentText()))

        w_label = QtWidgets.QLabel("Weights:")
        w_comboBox  = QtWidgets.QComboBox()
        w_comboBox.addItems(comboList)
        w_comboBox.currentIndexChanged.connect(lambda: self.setData('wdata', w_comboBox.currentText()))

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

    def setData(self, key, valname):
        """Change data"""
        if valname == '(none)':
            setattr(self, key, None)
        else:
            setattr(self, key, self.vars[valname])

    def initialize_method(self, gbox):
        """Initialize the method groupBox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())

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

    def initialize_display(self, gbox):
        """Initialize the display groupBox in the tab"""
        gbox.setLayout(QtWidgets.QGridLayout())


    def setDisplayText(self, result_textBox, model={}):
        # parse coefficients

        final_text = """
        {}:
            f(x) = {}
        """#.format(model['type'], model['formula'])

        # Coefficients
        coef_format = "{} = {} ({}, {})"   # var = mean (lower, upper)
        final_text = final_text + """
        Coefficeints (with 95% confidence bounds):
            {}
        """#.format()

        final_text = final_text + """
        Goodness of fit:
            SSE: {}
            RMSE: {}
            R-square: {}
            Adjusted R-square: {}
        """#.format(model['SSE'], model['RMSE'],  model['R-square'], model['Adjusted R-Square'])
        result_textBox.setText(final_text)

    def whatTab(self):
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
        vars_final = {}
        for kk in vars_list:
            for key, value in kk.items():
                if isinstance(value, np.ndarray):
                    vars_final[key] = value
                elif isinstance(value,  (list, tuple)):
                    vars_final[key] = np.array(value)
        return vars_final


    def curveFit(self):
        print("Not implemented")

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
    X = np.arange(0, 100, 0.1) + np.random.randn(1000)/100
    Y = 5 + np.arange(0, 100, 0.1) + np.random.randn(1000)/100
    sys.excepthook = my_excepthook   # helps prevent uncaught exception crashing the GUI
    app = QtWidgets.QApplication(sys.argv)
    w = cftool_MainWindow()
    w.show()


    sys.exit(app.exec_())